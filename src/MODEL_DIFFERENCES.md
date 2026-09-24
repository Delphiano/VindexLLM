# Diferenças de modelo implementadas

Este documento resume as tecnologias e variações de arquitetura que o VindexLLM
ja trata ao carregar modelos GGUF. O ponto de entrada e `TVdxModel.LoadModel`,
que le `general.architecture`, resolve uma classe concreta pelo
`TVdxModelRegistry` e delega para a implementacao especifica.

## Arquiteturas registradas

As classes de modelo se registram durante a inicializacao das units:

- `llama`: `TVdxLlamaModel` em `VindexLLM.Model.Llama.pas`.
- `gemma3`: `TVdxGemma3Model` em `VindexLLM.Model.Gemma3.pas`.
- `gemma-embedding`: mesmo caminho de Gemma 3, com atencao bidirecional para
  embeddings.
- `mistral3`: `TVdxMinistral3Model` em `VindexLLM.Model.Ministral3.pas`.
- `qwen35`: `TVdxQwen35Model` em `VindexLLM.Model.Qwen35.pas`.

Cada implementacao le metadados no prefixo da sua familia (`llama.*`,
`gemma3.*`, `mistral3.*`, `qwen35.*`) e valida dimensoes, numero de camadas,
cabecas de atencao, tamanho de contexto e tensores obrigatorios antes de alocar
pesos na GPU.

## Tokenizacao

`VindexLLM.Tokenizer.pas` implementa dois caminhos principais:

- Byte-BPE estilo GPT-2 quando `tokenizer.ggml.model = gpt2`.
- BPE/SPM-like por scores quando o modelo GGUF nao usa `gpt2`.

No caminho byte-BPE, os pre-tokenizadores aceitos sao:

- `tekken`: usado por Ministral/Mistral, com regex Unicode propria e
  comportamento equivalente a `ignore_merges` para pre-tokens ja presentes no
  vocabulario.
- `qwen2`: regex no estilo Qwen2.
- `qwen35`: regex adaptada para Qwen 3.5.

O tokenizer tambem faz matching guloso de tokens especiais, de controle e
user-defined antes de processar texto comum. IDs de BOS/EOS sao lidos de
`tokenizer.ggml.bos_token_id` e `tokenizer.ggml.eos_token_id`.

## Atencao e posicao

O nucleo comum de atencao esta em `VindexLLM.Attention.pas`. Ele cobre:

- Q/K/V separados ou caminho fundido para QKV em pesos `Q4_0`.
- GQA/MQA por meio de `attention.head_count` e `attention.head_count_kv`.
- RoPE para Q e K no decode de token unico e no prefill em batch.
- Q/K norm opcional quando os tensores `attn_q_norm.weight` e
  `attn_k_norm.weight` existem.
- Atencao causal para modelos decoder-only.
- Atencao bidirecional em caminhos de embedding, especialmente
  `gemma-embedding`.

Gemma 3 pode variar a base de RoPE por camada via `GetRoPETheta`. Ministral 3
usa um caminho proprio com YaRN: a implementacao exige `rope.scaling.type =
yarn` e usa metadados como `rope.scaling.factor`,
`rope.scaling.original_context_length`, `yarn_beta_fast`, `yarn_beta_slow` e
`attention.temperature_scale`.

## Normalizacao

`VindexLLM.LayerNorm.pas` fornece RMSNorm em variantes de uso comum:

- in-place;
- copy + norm;
- add + norm;
- batch;
- pesos de norma por ramo de atencao e FFN.

Gemma 3 usa mais pesos de normalizacao por camada, incluindo normas pos-atencao,
pos-FFN e Q/K norm. LLaMA usa RMSNorm antes de atencao e FFN, sem Q/K norm.
Ministral 3 e Qwen 3.5 usam kernels densos proprios, mas tambem validam pesos
de norma F32.

## FFN e ativacoes

O projeto trabalha com o layout comum `gate`, `up` e `down`, mas a ativacao e o
caminho de execucao variam por familia:

- LLaMA usa SwiGLU/SiLU multiplicativo.
- Gemma 3 usa gate com GELU multiplicado por `up`.
- Ministral 3 usa seu shader `MINISTRAL_DENSE`, com operacoes densas
  especificas para a arquitetura.
- Qwen 3.5 usa `QWEN35_DENSE`, combinando camadas de atencao e camadas SSM.

Ha pipelines fundidos para reduzir dispatches em casos especificos, como
`gate + up + ativacao * mul` e QKV fundido quando o tipo de peso permite.

## Qwen 3.5 hibrido

`TVdxQwen35Model` implementa o caminho nativo para Qwen3.5 0.8B texto. A
validacao espera 24 camadas decodificadoras, com possivel camada MTP ignorada.
O modelo e hibrido: o log descreve 18 camadas SSM e 6 camadas de atencao.

Os metadados SSM validados incluem:

- `ssm.conv_kernel`;
- `ssm.state_size`;
- `ssm.group_count`;
- `ssm.time_step_rank`;
- `ssm.inner_size`;
- `full_attention_interval`.

Por causa do estado recorrente, o prefill desse caminho usa bloco de 1 token:
o estado avanca estritamente em ordem.

## Cache KV e contexto

O modelo base expoe uma superficie comum para cache:

- `CacheFormat`;
- `CacheBytesPerLayer`;
- `CacheBuffer`;
- `PrefillBatchSize`.

LLaMA, Gemma e Ministral usam cache KV por camada. Ministral usa cache F32 no
caminho nativo atual. Qwen 3.5 reserva slots uniformes para preservar o formato
de snapshot existente, mesmo misturando atencao e SSM.

O tamanho de contexto e sempre limitado pelo menor valor entre o contexto nativo
do GGUF e o `AMaxContext` solicitado. Algumas implementacoes tambem aplicam
limites praticos; Ministral e Qwen 3.5 restringem o contexto nativo atual a
1..65535.

## Quantizacao e tensores GGUF

O leitor GGUF e os kernels aceitam varios tipos GGML, conforme o caminho de
modelo:

- `F32`;
- `F16`;
- `Q4_0`;
- `Q4_1`;
- `Q8_0`;
- `Q3_K`;
- `Q4_K`;
- `Q6_K`.

Nem todo modelo aceita todos os tipos em todos os tensores. Por exemplo, alguns
caminhos so usam kernels fundidos em `Q4_0`, embeddings tem uma lista propria
de tipos aceitos, e os pesos de normalizacao normalmente devem ser F32.

## Weight paging

`TVdxWeightPager` permite dividir pesos entre GPU e arquivo GGUF mapeado. LLaMA,
Gemma, Ministral e Qwen 3.5 usam o pager para pesos de bloco (`Q/K/V/O` e
`gate/up/down`) quando necessario.

O orcamento padrao usa uma fracao conservadora da VRAM Vulkan disponivel e pode
ser reduzido por `VINDEXLLM_GPU_BUDGET_MB`. Pesos residentes ficam em buffers de
GPU; camadas nao residentes sao copiadas por staging antes do dispatch.

## Kernels Vulkan especializados

O projeto ja contem pipelines Vulkan para:

- embedding single-token e batch;
- RMSNorm single-token e batch;
- RoPE single-token e batch;
- Q/K norm;
- matvec/matmul;
- QKV fundido;
- FFN fundido em casos de `Q4_0`;
- unembed/logits;
- kernels densos especificos para Ministral 3 e Qwen 3.5.

Ministral e Qwen 3.5 embutem seus SPIR-V via resources (`Ministral.res` e
`Qwen35.res`), gerados a partir dos respectivos `dense.comp`.

## Templates e uso de alto nivel

Cada familia pode sobrescrever:

- `FormatPrompt`;
- `FormatEmbedding`;
- `GetStopTokenStrings`;
- `SupportsEmbedding`.

Isso separa diferencas de chat template, tokens de parada e modo de embedding
da execucao numerica do transformer.

## Limites atuais

- `mistral3` valida especificamente o Ministral 3 3B texto.
- `qwen35` valida especificamente Qwen3.5 0.8B texto.
- Camadas extras, adapters, biases ou tensores inesperados podem ser recusados.
- Algumas otimizacoes sao condicionais ao tipo de peso, especialmente `Q4_0`.
- O suporte de tokenizer byte-BPE aceita somente `tekken`, `qwen2` e `qwen35`.
- O caminho de SSM do Qwen 3.5 nao faz prefill em blocos grandes porque o estado
  recorrente precisa avancar token a token.
