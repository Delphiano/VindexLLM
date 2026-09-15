# Divisão de pesos entre RAM e GPU

Gemma 3 e Ministral 3 usam `TVdxWeightPager` automaticamente no carregamento.
Não é necessário alterar a chamada a `TVdxInference.LoadModel`.

O orçamento padrão é 75% do maior heap Vulkan DEVICE_LOCAL. O cálculo desconta
as alocações já realizadas (embeddings, normas, cache KV e buffers de trabalho),
um staging buffer e, quando necessário, sete buffers reutilizáveis para uma
camada. As primeiras camadas que couberem ficam residentes. Os demais pesos
permanecem no arquivo GGUF mapeado pelo processo; o Windows gerencia suas páginas
em RAM, sem exigir uma segunda cópia integral do modelo.

Antes de cada camada não residente, os pesos Q/K/V/O e gate/up/down passam pelo
staging buffer para os respectivos buffers compartilhados. Cópias e dispatches
são síncronos nesse modo, com fences e barreiras Vulkan, para que nenhum peso ou
descriptor seja reutilizado enquanto a GPU ainda o lê. Todos os cálculos continuam
nos shaders existentes; não há execução de camadas na CPU.

O Gemma de texto com contexto explícito usa prefill de até 32 tokens por bloco.
O cache mantém o contexto configurado. Gemma Embedding e o carregamento pelo
módulo Embeddings (contexto nativo, `AMaxContext=0`) preservam o prefill completo
para atenção bidirecional.

## Ajuste opcional

`VINDEXLLM_GPU_BUDGET_MB` permite reduzir o orçamento, em MiB, antes de carregar o
modelo. Por exemplo, `1200` limita o plano a 1200 MiB. O valor nunca aumenta o teto
padrão de 75%. Isso é útil quando outros aplicativos também usam a GPU.

O log informa quantas camadas ficam residentes e quantas são transferidas.
O orçamento é uma estimativa conservadora, não uma consulta dinâmica à memória
livre do driver. Falhas Vulkan continuam interrompendo o carregamento com diagnóstico.

## Limites

- Cache KV, embeddings e buffers de cálculo ainda precisam caber na GPU, junto
  com uma camada temporária. Se não couberem, reduza o contexto ou o modelo.
- Transferir pesos por token pode reduzir bastante a velocidade, especialmente
  em GPUs conectadas por PCIe com pouca VRAM.
- O GGUF deve permanecer disponível e inalterado enquanto o modelo estiver aberto.
- Os handles retornados pelo pager são emprestados; somente o pager os destrói.
- Não foram executados testes de inferência ou desempenho para esta implementação.
- O projeto VdxTestbed foi compilado em Win64; o executável não foi iniciado.
