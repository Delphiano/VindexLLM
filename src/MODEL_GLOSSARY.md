# Dicionário de termos de modelos

Este glossário explica os termos usados na documentação e no código do
VindexLLM, com foco em modelos de linguagem executados localmente.

## Arquivos e pesos

| Termo | Significado |
| --- | --- |
| **Modelo** | Conjunto de pesos e metadados que permite ao software prever o próximo token de um texto. |
| **Pesos** | Números aprendidos durante o treinamento; determinam o comportamento do modelo. |
| **Checkpoint** | Arquivo ou conjunto de arquivos que armazena os pesos de um modelo treinado. |
| **GGUF** | Formato de arquivo usado para armazenar pesos, tokenizer e metadados de modelos, otimizado para inferência local. |
| **Metadados** | Informações no GGUF que descrevem a arquitetura, dimensões, tokenizer, contexto e quantização. |
| **Tensor** | Arranjo multidimensional de números. Pesos, ativações e caches são tensores. |
| **Embedding** | Vetor numérico que representa um token. A tabela `token_embd.weight` transforma IDs de tokens em vetores. |
| **Logits** | Pontuações brutas do modelo para cada token possível antes da amostragem. |
| **Unembedding** | Projeção final que transforma o estado interno do modelo em logits. |
| **MTP** | *Multi-Token Prediction*: bloco opcional para prever vários tokens. A inferência autoregressiva comum não o utiliza. |

## Texto e geração

| Termo | Significado |
| --- | --- |
| **LLM** | *Large Language Model*, modelo de linguagem que prevê sequências de tokens. |
| **Inferência** | Uso do modelo já treinado para processar um prompt e gerar uma resposta. |
| **Prompt** | Texto de entrada fornecido ao modelo. |
| **Token** | Unidade de texto processada pelo modelo; pode ser uma palavra, parte de palavra, pontuação ou byte. |
| **Tokenização** | Conversão entre texto e IDs de tokens. |
| **Tokenizer** | Regras e vocabulário usados na tokenização. |
| **Vocabulário** | Lista de todos os tokens que o modelo conhece, cada um com um ID. |
| **Byte-BPE** | Tokenizador que constrói tokens a partir de bytes e fusões frequentes; é usado pela família Qwen. |
| **NFC** | Normalização Unicode que torna formas equivalentes de texto consistentes, por exemplo caracteres acentuados compostos. |
| **ChatML** | Formato de prompt com marcadores de papel, como sistema, usuário e assistente. |
| **Geração autoregressiva** | Geração de um token por vez, reutilizando os tokens anteriores como contexto. |
| **Prefill** | Processamento inicial de todos os tokens do prompt antes de começar a gerar novos tokens. |
| **Decodificação** | Etapa repetida que produz cada novo token após o prefill. |
| **Token de parada** | Token que encerra a geração, como fim de texto ou fim de turno. |
| **Amostragem** | Escolha do próximo token a partir dos logits. |
| **Temperatura** | Controla aleatoriedade na amostragem: menor valor favorece respostas mais previsíveis. |
| **Top-k** | Limita a escolha aos `k` tokens mais prováveis. |
| **Top-p** | Limita a escolha ao menor conjunto de tokens cuja probabilidade acumulada atinge `p`. |
| **Seed** | Valor inicial de aleatoriedade; com as mesmas condições ajuda a reproduzir uma geração. |

## Arquitetura neural

| Termo | Significado |
| --- | --- |
| **Camada / bloco** | Unidade repetida da rede que transforma as representações dos tokens. |
| **Decoder causal** | Arquitetura que, ao prever um token, só pode usar tokens anteriores. |
| **Estado oculto** | Vetor de trabalho que carrega a representação de um token entre camadas. |
| **Dimensão oculta** | Número de valores no estado oculto; no Qwen3.5-0.8B suportado é 1024. |
| **FFN** | *Feed-Forward Network*: sub-rede por token que amplia e transforma o estado oculto. |
| **SwiGLU** | Função de ativação usada na FFN que combina uma porta e uma projeção. |
| **Residual** | Soma da entrada de uma subcamada à sua saída, preservando informação durante a rede. |
| **RMSNorm** | Normalização baseada na média quadrática, aplicada para estabilizar os cálculos. |
| **Atenção** | Mecanismo que pondera quais posições anteriores são relevantes para a posição atual. |
| **Cabeça de atenção** | Parte independente do cálculo de atenção; várias cabeças capturam relações diferentes. |
| **Q, K e V** | *Query*, *Key* e *Value*: projeções usadas para calcular atenção. Queries consultam, keys indexam e values fornecem conteúdo. |
| **KV cache** | Cache de keys e values de tokens anteriores, evitando recalcular atenção a cada novo token. |
| **RoPE** | *Rotary Position Embedding*: técnica que incorpora a posição do token nas projeções de atenção. |
| **NeoX RoPE** | Variante de ordenação dos pares usada por alguns modelos com RoPE. |
| **SSM** | *State Space Model*: camada recorrente que mantém estado em vez de comparar diretamente todas as posições. |
| **Estado recorrente** | Estado persistente das camadas SSM; deve avançar na ordem dos tokens. |
| **Convolução causal** | Operação que usa apenas valores atuais e anteriores, preservando a causalidade. |
| **Qwen3.5 híbrido** | Arquitetura que combina blocos SSM e blocos de atenção completa. |

## Contexto, memória e desempenho

| Termo | Significado |
| --- | --- |
| **Contexto / janela de contexto** | Máximo de tokens que o modelo pode considerar em uma sessão. |
| **Posição** | Índice de um token dentro do contexto. |
| **Cache de contexto** | Memória que guarda dados de tokens anteriores, como KV cache e estado SSM. |
| **VRAM** | Memória da placa de vídeo, usada para pesos, buffers e cache durante a inferência. |
| **RAM** | Memória principal do computador. O GGUF pode ser mapeado nela pelo sistema operacional. |
| **Memory mapping** | Mapeamento do arquivo GGUF na memória virtual sem carregar todo o arquivo de uma vez. |
| **Buffer** | Região de memória, normalmente na GPU, usada para armazenar tensores intermediários. |
| **Paginação de pesos** | Estratégia que mantém parte dos pesos no arquivo mapeado e envia dados à GPU quando necessário. |
| **Prefill em lote** | Processamento de vários tokens do prompt de uma vez. O caminho Qwen3.5 usa passos unitários para preservar o estado SSM. |

## Quantização

| Termo | Significado |
| --- | --- |
| **Quantização** | Armazenamento aproximado dos pesos com menos bits para reduzir RAM, VRAM e tamanho do arquivo. |
| **Dequantização** | Reconstrução aproximada de valores numéricos a partir dos pesos quantizados durante o cálculo. |
| **GGML** | Convenções de tipos e formatos de tensores usados nos arquivos GGUF. |
| **F32** | Número de ponto flutuante de 32 bits; maior precisão e maior consumo de memória. |
| **F16** | Número de ponto flutuante de 16 bits; ocupa metade do F32. |
| **Q4_K** | Quantização GGML com aproximadamente 4 bits por peso e blocos de 256 valores. |
| **Q5_K** | Variante com aproximadamente 5 bits por peso; pode aparecer em arquivos quantizados de forma mista. |
| **Q6_K** | Variante com aproximadamente 6 bits por peso, geralmente mais precisa e maior. |
| **Q4_K_M** | Perfil de quantização misto: usa principalmente Q4_K, mas pode usar outros tipos, como Q5_K, em tensores selecionados. |
| **Tipo de tensor** | Formato numérico de um tensor no GGUF, como F32, F16, Q4_K ou Q5_K. |

## GPU, Vulkan e shaders

| Termo | Significado |
| --- | --- |
| **GPU** | Processador especializado em executar muitos cálculos em paralelo. |
| **Vulkan** | API gráfica e de computação usada pelo VindexLLM para executar cálculos na GPU. |
| **Shader** | Pequeno programa executado na GPU. No projeto, implementa operações de inferência paralela. |
| **Compute shader** | Shader voltado para cálculos gerais, sem desenhar gráficos. `dense.comp` é um compute shader. |
| **GLSL** | Linguagem de programação usada para escrever shaders. |
| **glslangValidator** | Compilador de referência que converte GLSL em SPIR-V. Só é necessário ao recompilar shaders. |
| **SPIR-V** | Formato binário intermediário aceito por drivers Vulkan. O shader compilado é `dense.spv`. |
| **Recurso Delphi (`.res`)** | Arquivo que incorpora o SPIR-V no executável. Neste projeto, é `Qwen35.res`. |
| **Pipeline de computação** | Configuração Vulkan que associa o shader, buffers e parâmetros para executar uma operação na GPU. |
| **Dispatch** | Comando que inicia um conjunto de execuções paralelas de um compute shader. |
| **Workgroup** | Grupo de execuções do shader que pode colaborar por memória compartilhada. |

## Termos específicos deste projeto

| Termo | Significado |
| --- | --- |
| **VindexLLM** | Implementação local de inferência em Delphi com computação Vulkan. |
| **`TVdxModel`** | Classe base que abre o GGUF, seleciona a arquitetura e coordena o carregamento. |
| **`TVdxQwen35Model`** | Implementação nativa da arquitetura Qwen3.5-0.8B de texto. |
| **`TVdxGGUFReader`** | Leitor do GGUF; interpreta metadados e fornece acesso aos tensores mapeados. |
| **`dense.comp`** | Código GLSL do shader de computação do caminho Qwen3.5. |
| **`dense.spv`** | Versão SPIR-V compilada de `dense.comp`. |
| **`Qwen35.res`** | Recurso Delphi que embute `dense.spv` no executável. |
| **`Qwen35Smoke.exe`** | Programa de teste rápido para abrir um GGUF e gerar uma resposta. |
