# Native Qwen3.5-0.8B (text)

Para definições dos termos de modelos, quantização, GPU e inferência usados
nesta documentação, consulte o [dicionário de termos](../MODEL_GLOSSARY.md).

`VindexLLM.Model.Qwen35.pas` registers `qwen35` through `VindexLLM.Inference`.
The VCL application and console demo default to `D:\Qwen3.5-0.8B.gguf`.
Inference runs in Delphi and embedded Vulkan compute shaders; no llama.cpp DLL,
server, Python runtime, or external tokenizer is needed by the application.

Implemented for this model: 24 decoder blocks (18 gated delta net / 6 full
attention), causal convolution, persistent recurrent state, Q/K normalization,
partial NeoX RoPE for text positions, attention gating, SwiGLU, tied embeddings,
Q4_K/Q6_K weights, Qwen35 byte-BPE with NFC, and ChatML with thinking disabled.
The optional 25th MTP block is not used for ordinary autoregressive decoding.
Other Qwen sizes and multimodal inputs are not supported by this implementation.

Prefill currently processes one token at a time, prioritizing correctness of
recurrent state over throughput. The default context is 4096; memory depends
on context size. Cache snapshots use format 2 with padded F32 slots for both
attention and SSM state. Starting again at position zero resets recurrent
history; saving/restoring includes both convolution and delta-rule state.

## Build

From a Delphi Win64 command environment:

```powershell
.\build.ps1
# Optional shader rebuild (the compiled Qwen35.res is included):
.\build.ps1 -RebuildShader -Glslang C:\path\to\glslang.exe
```

### glslang and shaders

`glslangValidator` is the reference compiler for GLSL shaders. In this project
it compiles `dense.comp`, the Vulkan GPU compute shader, into `build/dense.spv`
(SPIR-V), which is then embedded in `Qwen35.res` for the Delphi application.

It is **not required to run** the already compiled application: the repository
includes the compiled shader resource. It is only required when changing
`dense.comp` and rebuilding the shader. Pass the full path to
`glslangValidator.exe` with `-Glslang`, as in the command above.

The application is written to `../../VCL/Win64/Debug/Project1.exe`.
Open the VCL project, click **Carregar**, enter a prompt and click **Gerar**.
Each generation starts a fresh prompt; UI streaming is marshalled to the main
thread and model unload waits for the generation worker.

## Validation

The test programs are built by `build.ps1`. Python dependencies are only needed
for numerical development tests (`numpy`, `gguf`, `tokenizers`).

```powershell
python -m venv build/venv
.\build\venv\Scripts\python.exe -m pip install numpy gguf tokenizers
.\build\venv\Scripts\python.exe test_kernels.py
.\build\Qwen35ForwardTest.exe D:\Qwen3.5-0.8B.gguf build/forward.bin
.\build\venv\Scripts\python.exe test_forward.py D:\Qwen3.5-0.8B.gguf
.\build\Qwen35StateTest.exe D:\Qwen3.5-0.8B.gguf build/state.kvc
.\build\Qwen35Smoke.exe D:\Qwen3.5-0.8B.gguf "What is the capital of France?" 12
```

For tokenizer parity, download the publisher's tokenizer into
`reference/tokenizer.json` and run `test_tokenizer.py` (defaults to the model
path above). Tests cover combining accents/NFC, Portuguese, emoji, Chinese,
whitespace, contractions, digits and special tokens.

Validated on Delphi 37 Win64 and Intel HD Graphics 530: 12 numerical kernel
checks; all 24 layer outputs over 3 recurrent steps versus independent NumPy;
deterministic generation after reset and snapshot restore. The forward
comparison had about 3e-6 relative error at the final layer. These checks verify
implementation accuracy, not factual accuracy of model answers.

## References

Architecture and tensor layout were checked against the upstream sources on
2026-09-22. The new SSM shader operations are a direct implementation of the
recurrence, tested independently with NumPy; dense operations are adapted from
this project's existing native Ministral shader.

- https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp (MIT)
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py (Apache-2.0)
- https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/tokenizer.json
