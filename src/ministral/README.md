# Native Ministral 3 support

`VindexLLM.Model.Ministral3.pas` implements the text-only Ministral 3 3B
forward path for `general.architecture = mistral3`.  `dense.comp` contains its
Vulkan compute kernel; `Ministral.res` embeds the compiled `build/dense.spv`.

The implementation was derived from the tensor layout, YaRN, attention-scale,
and Tekken tokenization behavior in `ggml-org/llama.cpp`, revision fetched on
2026-09-14.  The upstream project is MIT licensed:
https://github.com/ggml-org/llama.cpp

To rebuild the embedded shader, compile `dense.comp` to `build/dense.spv` with
`glslangValidator -V`, then run `brcc32 -foMinistral.res Ministral.rc` from
this directory.  The resource is loaded by the Pascal unit at runtime.

`MinistralKernelTest.dpr` plus `test_kernels.py` provide independent numerical
checks for the supported GGUF weight layouts and the transformer primitives.
`MinistralSmoke.dpr` is an end-to-end loader and generation smoke test.
