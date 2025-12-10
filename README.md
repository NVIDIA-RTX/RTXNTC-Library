# RTX Neural Texture Compression Library (LibNTC)

This repository (or folder, depending on how you got here) contains the source code for the NTC library.

LibNTC can be built separately following the same instructions as the full [RTX Neural Texture Compression SDK](https://github.com/NVIDIA-RTX/RTXNTC), and does not depend on anything else in the SDK. This means it can be included into a larger project by copying only this folder or adding a submodule.

For the integration guide, please see the RTX Neural Texture Compression SDK.

## CMake Configuration options

- `NTC_BUILD_SHARED`: Configures whether LibNTC should be built as a static library (`OFF`) or a dynamic one (`ON`). Default is `ON`.
- `NTC_WITH_CUDA`: Enables the CUDA-based functionality like compression. Set this to `OFF` to build a compact version of LibNTC for integration into game engines.
- `NTC_CUDA_ARCHITECTURES`: List of CUDA architectures in the [CMake format](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html) for which the kernels should be compiled. A reduced list can make builds faster for development purposes.
- `NTC_WITH_DX12`: Enables building the DX12 shaders and weight conversion features (Windows only). Default is `ON`.
- `NTC_WITH_VULKAN`: Enables building the Vulkan shaders and weight conversion features. Default is `ON`.
- `NTC_WITH_PREBUILT_SHADERS`: Enables building the shaders for decompression on load, BCn compression, and image difference passes. Default is `ON`.
- `NTC_DEBUG_SHADERS`: Adds debug symbols into the prebuilt shaders, useful for profiling. Default is `OFF`.
- `LIBNTC_BIN_DIRECTORY`: Location for the build LibNTC binaries on all build configurations. When unset, CMake default locations are used.

## Directory structure

- [`external/dxc`](external/dxc): Binary builds of a regular recent version of the DirectX Shader Compiler
- [`external/slang`](external/slang): Binary builds of the [Slang compiler](https://github.com/shader-slang/slang) and a [custom version of DXC](https://github.com/NVIDIA-RTX/DirectXShaderCompiler/tree/CooperativeVector) with Cooperative Vector support
- [`include/libntc`](include/libntc): Public C++ headers for LibNTC
- [`include/libntc/shaders`](include/libntc/shaders): HLSL/Slang shader headers for things like Inference on Sample
- [`src`](src): Source code for LibNTC
- [`src/RegressionKernels.h`](src/RegressionKernels.h) Source code for the CUDA compression kernels
- [`src/Inference.cu`](src/Inference.cu) Source code for the CUDA decompression (inference) kernel
- [`src/shaders`](src/shaders): Source code for the prebuilt shaders used by LibNTC, such as decompression or BCn encoding
- [`src/tin`](src/tin): Source code for the Tiny Inline Networks (TIN) library, customized for NTC

## Decoder configuration

Since version 0.8.0, LibNTC is built supporting only one version of the image decoder network (multi-layer perceptron, MLP). There are multiple possible decoder configurations, and the default one is a balanced choice for high inference performance and good compression quality. Generally, more and larger MLP layers result in higher quality and lower performance, and it is left up to integrations of NTC to decide whether to use the default configuration or choose a different one. To adjust the configuration, edit the `NTC_MLP_LAYERS` and `NTC_MLP_HIDDEN*_CHANNELS` constants in [`include/libntc/shaders/InferenceConstants.h`](include/libntc/shaders/InferenceConstants.h).

It is convenient to describe a decoder configuration as 2 or 3 numbers, indicating the sizes of the hidden layers. The default configuration is `64-48-32`. The configuration used in LibNTC versions before 0.9.0 is `64-64-64`. Another good option is `48-48-32` which provides about 10-20% higher inference performance at the cost of about 7% worse compression (-0.5 dB PSNR).

## License

[NVIDIA RTX SDKs LICENSE](LICENSE.txt)
