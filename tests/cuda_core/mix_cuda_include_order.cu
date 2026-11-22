// Resident decoders can declare strix::cuda::core before including the mix.
#include "cuda/core/head_argmax.cuh"
#include "cuda/core/mix_cuda.cuh"

#include <type_traits>

static_assert(std::is_same_v<decltype(&::strix::cuda::mix::enqueue),
    cudaError_t (*)(const ::strix::core::residual_mix::MixArgs&, cudaStream_t)>);
