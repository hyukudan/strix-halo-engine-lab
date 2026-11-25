// R1 only. Q/K heads and V heads own disjoint convolution channels. The
// recurrent value tiles consume shared Q/K heads, and gated RMSNorm consumes
// every value tile: both edges retain their grid barriers and original CTAs.
__global__ __launch_bounds__(kRecurrentThreads)
void decode_prepare_kernel(
    float* qkv, std::uint16_t* conv_state, const void* conv_weights, SmallWeightKind conv_kind,
    float* beta, float* gate, const void* dt, SmallWeightKind dt_kind, const void* a, SmallWeightKind a_kind,
    std::uint32_t key_heads, std::uint32_t value_heads, std::uint32_t head_dim,
    std::uint32_t kernel_size, float epsilon) {
    const unsigned head = blockIdx.x;
    const unsigned channel = head * head_dim + threadIdx.x;
    core::causal_conv::ring_update<GdnBf16, GdnSmallLoad>(qkv, conv_state, conv_weights, conv_kind, kernel_size, channel);
    if (head < 2U * key_heads) {
        __shared__ float scratch[kRecurrentThreads];
        __syncthreads();
        // Keep head_dim a runtime argument, as in normalize_qk_kernel: folding
        // rsqrtf(128) at compile time could differ from the device instruction.
        normalize_qk_body(qkv, key_heads, head_dim, epsilon, nullptr, head, scratch);
    } else if (head < 2U * key_heads + value_heads && threadIdx.x == 0) {
        ops::recur::gated_delta_gate_body<GdnGate, GdnBf16, GdnSmallLoad>(beta, gate, dt, dt_kind, a, a_kind, head - 2U * key_heads);
    }
}

// Guarded graphs must finish every convolution before a normalization failure
// can publish graph_status. Convolution and beta/gate have no such dependency;
// their disjoint work shares a launch, retaining the original rejection rules.
__global__ __launch_bounds__(kRecurrentThreads)
void decode_conv_gate_kernel(
    float* qkv, std::uint16_t* conv_state, const void* conv_weights, SmallWeightKind conv_kind,
    float* beta, float* gate, const void* dt, SmallWeightKind dt_kind, const void* a, SmallWeightKind a_kind,
    std::uint32_t channels, std::uint32_t heads, std::uint32_t kernel_size, std::uint32_t* graph_status) {
    const unsigned conv_blocks = (channels + kRecurrentThreads - 1U) / kRecurrentThreads;
    if (blockIdx.x < conv_blocks) {
        if (graph_status != nullptr && *graph_status != 0) return;
        const unsigned channel = blockIdx.x * kRecurrentThreads + threadIdx.x;
        if (channel < channels) core::causal_conv::ring_update<GdnBf16, GdnSmallLoad>(qkv, conv_state, conv_weights, conv_kind, kernel_size, channel);
    } else {
        const unsigned head = (blockIdx.x - conv_blocks) * kRecurrentThreads + threadIdx.x;
        if (head < heads) ops::recur::gated_delta_gate_body<GdnGate, GdnBf16, GdnSmallLoad>(beta, gate, dt, dt_kind, a, a_kind, head);
    }
}
