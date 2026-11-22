// Included beside mma_dot. Each CTA owns one eight-output half of the
// incumbent m16 tile, including its original MMA fragment slots. All eight
// warps still own the same K32 blocks and issue the same lo/mid/hi MMA chain.
__global__ void __launch_bounds__(kBf16NarrowThreads) mma3_r1_output_split(
    const void* weights, std::uint32_t in, const float* input, float* output) {
    __shared__ float partial[8][kBf16NarrowOutputs];
    pdl_trigger();
    const auto first = blockIdx.x * kBf16NarrowOutputs;
    const bool upper = (blockIdx.x & 1U) != 0U;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, qd = lane & 3;
    float accumulator[1][4]{};
    pdl_wait();
    mma_dot<WeightFormat::bf16, 1, 3, 4, true>(
        weights, in, first, input, 1U, warp, g, qd, accumulator, nullptr, lane, upper);
    // Only the first activation column is live. It is register 0 or 2 in
    // the quad's first lane, depending on the unchanged output fragment half.
    if (qd == 0) partial[warp][g] = accumulator[0][upper ? 2 : 0];
    __syncthreads();
    if (threadIdx.x < kBf16NarrowOutputs) {
        const unsigned o = threadIdx.x;
        float value = partial[0][o];
#pragma unroll
        for (int w = 1; w < 8; ++w) value = __fadd_rn(value, partial[w][o]);
        output[first + o] = attn::group_indexer::bf16_round(value);
    }
}
