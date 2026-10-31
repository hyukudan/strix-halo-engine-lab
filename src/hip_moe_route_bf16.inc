// Shared only by the DA16 storage epilogue and its matching ordered merge readers.
template<bool BF16>
__device__ __forceinline__ float moe_route_load(const float* base, std::uint64_t index) {
    if constexpr (BF16)
        return __uint_as_float(static_cast<unsigned>(reinterpret_cast<const std::uint16_t*>(base)[index]) << 16U);
    else return base[index];
}
template<bool BF16>
__device__ __forceinline__ float moe_route_store(float* base, std::uint64_t index, float dot) {
    if constexpr (BF16) {
        const auto bits = strix::moe_route_bf16_bits(__float_as_uint(dot));
        reinterpret_cast<std::uint16_t*>(base)[index] = bits;
        return __uint_as_float(static_cast<unsigned>(bits) << 16U);
    } else { base[index] = dot; return dot; }
}
