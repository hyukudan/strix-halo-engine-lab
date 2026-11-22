#pragma once
// Low-rank controller output for one element (shared by the pointwise controller and the gated projection epilogue,
// so both store the same bits): out = (1 - tanh(forget)) * current + tanh(gain) * message, each operation rounded.
// Includers bring their device runtime header first (shared headers stay SDK-free).
#if defined(__HIPCC__) || defined(__CUDACC__)

namespace strix::core {

__device__ __forceinline__ float controller_mix(float current, float message, float forget, float gain) {
    const float retained = __fmul_rn(__fsub_rn(1.0F, tanhf(forget)), current);
    const float incoming = __fmul_rn(tanhf(gain), message);
    return __fadd_rn(retained, incoming);
}

}  // namespace strix::core
#endif
