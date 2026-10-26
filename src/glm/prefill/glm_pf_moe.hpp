#pragma once
// GLM-5.3-Flash prefill MoE (one rank, gfx1151): every routed expert of a T-row chunk in a fixed number of grouped
// launches (port of the MiniMax v4 grouped core, 3e1897a1 src/minimax/moe/mmx_prefill*).
//
// Identity (bitwise), per call:
//   for each expert e with routed rows (rows in ascending token order, X_e = those rows of x):
//     g = strix_gemm_pinned(gate_e, X_e, splits = 1); u = strix_gemm_pinned(up_e, X_e, 1)
//     a = dev::swiglu(g, u) (R15);                     y_e = strix_gemm_pinned(down_e, a, 1)
//   eo[t][k] = y_{ids[t][k]}[row of t];  out = dev::moe_combine(weights, eo, shared) (R16, route order k = 0..7)
// with strix-trellis (e7f801d). Experts without routed rows are skipped. No GEMV path anywhere: every
// row of every expert takes the same grouped WMMA chain, so a row's bits never depend on the group size.
//
// Launches (one per stage over all groups; GEMM stages one per row-tile class MT = 1, 2, 4, 8):
//   gather + input transform (gate and up suh, tile-major f16) -> gate/up grouped WMMA (f32 partials, one split)
//   -> finish (output transform of gate and up, R15 SwiGLU, down input transform, tile-major f16)
//   -> down grouped WMMA -> combine (down output transform per route + R16 + shared).
// Routes come from the host (one D2H per chunk by the caller). The call uploads its tables with hipMemcpyAsync and
// synchronizes the stream once before the first launch (the host tables die with the call), as the MiniMax core does.
#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

#include "strix_trellis.h"

namespace strix::glm::pf {

inline constexpr unsigned kMoeExperts = 288, kMoeTop = 8, kMoeHidden = 4096, kMoeInter = 1024;  // per rank (I slice)

// One EXL3 matrix [N outputs][K inputs], mul1 codebook, as prepared by strix_weights_prepare_parts (bytes verbatim):
//   trellis: [K/16][N/16][Bits * 8] uint32 (16 x 16 tile (k, n) at word (k * N/16 + n) * Bits * 8), 4-byte aligned;
//   suh: [K] f16 bits; svh: [N] f16 bits. Device pointers.
struct MoeExl3 {
    const std::uint32_t* trellis = nullptr;
    const std::uint16_t* suh = nullptr;
    const std::uint16_t* svh = nullptr;
};
struct MoeExpert {
    MoeExl3 gate, up;  // N = 1024, K = 4096
    MoeExl3 down;      // N = 4096, K = 1024
};

struct PrefillMoeView {
    std::uint32_t rows = 0;                // T
    const float* x = nullptr;              // [T][4096] device (the normed hidden rows, fp32)
    const std::uint32_t* ids = nullptr;    // [T][8] host; ids < 288 (the router writes them ascending per row)
    const float* weights = nullptr;        // [T][8] host, same order as ids
    const MoeExpert* experts = nullptr;    // host array [288]; only routed experts are read (must be non-null)
    unsigned bits_gu = 4, bits_dn = 4;     // trellis bits of gate/up and of down (3, 4, 5 or 6)
    const float* shared = nullptr;         // [T][4096] device, shared-expert output (computed separately)
    float* out = nullptr;                  // [T][4096] device (may alias neither x nor shared)
    float* eo_debug = nullptr;             // optional [T][8][4096] device: y_e of every route (= dev::moe_combine's eo)
};

// Device workspace of one call (depends on the routes: groups are padded to 16 rows).
std::size_t prefill_moe_workspace_bytes(const PrefillMoeView& v);
// Upper bound for any routes of T rows.
std::size_t prefill_moe_workspace_bound(std::uint32_t rows);
// Throws std::invalid_argument / std::runtime_error (fail closed) on bad views, small workspace or HIP errors.
// workspace: device, 256-byte aligned.
void enqueue_prefill_moe_glm(const PrefillMoeView& v, void* workspace, std::size_t workspace_bytes, hipStream_t stream);

// The three device pointers of a prepared strix_weights handle of an N x K matrix at `bits` (mul1, no bias), from
// strix_weights_device_span and the library's allocation layout (api.hip prepare_regions: trellis, suh, svh at
// STRIX_TRELLIS_WORKSPACE_ALIGNMENT-aligned offsets in that order). Throws if the span size does not match exactly.
MoeExl3 moe_exl3_from_weights(strix_weights w, unsigned N, unsigned K, unsigned bits);

// Profiling only: `marker(label, stream, user)` after "start" (tables uploaded, before the first launch) and after each
// stage ("gather", "gate_up", "finish", "down", "combine"); nullptr = off.
using PrefillMoeMarker = void (*)(const char* label, hipStream_t stream, void* user);
void set_prefill_moe_marker(PrefillMoeMarker marker, void* user);

}  // namespace strix::glm::pf
