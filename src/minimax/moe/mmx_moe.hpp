#pragma once
// minimax-m3 decode MoE and dense EXL3 linears for one TP2 rank (gfx1151), N = 1..8 rows.
//
// Identity (bitwise): per row, the same bits as the bring-up chain of tests/hip_minimax_g4.hip at <= 8 rows:
//   router      strix/minimax_moe.hpp route_row (FP32 logits, lane l chains j = l, l + 32, ...; butterfly; sigmoid +
//               bias choice; top-4 by (choice desc, id asc); w = s / (sum + 1e-20) x 2);
//   experts     strix_gemv (strix-trellis 0.3, the decode GEMV; not strix_gemm, which takes WMMA) for gate (w1) and up (w3) [6144 -> 1536 rank slice, K3], swiglu
//               (min(g, 7) sigmoid(1.702 min(g, 7)) (clamp(u, +-7) + 1)), strix_gemv down (w2) [1536 -> 6144];
//   combine     acc = 0; acc = fmaf(w_e, y_e, acc) over the row's 4 experts in ascending expert id; acc + shared;
//   shared      the same MLP with the shared expert (K5), added unscaled;
//   linears     strix_gemv (input transform, split GEMV core, output transform) for the dense sites.
// Every row's bits depend only on that row (batch invariance N = 1..8): the per-row chains, split boundaries and
// reduction orders never depend on the other rows of the call.
//
// Launch structure of one MoE layer (enqueue_moe, defaults; no host synchronisation):
//   1. router + shared expert gate/up in one launch (128 router CTAs, the last one does top-4 + dispatch table;
//      the other CTAs stream the shared expert's gate/up meanwhile)          STRIX_MMX_MOE_OVERLAP=0 disables;
//   2. routed gate/up: the library GEMV grid (CTA = 4 tile columns x 1 split) over all active slots, split input
//      transform in LDS, raw split partials; the last CTA per block does H, svh, SwiGLU and the down transform;
//   3. down, same grid; the last CTA per output block does H, svh and the fixed-order combine (+ shared).
// Flags (all paths bitwise): STRIX_MMX_MOE_WIDE unset = wide at N2-4 only (measurement), 1 = wide at every N (expert
// gate/up/down: CTA = 16 columns, wave = 4 columns
// streaming 16-byte k-row runs; 0 = 4-column CTAs), STRIX_MMX_MOE_V 4 (1/3 older forms), STRIX_MMX_MOE_DEPTH3 4 (8/12),
// STRIX_MMX_MOE_W 4 (8), STRIX_MMX_MOE_DEPTH8R 0 (2), STRIX_MMX_ROUTER 2 (1 = 16-CTA form; used when the
// post-exchange is fused, h != nullptr), STRIX_MMX_LINEAR_V unset = a measured per-site / row-count schedule of
// v4 / v8 / v9 (mmx_moe.hip scheduled_linear_version; 1, 3..9 force one form). Not for the head: the library's strix_gemv
// is faster there (it already runs at ~230 of the ~247 GB/s read ceiling).
#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

namespace strix::minimax::moe::opt {

inline constexpr unsigned kHidden = 6144, kInter = 1536 /* per rank */, kExperts = 128, kTopE = 4, kMaxRows = 8;
inline constexpr unsigned kMaxSlots = kMaxRows * kTopE;  // routed slots (distinct experts of a call)
inline constexpr unsigned kSharedSlot = kMaxSlots;        // slot index of the shared expert
inline constexpr unsigned kMaxPairs = kMaxSlots + kMaxRows;  // (slot, row) pairs: routed <= 32, shared <= 8

// One EXL3 matrix (rank slice) resident on the device, in the strix-trellis device layout (= file layout):
// trellis [input / 16][output / 16][bits * 8] uint32 words, suh [input] FP16, svh [output] FP16.
struct Exl3 {
    const std::uint32_t* trellis = nullptr;
    const std::uint16_t* suh = nullptr;
    const std::uint16_t* svh = nullptr;
    std::uint32_t input = 0, output = 0, bits = 0;
};
struct ExpertSlices {  // w1 (gate), w3 (up): 6144 -> 1536; w2 (down): 1536 -> 6144
    Exl3 gate, up, down;
};

// Written by the router kernel, read by the later MoE kernels (device memory).
struct Dispatch {
    std::uint32_t rows;                     // N of the call
    std::uint32_t count;                    // distinct routed experts (slots 0..count-1, ascending expert id)
    std::uint32_t expert[kMaxSlots];
    std::uint32_t nrows[kMaxSlots + 1];     // rows per slot (slot kSharedSlot = all rows)
    std::uint32_t first[kMaxSlots + 1];     // first (slot, row) pair index of the slot
    std::uint32_t row[kMaxSlots][kMaxRows]; // call row of each local row, ascending
    std::uint32_t row_pair[kMaxRows][kTopE];  // per call row: its 4 pairs in ascending expert id
    float row_weight[kMaxRows][kTopE];        // and their routing weights
    std::uint32_t row_expert[kMaxRows][kTopE];  // and their expert ids
    std::uint32_t ids[kMaxRows][kTopE];       // route_row output (choice order)
    float weights[kMaxRows][kTopE];
};

struct MoeWorkspace {  // one device allocation of moe_workspace_bytes(), zeroed once before first use
    void* base = nullptr;
};
std::size_t moe_workspace_bytes();

// decode (STRIX_MM_XB_PUBLISH): the producer kernel publishes its own exchange message (lean form): every lane releases
// its stores at agent scope, each CTA counts on `counter` (agent scope), and only the last CTA of the grid resets the
// counter, releases at system scope and stores `sequence` into the host-visible `flag`. flag == nullptr: no publish
// (today's code). Only the CTA that finishes each 128-output tile counts (tiles arrivals per message). Accepted by the decode forms the engine pins (producer_publish_supported()), else enqueue throws.
struct PublishTarget {
    std::uint64_t* flag = nullptr;
    unsigned* counter = nullptr;
    std::uint64_t sequence = 0;
    unsigned tiles = 0;  // set by the enqueue functions: output tiles whose finishers count (callers leave 0)
};

struct MoeView {
    std::uint32_t rows = 0;               // 1..8
    const float* y = nullptr;             // rows x 6144, normed hidden (post-exchange output; written when h is set)
    // Optional fused post-exchange (residual + Gemma norm of strix/minimax_moe.hpp post_exchange_row, same bits): when
    // h is set, the router computes h += part0 + part1 and y itself (one launch instead of post-exchange + router).
    float* h = nullptr;                   // rows x 6144 residual, updated in place
    const float* part0 = nullptr;         // rank 0's partial rows
    const float* part1 = nullptr;         // rank 1's partial rows
    const float* norm_w = nullptr;        // 6144 Gemma norm weights
    const float* router_w = nullptr;      // 128 x 6144 FP32 (replicated)
    const float* bias = nullptr;          // 128 score-correction bias
    const ExpertSlices* experts = nullptr;  // device array [128] (routed experts, K3)
    ExpertSlices shared{};                // shared expert slices (K5)
    float* out = nullptr;                 // rows x 6144: routed sum + shared (this rank's partial, to exchange)
    std::uint32_t* ids = nullptr;         // optional rows x 4 (route_row order)
    float* weights = nullptr;             // optional rows x 4
    MoeWorkspace ws{};
    PublishTarget pub{};                  // optional: the down/combine launch publishes `out` (STRIX_MM_XB_PUBLISH)
};
// Throws std::invalid_argument on unsupported shapes (bits other than 3/5 for experts/shared, rows outside 1..8).
void enqueue_moe(const MoeView& v, hipStream_t stream);
// Pieces of enqueue_moe (for the microbenchmark and for a caller that overlaps them with other work).
void enqueue_router(const MoeView& v, hipStream_t stream);
// prepare + gate_up + down; marks (optional, 2 events): recorded after prepare and after gate_up (profiling).
void enqueue_experts(const MoeView& v, hipStream_t stream, hipEvent_t* marks = nullptr);
const Dispatch* moe_dispatch(const MoeWorkspace& ws);        // device pointer

// ---------------------------------------------------------------- router for many rows (prefill chunks)
// Same bits as route_row / the fused post-exchange router per row (lane chains, butterfly, sigmoid + bias, top-4,
// weights), computed as a tiled pass: one wave per expert over 32 rows per CTA (router weights read once per 32 rows
// instead of once per row), then one thread per row for the top-4. scratch: rows x 128 x 2 floats.
void enqueue_router_rows(unsigned rows, const float* y, const float* router_w, const float* bias, float* scratch,
                         std::uint32_t* ids, float* weights, hipStream_t stream);

// ---------------------------------------------------------------- dense linears (attention projections, shared /
// dense MLP halves, head slice): y[r][0..output) = strix_gemm(x[r][0..input)) for r < rows <= 8, several matrices
// with their own inputs and outputs in one launch pair (prepare + core with the output transform in the last CTA of
// each 128-output block).
struct LinearJob {
    Exl3 w{};
    const float* x = nullptr;
    std::uint32_t x_stride = 0;  // floats between rows
    float* y = nullptr;
    std::uint32_t y_stride = 0;
};
inline constexpr unsigned kMaxJobs = 4;
std::size_t linear_workspace_bytes(const LinearJob* jobs, unsigned count);  // for 8 rows
// marks (optional, 1 event, profiling): recorded after the prepare launch of the two-launch forms (v1/v4).
void enqueue_linears(const LinearJob* jobs, unsigned count, unsigned rows, void* workspace, hipStream_t stream,
                     hipEvent_t* marks = nullptr);
// The same with the measured per-shape / N schedule regardless of STRIX_MMX_LINEAR_V (which is read once per process
// and pinned by the engine for its own sites): for callers such as the DSpark drafter. Same output bits.
void enqueue_linears_scheduled(const LinearJob* jobs, unsigned count, unsigned rows, void* workspace, hipStream_t stream);
// decode (STRIX_MM_XB_PREP): the input transform of the two-launch forms written by the producer of x instead of the
// prepare launch. linear_xt_offsets gives, per job, the offset in halves (_Float16) of its transformed input inside the
// workspace (row r of job i at workspace + offs[i] + r * input); the producer writes it with dev::transform_block (the
// prepare kernel's function, same bits) and the call then passes inputs_prepared = true (no prepare launch). Only the
// two-launch forms (STRIX_MMX_LINEAR_V 1, 4, 7, 8, 9, 10 / scheduled: all of them run the same prepare transform and
// accept prepared inputs); the one-launch forms 3, 5, 6 transform inside their core and throw.
void linear_xt_offsets(const LinearJob* jobs, unsigned count, std::size_t* offs);
void enqueue_linears_prepared(const LinearJob* jobs, unsigned count, unsigned rows, void* workspace, hipStream_t stream);
// The same call whose core launch publishes its outputs (lean producer publish, PublishTarget above); prepared as above.
void enqueue_linears_publish(const LinearJob* jobs, unsigned count, unsigned rows, void* workspace, hipStream_t stream,
                             const PublishTarget& pub, bool prepared);
// True when the process's linear form (v4) and MoE decode form (overlap, v4, 4 waves) carry the publish epilogue.
bool producer_publish_supported();
// Profiling only: when set (device buffer, 5 x 64-bit per core CTA), the v4/v7 core writes per CTA the realtime counter
// at start, after its inputs are in LDS, after its split, at exit (or finish start), and its HW_ID1. nullptr = off.
void set_linear_trace(std::uint64_t* trace);

}  // namespace strix::minimax::moe::opt
