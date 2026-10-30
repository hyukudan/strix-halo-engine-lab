#pragma once

// MiniMax DSpark drafter on the device (gfx1151, wave32): the kernels of the declared arithmetic "mm-dspark-v1"
// (minimax_dspark.hpp), bitwise equal to the host reference, and the per-sequence context rings.
//
// Placement (qualification notes section 2): both ranks run append() and block() on identical inputs (the target
// residual is replicated after every layer), so both hold the same y; the head is the target's (vocabulary half per
// rank) and the Markov chain runs on each rank's half with one (value, id) exchange per step, combined with combine().
//
// Usage per draft (one sequence slot):
//   append(slot, taps, n, pos0)                     committed target rows (taps [n][30720] f32, device)
//   forward_block(slot, anchor_row, noise_row, P)           y() = the block's final-norm rows [7][6144]
//   head(y) -> base logits of this rank's vocabulary range (the caller's resident head)
//   for k in 0..6: markov_step(prev, base + k * stride, v0, v1) -> candidate; exchange; combine -> prev
//   confidence(tokens t_0..t_6)                     per-row survival estimates
// Everything is enqueued on the given stream; only sample() and the candidate reads synchronize.

#include "minimax_dspark.hpp"

// hipStream_t without the HIP headers (the same declaration as hip_runtime_api.h), so plain C++ callers stay HIP-free.
struct ihipStream_t;
typedef struct ihipStream_t* hipStream_t;

#include <array>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <vector>

namespace strix::minimax::dspark {

struct ContextSnapshot {
    std::vector<std::uint16_t> k, v; // [5][4096][512], invalid slots canonical zero
    std::vector<std::int64_t> positions;
};

class DeviceDrafter {
public:
    struct Options {
        std::uint32_t slots = 1;          // independent sequences, one context ring each
        std::uint64_t capacity = 8256;    // highest position + block (RoPE table size)
        bool tiled_linears = false;       // multi-row linears with the activations staged in LDS (same bits); OFF until never slower
        std::uint32_t attn_chunk = attention_chunk;  // 64-key partials (ON: exact vs host, never slower in d4) or 0 = v1
        bool prefetch_linears = false;    // with tiled_linears: the lane's weight slice in registers first (same bits)
        bool multi_linears = false;       // MXFP8 linears with 4 outputs per wave (same bits); OFF until never slower
        // chunked attention kernels (1..3 gated bit-identical to 0 for finite values, T3; 4 / 5 NOT gated): 0 per-row scores + pv (d4), 1 one block per
        // (kv head, chunk) for all 7 rows (measurement: slower), 2 scores3 (32-key reduce-scatter, per row) + per-row pv, 3 scores3 + pv3,
        // 4 / 5 = 3 with scores3 compiled for >= 8 / 12 waves per EU (measurement: scores3 at 224 VGPRs)
        // default 2: measurement (K4fc, ABA x3) B/A 0.96-0.97 / 0.92 / 0.82 at ctx 300 / 1000 / 4095, never slower, T3 bitwise;
        // 3: one ctx-300 triplet 1.004 -> not default
        std::uint32_t attn_variant = 2;
    };
    DeviceDrafter(const Weights& host, Options options);

    // Draft linears through another implementation (e.g. the engine's strix-trellis gemv on an EXL3 requant of the draft,
    // qualification notes): kind 0..6 = q, k, v, o, gate, up, down of draft layer `layer`, kind 7 = fc (layer 0);
    // out[r][o] for `rows` rows of x (row stride = the linear's input width). A hook returns false for a linear it does not
    // carry (then the built-in kernel runs). Unset = the built-in kernels. Changes acceptance, never output.
    enum LinearKind : std::uint32_t { lin_q = 0, lin_k, lin_v, lin_o, lin_gate, lin_up, lin_down, lin_fc };
    using LinearHook = std::function<bool(std::uint32_t layer, std::uint32_t kind, const float* x, std::size_t rows, float* out,
                                          hipStream_t stream)>;
    void set_linear_hook(LinearHook hook) { hook_ = std::move(hook); }
    // Several linears of one layer that read the same x in one call (q + k + v, gate + up; k + v in append): outs[i]
    // receives kinds[i]. Returning false (or unset) runs each kind through linear() as before. Same contract as the hook.
    using LinearGroupHook = std::function<bool(std::uint32_t layer, const std::uint32_t* kinds, std::uint32_t count, const float* x,
                                               std::size_t rows, float* const* outs, hipStream_t stream)>;
    void set_linear_group_hook(LinearGroupHook hook) { group_hook_ = std::move(hook); }
    // hipDeviceSynchronize for callers without HIP headers (draft-cycle attribution).
    static void device_synchronize();
    static std::string runtime_identity();
    // Attribution: every linear() synchronised and timed per kind (0..7) and the number of calls; off by default.
    // linear_seconds = block passes only; append_linear_seconds = context appends (fc + k/v of every layer).
    void set_profile(bool on) { profile_ = on; }
    [[nodiscard]] const std::array<double, 8>& linear_seconds() const { return lin_s_; }
    [[nodiscard]] const std::array<double, 8>& append_linear_seconds() const { return lin_app_s_; }
    [[nodiscard]] const std::array<std::size_t, 8>& linear_calls() const { return lin_n_; }
    ~DeviceDrafter();
    DeviceDrafter(const DeviceDrafter&) = delete;
    DeviceDrafter& operator=(const DeviceDrafter&) = delete;

    // Context of n committed rows at positions pos0..pos0 + n - 1 (any n; rows below the window may be skipped).
    void append(std::uint32_t slot, const float* taps, std::size_t n, std::uint64_t pos0, hipStream_t stream);
    // The same rows staged (device copy of the taps) and appended later in one batch: by the next append or block pass
    // of the slot, or when 32 rows are staged. For cycles that do not draft (s10: a paused cycle's 1-row append cost
    // ~1.5 ms of GPU time). Batching changes the rows' bits only through the linear path, never the output.
    void append_deferred(std::uint32_t slot, const float* taps, std::size_t n, std::uint64_t pos0, hipStream_t stream);
    void flush(std::uint32_t slot, hipStream_t stream);
    // Forget staged rows (a new sequence starts in the slot).
    void drop_staged(std::uint32_t slot) { staged_n_.at(slot) = 0; }
    // The block pass from the anchor at `position`; anchor_row / noise_row = the target's BF16 embedding rows.
    void forward_block(std::uint32_t slot, const std::uint16_t* anchor_row, const std::uint16_t* noise_row, std::uint64_t position,
               hipStream_t stream);
    [[nodiscard]] const float* y() const noexcept { return y_; }
    // argmax over [v0, v1) of base[v - v0] + W2[v] . W1[prev] into `out` (device, 8 bytes: value, id).
    void markov_step(std::uint32_t prev, const float* base, std::uint32_t v0, std::uint32_t v1, Candidate* out, hipStream_t stream);
    // Confidence of the 7 block rows given t_0..t_6 (host values) into out (device [7] f32).
    void confidence(const std::array<std::uint32_t, block>& prev, float* out, hipStream_t stream);
    // markov_step into the drafter's own buffer, read back to the host (synchronizes).
    Candidate markov_step_host(std::uint32_t prev, const float* base, std::uint32_t v0, std::uint32_t v1, hipStream_t stream);
    // confidence() read back to the host (synchronizes).
    std::array<float, block> confidence_host(const std::array<std::uint32_t, block>& prev, hipStream_t stream);
    // One-rank chain over the full vocabulary: base [7][vocab] (device). Synchronizes.
    Draft sample(std::uint32_t anchor, const float* base, hipStream_t stream);

    // Switch the kernel variants between calls (A/B timing; both variants are checked against their host order).
    void set_attn_variant(std::uint32_t v) { opt_.attn_variant = v; }
    void set_kernels(bool tiled_linears, std::uint32_t attn_chunk, bool prefetch_linears = false, bool multi_linears = false) {
        opt_.tiled_linears = tiled_linears;
        opt_.attn_chunk = attn_chunk;
        opt_.prefetch_linears = prefetch_linears;
        opt_.multi_linears = multi_linears;
    }
    void reset_context(std::uint32_t slot);
    [[nodiscard]] ContextSnapshot export_context(std::uint32_t slot) const;
    void import_context(std::uint32_t slot, const ContextSnapshot&, std::uint64_t frontier);

    // Device bytes of the weights and of the rings.
    [[nodiscard]] std::uint64_t weight_bytes() const noexcept { return weight_bytes_; }
    [[nodiscard]] std::uint64_t ring_bytes() const noexcept { return ring_bytes_; }

    // Building blocks, exposed for the kernel tests (device pointers, rows <= 8 or 32).
    void mxfp8_linear(const Mxfp8Matrix& w_device, const float* x, std::size_t rows, float* out, hipStream_t stream) const;
    void bf16_linear(const Bf16Matrix& w_device, const float* x, std::size_t rows, float* out, hipStream_t stream) const;
    void rms_norm(const float* x, const std::uint16_t* weight, std::size_t rows, float* out, hipStream_t stream) const;
    // Per-head norm + RoPE of `rows` rows of `count` heads (row stride count * 128), row r at position pos0 + r.
    void head_norm_rope(float* x, std::size_t rows, std::uint32_t count, const std::uint16_t* weight, std::uint64_t pos0,
                        hipStream_t stream) const;
    [[nodiscard]] const Weights& device_weights() const noexcept { return dev_; }

private:
    void linear(std::uint32_t layer, std::uint32_t kind, const float* x, std::size_t rows, float* out, hipStream_t stream) const;
    void append_now(std::uint32_t slot, const float* taps, std::size_t n, std::uint64_t pos0, hipStream_t stream);
    std::vector<float*> staged_;                    // per slot [32][tap_width]
    std::vector<std::size_t> staged_n_;
    std::vector<std::uint64_t> staged_pos_;
    // profile: a group's time goes to kinds[0] (one call); its other kinds count calls only.
    void linear_group(std::uint32_t layer, std::initializer_list<std::uint32_t> kinds, const float* x, std::size_t rows,
                      std::initializer_list<float*> outs, hipStream_t stream) const;
    LinearHook hook_;
    LinearGroupHook group_hook_;
    bool profile_ = false;
    mutable std::array<double, 8> lin_s_{}, lin_app_s_{};
    bool in_append_ = false;
    mutable std::array<std::size_t, 8> lin_n_{};
    void linear_impl(std::uint32_t layer, std::uint32_t kind, const float* x, std::size_t rows, float* out, hipStream_t stream) const;
    Options opt_;
    Weights dev_;                                   // the same views, device pointers
    std::vector<void*> allocations_;
    std::uint64_t weight_bytes_ = 0, ring_bytes_ = 0;
    float *rope_cos_ = nullptr, *rope_sin_ = nullptr;  // [capacity][32]
    float* e4m3_lut_ = nullptr;                     // [256]
    std::vector<std::uint16_t*> ring_k_, ring_v_;   // per slot [layers][window][512] BF16
    std::vector<std::vector<std::int64_t>> ring_pos_;  // host copy of the positions each slot holds
    // activations
    float *r_ = nullptr, *x_ = nullptr, *q_ = nullptr, *kb_ = nullptr, *vb_ = nullptr, *att_ = nullptr, *o_ = nullptr,
          *gate_ = nullptr, *up_ = nullptr, *y_ = nullptr, *ctx_fc_ = nullptr, *ctx_c_ = nullptr, *ctx_k_ = nullptr,
          *ctx_v_ = nullptr;
    std::uint16_t *kb16_ = nullptr, *vb16_ = nullptr;  // block K/V as BF16 [7][512]
    float *att_s_ = nullptr, *att_po_ = nullptr, *att_ps_ = nullptr;  // chunked attention: scores, partial o, partial sums
    Candidate* cand_ = nullptr;                     // per-block Markov candidates
    Candidate* pick_ = nullptr;                     // sample()'s step result
    float* conf_ = nullptr;
    std::uint32_t* prev_dev_ = nullptr;
    template <class T>
    T* alloc(std::size_t count);
};

} // namespace strix::minimax::dspark
