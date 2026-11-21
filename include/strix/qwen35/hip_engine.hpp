#pragma once
// Qwen3.8-27B dense family (`qwen35`) on one HIP device: window engine over the shared HIP cores (dense_linear,
// attn_core dense_decode, hip_gdn_decode_step_dims, ops/norm). Same drafting contract as CudaEngine:
//   forward_rows(tokens[R]) runs R rows at positions committed() .. committed()+R-1 (R <= max_rows()) and leaves per-row
//   logits / argmax on the device. Nothing is committed: the GDN matrix + conv state are snapshotted first, the KV rows are
//   written past the committed length (dead until committed). commit(k) accepts the first k rows of the LAST window:
//   k == R keeps the advanced state (free); 0 < k < R restores the snapshot and replays the first k rows of every GDN layer
//   from the saved raw inputs; k == 0 restores only. Rows are bitwise equal to row-by-row decode (every reduction order is
//   independent of the row count).
// Home: specific (model layer composition: qwen35 GDN/attention interleave); every kernel is shared (backend core).
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "strix/qwen35/config.hpp"
#include "strix/qwen35/engine_types.hpp"

namespace strix::spec { class DraftRows; }

namespace strix::qwen35 {

struct HipEngineOptions {
    std::uint32_t max_positions = 4096;  // KV capacity per full-attention layer
    bool quiet = false;
    // forward_commit window width. <= 16: the 16-row exact windows. > 16 with wide_exact: EXACT wide windows (below).
    // > 16 without wide_exact: FAST policy dense-q8a8-f16-v1 (shared core::f16_dense_tall::q8a8: A8 activations x Q8_0
    // weights, FP16 WMMA; IU4 weights: dense-iu4a16-f16-v1 of core::f16_dense_tall::iu4a16, FP16 activations x decoded
    // IU4) for the dense projections of those wide windows; GDN and attention keep their exact kernels. Decode windows
    // (forward_rows) stay <= 16 rows.
    // Default: 64-row EXACT windows. Measured on Strix 27B IU4 (one node): every row bitwise the 16-row result (last-row
    // logits + generated ids at 2K and 4K, widths 32..512), 2K warm 131.6-132.8 -> 185.2-187.4 tok/s (3 ABAB pairs),
    // 4K 128.3 -> 160.6 tok/s; 64 rows beat 128 / 256 / 512 (the exact decode attention costs more per row in wider
    // grids: 4K attend 0.79 -> 2.10 ms/row at 512 rows). Rollback: prefill_rows = 16 (or wide_exact = false for FAST).
    std::uint32_t prefill_rows = 64;
    // prefill_rows > 16 in the EXACT class instead of FAST: the 16-row arithmetic over wide windows (IU4 dense projections
    // through core::dense_iu4::run's row-block grid: the 8-row blocks of a column block share its weight slice through the
    // cache instead of re-streaming it per 8 rows; Q8_0 / BF16 keep 16-row sweeps; exact decode attention, GDN dims rows =
    // window). Every row is bitwise the 16-row result by construction (and by the tickets above).
    bool wide_exact = true;
    // Owned layer range [layer_begin, layer_end) of a multi-rank layer split (layer_end 0 = every layer: one rank, today's
    // engine). Only the owned layers' weights, KV cache and recurrent state are loaded; the embedding is loaded when the
    // range starts at layer 0, the final norm and head when it ends at the last layer.
    std::uint32_t layer_begin = 0;
    std::uint32_t layer_end = 0;
};

// Window rows the engine uses for a requested forward_commit width. EXACT wide windows (wide_exact, rows > 16) were
// measured on IU4 dense projections only (Strix 27B IU4: every large projection and the head IU4, ssm_alpha / ssm_beta
// Q8_0). When any large dense projection the engine loads is in another format (all_large_iu4 false) the window
// resolves to the 16-row exact form: the rows are bitwise the same either way, only the unmeasured wide form is not
// taken. Provisional until an identity + timing ticket on a Q8_0 or BF16 27B file. The FAST class is not affected.
[[nodiscard]] constexpr std::uint32_t resolve_prefill_rows(std::uint32_t requested, bool wide_exact, bool all_large_iu4) noexcept {
    return wide_exact && requested > 16U && !all_large_iu4 ? 16U : requested;
}

struct HipRowsOut {
    const float* logits = nullptr;       // device [rows][vocab] (all) or [1][vocab] (last); null for none
    const std::uint32_t* top = nullptr;  // device [rows] argmax (lowest index on ties) (all) or [1] (last)
    // Only with an MTP layer loaded (load_mtp): per-row trunk hidden of EVERY row of the window (valid until the next forward).
    const float* hidden_post = nullptr;  // device [rows][hidden] after the final norm
    const float* hidden_pre = nullptr;   // device [rows][hidden] residual stream before the final norm
};

// MTP layer outputs (device pointers valid until the next mtp_forward); same contract as the CUDA engine's MtpOut.
struct HipMtpOut {
    const float* logits = nullptr;  // [rows][vocab] (want_top)
    const std::uint32_t* top = nullptr;
    const float* hidden_pre = nullptr;   // MTP block output before shared_head_norm
    const float* hidden_post = nullptr;  // after shared_head_norm
};

class HipEngine {
public:
    using Rows = HipRowsOut;   // shared MtpChain binding (strix/spec/mtp_chain.hpp)
    using MtpOut = HipMtpOut;
    using LogitRows = qwen35::LogitRows;
    ~HipEngine();
    HipEngine(const HipEngine&) = delete;
    HipEngine& operator=(const HipEngine&) = delete;
    static std::unique_ptr<HipEngine> from_gguf(const std::string& path, const HipEngineOptions& opt);

    [[nodiscard]] const Config& config() const;
    [[nodiscard]] std::uint32_t vocab() const;
    [[nodiscard]] static constexpr std::uint32_t max_rows() { return 16U; }
    [[nodiscard]] std::uint32_t prefill_rows() const;  // widest forward_commit window (options)
    [[nodiscard]] std::uint32_t committed() const;
    [[nodiscard]] std::uint64_t weight_bytes() const;          // bytes read per pass (all weights but the embedding table)
    [[nodiscard]] std::uint64_t head_weight_bytes() const;     // output head bytes (read by every full-head row pass)
    [[nodiscard]] std::uint64_t mtp_weight_bytes() const;      // MTP layer bytes (0 before load_mtp)
    [[nodiscard]] std::uint64_t state_bytes_per_pass() const;  // GDN state read + write per pass
    [[nodiscard]] std::uint32_t kv_bytes_per_position() const;
    void reset();  // committed length 0, recurrent state zero, window discarded

    // rows 1..max_rows(). tokens: host array. Async on the stream; read outputs after read_top / sync.
    HipRowsOut forward_rows(const std::uint32_t* tokens, std::uint32_t rows, LogitRows want = LogitRows::all);
    void commit(std::uint32_t k);  // accepts the first k rows of the last forward_rows window (0 <= k <= rows)
    // Same arithmetic, every row accepted immediately: no snapshot, no replay inputs (prefill chunks, serial decode).
    HipRowsOut forward_commit(const std::uint32_t* tokens, std::uint32_t rows, LogitRows want = LogitRows::all);

    // Multi-rank layer split (one rank runs a contiguous range; only the residual rows cross a cut: KV cache, GDN matrix and
    // conv state stay with the rank that owns the layer). forward_range(first, last, ...) is forward_commit restricted to
    // layers [first, last) inside the owned range: first == 0 embeds `tokens`, otherwise the input rows must already be in
    // hidden_rows() (import_hidden, or a stream-ordered device copy); logits only when last == layers. The window is committed
    // (position advances) when `last` is the owned end, so one engine that owns every layer can run [0, k) then [k, L) of the
    // same window (the one-process cut test). forward_commit == forward_range(0, layers) on a full engine: same launches.
    HipRowsOut forward_range(std::uint32_t first, std::uint32_t last, const std::uint32_t* tokens, std::uint32_t rows,
                             LogitRows want = LogitRows::none);
    [[nodiscard]] std::uint32_t layer_begin() const;   // owned range
    [[nodiscard]] std::uint32_t layer_end() const;
    [[nodiscard]] float* hidden_rows() const;           // device FP32 [rows][hidden]: the residual rows at the cut
    [[nodiscard]] std::size_t hidden_row_bytes() const;
    void export_hidden(float* host, std::uint32_t rows) const;       // device -> host, synchronizes
    void import_hidden(const float* host, std::uint32_t rows) const; // host -> hidden_rows(), stream-ordered
    // Stage range of a rank that owns more layers than it runs in the prefill (the decoder of a relayed plan): windows run
    // [first, last) and commit at `last`. Default = the owned range (unchanged behaviour); set_stage(layer_begin(),
    // layer_end()) restores it before the decode.
    void set_stage(std::uint32_t first, std::uint32_t last);
    // Layer state of [first, last) at the committed positions: per layer in order, GDN = recurrent matrix + conv history,
    // attention = K then V rows [0, committed) (position-major). export synchronizes; import requires no pending window
    // and the same committed count, overwrites those layers' state and drops the V^T copy (rebuilt from the cache by the
    // next wide window: same bytes). Bytes cross unchanged, so a decoder that imports every other stage's layers holds
    // exactly the single-rank state.
    [[nodiscard]] std::size_t state_bytes(std::uint32_t first, std::uint32_t last) const;
    void export_state(std::uint32_t first, std::uint32_t last, void* host) const;
    void import_state(std::uint32_t first, std::uint32_t last, std::uint32_t positions, const void* host);
    [[nodiscard]] void* stream_handle() const;          // hipStream_t of every launch (staging copies order on it)

    void read_top(const HipRowsOut& o, std::uint32_t rows, std::uint32_t* host) const;  // synchronizes
    void read_logits(const HipRowsOut& o, std::uint32_t rows, float* host) const;       // [rows][vocab], synchronizes
    void read_logits_rows(const HipRowsOut& o, std::uint32_t rows, float* host) const {
        if (!o.logits) throw std::logic_error("qwen35 verify: sampled rows require full logits");
        read_logits(o, rows, host);
    }
    void sync() const;

    // MTP layer (blk.<layers>.* of a side GGUF: attn_norm, post_attention_norm, ffn_*, attn_q/k/v/output, q/k norms, nextn.enorm /
    // hnorm / eh_proj / shared_head_norm; bf16 or q8_0 matrices, F32 norms). The embedding and lm_head are the trunk's.
    // Call once after from_gguf; afterwards forward_rows / forward_commit also return hidden_pre / hidden_post (one extra final
    // norm launch per pass; the logits path is unchanged). Without load_mtp nothing changes.
    void load_mtp(const std::string& path);
    [[nodiscard]] bool has_mtp() const;
    // True when the GGUF at `path` carries the next-token-prediction layer itself (blk.<last>.nextn.eh_proj; the IU4 pack
    // keeps it as Q8_0), so load_mtp(path) needs no side file. Header read only.
    [[nodiscard]] static bool has_mtp_layer(const std::string& path);
    // Draft-only MTP head over a vocabulary subset (strix/spec/draft_rows.hpp, group 16 = one IU4 tile column of the head,
    // include/strix/qwen35/iu4_draft_head.hpp): before each head launch the engine gathers the rows' new tile columns into a
    // compact head (device-to-device, no re-quantisation) and maps the argmax back to token ids on the device, so
    // HipMtpOut::top stays a token id; HipMtpOut::logits then hold the compact rows. nullptr = the full head (default).
    // Requires an IU4 lm_head; the caller keeps `rows` alive while set. Drafts only steer acceptance: verify emits the
    // target's own greedy tokens either way.
    void set_draft_rows(strix::spec::DraftRows* rows);
    [[nodiscard]] static constexpr std::uint32_t mtp_max_rows() { return 16U; }
    // Row r runs at position pos0 + r with input token next_tokens[r] (the token FOLLOWING the hidden's position) and
    // hidden_dev[r] = the trunk hidden of position pos0 + r. Rows are causal inside the call; earlier positions must have been
    // written by earlier calls (own bf16 KV, positions explicit). want_top=false skips lm_head + argmax.
    HipMtpOut mtp_forward(const std::uint32_t* next_tokens, const float* hidden_dev, std::uint32_t rows, std::uint32_t pos0,
                          bool want_top = true);
    // Device-fed MTP chain (strix/spec/mtp_chain.hpp, MtpFeedEngine): the chain enqueues a whole draft without host round
    // trips and reads the ids once. mtp_step_fed = mtp_forward on one row (same launches and operands) whose input id is
    // *in_host (by value) or feed slot in_slot (an earlier step's pick), its pick written into slot out_slot.
    static constexpr std::uint32_t kMtpFeedSlots = 64;
    HipMtpOut mtp_step_fed(const std::uint32_t* in_host, std::uint32_t in_slot, std::uint32_t out_slot, const float* hidden_dev,
                           std::uint32_t pos, bool want_top);
    void mtp_feed_put(std::uint32_t slot, const std::uint32_t* top_dev);  // slot = *top_dev (async, device to device)
    void mtp_feed_read(std::uint32_t n, std::uint32_t* host) const;       // host[0..n) = slots 0..n-1 (synchronizes)
    // Device helpers for the shared chain glue (all on the engine stream).
    [[nodiscard]] float* alloc_f32(std::size_t n) const;
    void free_f32(float* p) const;
    void copy_f32_async(float* dst, const float* src, std::size_t n) const;
    // Timing only: set the committed length to `pos` WITHOUT computing (KV / state keep their contents; positions below pos are
    // whatever the cache holds). Used to sample window cost at long context without running the whole prefill.
    void bench_seek(std::uint32_t pos);

    // Exact tensor-parallel decode over N processes (one per device), each holding the whole model and the same
    // committed state (a replicated one-rank prefill). From this call on, every window (rows <= max_rows, whole model)
    // computes only this rank's columns / heads (shares by `rates`, strix/pair/column_split.hpp) and joins the others'
    // through `gather`: fn(ctx, stream, slots, slot_bytes, bytes), ordered on `stream` after the producers of this
    // rank's slot (slots + rank * slot_bytes, first `bytes`), must fill the first `bytes` of every other slot with that
    // rank's and order later work on `stream` after it; every rank makes the same calls in the same order. Ids are the
    // one-rank ids bit for bit; logits rows are gathered (and bitwise) only with `full_logits`, otherwise logits stay
    // null and read_top carries the split argmax. The committed state is sliced to this rank's heads (one way).
    struct TpGather {
        void* ctx = nullptr;
        void (*fn)(void* ctx, void* stream, void* slots, std::size_t slot_bytes, std::size_t bytes) = nullptr;
    };
    void enable_tp(std::uint32_t rank, const std::vector<double>& rates, TpGather gather, bool full_logits = false);
    [[nodiscard]] bool tp_on() const;
    [[nodiscard]] std::string tp_receipt() const;  // shares per tensor class, gathers per token

private:
    HipEngine();
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace strix::qwen35
