#pragma once
// Qwen3.8-27B dense family (`qwen35`) on one CUDA device: reusable engine over the shared cuda_core kernels.
// Home: specific (model layer composition); every arithmetic kernel lives in src/cuda_core (shared).
//
// Drafting contract (Gemma-style target):
//   forward_rows(tokens[R]) runs R rows at positions committed() .. committed()+R-1 and leaves per-row logits / argmax /
//   post-final-norm hidden on the device. NOTHING is committed: the GDN recurrent state and conv state are written to a
//   shadow copy, the KV rows are written past the committed length. commit(k) then accepts the first k rows of the LAST
//   window (0..R): k == R flips the state parity (free); 0 < k < R re-applies the first k rows of the window to the
//   pre-window state in one state pass (the window's per-row GDN inputs are kept on the device); k == 0 discards.
//   Rows 1..R are bitwise equal to row-by-row decode (every reduction order is independent of the row count).
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "strix/cuda/core/nvfp4_tile.hpp"
#include "strix/cuda_core/attention_policy.hpp"
#include "strix/cuda_core/dense.hpp"
#include "strix/cuda_core/operand_cache_plan.hpp"
#include "strix/qwen35/config.hpp"
#include "strix/split_k_policy.hpp"
#include "strix/qwen35/engine_types.hpp"

namespace strix::spec { class DraftRows; }

namespace strix::qwen35 {

enum class PackFormat : std::uint8_t { bf16, q8_0 };
// TURBO weight source (owner requirement): off = exact pack as is; native = TURBO policy without weight change;
// onfly = BF16 2-D weights rounded at load to NVFP4 (RTN per 16, kept only in VRAM). converted = the same NVFP4 bytes read from a pack written by save_converted_pack (no BF16 source,
// no conversion; the pack decides which tensors are NVFP4, e.g. lm_head).
enum class TurboWeights : std::uint8_t { off, native, onfly, converted };
// Prefill arithmetic policy for CudaEngine::prefill (the large-M path). exact: dense on the large-M tensor-core GEMM with the decode
// reduction order + decode attention in 32-row windows + sequential GDN: bitwise equal to forward_rows windows + commit.
// flash: exact dense, causal flash attention (bf16 P, online softmax; FAST attention, differs from decode attention).
// fast: flash + activation split of ONE bf16 term (mma1; nvfp4 weights are unchanged: their activation is f16 already).
// fast2: flash + two bf16 terms (mma2).
// fast8: fast + nvfp4 weights on the e4m3 x e4m3 mma (activation per-row e4m3, weight e2m1*e4m3 rounded to e4m3): TURBO/FAST, quality-gated.
// fast4: fast + nvfp4 weights on the block-scaled e2m1 x e2m1 mma (activation per-row NVFP4: row scale, e4m3 per 16, e2m1), gate and
//        up fused (strix/cuda/core/nvfp4_tile.hpp, nvfp4-tile-mma-f32-v1): W4A4 FAST, quality-gated, opt-in only.
enum class PrefillArith : std::uint8_t { exact, flash, fast, fast2, fast8, fast4 };
[[nodiscard]] const char* prefill_arith_name(PrefillArith a);
// GDN delta rule of the large-M prefill (non-exact prefill policies only; the exact policy keeps the decode windows):
//   serial  = the decode recurrence row by row (gdn.hpp gdn_prefill_layer), bitwise equal to the windows.
//   chunk16 = FAST gdn-chunk16-f32-v1 (strix/cuda_core/gdn_chunk.hpp): 16-row chunks, FP32, all SMs; bits differ (identity
//             suffix -gdnc16), opt-in until its logit gate passes.
enum class PrefillGdn : std::uint8_t { serial, chunk16 };
// CLI / option spelling of PrefillGdn ("serial" | "chunk16"); anything else is rejected, never mapped to a default.
[[nodiscard]] inline PrefillGdn parse_prefill_gdn(const std::string& s) {
    if (s == "serial") return PrefillGdn::serial;
    if (s == "chunk16") return PrefillGdn::chunk16;
    throw std::invalid_argument("--prefill-gdn expects serial|chunk16, got '" + s + "'");
}
// CLI spelling of EngineOptions::prefill_norm_fold ("on" | "off"); anything else is rejected.
[[nodiscard]] inline bool parse_prefill_norm_fold(const std::string& s) {
    if (s == "on") return true;
    if (s == "off") return false;
    throw std::invalid_argument("--prefill-norm-fold expects on|off, got '" + s + "'");
}
// CLI spelling of EngineOptions::prefill_tile_feed ("bulk" | "lsu", nvfp4::tile::parse_feed); anything else is rejected.
[[nodiscard]] inline cuda::core::nvfp4::tile::Feed parse_prefill_tile_feed(const std::string& s) {
    cuda::core::nvfp4::tile::Feed f{};
    if (cuda::core::nvfp4::tile::parse_feed(s.c_str(), &f)) return f;
    throw std::invalid_argument("--prefill-tile-feed expects bulk|lsu, got '" + s + "'");
}

struct EngineOptions {
    cuda_core::DenseArith dense_arithmetic = cuda_core::dense_arith();
    std::uint32_t max_positions = 4096;  // KV capacity per attention layer (and for the MTP layer)
    bool use_graphs = true;              // CUDA graph per (rows, logits mode)
    bool multi_gemv = true;              // one launch for GEMVs sharing x (q|k|v, qkv|z|beta|alpha); bitwise equal either way
    bool pdl = false;                    // experimental, default off: programmatic dependent launch + weight L2 prefetch between kernels
    bool load_mtp = true;                // read blk.<layers> (MTP) tensors when present
    bool quiet = false;
    // TURBO levers (default off = exact BF16/q8_0 identity). Each lever is individually switchable.
    TurboWeights turbo_weights = TurboWeights::off;
    std::uint32_t turbo_bits = 4;   // 4 = NVFP4 (3/2 later)
    bool turbo_lm_head = true;      // onfly: lm_head follows the source (true) or stays BF16 (false)
    // multi_gemv (above) is also the TURBO multi-launch lever.
    // Large-M prefill (CudaEngine::prefill). Default exact = bitwise equal to the windowed path.
    PrefillArith prefill_arith = PrefillArith::exact;
    PrefillGdn prefill_gdn = PrefillGdn::serial;  // ignored (and not in the identity) under prefill_arith exact
    // FAST4 only: the hidden RMSNorms (attn / post) and, on the chunked GDN path, the gated RMSNorm of the ssm_out input are folded
    // into the A-operand quantiser (cuda_core dense_big_norm_split4 / dense_big_gated_norm_split4), so no f32 normalised rows are
    // written and re-read. Bitwise equal to norm + quantise (no identity suffix); a non-FAST4 consumer gets the f32 rows materialised
    // on demand. Default on: identity PASS + strict A/B x3 on the device for the hidden fold (8K FAST4 + chunk16 -4.4 %);
    // --prefill-norm-fold off is the rollback.
    bool prefill_norm_fold = true;
    // FAST4 only: stage feed of the tile GEMMs (cuda/core nvfp4_tile Feed). bulk = one producer warpgroup with 1-D bulk copies
    // into an mbarrier ring; lsu = every thread's cp.async + one CTA barrier per stage. Same bits (no identity suffix).
    // Default lsu preserves the incumbent kernel; opt in to the measured operator-level bulk feed with --prefill-tile-feed bulk.
    // End-to-end prefill validation is still required before a default switch.
    cuda::core::nvfp4::tile::Feed prefill_tile_feed = cuda::core::nvfp4::tile::kDefaultFeed;
    std::uint32_t prefill_chunk = 2048;  // rows per large-M chunk (multiple of 128, <= 8192); buffers are allocated on first use
    // Wave-aligned steps (strix/wave_fit.hpp, scheduling only, bitwise): prompts of >= this many tokens step by the largest
    // rows <= prefill_chunk whose flash grid fills whole waves (4096 -> 3968 on 188 SMs: flash +16 % at 128K, bench). 0 = off
    // (opt-in until the 128K/256K wall ABAB; then the measured crossover becomes the default).
    std::uint32_t prefill_wave_min_tokens = 0;
    std::uint32_t prefill_slots = 2;     // chunks in flight (1..2): 2 overlaps the sequential GDN kernel of one chunk with the GEMMs of the next
    // bounded submission (cuda_core/submit_window.hpp): the host queues at most this many chunks ahead of the GPU (waits on
    // chunk i - N before queuing chunk i). Scheduling only, bitwise. 0 = unbounded (whole prefill queued before one sync).
    std::uint32_t prefill_inflight = 3;
    std::uint32_t prefill_ramp = 0;      // > 0: first chunks grow ramp, 2*ramp, ... (< prefill_chunk) from idle; 0 = off
    bool prefill_big = true;             // false: prefill() runs forward_rows windows of 32 + commit (reference / fallback)
    // Long-context prefill attention (non-exact prefill policies only; the exact policy keeps attention_decode).
    // flash_pipe: shared-memory pipeline of the flash kernel (scheduling only, bitwise equal across pipes).
    cuda_core::FlashPipe flash_pipe = cuda_core::FlashPipe::pad2;
    // flash_tail: wave-tail split of the flash grid (scheduling only, bitwise equal; measured +4.5 % flash at 4096-row chunks). off = old grid.
    cuda_core::FlashTail flash_tail = cuda_core::FlashTail::split;
    // attn8: e4m3 flash attention over an e4m3 shadow of the KV cache (bit-changing, own identity suffix "-a8"; FAST gate required).
    bool prefill_attn8 = false;
    // decode_attn8: decode / verify attention on e4m3 tensor cores over the same e4m3 shadow (cuda_core attention_decode8; FAST class
    // d8, bit-changing vs BF16 attention_decode, own identity suffix "-d8"; opt-in until its quality gate passes). Main session only.
    bool decode_attn8 = false;
    // Resident expanded prefill operands (strix/cuda_core/operand_cache_plan.hpp, shared dense_big core): each NVFP4 / q8_0 weight
    // is expanded into its MMA operand once per engine instead of once per chunk, slot and request. Bitwise neutral (same bytes,
    // same kernel). auto = FAST8 e4m3 operands only (the measured per-call cost), on = also the exact f16 / q8 operands; both
    // under free VRAM - max(4 GiB, 10 % total) and prefill_operand_cache_cap_mib (0 = no cap). off = per-call expansion.
    cuda_core::operand_cache::Mode prefill_operand_cache = cuda_core::operand_cache::Mode::automatic;
    std::uint32_t prefill_operand_cache_cap_mib = 0;
    // NVFP4 GEMV split-K policy (strix/split_k_policy.hpp; resolved by the profile resolver strix/profiles/qwen_dense_cuda.hpp, lever split_k): exact = target
    // 1000 (today's bits); fast = FAST nvfp4-splitk-fast-v1 (per-shape targets for the device, identity suffix "-skf1").
    // Process-wide: the policy is core state (nvfp4_set_split_profile), so every live CudaEngine of a process shares one
    // split policy and the last init() wins. The "-skf1" suffix names the request; on a device without FAST rows the bits
    // are EXACT (print_split_policy reports the effective policy and the device key).
    split_k::Profile nvfp4_split = split_k::Profile::exact;
};

// Named identity of the arithmetic + weight source; also the prefix-cache key (caches never mix identities).
[[nodiscard]] inline std::string engine_identity(const EngineOptions& o) {
    static const char* names[] = {"fp32", "mma3", "mma2", "mma1"};
    static const char* pf_names[] = {"exact", "flash", "fast", "fast2", "fast8", "fast4"};
    const std::string a = names[static_cast<unsigned>(o.dense_arithmetic)];
    // a non-exact prefill policy is a different arithmetic identity (prefix caches never mix them)
    // decode_attn8 changes the decode bits and so every later K/V row: its own suffix (FAST class d8, never an exact identity)
    const std::string pf = (o.prefill_arith == PrefillArith::exact
        ? std::string()
        : std::string("-pf-") + pf_names[static_cast<unsigned>(o.prefill_arith)] + (o.prefill_attn8 ? "-a8" : "") +
              (o.prefill_gdn == PrefillGdn::chunk16 ? "-gdnc16" : "")) + (o.decode_attn8 ? "-d8" : "");
    if (o.turbo_weights == TurboWeights::off) return "qwen35-cuda-exact-" + a + "-v1" + pf;
    // the FAST split policy changes the NVFP4 GEMV summation order (decode, verify and the exact prefill's split ranges)
    const std::string sk = o.nvfp4_split == split_k::Profile::fast ? "-skf1" : "";
    if (o.turbo_weights == TurboWeights::native) return "qwen35-turbo-native-" + a + "-v1" + pf + sk;
    // converted packs hold the onfly bytes: same arithmetic identity as onfly (prefix caches may be shared)
    return "qwen35-turbo-nvfp4-onfly-rtn16-v2-retained-" + a + (o.turbo_lm_head ? "" : "-lmhead-bf16") + pf + sk;
}
// Writes the onfly NVFP4 weights of the BF16 gguf `src` as a converted pack at `out` (use with TurboWeights::converted). Returns bytes.
std::uint64_t save_converted_pack(const std::string& src, const std::string& out, const EngineOptions& opt);
// Effective-config text (identity, prefix-cache key, every lever).
void print_effective_config(const EngineOptions& o, const char* dense_arith, std::FILE* f);
// The "  split-k:" line of the effective config: the split policy the device really runs (identity, SM count, compute
// capability); FAST requested on a device without rows prints "FAST requested, bits EXACT". Turbo weights only.
void print_split_policy(const EngineOptions& o, std::FILE* f);

// Outputs of the last forward (device pointers, valid until the next forward on this engine).
struct RowsOut {
    const float* logits = nullptr;    // [rows][vocab] (all) or [1][vocab] (last), null for none
    const std::uint32_t* top = nullptr;  // [rows] argmax (lowest index on ties) (all) or [1] (last)
    const float* hidden_post = nullptr;  // [rows][hidden] after the final norm (always)
    const float* hidden_pre = nullptr;   // [rows][hidden] residual stream before the final norm
};

// Multi-session decode (slots): one batch = groups; a group is `rows` consecutive rows of ONE session slot (1 = a decode lane,
// more = a verify / prefill window). Total rows over all groups <= 32.
struct SlotRows {
    std::uint32_t slot = 0;
    std::uint32_t rows = 1;
};

class CudaEngine {
public:
    using Rows = RowsOut;  // shared MtpChain binding (strix/spec/mtp_chain.hpp)
    using LogitRows = qwen35::LogitRows;
    ~CudaEngine();
    CudaEngine(const CudaEngine&) = delete;
    CudaEngine& operator=(const CudaEngine&) = delete;

    static std::unique_ptr<CudaEngine> from_gguf(const std::string& path, const EngineOptions& opt);
    // Real dimensions of `cfg` (defaults = Qwen3.8-27B) with weights generated on the device by a counter-based RNG
    // (deterministic in seed, scaled so activations stay finite). No disk. Timing equals the real model (memory bound).
    static std::unique_ptr<CudaEngine> synthetic(const Config& cfg, PackFormat pack, const EngineOptions& opt,
                                                 std::uint64_t seed);

    [[nodiscard]] const Config& config() const;
    [[nodiscard]] const EngineOptions& options() const;
    [[nodiscard]] std::string identity() const;  // engine_identity(options(), current dense arith)
    [[nodiscard]] std::uint32_t nvfp4_tensors() const;  // number of tensors stored as NVFP4
    [[nodiscard]] bool has_mtp() const;
    [[nodiscard]] cudaStream_t stream() const;
    [[nodiscard]] std::uint32_t committed() const;             // committed sequence length
    [[nodiscard]] std::uint64_t weight_bytes() const;          // bytes read per pass by the trunk + lm_head
    [[nodiscard]] std::uint64_t state_bytes_per_pass() const;  // GDN state read + write per pass

    void reset();  // committed length 0, all recurrent state zero, window discarded

    // rows 1..32 (cuda_core::kMaxRows). tokens: host array. Async on stream(); read outputs after synchronizing.
    RowsOut forward_rows(const std::uint32_t* tokens, std::uint32_t rows, LogitRows want = LogitRows::all);
    // Prefill n tokens (any n >= 1) at positions committed() .. committed()+n-1 and COMMIT them (same committed state as forward_rows
    // windows + commit(rows) under PrefillArith::exact). Returns the logits / argmax / hidden of the LAST token (want = last; none
    // skips the lm_head). Runs the large-M tensor-core path in chunks of options().prefill_chunk rows; falls back to windows when
    // prefill_big is off, dense arith is fp32 or a weight shape is unsupported. Synchronises the host at the end (own streams); no CUDA graphs.
    RowsOut prefill(const std::uint32_t* tokens, std::uint32_t n, LogitRows want = LogitRows::last);
    [[nodiscard]] bool prefill_is_big() const;  // whether prefill() takes the large-M path for this engine
    // Accepts the first k rows of the last forward_rows window (0 <= k <= rows). Async on stream().
    void commit(std::uint32_t k);

    // MTP layer (single full-attention block with its own KV; positions are explicit, no hidden state to commit).
    // Row r runs at position pos0 + r with input token next_tokens[r] (the token FOLLOWING the hidden's position) and
    // hidden_dev[r] = the trunk hidden of position pos0 + r (use RowsOut::hidden_post or hidden_pre: the convention is the
    // caller's choice). Rows are causal inside the call; earlier positions must have been written by earlier calls.
    struct MtpOut {
        const float* logits = nullptr;  // [rows][vocab]
        const std::uint32_t* top = nullptr;
        const float* hidden_pre = nullptr;   // MTP block output before mtp.norm (feed back for chains, convention 'pre')
        const float* hidden_post = nullptr;  // after mtp.norm
    };
    MtpOut mtp_forward(const std::uint32_t* next_tokens, const float* hidden_dev, std::uint32_t rows, std::uint32_t pos0,
                    bool want_top = true);  // want_top=false: skip lm_head + argmax (logits/top null): KV/hidden only
    // Device-fed MTP chain (strix/spec/mtp_chain.hpp): one row at `pos` whose input id is *in_host (by value) or, with
    // in_host null, the id in feed slot in_slot on the device; want_top also stores the row's draft id in slot out_slot.
    // The chain enqueues a whole draft without host round trips and reads the ids once (mtp_feed_read). Same kernels and
    // operands as mtp_forward(rows 1), so the drafted ids are bitwise those of the host-fed chain.
    static constexpr std::uint32_t kMtpFeedSlots = 64;
    MtpOut mtp_step_fed(const std::uint32_t* in_host, std::uint32_t in_slot, std::uint32_t out_slot, const float* hidden_dev,
                        std::uint32_t pos, bool want_top = true);
    void mtp_feed_put(std::uint32_t slot, const std::uint32_t* top_dev);  // slot = *top_dev (async, device to device)
    void mtp_feed_read(std::uint32_t n, std::uint32_t* host) const;       // host[0..n) = slots 0..n-1 (synchronizes)
    // Verify pre-launch (strix/spec/block_drafter.hpp): mtp_feed_read_async(n) queues the copy of slots 0..n-1 to pinned
    // host staging (no wait); mtp_feed_wait(n, host) waits for THAT copy only (not for work queued after it).
    void mtp_feed_read_async(std::uint32_t n);
    void mtp_feed_wait(std::uint32_t n, std::uint32_t* host);
    // forward_rows whose first nhost row ids are host values (host_ids) and whose next k rows take their ids on the device
    // from feed slots slot[j] (or host_vals[j] when slot[j] == strix::spec::DraftFeed::kHost = 0xFFFFFFFF). Same graph,
    // same operands as forward_rows(tokens) with tokens = host ids + the feed ids; nothing is read or waited for.
    RowsOut forward_rows_fed(const std::uint32_t* host_ids, std::uint32_t nhost, const std::uint32_t* slot,
                             const std::uint32_t* host_vals, std::uint32_t k, LogitRows want = LogitRows::all);
    // Draft-only MTP head over a vocabulary subset (strix/spec/draft_rows.hpp, group 16 = one NVFP4 tile of the head):
    // before each head launch the engine copies the rows' new tiles into a compact head (device-to-device, no
    // re-quantisation) and maps the argmax back to token ids on the device, so MtpOut::top stays a token id. MtpOut::logits
    // then hold the compact rows. nullptr = the full head (default). Requires an NVFP4 lm_head; the caller keeps `rows`
    // alive while set. Drafts only steer acceptance: verify emits the target's own tokens either way.
    void set_draft_rows(strix::spec::DraftRows* rows);

    // Host copies (synchronize the stream).
    void read_logits(const RowsOut& o, std::uint32_t row, float* host_vocab) const;
    void read_top(const RowsOut& o, std::uint32_t rows, std::uint32_t* host) const;
    void read_hidden(const float* dev, std::uint32_t rows, float* host) const;
    void sync() const;

    // ---- Multi-session (slots). Home: shared mechanism (row map + slot tables live in cuda_core attention/rope/gdn); the layer
    // composition below is model-specific. Slots have their own KV, GDN/conv state, parity and length; the single-session API above
    // is a separate session (do not interleave its pending window with a slot batch: both reuse the window buffers).
    // Arithmetic per row is bitwise equal to running that session alone (every reduction is independent of the row count/mix).
    void configure_slots(std::uint32_t n);  // allocate state for n sessions (once, before graphs); n >= 1
    [[nodiscard]] std::uint32_t slot_count() const;
    [[nodiscard]] std::uint32_t slot_committed(std::uint32_t slot) const;
    void reset_slot(std::uint32_t slot);
    void copy_slot(std::uint32_t dst, std::uint32_t src);  // dst becomes a copy of src (KV prefix, GDN state, length)
    // tokens: host [sum rows] in group order; group g row i runs at position slot_committed(slot)+i. want: none/all (last == all for
    // 1-row groups; for wider groups it also returns all rows). auto_commit: every row is accepted (state committed inside the graph);
    // false keeps the window until commit_slots. Logits / top / hidden are indexed by batch row.
    RowsOut forward_slots(const SlotRows* groups, std::uint32_t n, const std::uint32_t* tokens, LogitRows want = LogitRows::all,
                          bool auto_commit = true);
    // After forward_slots(..., auto_commit=false): accept[g] rows of group g (0..rows) are committed, the rest discarded.
    void commit_slots(const std::uint32_t* accept);

    // Diagnostics (no effect on arithmetic).
    void set_graphs(bool on);
    [[nodiscard]] std::uint64_t launches_last_forward() const;

private:
    CudaEngine();
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace strix::qwen35
