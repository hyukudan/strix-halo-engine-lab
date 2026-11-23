#pragma once
// Hy4-flash HIP forward: the stage map of hy4_hip_plan.hpp executed on shared entries only (no kernel in this
// directory). Batch 1, one position per step (the prompt is fed position by position: the correctness form; prefill
// rows and graph capture are tuning work). Device arithmetic class: FAST (A8 routed experts, absorbed MLA, device
// reduction orders); compared with the CPU oracle src/hy4/ref by ids and logits distance, never called exact.
//
// A Forward owns the trunk layers [first, last) of a model that may be truncated to `depth` layers
// (Config::truncated): first == 0 embeds the token, last == depth runs the head. A layer range is one rank of a
// pipeline split: the 4-stream state X [4][hidden] (FP32 holding BF16 values) crosses ranks.
//
// Indexed layers: every position stores its token key (token ring), each closing group of 4 is pooled into the group
// key cache; from slot 2048 on (group_indexer::dense false) the query scores the closed groups and attends the selected
// list (shared pool / score scan / G6 select / sparse flash List). Below 2048 the list is the causal prefix.
#include "hy4/ref/hy4_host_ref.hpp"
#include "strix/pair/gather_fn.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace strix::hy4::hipbind {

// Head of the next-token block's drafts (want_top). full: the pack's BF16 output.weight through the shared device greedy
// (the verify's own head). q8 / q4: the shared MTP draft head (strix/hip_mtp_draft_head.hpp) over a draft-only Q8_0 (Q4_0)
// copy of every output.weight row, built at first use (ggml Q8_0 rule; 394 / 209 MB): FAST-class draft logits, the verify
// head is unchanged, so emitted ids are not affected; only drafts can differ.
// Default q4 (measured, cluster1 300 ids, ids == plain on every run: code 27.93 -> 29.22, think 27.18 -> 28.72, prose
// 20.83 -> 21.34 tok/s; the 191-row device check agrees with the BF16 head on 186 rows at 2.19 ms less per draft).
enum class NextnHead : std::uint8_t { full, q8, q4 };

struct ForwardOptions {
    std::vector<std::filesystem::path> parts;  // the pack's GGUF parts (all of them; the Engram table is read by row)
    std::uint32_t depth = 0;                   // 0: the pack's depth; else Config::truncated(depth)
    std::uint32_t first = 0, last = 0;         // trunk layers of this rank; last 0 = depth
    std::uint32_t context = 4096;              // positions this forward may hold
    std::uint32_t prompt_rows = 0;             // batched prompt window (rows per launch, multiple of 8, >= 16); 0 = none
    bool head_q8 = false;                      // FAST weights.head.q8_0@1: Q8_0 copy of the BF16 head for the decode head
    std::uint32_t verify_rows = 0;             // rows of a verify pass (forward_rows, 1..8); rings sized for them; 0 = none
    bool nextn = false;                        // also load the pack's next-token (MTP) block: nextn_rows() (whole-model rank)
    NextnHead nextn_head = NextnHead::q4;      // the drafts' head (want_top of nextn_rows)
};

struct LoadReport {
    double seconds = 0;
    std::uint64_t bytes = 0;
    std::uint32_t tensors = 0;
    std::string queue_line;  // core::weight_load_line of each part
};

struct StepTimes {  // ms of the last step, by stage family (device-synchronised per stage only when timing is on)
    double embed = 0, engram = 0, attention = 0, ffn = 0, head = 0, total = 0;
};

class Forward {
public:
    explicit Forward(const ForwardOptions& options);
    ~Forward();
    Forward(const Forward&) = delete;
    Forward& operator=(const Forward&) = delete;

    [[nodiscard]] const Config& config() const noexcept;
    [[nodiscard]] const LoadReport& load_report() const noexcept;
    [[nodiscard]] std::uint32_t position() const noexcept;
    [[nodiscard]] bool has_head() const noexcept;
    [[nodiscard]] bool has_embedding() const noexcept;

    // One position. Every rank sees every token (the Engram window reads the token history).
    //   has_embedding(): `state` is ignored on input; else it holds X from the previous rank.
    //   has_head(): returns the FP32 logits; else `state` receives X for the next rank and the return is empty.
    const std::vector<float>& step(std::uint32_t token, std::vector<float>& state);
    // One position on a whole-model rank, next token by the shared device greedy rule (first max, lowest token on ties:
    // the host rule of strix-hy4-rank) without copying the logits to the host.
    std::uint32_t step_greedy(std::uint32_t token);

    // The prompt in windows of rows (options.prompt_rows): every stage runs its shared rows form over the window (Q8_0
    // Linears on the A8 rows GEMM, routed experts on the grouped FP16 prefill core, the flash family with the indexer
    // select per row). Positions before an 8-aligned window start and a tail shorter than 16 run through step(). FAST
    // class like step(): compared with the position-by-position device path by ids and logits distance. One rank with
    // the embedding and the head; a stage of a layer split runs prefill_window() under the shared pipeline. Returns the
    // logits of the last prompt position.
    const std::vector<float>& prefill(const std::vector<std::uint32_t>& prompt);

    // ---- one stage of the shared layer pipeline (strix/pair/layer_pipeline.hpp StageOps / DecodeOps), any layer range ----
    // One window of layers::windows(prompt, prompt_rows) starting at position() (hy4_hip_plan stage_window_plan: the rows
    // form over its admissible head, steps for the rest). A stage without the embedding reads the window's input rows
    // from seam_rows(); a stage without the head leaves its output rows there. head (head stage, last window): returns
    // the last row's logits, else an empty vector. Same work and order as prefill(): bitwise the one-node prefill.
    const std::vector<float>& prefill_window(std::span<const std::uint32_t> tokens, std::uint32_t start, bool head);
    [[nodiscard]] void* seam_rows() const noexcept;        // device [prompt_rows][4][hidden] FP32 (BF16 values); null on a whole-model rank
    [[nodiscard]] std::size_t seam_row_bytes() const noexcept;  // 4 x hidden x 4 B: one seam row = one decode state
    // ---- pair prime on a whole-model rank: the stages of a StageMap as layer ranges of THIS forward, then the state
    // re-laid out for the TP decode (strix/pair/relayout.hpp), KV in place. Prompt from position 0 after reset():
    //   window_rows(P): rows the one-node prompt plan runs as windows (the stage pipeline runs exactly these windows;
    //     the P - window_rows(P) tail positions run as steps through prompt_tail, TP steps after enable_tp);
    //   range_window(ids, start, a, b, head): one window of layers [a, b) (prefill_window's work with this forward's
    //     caches; input / output rows through range_seam() unless a == 0 / b == layers; b == layers feeds the
    //     prompt-hidden sink with the window's rows); one-node like prefill() (no gather);
    //   cache_bytes / export_cache / import_cache(layer, P): the layer's state for P positions (latent rows or ring slots,
    //     indexer token-key ring and group keys), synchronous host copies;
    //   seat(P): position P after every window ran here or was imported; feed_prompt_hidden(rows, p0, n): the sink with
    //     rows produced on the peer (device rows); prompt_tail(ids): steps with the prompt-hidden feed, last logits.
    // Bits: every cache row is the stage owner's = the one-node prefill's at the same prompt rows (StageMap invariant);
    // the tail steps are TP steps = the one-node steps (TP invariant): decode ids equal the one-node-primed run.
    [[nodiscard]] std::uint32_t window_rows(std::uint32_t prompt) const;
    [[nodiscard]] void* range_seam();
    // resident (pair window ring, strix/pair/window_ring.hpp): the window's X rows stay in the forward's own rows between
    // consecutive ranges of the SAME window ([0, a1), [a1, a2), ... in layer order, no other window in between): no seam
    // copies, and the window's Engram rows are read ahead at its first range and released after its last.
    const std::vector<float>& range_window(std::span<const std::uint32_t> tokens, std::uint32_t start, std::uint32_t first_layer,
                                           std::uint32_t last_layer, bool head, bool resident = false);
    // The prompt's window ids recorded at once (position 0 after reset(); `ids` = the window_rows(P) positions), so the
    // range windows of a context split may run out of prompt order (the Engram hash reads the earlier ids).
    void record_prompt(std::span<const std::uint32_t> ids);
    [[nodiscard]] std::uint64_t cache_bytes(std::uint32_t layer, std::uint32_t positions) const;
    void export_cache(std::uint32_t layer, std::uint32_t positions, std::span<std::byte> out);
    void import_cache(std::uint32_t layer, std::uint32_t positions, std::span<const std::byte> in);
    // The layer's cache rows written by positions [p0, p1) only (every slot of them: latent rows / ring slots, indexer token
    // ring slots, the group keys the range closes; p0, p1 multiples of the index ratio): the window ring's message.
    // Synchronous host copies; the same piece order on both ranks.
    [[nodiscard]] std::uint64_t rows_bytes(std::uint32_t layer, std::uint32_t p0, std::uint32_t p1) const;
    void export_rows(std::uint32_t layer, std::uint32_t p0, std::uint32_t p1, std::span<std::byte> out);
    void import_rows(std::uint32_t layer, std::uint32_t p0, std::uint32_t p1, std::span<const std::byte> in);
    void seat(std::uint32_t positions);
    void feed_prompt_hidden(const float* rows, std::uint32_t p0, std::uint32_t n);
    const std::vector<float>& prompt_tail(std::span<const std::uint32_t> tokens);
    [[nodiscard]] void* stream_handle() const noexcept;    // the hipStream_t every device operation of this forward runs on
    void sync();
    // step_greedy on a head stage of any range: `state` = X from the previous stage (ignored with the embedding).
    std::uint32_t step_greedy(std::uint32_t token, std::vector<float>& state);
    // Back to position 0 for a repeat run of the same process: every cache row is written in position order before a
    // read, so a repeat reproduces the bits (the repeat's logits digest is the check).
    void reset() noexcept;
    // State image (strix/session/state_image.hpp): the target's cross-position caches (latent rings, indexer token rings,
    // group keys) + position + token history, so a long teacher-forced reference stops at a ticket boundary and goes on
    // in a new process with the same bits. load_state() needs the same layout (model depth, rank range, context, ring
    // rows): another layout is refused. The next-token block's caches are not in the image (refused with nextn on).
    // Synchronises. Diagnostic / gate tool, no product path.
    [[nodiscard]] std::string state_layout() const;
    [[nodiscard]] const std::vector<std::uint32_t>& token_history() const noexcept;  // ids at positions 0 .. position() - 1
    void save_state(const std::filesystem::path& file) const;
    void load_state(const std::filesystem::path& file);
    // Undo the last step (pipelined speculation, strix/pair/layer_pipeline.hpp DecodeOps::rewind): position and token
    // history go back one. Every device row a step writes is addressed by its position (latent / token-key rings, the
    // group key the position closes), so the redo at the same position rewrites all of them before any read; a sliding
    // ring slot reused by position p held p - ring (ring >= window), outside p's window. step() synchronizes at its end.
    void rewind();

    // ---- verify rows (hy4_hip_rows.hpp / .inc), whole-model rank, options.verify_rows > 0 ----
    // R = tokens.size() rows at positions position() .. + R - 1 in one pass, each row bitwise what step() gives at that
    // position (class BASE). Greedy id per row in rows_greedy(); the logits rows [R][vocab] in rows_logits() when asked.
    // Synchronises; commit_rows(k) (0 <= k <= R) keeps the first k rows before the next pass or step.
    void forward_rows(std::span<const std::uint32_t> tokens, bool logits);
    [[nodiscard]] std::span<const std::uint32_t> rows_greedy() const noexcept;
    [[nodiscard]] const std::vector<float>& rows_logits() const noexcept;
    void commit_rows(std::uint32_t k);
    void set_rows_expert_union(bool on) noexcept;  // routed experts of a window read once (bitwise); A/B aid of the bench
    // Device [R][hidden] rows of output_norm(hidden) (the head's input) of the last forward_rows pass; valid until the next
    // pass or the next nextn_rows call (its rows form uses the verify workspace). The MTP teacher input of the committed rows.
    [[nodiscard]] const float* rows_hidden() const noexcept;

    // ---- next-token (MTP) block (options.nextn, whole-model rank; GGUF block config().layers), host oracle
    // HostRef::nextn_step. Rows r = 0 .. R-1 at block positions pos0 + r, its own K/V (positions < pos0 cached by earlier
    // calls; a call rewrites every cache row it reads at or after pos0, so a redo at the same positions replaces a
    // rejected chain):
    //   x = eh_proj(cat(enorm(e(tokens[r])), hnorm(hidden[r]))); x = x + attn(L, attn_norm(x)); x = x + ffn(L, ffn_norm(x))
    //   post = shared_head_norm(x)  (the head's input; a chain step feeds it back as `hidden`)
    // tokens: host [R]; hidden: device [R][hidden]. want_top: the shared device greedy id of the LAST row's logits (first
    // max, lowest id on ties) at top[R - 1]; the other entries are 0. The block's numerics are those of the pack (BF16
    // weights as stored, routed experts by per-expert dense rows): class FAST, a drafter only (the verify decides every id;
    // never touches the target's caches, position or token history). Synchronises.
    // Rows form (options.verify_rows >= 2): the call's rows run in chunks of up to verify_rows through the verify rows form
    // (every non-expert weight read once per chunk; experts per row), each row bitwise equal to the per-row loop;
    // set_nextn_row_loop(true) runs one row at a time (the identity reference of the device check).
    struct NextnOut {
        const std::uint32_t* top = nullptr;   // device [R]
        const float* hidden_pre = nullptr;    // device [R][hidden]: x before shared_head_norm
        const float* hidden_post = nullptr;   // device [R][hidden]: shared_head_norm(x)
    };
    static constexpr std::uint32_t kNextnRows = 32;  // widest nextn_rows call
    [[nodiscard]] bool has_nextn() const noexcept;
    void set_nextn_row_loop(bool on) noexcept;
    // The drafts' head from the next nextn_rows call on (the device check compares the forms on the same rows).
    void set_nextn_head(NextnHead head);
    // After enable_tp: nextn_rows runs tensor-parallel (this rank's heads, BF16 expert slots 8 + 8 and shared rows, joined;
    // the drafts' head on both ranks) instead of the one-node form. Both ranks must then make every nextn_rows call in the
    // same order (strix/spec/pair_round.hpp draft rounds). Bitwise the one-node block: the same joins as the trunk's.
    void set_nextn_tp(bool on);
    // outputs = trailing rows whose top / hidden the caller reads (default all). Rows [0, rows - outputs) only write the
    // block's caches (bitwise the full form's: the latent row and the indexer keys depend only on the row's input); their
    // pre / post rows are unspecified and their top is 0. want_top needs outputs >= 1. The row loop computes every row.
    static constexpr std::uint32_t kAllRows = ~0U;
    NextnOut nextn_rows(const std::uint32_t* tokens, const float* hidden, std::uint32_t rows, std::uint32_t pos0, bool want_top,
                        std::uint32_t outputs = kAllRows);
    // Logits of the last nextn_rows row (host copy; want_top must have been set): the device-vs-oracle check. Always the
    // full BF16 head's logits (recomputed from the row when the drafts use a q8 / q4 head).
    [[nodiscard]] const std::vector<float>& nextn_logits();
    // prefill() reports every prompt row's output_norm(hidden) (device rows, valid during the call) with its first position:
    // the MTP teacher input of the prompt. Empty function = off (the default; no extra work).
    void set_prompt_hidden(std::function<void(const float* rows, std::uint32_t p0, std::uint32_t n)> sink);
    // Device glue of the shared MTP chain (strix/spec/mtp_chain.hpp EngineChainDev) and a synchronising id read.
    float* alloc_f32(std::size_t floats);
    void free_f32(float* p) noexcept;
    void copy_f32_async(float* dst, const float* src, std::size_t floats);
    void read_ids(const std::uint32_t* device, std::uint32_t n, std::uint32_t* host);
    void read_f32(const float* device, std::size_t floats, float* host);  // synchronises (taps / checks)

    void set_timing(bool on) noexcept;  // per-stage device sync for StepTimes (a first-receipt aid, not a tuned path)
    [[nodiscard]] const StepTimes& times() const noexcept;

    // ---- exact tensor parallelism over two whole-model ranks (decode steps; strix/pair/gather_link.hpp) ----
    // After enable_tp every step() computes this rank's share and joins it with the peer's through `gather` (the shared
    // GatherLink::fn binding): attention by heads (gate, q_b, rope, K_b, query image, flash, V_b, gate) then o joined,
    // o_proj by output rows then joined; layer-0 FFN gate/up by rows (act joined), down by rows (joined); MoE routed
    // slots 8 + 8 (A8 probe / down at ids + 8 rank) and shared gate/up by rows, one join of [slot rows | shared act];
    // head by vocabulary rows (logits joined). Router, latent down/up, shared down, HC sites, norms, Engram, indexer
    // and the greedy pick stay replicated. Every split output is a whole per-head / per-row / per-slot result of the
    // same kernel and the joins are concatenations (the combine runs over all 16 slot rows in slot order): bitwise the
    // one-node step. Both ranks must run the same step sequence (each gather is a symmetric swap). forward_rows (verify
    // rows) splits the same way over its R rows (heads, o_proj / FFN / shared rows, routed slots 8 + 8 per row; the head
    // stays replicated): bitwise the one-node pass; both ranks run the same passes (strix/spec/pair_round.hpp). prefill()
    // and the next-token block (nextn_rows / nextn_logits, the drafter) run the one-node form on each rank that calls them
    // (no gather), unless set_nextn_tp(true) (nextn_rows then splits like a step on both ranks). Whole-model rank only.
    using TpGather = strix::pair::GatherFn;
    void enable_tp(std::uint32_t rank, TpGather gather);
    [[nodiscard]] std::uint32_t tp_gathers_per_step() const noexcept;  // 0 before enable_tp
    [[nodiscard]] std::string tp_receipt() const;                      // one line: rank, shares, gathers and bytes per step
    [[nodiscard]] std::uint64_t tensor_bytes(const std::string& name) const noexcept;  // loaded tensor bytes (0: absent)

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::hy4::hipbind
