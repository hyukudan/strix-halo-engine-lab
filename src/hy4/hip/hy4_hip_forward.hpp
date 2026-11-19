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

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace strix::hy4::hipbind {

struct ForwardOptions {
    std::vector<std::filesystem::path> parts;  // the pack's GGUF parts (all of them; the Engram table is read by row)
    std::uint32_t depth = 0;                   // 0: the pack's depth; else Config::truncated(depth)
    std::uint32_t first = 0, last = 0;         // trunk layers of this rank; last 0 = depth
    std::uint32_t context = 4096;              // positions this forward may hold
    std::uint32_t prompt_rows = 0;             // batched prompt window (rows per launch, multiple of 8, >= 16); 0 = none
    std::uint32_t verify_rows = 0;             // rows of a verify pass (forward_rows, 1..8); rings sized for them; 0 = none
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
    [[nodiscard]] void* stream_handle() const noexcept;    // the hipStream_t every device operation of this forward runs on
    void sync();
    // step_greedy on a head stage of any range: `state` = X from the previous stage (ignored with the embedding).
    std::uint32_t step_greedy(std::uint32_t token, std::vector<float>& state);
    // Back to position 0 for a repeat run of the same process: every cache row is written in position order before a
    // read, so a repeat reproduces the bits (the repeat's logits digest is the check).
    void reset() noexcept;
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

    void set_timing(bool on) noexcept;  // per-stage device sync for StepTimes (a first-receipt aid, not a tuned path)
    [[nodiscard]] const StepTimes& times() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::hy4::hipbind
