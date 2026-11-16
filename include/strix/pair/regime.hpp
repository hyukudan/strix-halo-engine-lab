#pragma once
// Per-request two-node mode selection (regime auto-select: a lever faster only in some regimes ships with
// per-regime selection). Pure policy, shared by every engine's pair route and server scheduler; engines
// supply measured thresholds. Modes:
//   single    - the request runs on one node (the other node serves its own queue);
//   pipeline  - both nodes prefill the prompt as a two-stage layer pipeline, one node then decodes;
//   disaggregate - the idle node prefills, the K/V state moves to the node already decoding, which
//               batches the new stream with its active ones (decode never waits for prompt chunks).
#include <cstddef>
#include <cstdint>
#include <span>

namespace strix::pair {
enum class Mode { single, pipeline, disaggregate };
struct Load {
    std::size_t prompt = 0;          // prompt tokens of the request being placed
    std::size_t chunks = 0;          // its pipeline chunks under the engine's chunk schedule
    std::size_t queued = 0;          // requests waiting behind it (both nodes)
    std::size_t active_decodes = 0;  // streams decoding on the decode node
    bool peer_idle = true;           // the other node has no prefill or decode of its own
};
struct Policy {
    std::size_t min_pipeline_chunks = 2;   // one chunk has nothing to overlap: the seam only adds latency
    std::size_t min_disaggregate_prompt = 512; // below this the prompt chunk stalls decode less than the move costs
};
[[nodiscard]] inline Mode choose(const Load& load, const Policy& policy) {
    // A busy peer or a queue: two independent nodes give more throughput than one shared prompt.
    if (!load.peer_idle || load.queued > 0) return Mode::single;
    if (load.active_decodes > 0)
        return load.prompt >= policy.min_disaggregate_prompt ? Mode::disaggregate : Mode::single;
    return load.chunks >= policy.min_pipeline_chunks ? Mode::pipeline : Mode::single;
}

// Length-aware two-stage pipeline rule for an idle pair (engines supply their measured costs).
// Per chunk of c rows both stages together cost P(c) = fixed_ms + ms_per_row x c. The critical path of
// N balanced chunks is stage A of chunk 0, then every chunk of stage B:
//   wall(N) = (N + 1) / 2 x P(n / N) + seam_ms + handoff_fixed_ms + handoff_ms_per_row x n.
// The chunk minimising wall wins; the pipeline is chosen only if that beats single_ms_per_row x n.
// This estimate averages rows across chunks and assumes balanced stages. Execution uses full chunks
// and a shorter tail: qualify uneven tails and unbalanced cuts before treating estimates as timings.
struct PipelineCost {
    double fixed_ms = 0.0;           // per chunk, both stages
    double ms_per_row = 0.0;         // per chunk row, both stages
    double seam_ms = 0.0;            // first-frame seam latency (reaches the critical path once)
    double handoff_fixed_ms = 0.0;   // state hand-off to the decode rank after the prefill
    double handoff_ms_per_row = 0.0;
    double single_ms_per_row = 0.0;  // one node with its own tiles, all in
};
struct PipelineChoice {
    Mode mode = Mode::single;
    std::uint32_t chunk = 0;         // rows per chunk of the best pipeline (0: no legal chunk)
    std::uint64_t chunks = 0;
    double pipeline_ms = 0.0;        // predicted wall of that pipeline
    double single_ms = 0.0;          // predicted one-node wall
};
[[nodiscard]] inline double pipeline_wall_ms(std::uint64_t rows, std::uint32_t chunk, const PipelineCost& cost) {
    if (rows == 0U || chunk == 0U) return 0.0;
    const auto chunks = static_cast<double>((rows + chunk - 1U) / chunk);
    const auto n = static_cast<double>(rows);
    return (chunks + 1.0) / 2.0 * (cost.fixed_ms + cost.ms_per_row * n / chunks) + cost.seam_ms +
        cost.handoff_fixed_ms + cost.handoff_ms_per_row * n;
}
// `candidates` in ascending order (ties keep the smaller chunk); `legal(chunk)` is the engine's tile rule.
// A chunk covering the whole prompt has nothing to overlap and is never a pipeline.
template <class Legal>
[[nodiscard]] PipelineChoice choose_pipeline(std::uint64_t rows, const PipelineCost& cost,
    std::span<const std::uint32_t> candidates, Legal&& legal) {
    PipelineChoice best;
    best.single_ms = cost.single_ms_per_row * static_cast<double>(rows);
    for (const auto chunk : candidates) {
        if (chunk == 0U || chunk >= rows || !legal(chunk)) continue;
        const auto wall = pipeline_wall_ms(rows, chunk, cost);
        if (best.chunk == 0U || wall < best.pipeline_ms) {
            best.chunk = chunk;
            best.chunks = (rows + chunk - 1U) / chunk;
            best.pipeline_ms = wall;
        }
    }
    if (best.chunk != 0U && best.pipeline_ms < best.single_ms) best.mode = Mode::pipeline;
    return best;
}
} // namespace strix::pair
