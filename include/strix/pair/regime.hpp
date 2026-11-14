#pragma once
// Per-request two-node mode selection (regime auto-select: a lever faster only in some regimes ships with
// per-regime selection). Pure policy, shared by every engine's pair route and server scheduler; engines
// supply measured thresholds. Modes:
//   single    - the request runs on one node (the other node serves its own queue);
//   pipeline  - both nodes prefill the prompt as a two-stage layer pipeline, one node then decodes;
//   disaggregate - the idle node prefills, the K/V state moves to the node already decoding, which
//               batches the new stream with its active ones (decode never waits for prompt chunks).
#include <cstddef>

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
} // namespace strix::pair
