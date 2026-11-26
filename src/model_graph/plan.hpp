#pragma once

#include "strix/model_graph.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strix::model_graph {

// A descriptive, ordered logical schedule, not a new executor or cache identity.
// Family operations and arithmetic remain engine-owned; kind is a shared category.
enum class OpKind { embedding, hyper_connection, normalization, projection, state_update,
                    attention, router, expert, combine, head, selection, family_specific };
enum class BufferLifetime { decode_step, sequence };
struct PlanBuffer {
    std::string id;
    std::uint64_t bytes = 0;
    BufferLifetime lifetime = BufferLifetime::decode_step;
    std::uint64_t first_use = 0, last_use = 0;
    // Sequence storage is symbolic: bytes + bytes_per_position * retained positions.
    // This is not the physical allocator's capacity or an admission estimate.
    std::uint64_t bytes_per_position = 0;
    std::int64_t layer = -1;
    std::uint32_t rank = 0;
    std::string state_role;  // graph StateRequirement role, empty for activations
    bool external = false;  // source outside the selected layer range
    bool live_in = false, live_out = false;
    // Nonempty when byte sizing is deliberately unprojected; refers to graph shape.
    // A symbolic activation may report bytes=0 only with a nonempty extent.
    std::string extent;
};
struct PlanNode {
    OpKind kind = OpKind::family_specific;
    std::string operation, site, owner, arithmetic_identity;
    std::int64_t layer = -1;
    std::uint32_t rank = 0;
    std::vector<std::string> inputs, outputs, state_reads, state_writes, weights;
    std::vector<std::uint64_t> dependencies;  // earlier node indices
    // Logical routed-expert ownership; other operations are rank-local replicas
    // of the existing logical plan, not a claim about kernel/collective dispatch.
    std::uint32_t expert_begin = 0, expert_end = 0;
    bool graph_safe_candidate = true, dynamic_shape = false;
};
struct ExecutionPlan {
    std::string graph_identity, source;
    std::uint32_t rank = 0, ranks = 0;
    std::uint64_t first_layer = 0, end_layer = 0;
    std::vector<std::string> owners;  // engine-supplied implementation owner vocabulary
    std::vector<PlanBuffer> buffers;
    std::vector<PlanNode> nodes;
    // Optional descriptive policy and coverage, never executor inputs. Omitted
    // from dumps when empty so existing projections retain their serialization.
    std::map<std::string, std::string> policies;
    std::vector<std::string> engine_local;
};

void validate(const ExecutionPlan& plan, const ModelGraph& graph);
[[nodiscard]] std::string dump(const ExecutionPlan& plan);

}  // namespace strix::model_graph
