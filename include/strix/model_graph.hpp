#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace strix::model_graph {

// Description only: no device, address, rank placement, kernel or execution policy.
enum class BlockKind { transformer, recurrent, auxiliary };
enum class AttentionFamily { none, dense, gdn, qsa, kda, mla, compressed_indexed, msa };
enum class MlpKind { dense, moe, shared_expert };
enum class NormPlacement { pre_attention, post_attention, pre_mlp, post_mlp, input, output };
enum class HyperConnectionKind { hc, mhc };
enum class AttachmentKind { ple, mtp, draft };
enum class EdgeKind { data, residual, hyper_connection, cross_layer };

// Formats/extents are symbolic requirements. Ownership describes model semantics
// (e.g. vocabulary/expert partitionability), never a deployment rank assignment.
struct TensorRequirement {
    std::string role, format;
    std::vector<std::string> shape;
    std::string ownership;
};
struct StateRequirement {
    TensorRequirement tensor;
    std::string lifetime;
};
struct NormDesc {
    std::string kind, epsilon;
    NormPlacement placement = NormPlacement::pre_attention;
};
struct MlpDesc {
    MlpKind kind = MlpKind::dense;
    std::uint64_t intermediate = 0, experts = 0, selected = 0;
};
struct HyperConnectionDesc {
    HyperConnectionKind kind = HyperConnectionKind::hc;
    std::uint64_t streams = 0, iterations = 0;
};
struct LayerDesc {
    std::uint64_t id = 0;
    BlockKind block = BlockKind::transformer;
    AttentionFamily attention = AttentionFamily::none;
    std::vector<MlpDesc> mlps;
    std::vector<NormDesc> norms;
    std::optional<HyperConnectionDesc> hyper_connection;
    std::vector<TensorRequirement> tensors;
    std::vector<StateRequirement> states;
    // Family-neutral semantic parameters (never placement or execution policy).
    std::map<std::string, std::string> semantics;
    std::map<std::string, std::uint64_t> dimensions;
};
struct AttachmentDesc {
    AttachmentKind kind = AttachmentKind::draft;
    std::string name, target;
    std::vector<LayerDesc> layers;
};
struct EdgeDesc {
    std::string from, to;
    EdgeKind kind = EdgeKind::data;
};
struct ModelGraph {
    static constexpr std::uint64_t schema_version = 1;
    std::string model, semantic_version;
    // Adapter/source contract names and arithmetic declarations, for diagnostics.
    std::map<std::string, std::string> semantics;
    std::map<std::string, std::uint64_t> dimensions;
    std::vector<TensorRequirement> inputs, outputs, tensors;
    std::vector<LayerDesc> layers;  // trunk order; optional attachments are separate
    std::vector<AttachmentDesc> attachments;
    std::vector<EdgeDesc> edges;
    std::vector<StateRequirement> states;  // model-wide sequence state
};

// Canonical JSON, with FNV-1a-64 over its UTF-8 bytes (not native struct storage).
// This is a descriptive identity, never an execution, numerical or cache identity.
[[nodiscard]] std::string canonical(const ModelGraph& graph);
[[nodiscard]] std::string identity(const ModelGraph& graph);
[[nodiscard]] std::string dump(const ModelGraph& graph);
void validate(const ModelGraph& graph);

}  // namespace strix::model_graph
