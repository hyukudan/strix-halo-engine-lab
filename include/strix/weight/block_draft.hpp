#pragma once

#include "strix/weight/admission.hpp"
#include <map>

namespace strix::weight {

// Shared, version-one parallel block decoder schema. The target supplies its
// embedding, output head, real vocabulary and captured feature rows separately.
struct BlockDraftConfig {
    std::uint32_t hidden{}, intermediate{}, layers{}, heads{}, kv_heads{}, head_dim{}, vocab{};
    std::uint32_t block{}, group{}, taps{}, rank{}, top_k{}, window{}, mask{};
    std::vector<std::uint32_t> target_layers;
    double epsilon{}, theta{}, softcap{};
    bool post_softcap = true;
};
using DraftShapes = std::map<std::string, std::vector<std::uint64_t>>;
BlockDraftConfig parse_block_draft_config(std::string_view json);
DraftShapes block_draft_shapes(const BlockDraftConfig&);
// Throws when the parsed geometry cannot bind to the version-one candidate adapter.
void validate_block_candidate_geometry(const BlockDraftConfig&);

struct BlockDraftCheckpoint {
    BlockDraftConfig config;
    TensorIndex index;
    Digest256 config_digest{}; // exact JSON bytes, distinct from the header digest
    std::uint64_t payload_bytes{};
};
// Header only. No payload is read, interpreted, imported or executed here.
// Uses the common bounded, duplicate-key rejecting safetensors reader.
// The source must carry a caller-chosen checkpoint name (Source::shard_name), unique per checkpoint: tensor ids
// are scoped by that name; an unnamed source is refused. Identity of the bytes is digest_block_draft_payload.
Status index_block_draft(std::shared_ptr<const Source>, std::string_view config_json,
                         BlockDraftCheckpoint&);
// The same admission with a config source checked for size before any config read.
Status index_block_draft(std::shared_ptr<const Source>, const Source& config, BlockDraftCheckpoint&);
// Explicit payload operation for a caller-owned trusted source. Not performed
// by admission. Digests bind bytes, not merely shapes or filesystem identity.
Status digest_block_draft_payload(const BlockDraftCheckpoint&,
                                  std::map<std::string, Digest256>&);
Status read_block_draft_tensor(const BlockDraftCheckpoint&, std::string_view name,
                               std::vector<float>&);
} // namespace strix::weight
