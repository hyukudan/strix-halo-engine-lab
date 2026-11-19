#pragma once
// Hy4-flash DFly2 drafter: the official checkpoint shard (model-00148-of-00148.safetensors, BF16, every tensor named
// model.dfly_layers.*) described as DATA on the shared tensor_map checker. The drafter has no embedding and no head of
// its own (the target's are used). Roles on the shared parallel-block-drafter contract (strix/spec/block_parallel.hpp,
// preset hy4_dfly2): component "dfly.fusion" = slot-0 fusion, "dfly.correction" = hidden correction, "dfly.attention.*"
// = the five MLA draft layers (shared MLA contract geometry, strix/hy4/presets.hpp dsa), "dfly.mlp.*" = dense SwiGLU,
// "dfly.confidence" = per-row confidence head, "dfly.norm" = norms.
#include <string>
#include <vector>

#include "strix/weight/admission.hpp"
#include "strix/weight/tensor_map.hpp"

namespace strix::weight::tensor_map {

[[nodiscard]] std::span<const Entry> hy4_dfly2_table();
// Geometry of the official drafter (config.json text_config + dfly_num_hidden_layers 5, intermediate 9216 from the shard).
[[nodiscard]] Model hy4_dfly2_model();
// Any safetensors index in the checker's form: extents reversed to GGUF order (contiguous axis first), dtype mapped to the
// GGML id (F32 0, F16 1, BF16 30). False + error for another dtype or an opaque (rank > 4) tensor.
[[nodiscard]] bool safetensors_infos(const TensorIndex& index, std::vector<gguf::TensorInfo>& out, std::string& error);

}  // namespace strix::weight::tensor_map
