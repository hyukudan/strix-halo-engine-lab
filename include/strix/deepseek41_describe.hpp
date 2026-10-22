#pragma once

#include "strix/deepseek41_model.hpp"
#include "strix/serve/json.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

// tensormeld/engine-describe-v1 for the deepseek41 engines (`--describe-json`), and the tensormeld/model-parts-v1
// descriptor it carries, generated from the model file's header alone (metadata and tensor table: no weight is read).
//
// Parts, per layer (bytes are the GGUF tensors' bytes):
//   L<l>.attn       anchor  q_a/q_b/kv/o_a/o_b (Q8_0), the attention norms and sinks, and the compressor / indexer of the
//                           layer (except layer first_decoder's KV-source path, see ced_bridge); TP: q_b by heads, o_a by
//                           groups and wo_b by K halves are split over the ranks, the rest is replicated
//                           (replicated_fraction = replicated bytes / part bytes); state: the window ring, the ratio-2
//                           carry and, on KV sources, the compressed KV + index keys per position.
//   L<l>.mix        anchor  hyper-connection mixes (attn and ffn sites), the FFN norm and the router.
//   L<l>.engram_kv  anchor  Engram layers: engram_kv and its norms. L<l>.engram_table: lookup, the table on disk.
//   L<l>.shared     branch  the shared expert (Q8_0); split by rows over the ranks.
//   L<l>.experts    branch  the routed experts (IQ2_XXS gate/up + Q2_K down); halves by expert id (EP), per-shard combine.
//   L<first_decoder-1>.ced_bridge  anchor  the KV-source path of the first decoder layer (its attn mix and norm, the
//                           compressor and the index-key projection), which every prompt row needs.
//   embed (lookup, layer -1) and head (layer n).
// Prefill rows come from the engine's exact suffix replay plan (ds41::plan_suffix_replay): encoder layers see every row,
// decoder layer l computes a bounded cone. Cuts are the pipeline boundary's side bytes (ds41_pipeline_wire.hpp).
namespace strix::deepseek41 {

struct PartsOptions {
    std::string model_id = "deepseek-v4.1-flash";
    std::string revision;          // e.g. "<file name>@<layout identity>"
    std::string numerics = "ds41-exact-v1";
    std::uint32_t nominal_context = 8192;  // context used for context-dependent cut bytes (candidate blocks)
};
[[nodiscard]] serve::Json model_parts(const TargetModel& model, const PartsOptions& options);

struct EngineDescribeOptions {
    std::string engine_name, revision, executable_sha256, backend = "hip";
    std::vector<std::string> roles;  // whole, pipeline-stage, prefill (needs state export), decode (needs state import)
    bool state_export = false, state_import = false;
    std::vector<std::string> state_transports;
    std::uint32_t max_positions = 4096;
    serve::Json argv_template;       // object role -> argv (placeholders in braces), or null
    std::string notes;
    PartsOptions parts;
};
// The six capabilities of the deepseek41 parts.
[[nodiscard]] std::vector<std::string> deepseek41_capabilities();
[[nodiscard]] serve::Json engine_describe(const TargetModel& model, const EngineDescribeOptions& options);

// "<file name>@<layout identity hex>": names the file and its tensor layout without hashing the weights.
[[nodiscard]] std::string model_revision(const std::string& file_name, const TargetModel& model);

// For the apps' --describe-json: reads the model file's header (metadata and tensor table only), fills the engine's
// revision (the build's source revision), its executable SHA-256 (/proc/self/exe) and the model revision, and returns
// the compact JSON (one line).
[[nodiscard]] std::string describe_engine_file(const std::filesystem::path& gguf, EngineDescribeOptions options);
[[nodiscard]] std::string source_revision();  // STRIX_SOURCE_REVISION at configure time, or "unknown"
[[nodiscard]] std::string self_executable_sha256();

}  // namespace strix::deepseek41
