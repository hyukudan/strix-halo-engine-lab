#pragma once

#include "gemma4/engine/coordinator.hpp"
#include "gemma4/model/model.hpp"
#include "gemma4/serve/adapter.hpp"
#include <optional>

namespace strix::gemma4::features {

using strix::serve::Json;
struct Identity {
    std::string weights, route, tokenizer, chat_template, rope;
    std::string decode = arithmetic_id, prefill = "prefill-exact", cache = "F32";
    [[nodiscard]] Json json() const;
    bool operator==(const Identity&) const = default;
};
// Route includes original tensor formats/shapes/strides; template hash identifies
// the native renderer source. Empty tokenizer denotes raw-token-only input.
[[nodiscard]] Identity freeze(const ModelManifest& model, std::string tokenizer_hash = {});
[[nodiscard]] Identity backend_identity(const engine::Backend& backend);
struct Branch {
    std::string id;
    std::vector<std::uint32_t> tokens;
    std::size_t read_position = 0;
    std::vector<std::size_t> option_close_positions;
};
struct Request {
    Identity identity;
    std::optional<std::vector<std::uint32_t>> prefix;
    std::string stream_handle;
    std::vector<Branch> branches;
    std::vector<std::uint32_t> code_ids;
    bool retain_prefix = false;
    bool release_prefix = false;
    std::size_t chunk_rows = 8; // 1 is the exact serial fallback
};
struct HiddenRow {
    std::size_t branch_position = 0;
    std::uint64_t absolute_position = 0;
    std::vector<float> values;
};
struct BranchResult {
    std::string id;
    HiddenRow read;
    std::vector<HiddenRow> options;
    std::vector<float> code_logits; // capped, caller code_ids order, at read only
    std::size_t processed_rows = 0;
};
struct Result {
    Identity identity;
    std::string stream_handle, before, after;
    std::uint64_t frontier = 0;
    std::vector<BranchResult> branches;
    std::vector<std::uint32_t> code_ids;
    double prefix_ms = 0, branch_ms = 0;
    [[nodiscard]] Json json(bool include_features = true) const;
};

// Single-owner API, like Backend. Serialize calls with ordinary generation.
// Prefixes are immutable after admission. Branch transactions are always discarded.
class Exporter {
public:
    Exporter(engine::Backend& backend, Identity identity, std::size_t max_cached_prefixes = 8);
    [[nodiscard]] Result export_features(const Request& request);
    // Borrow an existing canonical stream. Caller guarantees matching identity and exclusive ownership.
    [[nodiscard]] Result export_features(const Request& request, engine::StreamState& committed_prefix);
    void release(const std::string& handle);
    [[nodiscard]] const Identity& identity() const { return identity_; }
private:
    struct Prefix { std::unique_ptr<engine::StreamState> state; std::uint32_t admitted = 0; };
    engine::Backend& backend_;
    Identity identity_;
    std::size_t limit_;
    std::uint64_t sequence_ = 0;
    std::map<std::string,Prefix> prefixes_;
};
// Strict portable JSON surface. Chat rendering uses the existing family adapter.
[[nodiscard]] Request parse_request(const Json& json, const gemma4::serve::Adapter* adapter = nullptr);
void write_npz(const std::filesystem::path& path, const Result& result);

} // namespace strix::gemma4::features
