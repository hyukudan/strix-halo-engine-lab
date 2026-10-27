#pragma once
// Metadata-only layer-45 inventory. Never opens tensor payloads or constructs a HIP object.
#include "../src/glm/engine/glm_json.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <utility>

namespace strix::glm::rank_defaults {
struct ModelFacts { bool mtp_present = false, mtp_complete = false; std::string reason = "missing mtp.safetensors"; };
struct MtpTensorRole {
    std::string name, dtype;
    std::vector<std::uint64_t> shape;
    bool trellis = false;  // Last dimension is 16*b, integer b in [1,8].
};
inline std::vector<MtpTensorRole> mtp_tensor_roles() {
    const std::string p = "model.language_model.layers.45.";
    std::vector<MtpTensorRole> roles;
    auto fixed = [&](const std::string& n, const char* dtype, std::vector<std::uint64_t> shape) {
        roles.push_back({p + n, dtype, std::move(shape), false});
    };
    for (const char* n : {"input_layernorm.weight", "post_attention_layernorm.weight", "enorm.weight", "hnorm.weight",
                         "shared_head.norm.weight"}) fixed(n, "BF16", {4096});
    fixed("self_attn.q_a_layernorm.weight", "BF16", {1536});
    fixed("self_attn.kv_a_layernorm.weight", "BF16", {512});
    fixed("self_attn.kv_b_proj.weight", "F16", {32768, 512});
    fixed("self_attn.indexer.wk.weight", "F16", {128, 4096});
    fixed("self_attn.indexer.index_kpool_compress_gate", "F16", {128, 4096});
    fixed("self_attn.indexer.weights_proj.weight", "F16", {32, 4096});
    fixed("self_attn.indexer.k_norm.weight", "F16", {128});
    fixed("self_attn.indexer.k_norm.bias", "F16", {128});
    fixed("self_attn.indexer.index_kpool_compress_ape", "F32", {4, 128});
    fixed("mlp.gate.weight", "F16", {288, 4096});
    fixed("mlp.gate.e_score_correction_bias", "F32", {288});
    auto linear = [&](const std::string& n, std::uint64_t in, std::uint64_t out) {
        roles.push_back({p + n + ".trellis", "I16", {in / 16, out / 16, 16}, true});
        fixed(n + ".suh", "F16", {in});
        fixed(n + ".svh", "F16", {out});
    };
    linear("eh_proj", 8192, 4096);
    linear("self_attn.q_a_proj", 4096, 1536);
    linear("self_attn.kv_a_proj_with_mqa", 4096, 512);
    linear("self_attn.indexer.wq_b", 1536, 4096);
    linear("self_attn.q_b_proj", 1536, 16384);
    linear("self_attn.o_proj", 16384, 4096);
    for (unsigned e = 0; e <= 288; ++e) {
        const std::string m = e == 288 ? "mlp.shared_experts." : "mlp.experts." + std::to_string(e) + ".";
        linear(m + "gate_proj", 4096, 2048);
        linear(m + "up_proj", 4096, 2048);
        linear(m + "down_proj", 2048, 4096);
    }
    return roles;
}
inline std::vector<std::string> mtp_tensor_names() {
    std::vector<std::string> names;
    for (const auto& role : mtp_tensor_roles()) names.push_back(role.name);
    return names;
}
inline ModelFacts probe_model_mtp(const std::filesystem::path& dir) {
    ModelFacts facts;
    try {
        facts.mtp_present = std::filesystem::is_regular_file(dir / "mtp.safetensors");
        // Match PackReader: unique weight_map shards, then every unindexed safetensors shard.
        std::set<std::string> listed;
        std::vector<std::string> files;
        const auto index = dir / "model.safetensors.index.json";
        if (std::filesystem::is_regular_file(index)) {
            std::ifstream f(index, std::ios::binary);
            const std::string text{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
            const auto json = engine::detail::parse_json(text.data(), text.data() + text.size());
            for (const auto& [name, shard] : json.obj().at("weight_map").obj()) {
                (void)name;
                if (listed.insert(shard.str()).second) files.push_back(shard.str());
            }
        }
        std::vector<std::string> unlisted;
        for (const auto& e : std::filesystem::directory_iterator(dir)) {
            const auto name = e.path().filename().string();
            if (e.is_regular_file() && e.path().extension() == ".safetensors" && !listed.count(name))
                unlisted.push_back(name);
        }
        std::sort(unlisted.begin(), unlisted.end());
        files.insert(files.end(), unlisted.begin(), unlisted.end());
        struct Metadata { engine::detail::Json json; std::uint64_t payload; };
        std::map<std::string, Metadata> tensors;
        for (const auto& file : files) {
            const auto path = dir / file;
            std::ifstream f(path, std::ios::binary);
            unsigned char bytes[8] = {};
            if (!f.read(reinterpret_cast<char*>(bytes), 8)) throw std::runtime_error("short header length: " + file);
            std::uint64_t length = 0;
            for (unsigned i = 0; i < 8; ++i) length |= std::uint64_t{bytes[i]} << (8 * i);
            if (length == 0 || length > (std::uint64_t{64} << 20)) throw std::runtime_error("invalid header length: " + file);
            const auto size = std::filesystem::file_size(path);
            if (size < 8 || length > size - 8) throw std::runtime_error("short JSON header: " + file);
            std::string header(static_cast<std::size_t>(length), '\0');
            if (!f.read(header.data(), static_cast<std::streamsize>(length))) throw std::runtime_error("short JSON header: " + file);
            const auto json = engine::detail::parse_json(header.data(), header.data() + header.size());
            for (const auto& [name, metadata] : json.obj()) {
                if (name == "__metadata__") continue;
                if (name.starts_with("model.language_model.layers.45.")) facts.mtp_present = true;
                if (!tensors.emplace(name, Metadata{metadata, size - 8 - length}).second)
                    throw std::runtime_error("tensor defined twice: " + name);
            }
        }
        const std::map<std::string, std::uint64_t> sizes = {
            {"BOOL", 1}, {"U8", 1}, {"I8", 1}, {"F8_E4M3", 1}, {"F8_E5M2", 1},
            {"U16", 2}, {"I16", 2}, {"F16", 2}, {"BF16", 2},
            {"U32", 4}, {"I32", 4}, {"F32", 4}, {"U64", 8}, {"I64", 8}, {"F64", 8}};
        auto integer = [](const engine::detail::Json& j) -> std::uint64_t {
            const double n = j.num();
            if (!std::isfinite(n) || n < 0 || n >= std::ldexp(1.0, 64) || std::floor(n) != n)
                throw std::runtime_error("invalid nonnegative integer");
            return static_cast<std::uint64_t>(n);
        };
        for (const auto& role : mtp_tensor_roles()) {
            const auto& name = role.name;
            if (!tensors.count(name)) { facts.reason = "incomplete head: missing " + name; return facts; }
            try {
                const auto& tensor = tensors.at(name);
                const auto& metadata = tensor.json.obj();
                if (!metadata.count("dtype")) throw std::runtime_error("missing dtype");
                const auto dtype = metadata.at("dtype").str();
                if (!sizes.count(dtype)) throw std::runtime_error("unknown dtype " + dtype);
                if (!metadata.count("shape")) throw std::runtime_error("missing shape");
                std::vector<std::uint64_t> shape;
                for (const auto& dim : metadata.at("shape").arr()) shape.push_back(integer(dim));
                if (dtype != role.dtype) throw std::runtime_error("role dtype must be " + role.dtype);
                auto expected_shape = role.shape;
                if (role.trellis && shape.size() == 3 && shape[2] >= 16 && shape[2] <= 128 && shape[2] % 16 == 0)
                    expected_shape[2] = shape[2];
                if (shape != expected_shape) throw std::runtime_error("role shape mismatch");
                std::uint64_t expected = sizes.at(dtype);
                for (const auto dim : shape) {
                    if (dim != 0 && expected > std::numeric_limits<std::uint64_t>::max() / dim)
                        throw std::runtime_error("tensor byte count overflow");
                    expected *= dim;
                }
                if (!metadata.count("data_offsets")) throw std::runtime_error("missing data_offsets");
                const auto& offsets = metadata.at("data_offsets").arr();
                if (offsets.size() != 2) throw std::runtime_error("invalid data_offsets");
                const auto begin = integer(offsets[0]), end = integer(offsets[1]);
                if (end < begin || end - begin != expected) throw std::runtime_error("tensor byte count mismatch");
                if (end > tensor.payload) throw std::runtime_error("tensor offsets exceed file payload");
            } catch (const std::exception& e) {
                throw std::runtime_error(name + ": " + e.what());
            }
        }
        facts.mtp_complete = true;
        facts.reason = "complete layer-45 inventory";
    } catch (const std::exception& e) { facts.reason = std::string("invalid MTP header: ") + e.what(); }
    return facts;
}
}  // namespace strix::glm::rank_defaults
