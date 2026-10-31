#include "pack.hpp"
#include "format.hpp"
#include "strix/model_content_hash.hpp"
#include <algorithm>
#include <map>
#include <set>

namespace strix::gemma4::exl3 {
namespace {
std::string shard_name(const serve::Json& value) {
    const auto s = value.as_string();
    if (s.empty() || s.find_first_of("/\\:") != std::string::npos || s == "." || s == "..")
        throw std::runtime_error("unsafe arm asset name");
    return s;
}
std::string tensor_class(const std::string& name) {
    if (name == "lm_head.weight") return "head";
    if (name.find(".mlp.") != std::string::npos) return "ffn";
    if (name.find(".self_attn.q_proj.") != std::string::npos || name.find(".self_attn.o_proj.") != std::string::npos) return "attn_qo";
    if (name.find(".self_attn.k_proj.") != std::string::npos || name.find(".self_attn.v_proj.") != std::string::npos) return "attn_kv";
    if (name.find(".per_layer_input_gate.") != std::string::npos || name.find(".per_layer_projection.") != std::string::npos ||
        name == "model.language_model.per_layer_model_projection.weight") return "ple_lin";
    throw std::runtime_error("arm tensor is not a supported linear");
}
}
std::string pack_arm(const std::filesystem::path& pack) {
    return model_detail::require(read_json(pack / "manifest.json"), "arm").as_string();
}
ModelManifest load_arm(const std::filesystem::path& pack, const std::filesystem::path& bf16) {
    using namespace model_detail;
    const auto manifest = read_json(pack / "manifest.json");
    if (require(manifest, "schema").as_string() != "g4-exl3-pack-v1" ||
        require(manifest, "identity").as_string() != identity ||
        require(manifest, "ple_table_format").as_string() != "BF16" ||
        require(manifest, "embedding_format").as_string() != "BF16")
        throw std::runtime_error("unsupported EXL3 arm identity");
    const auto& recipe = require(manifest, "recipe");
    if (!require(recipe, "scaled_hessian").as_bool() || require(recipe, "fake_hess").as_bool() ||
        require(recipe, "backend").as_string() != "exl3" || require(recipe, "codebook").as_string() != "mul1")
        throw std::runtime_error("unsupported EXL3 arm recipe");
    // The pack format is v1; the decode identity is the process-wide selection.
    ModelManifest model; model.identity = active_identity(); model.config = parse_text_config(bf16 / "config.json");
    auto expected = required_tensors(model.config);
    auto binary = std::make_shared<MappedFile>(pack / shard_name(require(manifest, "weights")));
    model.storage.push_back(binary);
    if (integer(require(manifest, "quantized_file_bytes")) != binary->size()) throw std::runtime_error("arm file size mismatch");
    auto shapes = expected; shapes["lm_head.weight"] = {model.config.vocab, model.config.hidden};
    std::set<std::string> seen;
    std::uint64_t frontier = 0;
    for (const auto& t : require(manifest, "tensors").as_array()) {
        const auto name = require(t, "name").as_string();
        const auto it = shapes.find(name);
        std::vector<std::uint64_t> shape;
        for (const auto& n : require(t, "shape").as_array()) shape.push_back(integer(n));
        if (!seen.insert(name).second || it == shapes.end() || shape != it->second || shape.size() != 2)
            throw std::runtime_error("unexpected arm tensor/shape");
        const auto kind = tensor_class(name);
        if (require(t, "class").as_string() != kind) throw std::runtime_error("arm class mismatch");
        const auto class_k = integer(require(require(manifest, "classes"), kind.c_str()));
        if (require(t, "format").as_string() == "BF16") {
            if (class_k != 16 || kind != "ple_lin" || integer(require(t, "bytes")) != shape[0] * shape[1] * 2)
                throw std::runtime_error("invalid BF16 arm linear exemption");
            continue;
        }
        if (require(t, "format").as_string() != "EXL3_MUL1" ||
            name == "model.language_model.embed_tokens.weight" || name == "model.language_model.embed_tokens_per_layer.weight")
            throw std::runtime_error("invalid quantized arm tensor class");
        const auto k = integer(require(t, "K")), offset = integer(require(t, "offset")), size = integer(require(t, "bytes"));
        if (k != class_k || (k != 4 && k != 5 && k != 6 && k != 8)) throw std::runtime_error("arm bitrate mismatch");
        LinearView view{binary->data() + std::min<std::uint64_t>(offset, binary->size()), unsigned(shape[0]), unsigned(shape[1]),
                        std::size_t(shape[1] * k / 8), WeightFormat::exl3_mul1};
        if (!geometry(view) || offset != frontier || offset > binary->size() || size > binary->size() - offset ||
            size != weight_storage_bytes(view)) throw std::runtime_error("invalid arm byte range");
        frontier += size;
        const auto hash = hash_bytes(view.data, std::size_t(size));
        if (hash != require(t, "sha256").as_string()) throw std::runtime_error("arm tensor hash mismatch");
        validate(view);
        model.tensors.push_back({name, shape, view, hash}); expected.erase(name);
    }
    if (frontier != binary->size() || seen.size() != 344 || !seen.contains("lm_head.weight")) throw std::runtime_error("incomplete arm payload/head");
    const auto index = read_json(bf16 / "model.safetensors.index.json");
    const MappedFile index_bytes(bf16 / "model.safetensors.index.json");
    if (hash_bytes(index_bytes.data(), index_bytes.size()) != require(recipe, "model_index_sha256").as_string())
        throw std::runtime_error("arm BF16 source index hash mismatch");
    const auto& map = require(index, "weight_map");
    for (const auto& [name, file] : map.as_object()) {
        (void)file;
        if (name.starts_with("model.language_model.") && !shapes.contains(name) && !frozen_compatibility_tensor(name))
            throw std::runtime_error("unexpected BF16 text source tensor: " + name);
        if (!expected.contains(name)) model.ignored_tensors.push_back(name);
    }
    std::map<std::string, SafeFile> files;
    for (const auto& [name, shape] : expected) {
        const auto filename = shard_name(require(map, name.c_str()));
        if (!files.contains(filename)) {
            auto f = read_safetensors(bf16 / filename);
            model.storage.push_back(f.storage); files.emplace(filename, std::move(f));
        }
        const auto& f = files.at(filename);
        const auto it = std::find_if(f.tensors.begin(), f.tensors.end(), [&](const auto& t) { return t.name == name; });
        if (it == f.tensors.end() || it->shape != shape || it->dtype != "BF16") throw std::runtime_error("invalid BF16 arm exemption");
        auto view = make_view(f.storage->data() + it->offset, WeightFormat::bf16, shape);
        validate_finite(view);
        model.tensors.push_back({name, shape, view, hash_bytes(view.data, std::size_t(it->bytes))});
    }
    // Name/hash pairs bind both the arm and every adopted BF16 tensor, including
    // the two gather tables. Metadata/paths cannot alias different model bytes.
    std::sort(model.tensors.begin(), model.tensors.end(), [](const auto& a, const auto& b) { return a.name < b.name; });
    strix::Sha256Fast digest;
    for (const auto& t : model.tensors) {
        digest.update({reinterpret_cast<const std::byte*>(t.name.data()), t.name.size()});
        digest.update({reinterpret_cast<const std::byte*>(t.source_hash.data()), t.source_hash.size()});
    }
    model.model_hash = strix::model_content_identity_hex(digest.finalize());
    return model;
}
} // namespace strix::gemma4::exl3
