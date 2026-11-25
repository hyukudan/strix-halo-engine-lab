#pragma once
// A synthetic GLM-5.3 Flash projector file in memory for host tests: a GGUF v3 writer (FP32 tensors) and a small tower
// with the real metadata keys and tensor names.
#include "../src/glm/vision/glm_vision.hpp"

#include "strix/vision/gguf_weights.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace glm_vision_toy {

using namespace strix;

// ---- a GGUF v3 file in memory (FP32 tensors) ---------------------------------------------------------------------

using Value = std::variant<std::uint32_t, float, bool, std::string, std::vector<float>>;
struct TensorData {
    std::string name;
    std::vector<std::uint64_t> dims;  // GGUF order
    std::vector<float> data;
};

template <class T>
inline void put(std::string& b, T v) {
    b.append(reinterpret_cast<const char*>(&v), sizeof v);
}
inline void put_text(std::string& b, const std::string& s) {
    put<std::uint64_t>(b, s.size());
    b += s;
}

struct File {
    std::vector<std::pair<std::string, Value>> meta;
    std::vector<TensorData> tensors;
    std::uint64_t seed = 1;

    void set(const std::string& key, Value v) {
        for (auto& m : meta)
            if (m.first == key) {
                m.second = std::move(v);
                return;
            }
        meta.emplace_back(key, std::move(v));
    }
    void erase(const std::string& key) {
        std::erase_if(meta, [&](const auto& m) { return m.first == key; });
    }
    bool has(const std::string& name) const {
        return std::any_of(tensors.begin(), tensors.end(), [&](const auto& t) { return t.name == name; });
    }
    TensorData& tensor(const std::string& name) {
        for (auto& t : tensors)
            if (t.name == name) return t;
        throw std::logic_error("no tensor " + name);
    }
    void drop(const std::string& name) {
        std::erase_if(tensors, [&](const auto& t) { return t.name == name; });
    }
    // Deterministic values in offset + scale * [-1, 1).
    void add(const std::string& name, std::vector<std::uint64_t> dims, float scale, float offset = 0.0F) {
        std::size_t n = 1;
        for (const auto d : dims) n *= static_cast<std::size_t>(d);
        TensorData t{name, std::move(dims), std::vector<float>(n)};
        for (auto& v : t.data) {
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            v = offset + scale * static_cast<float>(static_cast<double>(seed >> 40U) / 8388608.0 - 1.0);
        }
        tensors.push_back(std::move(t));
    }
    std::string bytes() const {
        std::string b = "GGUF";
        put<std::uint32_t>(b, 3);
        put<std::uint64_t>(b, tensors.size());
        put<std::uint64_t>(b, meta.size());
        for (const auto& [key, v] : meta) {
            put_text(b, key);
            if (const auto* u = std::get_if<std::uint32_t>(&v)) {
                put<std::uint32_t>(b, 4);
                put<std::uint32_t>(b, *u);
            } else if (const auto* f = std::get_if<float>(&v)) {
                put<std::uint32_t>(b, 6);
                put<float>(b, *f);
            } else if (const auto* flag = std::get_if<bool>(&v)) {
                put<std::uint32_t>(b, 7);
                put<std::uint8_t>(b, *flag ? 1 : 0);
            } else if (const auto* list = std::get_if<std::vector<float>>(&v)) {
                put<std::uint32_t>(b, 9);
                put<std::uint32_t>(b, 6);
                put<std::uint64_t>(b, list->size());
                for (const float e : *list) put<float>(b, e);
            } else {
                put<std::uint32_t>(b, 8);
                put_text(b, std::get<std::string>(v));
            }
        }
        std::uint64_t offset = 0;
        for (const auto& t : tensors) {
            put_text(b, t.name);
            put<std::uint32_t>(b, static_cast<std::uint32_t>(t.dims.size()));
            for (const auto d : t.dims) put<std::uint64_t>(b, d);
            put<std::uint32_t>(b, 0);  // F32
            put<std::uint64_t>(b, offset);
            offset += (t.data.size() * 4U + 31U) / 32U * 32U;
        }
        while (b.size() % 32 != 0) b.push_back('\0');
        for (const auto& t : tensors) {
            const std::size_t start = b.size();
            b.append(reinterpret_cast<const char*>(t.data.data()), t.data.size() * 4U);
            b.append((t.data.size() * 4U + 31U) / 32U * 32U - (b.size() - start), '\0');
        }
        return b;
    }
    gguf::FileInfo info() const {
        const std::string b = bytes();
        std::istringstream in(b, std::ios::in | std::ios::binary);
        return gguf::read(in, b.size(), glm::vision_gguf_limits());
    }
    vision::GgufWeights weights() const {
        const std::string b = bytes();
        std::vector<std::byte> raw(b.size());
        std::memcpy(raw.data(), b.data(), b.size());
        return vision::GgufWeights(info(), std::move(raw));
    }
};

// 2 blocks, width 32, 2 heads (dim 16), SwiGLU 24, patch 2, 2x2 merge (cell 4), merge width 20, projection 40,
// projector MLP 12, 4 to 36 image rows (64 to 576 pixels).
inline constexpr std::uint64_t kW = 32, kM = 24, kD = 16, kC = 20, kP = 40, kF = 12, kPatch = 2, kMerge = 2;
inline File small_file(bool fused, bool biases, std::uint64_t projection = kP) {
    File f;
    f.set("general.architecture", std::string("clip"));
    f.set("clip.projector_type", std::string("glm5v"));
    f.set("clip.has_vision_encoder", true);
    f.set("clip.use_silu", true);
    f.set("clip.vision.block_count", std::uint32_t{2});
    f.set("clip.vision.embedding_length", std::uint32_t{kW});
    f.set("clip.vision.attention.head_count", std::uint32_t{2});
    f.set("clip.vision.feed_forward_length", std::uint32_t{kM});
    f.set("clip.vision.projection_dim", static_cast<std::uint32_t>(projection));
    f.set("clip.vision.patch_size", std::uint32_t{kPatch});
    f.set("clip.vision.spatial_merge_size", std::uint32_t{kMerge});
    f.set("clip.vision.image_min_pixels", std::uint32_t{64});
    f.set("clip.vision.image_max_pixels", std::uint32_t{576});
    f.set("clip.vision.attention.layer_norm_epsilon", 1e-5F);
    f.set("clip.vision.swiglu_clamp", 0.6F);
    f.set("clip.vision.image_mean", std::vector<float>{0.5F, 0.4F, 0.3F});
    f.set("clip.vision.image_std", std::vector<float>{0.25F, 0.5F, 0.2F});
    f.add("v.patch_embd.weight", {kPatch, kPatch, 3, kW}, 0.2F);
    f.add("v.patch_embd.weight.1", {kPatch, kPatch, 3, kW}, 0.2F);
    f.add("v.patch_embd.bias", {kW}, 0.1F);
    for (int l = 0; l < 2; ++l) {
        const std::string p = "v.blk." + std::to_string(l) + ".";
        f.add(p + "ln1.weight", {kW}, 0.2F, 1.0F);
        f.add(p + "ln2.weight", {kW}, 0.2F, 1.0F);
        // The same draws for both layouts: the fused tensor is the concatenation of q, k and v.
        std::vector<float> qkv, qkv_bias;
        for (const char* part : {"attn_q", "attn_k", "attn_v"}) {
            f.add(p + part + ".weight", {kW, kW}, 0.3F);
            qkv.insert(qkv.end(), f.tensors.back().data.begin(), f.tensors.back().data.end());
            f.add(p + part + ".bias", {kW}, 0.1F);
            qkv_bias.insert(qkv_bias.end(), f.tensors.back().data.begin(), f.tensors.back().data.end());
            if (fused || !biases) f.drop(p + part + ".bias");
            if (fused) f.drop(p + part + ".weight");
        }
        if (fused) {
            f.tensors.push_back({p + "attn_qkv.weight", {kW, 3 * kW}, qkv});
            if (biases) f.tensors.push_back({p + "attn_qkv.bias", {3 * kW}, qkv_bias});
        }
        f.add(p + "attn_q_norm.weight", {kD}, 0.3F, 1.5F);
        f.add(p + "attn_k_norm.weight", {kD}, 0.3F, 1.5F);
        f.add(p + "attn_out.weight", {kW, kW}, 0.2F);
        f.add(p + "attn_out.bias", {kW}, 0.1F);
        f.add(p + "ffn_up.weight", {kW, kM}, 0.4F);
        f.add(p + "ffn_gate.weight", {kW, kM}, 0.4F);
        f.add(p + "ffn_down.weight", {kM, kW}, 0.2F);
        f.add(p + "ffn_gate.bias", {kM}, 0.1F);
        f.add(p + "ffn_down.bias", {kW}, 0.1F);
        if (!biases)
            for (const char* name : {"attn_out.bias", "ffn_gate.bias", "ffn_down.bias"}) f.drop(p + name);
    }
    f.add("mm.patch_merger.weight", {kMerge, kMerge, kW, kC}, 0.15F);
    f.add("mm.patch_merger.bias", {kC}, 0.1F);
    f.add("mm.model.fc.weight", {kC, projection}, 0.3F);
    f.add("mm.post_norm.weight", {projection}, 0.2F, 1.0F);
    f.add("mm.post_norm.bias", {projection}, 0.1F);
    f.add("mm.up.weight", {projection, kF}, 0.4F);
    f.add("mm.gate.weight", {projection, kF}, 0.4F);
    f.add("mm.down.weight", {kF, projection}, 0.3F);
    f.add("mm.down.bias", {projection}, 0.1F);
    return f;
}

}  // namespace glm_vision_toy
