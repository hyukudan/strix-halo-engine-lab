#pragma once
#include "gemma4/engine/runtime.hpp"
#include "gemma4/features/features.hpp"
#include <random>

namespace feature_fixture {
namespace g = strix::gemma4;
inline g::ModelManifest model() {
    g::ModelManifest m; auto& c = m.config;
    m.model_hash = "synthetic-feature-weights-v1"; m.rope_hash = "synthetic-feature-rope-v1";
    c.hidden = 16; c.layers = 42; c.inter = 24; c.vocab = 37;
    c.heads = 4; c.kv_heads = 1; c.ple_dim = 8; c.window = 8; c.max_positions = 96;
    for (unsigned l = 0; l < c.layers; ++l) {
        const bool global = l % 6 == 5;
        c.layer.push_back({global ? g::AttentionKind::global : g::AttentionKind::sliding,8,l < 24 ? l : global ? 23U : 22U});
    }
    std::mt19937 rng(827);
    for (const auto& [name,shape] : g::required_tensors(c)) {
        const auto outputs = shape.size() == 2 ? std::uint32_t(shape[0]) : 1U;
        const auto inputs = std::uint32_t(shape.back());
        auto values = std::make_shared<std::vector<float>>(std::size_t(outputs)*inputs);
        for (auto& x : *values) {
            x = float(int(rng()%127)-63)*0.0009765625F;
            if (name.find("norm") != std::string::npos) x += 0.5F;
            if (name.ends_with("layer_scalar")) x = 0.984375F;
        }
        m.tensors.push_back({name,shape,{reinterpret_cast<const std::uint8_t*>(values->data()),outputs,inputs,inputs*4ULL,g::WeightFormat::f32},{}});
        m.storage.push_back(values);
    }
    return m;
}
struct Tables {
    std::vector<float> sliding = std::vector<float>(96*256), global = std::vector<float>(96*512);
    Tables() {
        for (std::size_t i = 0; i < sliding.size(); i += 2) sliding[i] = 1;
        for (std::size_t i = 0; i < global.size(); i += 2) global[i] = 1;
    }
    g::RopeTables view() const { return {sliding,global,96,"synthetic-feature-rope-v1"}; }
};
inline std::vector<std::uint32_t> tokens(std::size_t n, unsigned seed = 0) {
    std::vector<std::uint32_t> out;
    for (std::size_t i = 0; i < n; ++i) out.push_back(std::uint32_t((i*7+seed)%37));
    return out;
}
inline g::features::Request request(const g::features::Identity& identity) {
    g::features::Request r; r.identity = identity; r.prefix = tokens(13); r.code_ids = {36,0,11,11};
    r.branches = {{"a",tokens(21,1),19,{0,7,8,20,7}},{"b",tokens(9,2),8,{3,8}}};
    return r;
}
} // namespace feature_fixture
