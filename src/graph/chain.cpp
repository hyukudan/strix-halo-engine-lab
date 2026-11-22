#include "strix/graph/chain.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace strix::graph {
namespace {

constexpr std::array<std::string_view, 49> names{
    "", "store.f32", "store.bf16", "store.f16", "store.a8g32", "store.e4m3",
    "residual.add", "residual.add.round_bf16", "scale.const", "scale.layer", "bias.add",
    "act.silu_mul", "act.swiglu_clamp", "act.swiglu_softcap", "act.gelu_tanh_mul",
    "act.sigmoid", "gate.sigmoid_mul", "softcap", "clip.l2", "norm.rms", "rope",
    "quant.a8g32", "quant.a4had", "quant.e4m3", "blocksum", "kv.append", "conv.tap",
    "mix.expand", "route.weight_mul", "finite.status", "emit.argmax", "emit.top2", "emit.topn",
    "norm.layer", "norm.group", "mod.affine", "rope.inverse", "act.swiglu_oai",
    "rotate.hadamard", "load.round.bf16", "store.iu4prep", "quant.i8rot", "emit.lse", "bias.by_class",
    "xch.publish", "xch.join", "wait.host", "recur.commit", "index_pool.append"
};

std::uint64_t chain_identity(const Chain& chain) {
    std::uint64_t hash = 14695981039346656037ULL;
    const auto add = [&](std::uint8_t byte) { hash = (hash ^ byte) * 1099511628211ULL; };
    add(chain.count);
    for (std::size_t i = 0; i < chain.count; ++i) {
        const auto& link = chain.links[i];
        const auto form = static_cast<std::uint16_t>(link.form);
        add(static_cast<std::uint8_t>(form));
        add(static_cast<std::uint8_t>(form >> 8U));
        add(link.arg);
    }
    return hash;
}

}  // namespace

std::string_view form_name(Form form) noexcept {
    const auto id = static_cast<std::size_t>(form);
    return id < names.size() ? names[id] : std::string_view{};
}

std::optional<Form> parse_form(std::string_view name) noexcept {
    const auto it = std::find(names.begin() + 1, names.end(), name);
    if (it == names.end()) return std::nullopt;
    return static_cast<Form>(it - names.begin());
}

ChainId ChainTable::intern(const Chain& chain) {
    if (chain.count > chain.links.size()) throw std::invalid_argument("chain: too many links");
    if (chain.count == 0) return 0;
    Chain key{};
    key.count = chain.count;
    for (std::size_t i = 0; i < key.count; ++i) {
        if (form_name(chain.links[i].form).empty()) throw std::invalid_argument("chain: invalid form");
        key.links[i] = chain.links[i];
    }
    const auto hash = chain_identity(key);
    const auto [first, last] = ids_.equal_range(hash);
    for (auto it = first; it != last; ++it) {
        const auto& candidate = at(it->second);
        if (candidate.count == key.count && candidate.links == key.links) return it->second;
    }
    if (chains_.size() == std::numeric_limits<ChainId>::max()) throw std::length_error("chain: ID space exhausted");
    const auto id = static_cast<ChainId>(chains_.size() + 1U);
    const auto it = ids_.emplace(hash, id);
    try {
        chains_.push_back(key);
    } catch (...) {
        ids_.erase(it);
        throw;
    }
    return id;
}

const Chain& ChainTable::at(ChainId id) const {
    static constexpr Chain empty{};
    return id == 0 ? empty : chains_.at(static_cast<std::size_t>(id) - 1U);
}

std::uint64_t ChainTable::identity(ChainId id) const {
    return chain_identity(at(id));
}

}  // namespace strix::graph
