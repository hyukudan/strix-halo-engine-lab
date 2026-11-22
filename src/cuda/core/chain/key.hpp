#pragma once

#include "strix/graph/chain.hpp"

namespace strix::cuda::core::chain {

using graph::ChainArgs;
using graph::Form;

enum class Scope : std::uint8_t { element, warp, row, unsupported };
enum class Arg : std::uint8_t { none, scalar, aux, state };

struct FormInfo {
    Scope scope;
    Arg arg;
};

[[nodiscard]] constexpr FormInfo form_info(Form form) {
    switch (form) {
#define STRIX_CUDA_CHAIN_FORM(id, scope, arg) case Form::id: return {Scope::scope, Arg::arg};
#include "forms.inc"
#undef STRIX_CUDA_CHAIN_FORM
    default: return {Scope::unsupported, Arg::none};
    }
}

template <Form F, std::uint8_t Slot = 0>
struct Step {
    static constexpr graph::Link link{F, Slot};
};

template <class... Steps>
struct List {
    static_assert(sizeof...(Steps) <= 8);
    static constexpr graph::Chain description{{Steps::link...}, sizeof...(Steps)};
};

// Package-local chain component of the catalog key. The fixed-width encoding
// preserves order and bindings without relying on the local ChainTable IDs.
// The execution family, geometry and numeric contract belong to the caller.
struct KernelKey {
    std::array<std::uint32_t, 8> links{};
    std::uint8_t count = 0;
    Scope scope = Scope::element;
    friend constexpr bool operator==(const KernelKey&, const KernelKey&) = default;
};

[[nodiscard]] constexpr std::optional<KernelKey> kernel_key(const graph::Chain& chain) {
    KernelKey key{};
    if (chain.count > chain.links.size()) return std::nullopt;
    key.count = chain.count;
    for (std::size_t i = 0; i < chain.count; ++i) {
        const auto [form, slot] = chain.links[i];
        const auto [scope, arg] = form_info(form);
        if (scope == Scope::unsupported) return std::nullopt;
        const auto capacity = arg == Arg::none ? 1U : arg == Arg::state ? 2U : 4U;
        if (slot >= capacity) return std::nullopt;
        if (scope > key.scope) key.scope = scope;
        key.links[i] = (static_cast<std::uint32_t>(form) << 8U) | slot;
    }
    return key;
}

template <class ChainList>
inline constexpr auto static_key = kernel_key(ChainList::description);

template <Form F>
inline constexpr Scope form_scope = form_info(F).scope;

template <class ChainList>
inline constexpr Scope chain_scope =
    static_key<ChainList> ? static_key<ChainList>->scope : Scope::unsupported;

}  // namespace strix::cuda::core::chain
