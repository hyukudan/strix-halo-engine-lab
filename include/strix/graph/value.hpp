#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace strix::graph {

enum class ValueScope : std::uint8_t { step, visit, pass, model };

struct ValueDecl {
    std::string_view id;
    std::uint64_t streams = 1, width = 1;
    ValueScope scope = ValueScope::visit;
};

// L is the physical layer of the consuming visit.
enum class ValueSelector : std::uint8_t { current, absolute_layer, relative_layer, model_layer_slice };

struct ValueRef {
    std::string_view id;
    ValueSelector selector = ValueSelector::current;
    std::int64_t layer = 0;  // layer number for absolute_layer, signed offset for relative_layer
};

struct LayerVisit;

struct ResolvedValueRef {
    ValueDecl value;
    std::optional<std::uint32_t> logical_visit;
    std::optional<std::uint32_t> model_slice;
};

[[nodiscard]] ValueScope parse_value_scope(std::string_view);
// The identifier in the result borrows from the input.
[[nodiscard]] ValueRef parse_value_ref(std::string_view);
// The result borrows its identifier from the declaration. A null current visit
// denotes a graph edge. Absolute references there select the first visit.
[[nodiscard]] ResolvedValueRef resolve_value_ref(
    const ValueRef&, std::span<const ValueDecl>, std::span<const LayerVisit> schedule,
    const LayerVisit* current);

}  // namespace strix::graph
