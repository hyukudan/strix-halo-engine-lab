#include "strix/q51_slot_parallel_contract.hpp"

#include <array>
#include <limits>
#include <stdexcept>

namespace strix::q51_slot_parallel {
std::uint64_t required_rawdots_bytes(Shape shape) {
    (void)q51_grouped_down::required_bytes(shape);
    // Geometry was bounded before these products: K <= 10 and H <= 2560.
    return static_cast<std::uint64_t>(shape.top_k) * shape.hidden * 4U;
}

void validate(const Request& request, const Bindings& bindings) {
    if (!request.enabled || request.approved_numerical_identity != numerical_identity)
        throw std::invalid_argument("Q5_1 slot-parallel down requires explicit numerical identity");
    q51_grouped_down::validate(
        {true, q51_grouped_down::numerical_identity, request.shape},
        {bindings.intermediates, bindings.weights, bindings.routes, bindings.route_weights,
         bindings.status, bindings.accumulation});
    const auto bytes = required_rawdots_bytes(request.shape);
    const auto raw = bindings.rawdots;
    if (!raw.address || raw.address % 256U || raw.bytes != bytes ||
        raw.bytes > std::numeric_limits<std::uintptr_t>::max() - raw.address)
        throw std::invalid_argument("Q5_1 slot-parallel raw-dot view must be exact and 256-byte aligned");
    const auto end = raw.address + static_cast<std::uintptr_t>(raw.bytes);
    const std::array others{bindings.intermediates, bindings.weights, bindings.routes,
                            bindings.route_weights, bindings.status, bindings.accumulation};
    for (const auto other : others)
        if (raw.address < other.address + other.bytes && other.address < end)
            throw std::invalid_argument("Q5_1 slot-parallel raw-dot view aliases a grouped owner");
}
} // namespace strix::q51_slot_parallel
