#pragma once
#include <cstdint>
#include <string_view>

namespace strix::q51_grouped_down {
inline constexpr std::string_view numerical_identity =
    "q51-grouped-down-probe-v1:original-f32-activation:virtual256-tree:ascending-slot-f32-fma:positive-zero-seed";
struct Shape { std::uint32_t hidden=1, intermediate=32, experts=1, top_k=1; };
struct Request { bool enabled=false; std::string_view approved_numerical_identity; Shape shape; };
struct DeviceBytes { std::uintptr_t address=0; std::uint64_t bytes=0; };
struct Bindings { DeviceBytes intermediates,weights,routes,route_weights,status,accumulation; };
struct RequiredBytes {
    std::uint64_t intermediates=0,weights=0,routes=0,route_weights=0,status=4,accumulation=0;
};
[[nodiscard]] RequiredBytes required_bytes(Shape shape);
// Pure host preflight. Six exact 16-byte-aligned views. Writable status and
// accumulation cannot alias any owner; readonly aliases are permitted.
void validate(const Request&,const Bindings&);
} // namespace strix::q51_grouped_down
