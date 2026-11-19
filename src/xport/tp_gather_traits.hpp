#pragma once
// Wire identity of the tensor-parallel slot gathers on the shared Channel (xport::Channel<Traits>): plain bytes, the
// dist::Tag kinds by value, its own magic / plan so a peer running another exchange family is refused by the trailer.
// Slot leases, CRC, progress and device readiness are implemented once by xport::Channel.
// Home: shared (xport wire identity, model-agnostic). Bound by: src/xport/tp_gather_hip.hpp (pair gather link).
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace strix::xport::tp_gather {

// Values follow dist::TagKind (communicator.hpp) so the Runtime converts a dist::Tag by value.
enum class Kind : std::uint32_t { control = 0, half_sum = 1, raw_slots = 2, argmax = 3 };
struct Tag { Kind kind = Kind::raw_slots; std::uint32_t layer = 0; std::uint64_t step = 0; };
struct Schedule {
    static constexpr std::size_t row_bytes(unsigned rows) { return rows; }
};
inline constexpr std::uint64_t kMagic = 0x3150544852544150ULL;  // "PATRHTP1"
struct Trailer {
    std::uint64_t magic = kMagic, epoch = 0, plan = 0, sequence = 0, step = 0;
    std::uint32_t bytes = 0;
    std::uint16_t kind = 0, flags = 0;
    std::uint32_t layer = 0, rank = 0, crc32c = 0, prev_crc32c = 0;
};
static_assert(sizeof(Trailer) == 64);
struct Traits {
    using Payload = std::byte;
    using Schedule = tp_gather::Schedule;
    using Tag = tp_gather::Tag;
    using Kind = tp_gather::Kind;
    using Trailer = tp_gather::Trailer;
    static constexpr std::size_t kWidth = 1, kInlineCrc = 1024;
    static constexpr std::uint64_t kPlan = 0x31504854ULL, kMagic = tp_gather::kMagic;
    static constexpr std::uint16_t kDeferred = 1, kPrevious = 2;
    static constexpr std::uint32_t kHeadLayer = 0xFFFFU;
    // CRC on every message; above kInlineCrc it runs beside the wire (deferred), off the gather's critical path.
    static constexpr bool kCrc = true, kCrcDeferred = true;
    static constexpr const char* kEnvironment = "STRIX_TP_XPORT_";
    static constexpr const char* kMemoryChoices = "host";
    static constexpr int memory_kind(std::string_view value) { return value == "host" ? 0 : -1; }
    static constexpr const char* crc_description(bool enabled, bool deferred) {
        return enabled ? (deferred ? "deferred" : "inline") : "off";
    }
};

}  // namespace strix::xport::tp_gather
