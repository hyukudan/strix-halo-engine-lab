#pragma once
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace strix::deepseek41::prefill_xport {
// DS41 owns only the collective tags and wire identity. Slot leases, CRC and
// progress are implemented once by xport::Channel.
enum class Kind : std::uint32_t { control = 0, payload = 1, argmax = 2 };
struct Tag { Kind kind = Kind::payload; std::uint32_t layer = 0; std::uint64_t step = 0; };
struct Schedule {
    static constexpr std::size_t row_bytes(unsigned rows) { return rows; }
};
inline constexpr std::uint64_t kMagic = 0x3158465031345344ULL;
struct Trailer {
    std::uint64_t magic = kMagic, epoch = 0, plan = 0, sequence = 0, step = 0;
    std::uint32_t bytes = 0;
    std::uint16_t kind = 0, flags = 0;
    std::uint32_t layer = 0, rank = 0, crc32c = 0, prev_crc32c = 0;
};
static_assert(sizeof(Trailer) == 64);
struct Traits {
    using Payload = std::byte;
    using Schedule = prefill_xport::Schedule;
    using Tag = prefill_xport::Tag;
    using Kind = prefill_xport::Kind;
    using Trailer = prefill_xport::Trailer;
    static constexpr std::size_t kWidth = 1, kInlineCrc = 1024;
    static constexpr std::uint64_t kPlan = 0x31465031345344ULL, kMagic = prefill_xport::kMagic;
    static constexpr std::uint16_t kDeferred = 1, kPrevious = 2;
    static constexpr std::uint32_t kHeadLayer = 40;
    static constexpr bool kCrc = true, kCrcDeferred = false;
    static constexpr const char* kEnvironment = "STRIX_DS41_PF_XPORT_";
    static constexpr const char* kMemoryChoices = "host";
    static constexpr int memory_kind(std::string_view value) { return value == "host" ? 0 : -1; }
    static constexpr const char* crc_description(bool enabled, bool deferred) {
        return enabled ? (deferred ? "deferred" : "inline") : "off";
    }
};
} // namespace strix::deepseek41::prefill_xport
