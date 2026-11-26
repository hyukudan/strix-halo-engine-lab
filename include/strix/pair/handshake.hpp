#pragma once

#include "strix/pair/lockstep.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>

namespace strix::pair {

// One fixed-size host exchange, before either rank enters its first model collective. Strict fields
// describe the common contract. Offers are intentionally asymmetric, for policies such as a memory
// minimum. Specific bindings decide how to combine them after the strict fields have passed.
struct PairHello {
    static constexpr std::uint32_t kMagic = 0x31524850U;  // "PHR1"
    static constexpr std::uint32_t kVersion = 1;

    std::uint32_t magic = kMagic, version = kVersion;
    std::uint32_t protocol = 0, rank = 0;
    std::array<std::uint8_t, 32> executable_sha256{}, model_sha256{};
    std::array<char, 64> gpu_arch{};
    std::uint64_t profile = 0, route = 0, placement_reference = 0, context = 0;
    std::uint32_t placement = 0, transport_capabilities = 0, slots = 0, max_rows = 0;
    std::array<std::uint64_t, 10> contract{};
    std::array<std::uint64_t, 4> offers{};
};
static_assert(std::endian::native == std::endian::little, "the pair handshake requires little-endian hosts");
static_assert(std::is_trivially_copyable_v<PairHello> && sizeof(PairHello) == 304);
static_assert(offsetof(PairHello, contract) == 192 && offsetof(PairHello, offers) == 272);

inline void set_gpu_arch(PairHello& hello, std::string_view arch) {
    if (arch.empty() || arch.size() >= hello.gpu_arch.size()) throw ProtocolError("pair hello: invalid GPU architecture");
    hello.gpu_arch = {};
    std::copy(arch.begin(), arch.end(), hello.gpu_arch.begin());
}

// Return the first mismatch, so both peers can report the field before they close the transport.
[[nodiscard]] inline std::string_view hello_mismatch(const PairHello& mine, const PairHello& peer) noexcept {
    if (peer.magic != PairHello::kMagic || peer.version != PairHello::kVersion) return "wire version";
    if (peer.rank > 1U || mine.rank > 1U || peer.rank == mine.rank) return "rank (expected complementary ranks 0 and 1)";
    if (mine.protocol == 0 || peer.protocol != mine.protocol) return "engine protocol";
    if (peer.executable_sha256 != mine.executable_sha256) return "executable SHA-256";
    if (peer.model_sha256 != mine.model_sha256) return "model identity";
    if (peer.gpu_arch != mine.gpu_arch) return "GPU architecture";
    if (peer.profile != mine.profile) return "resolved profile fingerprint";
    if (peer.route != mine.route) return "resolved route fingerprint";
    if (peer.placement != mine.placement || peer.placement_reference != mine.placement_reference) return "placement and receipt";
    if (peer.transport_capabilities != mine.transport_capabilities) return "transport capabilities";
    if (peer.context != mine.context) return "context capacity";
    if (peer.slots != mine.slots) return "slot capacity";
    if (peer.max_rows != mine.max_rows) return "row capacity";
    if (peer.contract != mine.contract) return "engine contract";
    return {};
}

template<class Exchange>
[[nodiscard]] PairHello exchange_hello(const PairHello& mine, Exchange&& exchange,
                                      std::span<const std::string_view> fields = {}) {
    PairHello peer;
    exchange(std::as_bytes(std::span(&mine, 1)), std::as_writable_bytes(std::span(&peer, 1)));
    if (auto mismatch = hello_mismatch(mine, peer); !mismatch.empty()) {
        if (mismatch == "engine contract") {
            for (std::size_t i = 0; i < fields.size() && i < mine.contract.size(); ++i) {
                if (mine.contract[i] != peer.contract[i]) {
                    mismatch = fields[i];
                    break;
                }
            }
        }
        throw ProtocolError("pair hello mismatch: " + std::string(mismatch));
    }
    return peer;
}

}  // namespace strix::pair
