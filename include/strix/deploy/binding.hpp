#pragma once
#include "strix/deploy/plan.hpp"
#include <algorithm>
#include <array>
#include <functional>
#include <span>
#include <stdexcept>

namespace strix::deploy {
inline std::uint64_t checked_add(std::uint64_t a, std::uint64_t b) {
    if (b > UINT64_MAX - a) throw std::overflow_error("deployment byte overflow");
    return a + b;
}
inline std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > UINT64_MAX / a) throw std::overflow_error("deployment byte overflow");
    return a * b;
}
inline std::uint64_t aligned(std::uint64_t n, std::uint64_t a) {
    return checked_mul(checked_add(n, a - 1) / a, a);
}
using Exchange = std::function<void(std::span<const std::byte>, std::span<std::byte>)>;
// Host transport only: rank-ordered observations and canonical plan agreement,
// before any device owner is constructed. Never invent a peer observation.
inline Plan resolve_pair(const Spec& spec, const FootprintProvider& provider,
                         unsigned rank, const DeviceFacts& local, const Exchange& exchange) {
    if (rank > 1) throw std::invalid_argument("deployment: pair rank must be 0 or 1");
    struct Wire {
        std::uint64_t magic, rank, version, cu, total, free, unified;
        std::array<char, 32> gfx{}, source{};
    } mine{}, peer{};
    mine.magic = 0x53544445504c3031ULL; mine.rank = rank;
    mine.version = local.gfx_target_version; mine.cu = local.cu_count;
    mine.total = local.mem_total_bytes; mine.free = local.mem_free_bytes; mine.unified = local.unified;
    std::copy_n(local.gfx_target.begin(), std::min(local.gfx_target.size(), mine.gfx.size()-1), mine.gfx.begin());
    std::copy_n(local.source.begin(), std::min(local.source.size(), mine.source.size()-1), mine.source.begin());
    exchange(std::as_bytes(std::span(&mine, 1)), std::as_writable_bytes(std::span(&peer, 1)));
    if (peer.magic != mine.magic || peer.rank != 1-rank || peer.cu > UINT32_MAX || peer.unified > 1 ||
        peer.gfx.back() || peer.source.back()) throw std::runtime_error("deployment: incompatible peer facts");
    DeviceFacts remote;
    remote.gfx_target_version = peer.version; remote.gfx_target = peer.gfx.data();
    remote.cu_count = static_cast<std::uint32_t>(peer.cu);
    remote.mem_total_bytes = peer.total; remote.mem_free_bytes = peer.free;
    remote.unified = peer.unified != 0; remote.source = peer.source.data();
    std::vector<DeviceFacts> devices(2);
    devices[rank] = local; devices[1-rank] = remote;
    auto plan = resolve(spec, devices, provider);
    const auto id = plan.hash();
    std::array<char, 16> own{}, other{};
    std::copy(id.begin(), id.end(), own.begin());
    exchange(std::as_bytes(std::span(own)), std::as_writable_bytes(std::span(other)));
    if (own != other) throw std::runtime_error("deployment: peer plan hash mismatch");
    return plan;
}
inline std::vector<DeviceFacts> inspection_devices(const Spec& spec, unsigned rank, const DeviceFacts& local) {
    auto devices = spec.devices;
    if (devices.empty()) devices.push_back(local);
    else {
        if (rank >= devices.size()) throw std::invalid_argument("deployment: rank outside supplied inventory");
        devices[rank] = local;
    }
    return devices;
}
inline void require_plan(const Plan& plan) {
    if (plan.refused) throw std::runtime_error("deployment refused: " + plan.refusal +
        (plan.reasons.empty() ? "" : "; " + plan.reasons.back()));
}
} // namespace strix::deploy
