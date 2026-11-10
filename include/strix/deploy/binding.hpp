#pragma once
#include "strix/deploy/plan.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <span>
#include <stdexcept>

namespace strix::deploy {
inline std::uint64_t byte_add(std::uint64_t a, std::uint64_t b) {
    if (b > UINT64_MAX - a) throw std::overflow_error("deployment byte sum overflow");
    return a + b;
}
inline std::uint64_t byte_mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > UINT64_MAX / a) throw std::overflow_error("deployment byte product overflow");
    return a * b;
}
inline bool pinned(const Spec& s, const std::string& name) {
    return s.pins.contains(name) || s.pins.contains("memory." + name);
}
// Fresh rank-ordered observations, before device owners. Never clone local facts
// for a remote node. Wire fields deliberately exclude private observation paths.
template<class Transport>
std::vector<DeviceFacts> exchange_pair_facts(Transport& transport, std::uint32_t rank,
                                           const DeviceFacts& local) {
    struct Wire {
        std::uint64_t magic, rank, version, cu, total, free, unified;
        std::array<char, 64> gfx, source;
    } mine{}, peer{};
    if (rank > 1) throw std::invalid_argument("deployment rank must be 0 or 1");
    mine.magic = 0x53544445504c3033ULL; mine.rank = rank;
    mine.version = local.gfx_target_version; mine.cu = local.cu_count;
    mine.total = local.mem_total_bytes; mine.free = local.mem_free_bytes; mine.unified = local.unified;
    if (local.gfx_target.size() >= mine.gfx.size() || local.source.size() >= mine.source.size())
        throw std::invalid_argument("deployment observation string exceeds wire limit");
    std::copy(local.gfx_target.begin(), local.gfx_target.end(), mine.gfx.begin());
    std::copy(local.source.begin(), local.source.end(), mine.source.begin());
    transport.exchange(std::as_bytes(std::span(&mine, 1)), std::as_writable_bytes(std::span(&peer, 1)));
    if (peer.magic != mine.magic || peer.rank != 1U-rank || peer.cu > UINT32_MAX || peer.unified > 1 ||
        peer.gfx.back() || peer.source.back()) throw std::runtime_error("deployment peer facts mismatch");
    DeviceFacts remote;
    remote.gfx_target = peer.gfx.data(); remote.source = peer.source.data();
    remote.gfx_target_version = peer.version; remote.cu_count = static_cast<std::uint32_t>(peer.cu);
    remote.mem_total_bytes = peer.total; remote.mem_free_bytes = peer.free; remote.unified = peer.unified != 0;
    std::vector<DeviceFacts> facts(2); facts[rank] = local; facts[1U-rank] = remote;
    return facts;
}
template<class Transport> void agree_plan(Transport& transport, const Plan& plan) {
    std::array<char, 16> mine{}, peer{};
    const auto id = plan.hash(); std::copy(id.begin(), id.end(), mine.begin());
    transport.exchange(std::as_bytes(std::span(mine)), std::as_writable_bytes(std::span(peer)));
    if (mine != peer) throw std::runtime_error("deployment peer plan hash mismatch");
    if (plan.refused) throw std::runtime_error("deployment refused: " + plan.refusal +
        (plan.reasons.empty() ? "" : "; " + plan.reasons.back()));
}
} // namespace strix::deploy
