#pragma once
// CPU-only host placement for an ascending, single-pass expert schedule.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace strix::ds41 {

enum class HostPinOrder { layer, deadline, early, interleave };

inline HostPinOrder parse_host_pin_order(std::string_view value) {
    if (value == "layer") return HostPinOrder::layer;
    if (value == "deadline") return HostPinOrder::deadline;
    if (value == "early") return HostPinOrder::early;
    if (value == "interleave") return HostPinOrder::interleave;
    throw std::invalid_argument("--host-pin-order must be layer, deadline, early or interleave");
}
inline const char* host_pin_order_name(HostPinOrder order) {
    return order == HostPinOrder::deadline ? "deadline" : order == HostPinOrder::early ? "early"
         : order == HostPinOrder::interleave ? "interleave" : "layer";
}

// Zero bytes means fully device resident. The first S nonzero layers already occupy
// the idle-preload slots, regardless of tier. Deadline placement keeps those on FILE
// and pins the longest affordable suffix of the remaining streamed schedule. In
// particular, small decoder tails are considered before full encoder layers.
// Stop at the first non-fitting layer: filling the remainder with earlier HOST
// layers would put a host-copy burst before a later FILE deadline. The loader has
// no independent disk lookahead past that burst (its four staging pages are bounded).
// This is a suffix policy, not a general heterogeneous knapsack optimizer. It avoids
// internal HOST/FILE transitions and keeps the disk prefix continuous at short rows.
inline std::vector<std::uint8_t> deadline_host_layers(std::span<const std::uint64_t> bytes,
                                                    std::uint32_t slots, double budget_bytes) {
    std::vector<std::uint8_t> host(bytes.size(), 0);
    std::vector<std::size_t> streamed;
    for (std::size_t l = 0; l < bytes.size(); ++l)
        if (bytes[l]) streamed.push_back(l);
    const auto preloaded = std::min<std::size_t>(std::max(1U, slots), streamed.size());
    for (std::size_t j = streamed.size(); j > preloaded;) {
        const auto l = streamed[--j];
        if (budget_bytes < static_cast<double>(bytes[l])) break;
        budget_bytes -= static_cast<double>(bytes[l]);
        host[l] = 1;
    }
    return host;
}

// Early placement: the legacy ascending fill (every streamed layer that still fits, in
// layer order), minus the first S streamed layers that the idle preload already holds.
// Pinned layers then sit at the start of the schedule, where little compute hides a
// FILE read, while later FILE reads overlap the compute of the pinned ones (measured: the
// suffix order was no faster than the legacy one at 30K for that reason).
inline std::vector<std::uint8_t> early_host_layers(std::span<const std::uint64_t> bytes,
                                                 std::uint32_t slots, double budget_bytes) {
    std::vector<std::uint8_t> host(bytes.size(), 0);
    std::size_t seen = 0;
    const std::size_t preloaded = std::max(1U, slots);
    for (std::size_t l = 0; l < bytes.size(); ++l) {
        if (!bytes[l]) continue;
        if (seen++ < preloaded) continue;
        if (budget_bytes < static_cast<double>(bytes[l])) continue;
        budget_bytes -= static_cast<double>(bytes[l]);
        host[l] = 1;
    }
    return host;
}

// Interleaved placement: a two-resource schedule of the streamed layers in execution order. The copy stream moves every
// streamed byte (host or file) at pcie_gbps; a file-tier layer is first read from the disk at disk_gbps into the file
// ring (ring_bytes of pinned pages), which can run ahead of the copies while it has room. With the layer and early
// orders the file layers come as one block after the host ones, so the disk idles through the host copies and the copy
// stream then waits for the disk (measured on one RTX: 61.5 GB host at 25 GB/s then 12.2 GB file at 6.6 GB/s, serial).
// Here a layer goes to FILE when bytes must come from the file anyway and its read, started when the disk and the ring
// are free, ends before the copy stream would finish that layer as a host one (the read hides behind earlier copies);
// otherwise HOST while the budget lasts, FILE when it is spent. Unlike deadline / early, the first S streamed layers
// (the slots' preload, loaded after each run) are pinned first: with back-to-back requests there is no idle time and
// a file preload (7.6 GB: 1.2 s from the disk vs 0.19 s from pinned memory) ran into the next run (measured: 1.25 s
// wait at layer 0). Only the source of the same bytes changes: no resident set, permutation or arithmetic changes.
// The rates only steer the placement (a wrong rate costs time, never bits).
inline std::vector<std::uint8_t> interleave_host_layers(std::span<const std::uint64_t> bytes, std::uint32_t slots,
                                                      double budget_bytes, double ring_bytes, double pcie_gbps = 25.0,
                                                      double disk_gbps = 6.6) {
    std::vector<std::uint8_t> host(bytes.size(), 0);
    const std::size_t preloaded = std::max(1U, slots);
    double rest = 0.0;
    std::size_t seen = 0;
    for (std::size_t l = 0; l < bytes.size(); ++l) {
        if (!bytes[l]) continue;
        if (seen++ >= preloaded) rest += static_cast<double>(bytes[l]);
        else if (budget_bytes >= static_cast<double>(bytes[l])) {  // the preload, from pinned memory
            host[l] = 1;
            budget_bytes -= static_cast<double>(bytes[l]);
        }
    }
    double file_need = std::max(0.0, rest - budget_bytes);  // bytes that come from the file whatever the order
    double t_copy = 0.0, t_disk = 0.0, ring_free_at = 0.0, ring_held = 0.0;
    seen = 0;
    for (std::size_t l = 0; l < bytes.size(); ++l) {
        if (!bytes[l] || seen++ < preloaded) continue;
        const double b = static_cast<double>(bytes[l]);
        const double copy = b / (pcie_gbps * 1e9), read = b / (disk_gbps * 1e9);
        // the read starts when the disk is free and the ring has room (else after the previous file layer's copies);
        // a layer larger than the ring is read in step with its own copies
        double start = t_disk;
        if (b > ring_bytes) start = std::max(start, t_copy);
        else if (b + ring_held > ring_bytes) start = std::max(start, ring_free_at);
        const double read_end = start + read;
        const bool fits = budget_bytes >= b;
        const bool to_file = !fits || (file_need >= b && read_end <= t_copy + copy);
        if (to_file) {
            t_disk = read_end;
            t_copy = std::max(t_copy + copy, read_end);
            ring_held = std::min(b, ring_bytes);
            ring_free_at = t_copy;
            file_need = std::max(0.0, file_need - b);
        } else {
            host[l] = 1;
            budget_bytes -= b;
            t_copy += copy;
        }
    }
    return host;
}

// The host set of a non-layer order (bytes[l]: streamed bytes of layer l, 0 = resident; budget and ring in bytes).
inline std::vector<std::uint8_t> ordered_host_layers(HostPinOrder order, std::span<const std::uint64_t> bytes, std::uint32_t slots,
                                                   double budget_bytes, double ring_bytes) {
    if (order == HostPinOrder::early) return early_host_layers(bytes, slots, budget_bytes);
    if (order == HostPinOrder::interleave) return interleave_host_layers(bytes, slots, budget_bytes, ring_bytes);
    return deadline_host_layers(bytes, slots, budget_bytes);
}

}  // namespace strix::ds41
