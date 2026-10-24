#pragma once

// deepseek41 hot-expert placement: the reader of a versioned placement file
// (format "tensormeld/hot-expert-placement-v1", docs/models/deepseek-v4.1/hot-placement.md) and the split of a layer's
// routed rows into local and remote work under the canonical combine of ds41_moe.hpp:
//   y = (shared + low) + high, low = fold of the row's experts with id < split in ascending id from 0,
//   high likewise for ids >= split; each step one FP32 product then one FP32 add.
// A remote target returns the unweighted raw vector of each (row, expert) it computes; when the
// leading positions of one half of a row's ascending order are on one target, it may return them as
// one folded partial (bitwise the same leading steps of the fold); the parts-host continues the fold
// from it, and later positions of that target in the half come raw. No other pre-aggregation keeps
// the bits, so the split never asks for one.
//
// Host only: no device code, no device runtime.

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strix::ds41 {

inline constexpr std::string_view hot_placement_format = "tensormeld/hot-expert-placement-v1";
inline constexpr std::string_view hot_placement_combine = "ds41-canonical-halves-v1";

// Expert homes. rtx_pro: the parts-host GPU (dense layers, local experts, the combine). rtx_4090: a
// second CUDA device of the same host (PCIe), returns vectors. pc_ram: host memory of the parts-host,
// copied to the parts-host GPU on demand and computed there. amd: the expert servers (every expert not
// listed elsewhere; each server holds a contiguous id range).
enum class HotDevice : std::uint8_t { rtx_pro = 0, rtx_4090 = 1, pc_ram = 2, amd = 3 };
inline constexpr std::size_t hot_device_count = 4;
// File ids: "rtx-pro", "rtx-4090", "pc-ram", "amd".
[[nodiscard]] std::string_view hot_device_name(HotDevice device) noexcept;
[[nodiscard]] std::optional<HotDevice> hot_device_from_name(std::string_view name) noexcept;
// Devices whose experts are computed by another device and come back as vectors.
[[nodiscard]] constexpr bool hot_device_returns_vectors(HotDevice device) noexcept {
    return device == HotDevice::rtx_4090 || device == HotDevice::amd;
}

struct HotPlacementGeometry {
    std::uint32_t layers = 40, experts = 384, top_k = 6, canonical_split = 192;
    std::uint64_t expert_bytes = 9953280;
};
// What the caller requires of a file: the geometry must match exactly; a non-null model_identity of the
// file must equal `model_identity` when one is given (a null file identity is accepted).
struct HotPlacementExpect {
    HotPlacementGeometry geometry;
    std::optional<std::string> model_identity;
};

struct HotPlacementDevice {
    HotDevice device = HotDevice::rtx_pro;
    std::string kind, role;
    double budget_gib = 0.0, reserve_gib = 0.0;
    std::optional<double> expert_gib;
    // Bytes the device may hold in experts: expert_gib if given, else budget_gib - reserve_gib (0 if
    // negative), times 2^30. Not used for amd.
    [[nodiscard]] double expert_capacity_bytes() const noexcept;
};
struct HotAmdServer {
    std::string node;
    std::uint32_t first = 0, end = 0; // expert ids [first, end)
};

struct HotPlacementLayer {
    // Expert ids per device, ascending; experts[amd] is the complement of the other three.
    std::array<std::vector<std::uint32_t>, hot_device_count> experts;
    // [expert id]: its device, and its weight slot = index in experts[device] (dense numbering per
    // device and layer, the order in which a device uploads that layer's experts).
    std::vector<HotDevice> device;
    std::vector<std::uint32_t> slot;
};

struct HotPlacement {
    HotPlacementGeometry geometry;
    std::string model_family, model_file;
    std::optional<std::string> model_identity;
    HotDevice dense_device = HotDevice::rtx_pro;
    std::vector<HotPlacementDevice> devices; // in file order
    std::vector<HotAmdServer> amd_servers;   // ascending, covering [0, experts) without gaps
    std::vector<HotPlacementLayer> layers;
    std::string combine_identity, remote_return;

    [[nodiscard]] HotDevice device_of(std::uint32_t layer, std::uint32_t expert) const;
    [[nodiscard]] std::uint32_t slot_of(std::uint32_t layer, std::uint32_t expert) const;
    // Index into amd_servers of the server that holds `expert`.
    [[nodiscard]] std::uint32_t amd_server_of(std::uint32_t expert) const;
    // Experts resident on `device` over all layers.
    [[nodiscard]] std::uint64_t resident_experts(HotDevice device) const;
    [[nodiscard]] const HotPlacementDevice* find_device(HotDevice device) const noexcept;
};

// Parse and validate. Throws std::invalid_argument naming the first violation: unknown format or
// combine identity, geometry different from `expect`, model identity mismatch, unknown device ids,
// layers not 0..layers-1 in order, ids out of range, not strictly ascending or listed on two devices
// of a layer, AMD server ranges not covering [0, experts) contiguously, or a device over its memory
// (resident experts * expert_bytes > capacity).
[[nodiscard]] HotPlacement parse_hot_placement(std::string_view json_text, const HotPlacementExpect& expect = {});
[[nodiscard]] HotPlacement load_hot_placement(const std::string& path, const HotPlacementExpect& expect = {});

// ---- split of one layer's routed rows -------------------------------------------------------------

// Message sizes of the accounting: per row sent to a target, the A8 input with the row's ids and
// weights; per returned vector (raw or folded) one FP32 hidden vector.
struct HotRequestSizes {
    std::uint64_t input_bytes_per_row = 5808;
    std::uint64_t output_bytes_per_vector = 20480;
};

// A computing target that returns vectors: rtx_4090 (server 0) or one AMD server.
struct HotTarget {
    HotDevice device = HotDevice::amd;
    std::uint32_t server = 0;
    friend bool operator==(const HotTarget&, const HotTarget&) = default;
};

// One position of a row's canonical order.
struct HotSlot {
    std::uint32_t expert = 0;
    std::uint32_t route_slot = 0;  // row * top_k + k: where the router put it (raw vector slot)
    std::uint32_t weight_slot = 0; // HotPlacementLayer::slot of the expert
    float weight = 0.0F;
    HotDevice device = HotDevice::rtx_pro;
    std::uint32_t server = 0;      // amd: index into amd_servers; otherwise 0
    bool folded = false;           // covered by a folded partial of its target
};

enum class HotVectorKind : std::uint8_t { raw, folded_prefix };
// A vector to request from a target. raw: the unweighted vector of (row, expert) at canonical
// position `first` of its half. folded_prefix: positions [0, count) of the half, folded as the combine
// folds them (from 0, ascending id, weight times raw then add), using the routing weights.
struct HotVectorRequest {
    std::uint32_t row = 0;
    std::uint8_t half = 0; // 0: ids < split, 1: the rest
    HotVectorKind kind = HotVectorKind::raw;
    std::uint32_t first = 0, count = 1; // positions within the half
    std::uint32_t expert = 0;           // raw: the expert; folded: the first expert of the prefix
    std::uint32_t route_slot = 0;       // raw: row * top_k + k; folded: that of the first expert
};

struct HotTargetWork {
    HotTarget target;
    std::vector<HotVectorRequest> requests; // ascending (row, half, first)
    std::vector<std::uint32_t> experts;     // distinct experts the target computes, ascending
    std::uint32_t rows = 0;                 // rows with at least one request
    std::uint32_t raw_vectors = 0, folded_vectors = 0, folded_positions = 0;
    std::uint64_t input_bytes = 0, output_bytes = 0;
};

struct HotLayerSplit {
    std::uint32_t layer = 0, rows = 0, top_k = 0, split = 0;
    // [row * top_k + p]: canonical position p of the row (ascending id; the low half first).
    std::vector<HotSlot> order;
    std::vector<std::uint32_t> low_counts;                                 // [row]
    std::vector<std::uint32_t> folded_prefix;                              // [row * 2 + half]: leading positions folded remotely
    std::array<std::vector<std::uint32_t>, hot_device_count> experts;      // distinct experts used per device, ascending
    std::array<std::uint32_t, hot_device_count> positions{};               // (row, expert) pairs per device
    std::vector<HotTargetWork> targets;                                    // targets with work: rtx_4090, then AMD servers ascending
    std::uint64_t upload_bytes = 0;                                        // pc_ram distinct experts * expert_bytes
    // Sums over targets: network = the AMD servers, pcie = rtx_4090 (same host).
    std::uint64_t network_input_bytes = 0, network_output_bytes = 0;
    std::uint64_t pcie_input_bytes = 0, pcie_output_bytes = 0;
    [[nodiscard]] bool all_local() const noexcept { return targets.empty(); }
    // Positions computed on the parts-host GPU (rtx_pro and pc_ram), in canonical order per row.
    [[nodiscard]] std::uint32_t local_positions() const noexcept {
        return positions[static_cast<std::size_t>(HotDevice::rtx_pro)] + positions[static_cast<std::size_t>(HotDevice::pc_ram)];
    }
};

// ids / weights [row * top_k + k] in router order (ids distinct within a row, < experts). Pure function
// of the placement and the routes. Per (row, half) the leading run of positions on one target is folded
// when it has at least two positions; that target's later positions in the half (after a position
// elsewhere) are raw, so a target returns n - run + 1 vectors for a half with n positions on it and a
// leading run of length run >= 1 (run 1 is the raw case). A single leading position stays raw (same
// payload, no weight on the server).
[[nodiscard]] HotLayerSplit split_hot_layer(const HotPlacement& placement, std::uint32_t layer, std::span<const std::uint32_t> ids,
                                            std::span<const float> weights, std::uint32_t rows, const HotRequestSizes& sizes = {});

} // namespace strix::ds41
