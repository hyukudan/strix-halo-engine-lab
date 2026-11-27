#pragma once

#include "strix/device/topology.hpp"
#include "strix/dist/transport.hpp"
#include "strix/ernie/inventory.hpp"
#include "strix/pair/layer_pipeline.hpp"

#include <array>
#include <optional>

namespace strix::ernie {

inline constexpr std::uint32_t kPairCut = (kLayerCount + 1U) / 2U;
inline constexpr std::size_t kPairRowBytes =
    model::narrow<std::size_t>(data::extras::hc_mult * data::dims::hidden) * sizeof(float);
inline constexpr std::size_t kPairLogitsBytes =
    model::narrow<std::size_t>(data::dims::vocab) * sizeof(float);

enum class PairAttention { hca, sparse_mla };

struct PairLayer {
    std::uint32_t id{};
    PairAttention attention{};
    bool routed{};
    std::uint32_t history_rows{}, compressed_rows{}, compression_ratio{};
    std::uint64_t kv_bytes{}, carry_bytes{};
};

struct PairTensor {
    std::string name;
    std::string shard;
    Module module{};
    std::optional<unsigned> layer;
    std::string wire_dtype;
    std::vector<std::uint64_t> shape;
    weight::Slice source;
    std::uint64_t resident_offset{};
};

struct PairMemory {
    std::uint64_t usable{}, weight_payload{}, weight_resident{}, kv_payload{}, kv_resident{}, carry{};
    std::uint64_t logits{}, exchange{}, load_staging{}, scratch{}, reserve{}, required{};
    [[nodiscard]] bool admitted() const noexcept { return usable > 0 && required <= usable; }
};

struct PairRank {
    pair::layers::Placement placement;
    std::vector<PairLayer> layers;
    std::vector<PairTensor> tensors;
    PairMemory memory;
};

// Placement accounts one packed image and one context slot per rank. The three
// runtime budgets for each rank come from its allocator/shape probe and the
// operator's deployment reserve. All are charged against usable device memory.
struct PairLimits {
    std::uint32_t context{}, prefill_rows{};
    std::uint64_t alignment{}, load_chunk_bytes{};
    std::array<std::uint64_t, 2> scratch{}, reserve{};
};

struct PairPlan {
    std::array<PairRank, 2> ranks;
    PairLimits limits;
    std::string transport;
    std::uint64_t draft_bytes{};
    // Set for a rank-scoped plan: only this rank's tensors and memory are known.
    std::optional<std::uint32_t> local;
    [[nodiscard]] bool admitted() const noexcept {
        if (local) return *local < ranks.size() && ranks[*local].memory.admitted();
        return ranks[0].memory.admitted() && ranks[1].memory.admitted();
    }
};

// A fixed whole-layer cut for first-token bring-up. This does not price the
// transport or qualify the native numerics. Rebuild with observed usable bytes
// and measured scratch before native allocation.
// With local_rank, the checkpoint copy may omit shards: every absent target
// tensor must belong to the other rank (drafter tensors stay deferred), and only
// the local rank's tensors and memory are planned and admitted.
[[nodiscard]] PairPlan make_pair_plan(const Inventory& inventory, const device::Topology& topology,
                                      const PairLimits& limits, std::string transport,
                                      std::optional<std::uint32_t> local_rank = std::nullopt);
// Rank owning a target tensor in the fixed cut; empty for drafter tensors.
[[nodiscard]] std::optional<std::uint32_t> pair_owner(std::string_view name);

using PairWeightWrite = std::function<void(std::uint64_t resident_offset, std::span<const std::byte>)>;

// Bounded, serial source reads in shard/offset order. The writer consumes each
// chunk synchronously and publishes the new resident image only after success.
void stream_pair_weights(const PairRank& rank, std::span<std::byte> staging, const PairWeightWrite& write);

struct PairFirstTokenOps {
    decltype(pair::layers::StageOps::range) forward;
    decltype(pair::layers::StageOps::seam) seam;
    std::function<void()> fence;
    std::function<std::uint32_t()> top;
};

// Host coordination for the first-token run. The rank's native hooks own KV
// and execute only [first,last). Both ranks verify the same assets, geometry,
// numerics and prompt before the shared prefill pipe sends any activations.
// A fresh device session is required after failure.
[[nodiscard]] std::uint32_t run_pair_first_token(const PairPlan& plan, std::uint32_t rank,
    std::span<const std::uint32_t> prompt, std::string_view asset_sha256,
    std::string_view numeric_identity, dist::Transport& link, const PairFirstTokenOps& ops);

} // namespace strix::ernie
