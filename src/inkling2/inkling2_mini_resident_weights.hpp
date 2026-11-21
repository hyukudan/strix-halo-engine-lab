#pragma once

#include "inkling2_mini_checkpoint.hpp"
#include "inkling2_mini_moe_dispatch.hpp"

#include <functional>
#include <map>

namespace strix::inkling2 {

struct MiniResidentRange {
    weight::Slice source;
    std::size_t destination_offset{};
    std::size_t bytes{};
};

class MiniResidentWeightPlan {
public:
    static constexpr std::size_t kAlignment = 256U;

    // The input has already passed the Mini text schema. NVFP4 host-only
    // sidecars are fetched separately when the resident expert views bind.
    [[nodiscard]] static MiniResidentWeightPlan make(const MiniTextWeights& weights);
    [[nodiscard]] std::size_t bytes() const { return bytes_; }
    [[nodiscard]] std::span<const MiniResidentRange> ranges() const { return ranges_; }

private:
    friend class MiniResidentWeights;
    enum class Dtype { bytes, bf16, f32 };
    struct Tensor {
        std::size_t offset;
        std::size_t bytes;
        Dtype dtype;
    };

    std::size_t bytes_{};
    std::vector<MiniResidentRange> ranges_;
    std::map<std::string, Tensor, std::less<>> tensors_;
    std::array<std::array<MiniExpertLayerView, 2>, kPhysicalLayers> slabs_;
};

// Borrow immutable resident bytes. Once bound, the views are independent of
// the checkpoint file and valid for as long as the storage is allocated.
class MiniResidentWeights {
public:
    [[nodiscard]] static MiniResidentWeights bind(const MiniResidentWeightPlan& plan,
                                                  std::span<const std::uint8_t> storage);
    [[nodiscard]] std::span<const std::uint16_t> bf16(std::string_view name) const;
    [[nodiscard]] std::span<const float> f32(std::string_view name) const;
    [[nodiscard]] std::span<const MiniMoeExpert> experts(unsigned physical) const;

private:
    [[nodiscard]] std::span<const std::uint8_t> tensor(std::string_view name,
                                                      MiniResidentWeightPlan::Dtype dtype) const;
    std::span<const std::uint8_t> storage_;
    std::map<std::string, MiniResidentWeightPlan::Tensor, std::less<>> tensors_;
    std::vector<MiniMoeExpert> experts_;
};

// The loader owns its staging and drains outstanding copies before destruction.
// Only the completed resident view is returned; callers discard the destination
// after a failed transfer. The exact slices can also be scheduled by placement.
template <class Loader>
[[nodiscard]] MiniResidentWeights load_mini_resident_weights(
    const MiniResidentWeightPlan& plan, std::span<std::uint8_t> storage, Loader& loader) {
    auto weights = MiniResidentWeights::bind(plan, storage);
    for (const auto& range : plan.ranges()) {
        loader.load(range.source, storage.subspan(range.destination_offset, range.bytes));
    }
    loader.ready();
    return weights;
}

}  // namespace strix::inkling2
