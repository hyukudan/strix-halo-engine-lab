#pragma once

#include "strix/weight/formats/microscaled_matrix.hpp"
#include "strix/weight/typed_slice.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace strix::weight {

struct ResidentSliceRange {
    Slice source;
    std::size_t destination_offset{};
    std::size_t bytes{};
};

class ResidentSlicePlan {
public:
    using NamedTensor = std::pair<std::string_view, const RawTensor*>;

    // Offsets are sorted within a shard so queued reads follow file order.
    [[nodiscard]] static ResidentSlicePlan make(std::vector<NamedTensor> tensors, std::size_t alignment) {
        if (tensors.empty() || !std::has_single_bit(alignment) || alignment < alignof(float) || alignment < alignof(std::uint16_t))
            throw std::invalid_argument("invalid resident tensor alignment");
        for (const auto& [name, raw] : tensors)
            if (name.empty() || !raw || raw->payload.range.bytes == 0U)
                throw std::invalid_argument("missing resident tensor " + std::string(name));
        std::ranges::stable_sort(tensors, {}, [](const auto& tensor) { return tensor.second->payload.range.offset; });

        ResidentSlicePlan plan;
        plan.alignment_ = alignment;
        plan.ranges_.reserve(tensors.size());
        constexpr auto limit = std::numeric_limits<std::size_t>::max();
        const auto mask = alignment - 1U;
        for (const auto& [name, raw] : tensors) {
            Slice payload;
            if (const auto status = subslice(raw->payload, {0, raw->payload.range.bytes}, payload); !status)
                throw std::invalid_argument("invalid resident tensor " + std::string(name) + ": " + status.detail);
            if (plan.bytes_ > limit - mask)
                throw std::overflow_error("resident alignment exceeds address space");
            const auto offset = (plan.bytes_ + mask) & ~mask;
            if (raw->payload.range.bytes > limit - offset)
                throw std::overflow_error("resident weights exceed address space");
            const auto bytes = static_cast<std::size_t>(raw->payload.range.bytes);
            const auto dtype = slice_dtype(raw->wire_dtype);
            if (!plan.tensors_.emplace(std::string(name), Tensor{offset, bytes, dtype}).second)
                throw std::invalid_argument("duplicate resident tensor " + std::string(name));
            plan.ranges_.push_back({std::move(payload), offset, bytes});
            plan.bytes_ = offset + bytes;
        }
        return plan;
    }

    [[nodiscard]] std::size_t bytes() const { return bytes_; }
    [[nodiscard]] std::span<const ResidentSliceRange> ranges() const { return ranges_; }

    void require_storage(std::span<const std::uint8_t> storage) const {
        if (bytes_ == 0U || storage.size() < bytes_ ||
            reinterpret_cast<std::uintptr_t>(storage.data()) % alignment_ != 0U)
            throw std::invalid_argument("incompatible resident weight storage");
    }

private:
    friend class ResidentSlices;
    struct Tensor {
        std::size_t offset;
        std::size_t bytes;
        SliceDtype dtype;
    };

    std::size_t bytes_{};
    std::size_t alignment_{};
    std::vector<ResidentSliceRange> ranges_;
    std::map<std::string, Tensor, std::less<>> tensors_;
};

class ResidentSlices {
public:
    [[nodiscard]] static ResidentSlices bind(const ResidentSlicePlan& plan,
                                              std::span<const std::uint8_t> storage) {
        plan.require_storage(storage);
        return {storage.first(plan.bytes()), plan};
    }

    [[nodiscard]] std::span<const std::uint16_t> bf16(std::string_view name) const {
        const auto bytes = tensor(name, SliceDtype::bf16);
        if (bytes.size() % sizeof(std::uint16_t) != 0U)
            throw std::invalid_argument("incomplete scalar in resident tensor " + std::string(name));
        return {reinterpret_cast<const std::uint16_t*>(bytes.data()), bytes.size() / sizeof(std::uint16_t)};
    }

    [[nodiscard]] std::span<const float> f32(std::string_view name) const {
        const auto bytes = tensor(name, SliceDtype::f32);
        if (bytes.size() % sizeof(float) != 0U)
            throw std::invalid_argument("incomplete scalar in resident tensor " + std::string(name));
        return {reinterpret_cast<const float*>(bytes.data()), bytes.size() / sizeof(float)};
    }

    Status bind_expert_slab(const microscaled::ExpertSlabView& slab, std::string_view codes_name,
                            std::string_view scales_name, std::span<microscaled::PackedMatrix> out) const {
        const auto codes = tensor(codes_name, SliceDtype::bytes);
        const auto scales = tensor(scales_name, SliceDtype::bytes);
        return slab.bind_storage(codes, scales, out);
    }

private:
    ResidentSlices(std::span<const std::uint8_t> storage, const ResidentSlicePlan& plan)
        : storage_(storage), tensors_(plan.tensors_) {}

    [[nodiscard]] std::span<const std::uint8_t> tensor(std::string_view name, SliceDtype dtype) const {
        const auto it = tensors_.find(name);
        if (it == tensors_.end() || it->second.dtype != dtype)
            throw std::invalid_argument("incompatible resident tensor " + std::string(name));
        return storage_.subspan(it->second.offset, it->second.bytes);
    }

    std::span<const std::uint8_t> storage_;
    std::map<std::string, ResidentSlicePlan::Tensor, std::less<>> tensors_;
};

}  // namespace strix::weight
