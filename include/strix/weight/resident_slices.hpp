#pragma once

#include "strix/weight/formats/microscaled_matrix.hpp"
#include "strix/weight/host_q8.hpp"
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

struct ResidentMatrixView {
    std::span<const std::uint8_t> bytes;
    FormatId storage;
    std::uint64_t rows, columns, row_bytes;
    // Every output of the matrix carries a factor 2^outer_exponent outside its group scales (a Strix pack entry's
    // per-tensor base, strix/weight/host_lowbit.hpp sp::); 0 for every other storage.
    std::int32_t outer_exponent = 0;
};

struct ResidentBf16Rows {
    const std::uint16_t* data;
    std::uint64_t rows, columns, row_bytes;
};

class ResidentSlicePlan {
public:
    struct NamedTensor {
        std::string_view name;
        const RawTensor* raw;
        // An explicit format declares a compact [N,K] BF16-source matrix.
        std::optional<FormatId> matrix_storage{};
        std::uint64_t row_pitch{};  // Zero uses the natural layout.
        // Pre-encoded bytes of matrix_storage (geometry.bytes long); replaces the BF16 conversion.
        std::optional<Slice> prepacked{};
        // Outer power of two of the prepacked matrix (LowbitPack::find); only a prepacked matrix may carry one.
        std::int32_t outer_exponent{};
    };

    // Offsets are sorted within a shard so queued reads follow file order.
    [[nodiscard]] static ResidentSlicePlan make(std::vector<NamedTensor> tensors, std::size_t alignment) {
        if (tensors.empty() || !std::has_single_bit(alignment) || alignment < alignof(float) || alignment < alignof(std::uint16_t))
            throw std::invalid_argument("invalid resident tensor alignment");
        for (const auto& tensor : tensors)
            if (tensor.name.empty() || !tensor.raw || tensor.raw->payload.range.bytes == 0U)
                throw std::invalid_argument("missing resident tensor " + std::string(tensor.name));
        std::ranges::stable_sort(tensors, {}, [](const auto& tensor) { return tensor.raw->payload.range.offset; });

        ResidentSlicePlan plan;
        plan.alignment_ = alignment;
        plan.ranges_.reserve(tensors.size());
        constexpr auto limit = std::numeric_limits<std::size_t>::max();
        const auto mask = alignment - 1U;
        for (const auto& [name, raw, matrix_storage, row_pitch, prepacked, outer_exponent] : tensors) {
            if (outer_exponent != 0 && !prepacked)
                throw std::invalid_argument("outer exponent without a prepacked matrix " + std::string(name));
            Slice payload{};
            if (const auto status = subslice(raw->payload, {0, raw->payload.range.bytes}, payload); !status)
                throw std::invalid_argument("invalid resident tensor " + std::string(name) + ": " + status.detail);
            auto dtype = slice_dtype(raw->wire_dtype);
            MatrixLoadGeometry geometry{};
            if (prepacked && (!matrix_storage || *matrix_storage == kBf16WeightStorage))
                throw std::invalid_argument("prepacked resident tensor needs a packed matrix storage " + std::string(name));
            if (matrix_storage) {
                if (dtype != SliceDtype::bf16)
                    throw std::invalid_argument("resident matrix requires BF16 source " + std::string(name));
                if (const auto status = matrix_load_geometry(*matrix_storage, raw->shape, geometry); !status)
                    throw std::invalid_argument("invalid resident matrix " + std::string(name) + ": " + status.detail);
                if (prepacked) {
                    if (prepacked->range.bytes != geometry.bytes)
                        throw std::invalid_argument("prepacked resident matrix has wrong byte count " + std::string(name));
                    payload = *prepacked;
                } else if (const auto status = prepare_bf16_matrix_slice(raw->payload, raw->shape, *matrix_storage, payload); !status) {
                    throw std::invalid_argument("invalid resident matrix " + std::string(name) + ": " + status.detail);
                }
                if (*matrix_storage != kBf16WeightStorage) dtype = SliceDtype::bytes;
            }
            const auto rows = raw->shape.dims[0];
            const auto columns = raw->shape.dims[1];
            const bool matrix_shape = raw->shape.rank == 2U && rows != 0U && columns != 0U;
            const auto stored_row_bytes = matrix_shape && payload.range.bytes % rows == 0U
                ? payload.range.bytes / rows : 0U;
            auto natural = stored_row_bytes;
            if (dtype == SliceDtype::bf16)
                natural = matrix_shape && columns <= std::numeric_limits<std::uint64_t>::max() / sizeof(std::uint16_t)
                    ? columns * sizeof(std::uint16_t) : 0U;
            const bool bf16_rows = dtype == SliceDtype::bf16 && natural != 0U && stored_row_bytes == natural;
            const bool pitched = row_pitch != 0U && row_pitch != natural;
            if (pitched && (!bf16_rows || matrix_storage || row_pitch < natural || row_pitch % 16U != 0U))
                throw std::invalid_argument("invalid resident row pitch " + std::string(name));
            if (bf16_rows) geometry.row_bytes = pitched ? row_pitch : natural;

            if (plan.bytes_ > limit - mask)
                throw std::overflow_error("resident alignment exceeds address space");
            const auto offset = (plan.bytes_ + mask) & ~mask;
            if (pitched && row_pitch > (limit - offset) / rows)
                throw std::overflow_error("resident rows exceed address space");
            const auto size = pitched ? rows * row_pitch : payload.range.bytes;
            if (size > limit - offset)
                throw std::overflow_error("resident weights exceed address space");
            const auto bytes = static_cast<std::size_t>(size);
            if (!plan.tensors_.emplace(std::string(name), Tensor{offset, bytes, dtype, matrix_storage,
                    rows, columns, geometry.row_bytes, pitched, outer_exponent}).second)
                throw std::invalid_argument("duplicate resident tensor " + std::string(name));
            if (pitched) {
                // Kernels read only natural row bytes. The padding retains the storage's prior contents.
                for (std::uint64_t row = 0; row < rows; ++row) {
                    Slice slice{};
                    if (const auto status = subslice(payload, {row * natural, natural}, slice); !status)
                        throw std::invalid_argument("invalid resident row " + std::string(name) + ": " + status.detail);
                    plan.ranges_.push_back({std::move(slice), offset + static_cast<std::size_t>(row * row_pitch),
                                            static_cast<std::size_t>(natural)});
                }
            } else {
                plan.ranges_.push_back({std::move(payload), offset, bytes});
            }
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
        std::optional<FormatId> matrix_storage;
        std::uint64_t rows, columns, row_bytes;
        bool pitched;
        std::int32_t outer_exponent;
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

    [[nodiscard]] ResidentBf16Rows bf16_rows(std::string_view name) const {
        const auto it = tensors_.find(name);
        if (it == tensors_.end() || it->second.dtype != SliceDtype::bf16 || it->second.row_bytes == 0U)
            throw std::invalid_argument("incompatible resident BF16 rows " + std::string(name));
        const auto& entry = it->second;
        return {reinterpret_cast<const std::uint16_t*>(storage_.data() + entry.offset),
                entry.rows, entry.columns, entry.row_bytes};
    }

    [[nodiscard]] std::span<const float> f32(std::string_view name) const {
        const auto bytes = tensor(name, SliceDtype::f32);
        if (bytes.size() % sizeof(float) != 0U)
            throw std::invalid_argument("incomplete scalar in resident tensor " + std::string(name));
        return {reinterpret_cast<const float*>(bytes.data()), bytes.size() / sizeof(float)};
    }

    [[nodiscard]] ResidentMatrixView matrix(std::string_view name) const {
        const auto it = tensors_.find(name);
        if (it == tensors_.end() || !it->second.matrix_storage)
            throw std::invalid_argument("incompatible resident matrix " + std::string(name));
        const auto& entry = it->second;
        return {storage_.subspan(entry.offset, entry.bytes), *entry.matrix_storage,
                entry.rows, entry.columns, entry.row_bytes, entry.outer_exponent};
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
        if (it == tensors_.end() || it->second.dtype != dtype || it->second.pitched)
            throw std::invalid_argument("incompatible resident tensor " + std::string(name));
        return storage_.subspan(it->second.offset, it->second.bytes);
    }

    std::span<const std::uint8_t> storage_;
    std::map<std::string, ResidentSlicePlan::Tensor, std::less<>> tensors_;
};

}  // namespace strix::weight
