#pragma once

// Explicit host-only substitution of MXFP4 checkpoint matrices by their strixpack sidecar form, for quality
// comparisons. The sidecar must be published for the same checkpoint; the model supplies the predicate that names
// the matrices a sidecar may carry.

#include "moe_core/grouped_strixpack.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::moe_core {

class PackedMatrixOverrides {
public:
    // `accepts(tensor)` is true for the checkpoint tensors a sidecar entry may replace.
    template <class Accepts>
    PackedMatrixOverrides(const std::filesystem::path& checkpoint, const std::filesystem::path& sidecar,
                          std::size_t max_pack_bytes, const Accepts& accepts) {
        one_row_.grouping.assignment_rows = {0};
        one_row_.grouping.expert_offsets = {0, 1};
        one_row_.tiles = {{0, 0, 1}};
        weight::PackCheckpoint source;
        require(weight::PackCheckpoint::open(checkpoint.string(), source));
        weight::Digest256 identity{};
        require(source.digest(identity));
        weight::LowbitPack pack;
        require(weight::LowbitPack::open_host(sidecar.string(), identity, pack));
        if (pack.entries().empty()) throw std::invalid_argument("empty packed matrix sidecar");
        std::size_t bytes = 0;
        for (const auto& entry : pack.entries()) {
            if (!accepts(std::string_view(entry.tensor)) || !weight::is_strixpack_storage(entry.storage))
                throw std::invalid_argument("invalid packed matrix " + entry.tensor);
            if (entry.bytes > max_pack_bytes - bytes)
                throw std::runtime_error("packed matrices exceed the resident byte budget");
            bytes += static_cast<std::size_t>(entry.bytes);
            const auto* raw = source.find(entry.tensor);
            if (!raw) throw std::invalid_argument("checkpoint lacks packed matrix " + entry.tensor);
            weight::PackMatrixSource matrix;
            require(weight::PackMatrixSource::bind(source, *raw, matrix));
            if (matrix.encoding != weight::PackMatrixSource::Encoding::mxfp4)
                throw std::invalid_argument("packed matrix source is not MXFP4: " + entry.tensor);
            const auto [it, inserted] = matrices_.try_emplace(entry.tensor);
            if (!inserted) throw std::invalid_argument("duplicate packed matrix " + entry.tensor);
            require(it->second.load(pack, entry, matrix));
        }
        require(pack.verify_identity());
    }

    [[nodiscard]] std::size_t matrix_count() const noexcept { return matrices_.size(); }

    // One input row through the packed matrix `name`; false when `name` is not packed.
    [[nodiscard]] bool project(std::string_view name, std::span<const float> input, std::span<float> output) const {
        const auto matrix = matrices_.find(name);
        if (matrix == matrices_.end()) return false;
        const std::array views{matrix->second.view()};
        require(grouped_strixpack::project(one_row_, views, grouped_strixpack::InputRows::token, input.size(),
                                           output.size(), input, output));
        return true;
    }

private:
    static void require(weight::Status status) {
        if (!status) throw std::runtime_error(status.detail);
    }

    std::map<std::string, grouped_strixpack::Matrix, std::less<>> matrices_;
    ExpertMajorPlan one_row_;
};

}  // namespace strix::moe_core
