#pragma once

#include "strix/numeric/logit_comparison.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string_view>

namespace strix::moe { enum class ExpertProjection; }

namespace strix::ernie {

// One observer per serial reference run. The gate/up projection records the
// route and the latent input. The down projection records the post-GLU input.
class ExpertCalibration {
public:
    ExpertCalibration();
    ~ExpertCalibration();

    [[nodiscard]] std::size_t add_window(std::span<const std::uint32_t> ids);
    void extend_window(std::size_t window, std::span<const std::uint32_t> processed_ids);
    void observe(std::size_t window, unsigned layer, unsigned expert, moe::ExpertProjection projection,
                 std::span<const float> values);

    [[nodiscard]] std::uint64_t route_count(unsigned layer, unsigned expert) const;
    [[nodiscard]] std::size_t moment_bytes() const noexcept;
    // The destination directory must be new.
    void write(const std::filesystem::path& directory, const std::filesystem::path& model_description) const;
    // A pilot replaces the most used experts in these layers. Each selected
    // expert contributes gate, up and down matrices, all at the same width.
    void write_slice_plan(const std::filesystem::path& path, std::span<const unsigned> layers,
                          std::size_t experts_per_layer, std::string_view storage) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Explicit host-only pack substitution for quality comparisons.
class PackedExpertOverrides {
public:
    PackedExpertOverrides(const std::filesystem::path& checkpoint, const std::filesystem::path& sidecar,
                          std::size_t max_pack_bytes = std::size_t{512} << 20U);
    ~PackedExpertOverrides();

    [[nodiscard]] std::size_t matrix_count() const noexcept;
    [[nodiscard]] bool project(std::string_view name, std::span<const float> input, std::span<float> output) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

using numeric::compare_logits;
using numeric::LogitComparison;

}  // namespace strix::ernie
