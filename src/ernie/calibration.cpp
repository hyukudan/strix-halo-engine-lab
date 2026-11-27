#include "strix/ernie/calibration.hpp"

#include "moe_core/packed_matrix_overrides.hpp"
#include "moe_core/reference.hpp"
#include "strix/ernie/inventory.hpp"
#include "strix/serve/json.hpp"
#include "strix/weight/calib_moments.hpp"
#include "strix/weight/pack_plan.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strix::ernie {
namespace {

namespace wei = weight;
namespace cal = wei::calib;
using serve::Json;

constexpr auto kLatentWidth = model::narrow<unsigned>(data::moe::moe_latent_size);
constexpr auto kInnerWidth = model::narrow<unsigned>(data::moe::moe_intermediate_size);
constexpr auto kMoeLayers = [] {
    unsigned count = 0;
    for (unsigned i = 0; i < kLayerCount; ++i)
        if (layer_has_moe(i)) ++count;
    return count;
}();
constexpr auto kMomentBytes = (std::size_t{kMoeLayers} * kExpertCount * (kLatentWidth + kInnerWidth) +
                               std::max(kLatentWidth, kInnerWidth)) * sizeof(double);
using RouteCounts = std::array<std::uint64_t, kExpertCount>;

std::string expert_prefix(unsigned layer, unsigned expert) {
    return target_layer_prefix(layer) + "block_sparse_moe.experts." + std::to_string(expert);
}

bool target_expert_matrix(std::string_view name) {
    constexpr std::string_view prefix = "model.layers.";
    constexpr std::string_view infix = ".block_sparse_moe.experts.";
    if (!name.starts_with(prefix)) return false;
    name.remove_prefix(prefix.size());
    const auto read_id = [&](unsigned& id) {
        if (name.empty()) return false;
        const auto end = name.data() + name.size();
        const auto result = std::from_chars(name.data(), end, id);
        if (result.ec != std::errc{}) return false;
        name.remove_prefix(static_cast<std::size_t>(result.ptr - name.data()));
        return true;
    };
    unsigned layer = 0, expert = 0;
    if (!read_id(layer) || !layer_has_moe(layer) || !name.starts_with(infix)) return false;
    name.remove_prefix(infix.size());
    return read_id(expert) && expert < kExpertCount &&
           (name == ".w1.weight" || name == ".w2.weight" || name == ".w3.weight");
}

void verify_counts(const cal::MomentSet& moments, const RouteCounts& routes, unsigned layer) {
    for (unsigned expert = 0; expert < kExpertCount; ++expert) {
        const auto prefix = expert_prefix(layer, expert);
        const auto* latent = moments.find(prefix + ".latent");
        const auto* inner = moments.find(prefix + ".inner");
        if (!latent && !inner && routes[expert] == 0U) continue;
        if (!latent || !inner || latent->rows != routes[expert] || inner->rows != latent->rows ||
            inner->window_rows != latent->window_rows)
            throw std::logic_error("incomplete calibration projection observations for " + prefix);
    }
}

}  // namespace

struct ExpertCalibration::Impl {
    cal::MomentSet moments{1U, cal::MomentLayout::diagonal, kMomentBytes};
    std::array<RouteCounts, kLayerCount> routes{};
};

ExpertCalibration::ExpertCalibration() : impl_(std::make_unique<Impl>()) {}
ExpertCalibration::~ExpertCalibration() = default;

std::size_t ExpertCalibration::add_window(std::span<const std::uint32_t> ids) {
    return impl_->moments.add_window(ids);
}

void ExpertCalibration::extend_window(std::size_t window, std::span<const std::uint32_t> processed_ids) {
    impl_->moments.extend_window(window, processed_ids);
}

void ExpertCalibration::observe(std::size_t window, unsigned layer, unsigned expert,
                                moe::ExpertProjection projection, std::span<const float> values) {
    const bool latent = projection == moe::ExpertProjection::gate_up;
    const auto width = latent ? kLatentWidth : kInnerWidth;
    if (!layer_has_moe(layer) || expert >= kExpertCount ||
        window >= impl_->moments.provenance().windows().size() || values.size() != width ||
        !std::ranges::all_of(values, [](float value) { return std::isfinite(value); }))
        throw std::invalid_argument("invalid ERNIE expert calibration observation");
    auto& count = impl_->routes[layer][expert];
    if (latent && count == static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
        throw std::overflow_error("ERNIE expert route count exceeds pack plan range");
    const auto prefix = expert_prefix(layer, expert);
    const std::array<std::string, 2> names = latent ?
        std::array{prefix + ".w1.weight", prefix + ".w3.weight"} : std::array{prefix + ".w2.weight", std::string{}};
    const auto consumers = std::span(names).first(latent ? 2U : 1U);
    auto* moment = &impl_->moments.declare(prefix + (latent ? ".latent" : ".inner"), width, consumers);
    impl_->moments.add(window, std::span(&moment, 1U), values.data(), 1U);
    if (latent) ++count;
}

std::uint64_t ExpertCalibration::route_count(unsigned layer, unsigned expert) const {
    return impl_->routes.at(layer).at(expert);
}

std::size_t ExpertCalibration::moment_bytes() const noexcept {
    return impl_->moments.moment_bytes();
}

void ExpertCalibration::write(const std::filesystem::path& directory,
                              const std::filesystem::path& model_description) const {
    if (moment_bytes() == 0) throw std::logic_error("ERNIE expert calibration has no observations");
    std::vector<wei::PackRouteCount> counts;
    std::vector<wei::MeasuredPackLayer> active_layers;
    for (unsigned layer = 0; layer < kLayerCount; ++layer) {
        if (!layer_has_moe(layer)) continue;
        const auto& routes = impl_->routes[layer];
        verify_counts(impl_->moments, routes, layer);
        if (std::ranges::all_of(routes, [](std::uint64_t count) { return count == 0U; })) continue;
        active_layers.push_back({layer, target_layer_prefix(layer)});
        for (unsigned expert = 0; expert < kExpertCount; ++expert)
            if (routes[expert]) counts.push_back({layer, expert, routes[expert]});
    }
    wei::write_measured_pack_description(directory, model_description, counts, impl_->moments, active_layers);
}

void ExpertCalibration::write_slice_plan(const std::filesystem::path& path, std::span<const unsigned> layers,
                                         std::size_t experts_per_layer, std::string_view storage) const {
    if (layers.empty() || experts_per_layer == 0 || experts_per_layer > kExpertCount ||
        (storage != "sp2" && storage != "sp3"))
        throw std::invalid_argument("invalid ERNIE pack pilot selection");
    std::array<bool, kLayerCount> seen{};
    std::vector<wei::PackSliceMatrix> matrices;
    for (const auto layer : layers) {
        if (!layer_has_moe(layer) || seen[layer]) throw std::invalid_argument("invalid or duplicate pilot layer");
        seen[layer] = true;
        const auto& routes = impl_->routes[layer];
        verify_counts(impl_->moments, routes, layer);
        std::vector<unsigned> active;
        for (unsigned expert = 0; expert < kExpertCount; ++expert)
            if (routes[expert]) active.push_back(expert);
        if (active.size() < experts_per_layer)
            throw std::invalid_argument("not enough observed experts for pilot layer " + std::to_string(layer));
        std::ranges::stable_sort(active, std::greater{}, [&](unsigned expert) { return routes[expert]; });
        for (const auto expert : std::span(active).first(experts_per_layer)) {
            const auto prefix = expert_prefix(layer, expert);
            for (const auto projection : {".w1.weight", ".w2.weight", ".w3.weight"}) {
                const bool down = std::string_view(projection) == ".w2.weight";
                matrices.push_back({prefix + projection, down ? kLatentWidth : kInnerWidth,
                                    down ? kInnerWidth : kLatentWidth});
            }
        }
    }
    wei::write_pack_slice_plan(path, storage, matrices);
}

struct PackedExpertOverrides::Impl {
    moe_core::PackedMatrixOverrides overrides;
};

PackedExpertOverrides::PackedExpertOverrides(const std::filesystem::path& checkpoint,
                                             const std::filesystem::path& sidecar, std::size_t max_pack_bytes)
    : impl_(std::make_unique<Impl>(Impl{{checkpoint, sidecar, max_pack_bytes, target_expert_matrix}})) {}

PackedExpertOverrides::~PackedExpertOverrides() = default;

std::size_t PackedExpertOverrides::matrix_count() const noexcept { return impl_->overrides.matrix_count(); }

bool PackedExpertOverrides::project(std::string_view name, std::span<const float> input,
                                    std::span<float> output) const {
    return impl_->overrides.project(name, input, output);
}

}  // namespace strix::ernie
