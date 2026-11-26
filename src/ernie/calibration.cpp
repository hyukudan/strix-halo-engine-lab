#include "strix/ernie/calibration.hpp"

#include "moe_core/grouped_strixpack.hpp"
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
namespace grouped = moe_core::grouped_strixpack;
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

void require(wei::Status status) {
    if (!status) throw std::runtime_error(status.detail);
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
    std::ifstream in(model_description, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open pack model description " + model_description.string());
    const std::string encoded{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    auto description = serve::parse_json_unique(encoded);
    const auto* schema = description.find("schema");
    if (!schema || schema->as_string() != "strix-pack-model/v1")
        throw std::invalid_argument("invalid pack model description");
    auto counts = Json::array();
    std::vector<unsigned> active_layers;
    for (unsigned layer = 0; layer < kLayerCount; ++layer) {
        if (!layer_has_moe(layer)) continue;
        const auto& routes = impl_->routes[layer];
        verify_counts(impl_->moments, routes, layer);
        if (std::ranges::all_of(routes, [](std::uint64_t count) { return count == 0U; })) continue;
        active_layers.push_back(layer);
        for (unsigned expert = 0; expert < kExpertCount; ++expert) {
            const auto count = routes[expert];
            if (count) counts.push(Json::Array{std::uint64_t{layer}, std::uint64_t{expert}, count});
        }
    }
    description.set("routes", std::move(counts));
    if (!std::filesystem::create_directory(directory))
        throw std::runtime_error("calibration capture directory already exists: " + directory.string());
    for (const auto layer : active_layers) {
        const auto prefix = target_layer_prefix(layer);
        impl_->moments.write_partition((directory / ("layer-" + std::to_string(layer) + ".moments")).string(),
                                       [&](const auto& input) { return input.name.starts_with(prefix); });
    }
    const auto plan_path = directory / "measured-pack-description.json";
    std::ofstream out(plan_path, std::ios::binary);
    out << description.dump() << '\n';
    out.close();
    if (!out) throw std::runtime_error("cannot write " + plan_path.string());
}

void ExpertCalibration::write_slice_plan(const std::filesystem::path& path, std::span<const unsigned> layers,
                                         std::size_t experts_per_layer, std::string_view storage) const {
    if (layers.empty() || experts_per_layer == 0 || experts_per_layer > kExpertCount ||
        (storage != "sp2" && storage != "sp3"))
        throw std::invalid_argument("invalid ERNIE pack pilot selection");
    const auto format = wei::strixpack_format(storage);
    const auto gate_bytes = wei::strixpack_matrix_bytes(format, kInnerWidth, kLatentWidth);
    const auto down_bytes = wei::strixpack_matrix_bytes(format, kLatentWidth, kInnerWidth);
    std::array<bool, kLayerCount> seen{};
    std::vector<std::string> rows;
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
                const auto bytes = std::string_view(projection) == ".w2.weight" ? down_bytes : gate_bytes;
                rows.push_back(prefix + projection + '\t' + std::string(storage) + '\t' + std::to_string(bytes));
            }
        }
    }
    if (std::filesystem::exists(path)) throw std::runtime_error("pack plan already exists: " + path.string());
    std::ofstream out(path, std::ios::binary);
    out << "tensor\tformat\tbytes\n";
    for (const auto& row : rows) out << row << '\n';
    out.close();
    if (!out) throw std::runtime_error("cannot write " + path.string());
}

struct PackedExpertOverrides::Impl {
    std::map<std::string, grouped::Matrix, std::less<>> matrices;
    moe_core::ExpertMajorPlan one_row;

    Impl() {
        one_row.grouping.assignment_rows = {0};
        one_row.grouping.expert_offsets = {0, 1};
        one_row.tiles = {{0, 0, 1}};
    }
};

PackedExpertOverrides::PackedExpertOverrides(const std::filesystem::path& checkpoint,
                                             const std::filesystem::path& sidecar, std::size_t max_pack_bytes)
    : impl_(std::make_unique<Impl>()) {
    wei::PackCheckpoint source;
    require(wei::PackCheckpoint::open(checkpoint.string(), source));
    wei::Digest256 identity{};
    require(source.digest(identity));
    wei::LowbitPack pack;
    require(wei::LowbitPack::open_host(sidecar.string(), identity, pack));
    if (pack.entries().empty()) throw std::invalid_argument("empty ERNIE packed expert sidecar");
    std::size_t bytes = 0;
    for (const auto& entry : pack.entries()) {
        if (!target_expert_matrix(entry.tensor) || !wei::is_strixpack_storage(entry.storage))
            throw std::invalid_argument("invalid packed ERNIE routed expert matrix " + entry.tensor);
        if (entry.bytes > max_pack_bytes - bytes)
            throw std::runtime_error("packed ERNIE expert matrices exceed the resident byte budget");
        bytes += static_cast<std::size_t>(entry.bytes);
        const auto* raw = source.find(entry.tensor);
        if (!raw) throw std::invalid_argument("checkpoint lacks packed matrix " + entry.tensor);
        wei::PackMatrixSource matrix;
        require(wei::PackMatrixSource::bind(source, *raw, matrix));
        if (matrix.encoding != wei::PackMatrixSource::Encoding::mxfp4)
            throw std::invalid_argument("packed ERNIE expert source is not MXFP4: " + entry.tensor);
        const auto [it, inserted] = impl_->matrices.try_emplace(entry.tensor);
        if (!inserted) throw std::invalid_argument("duplicate packed matrix " + entry.tensor);
        require(it->second.load(pack, entry, matrix));
    }
    require(pack.verify_identity());
}

PackedExpertOverrides::~PackedExpertOverrides() = default;

std::size_t PackedExpertOverrides::matrix_count() const noexcept { return impl_->matrices.size(); }

bool PackedExpertOverrides::project(std::string_view name, std::span<const float> input, std::span<float> output) const {
    const auto matrix = impl_->matrices.find(name);
    if (matrix == impl_->matrices.end()) return false;
    const std::array views{matrix->second.view()};
    require(grouped::project(impl_->one_row, views, grouped::InputRows::token, input.size(), output.size(), input, output));
    return true;
}

}  // namespace strix::ernie
