#pragma once

#include "strix/exec/profiler.hpp"
#include "strix/receipt/writer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>

namespace strix::exec {

// Logical operand traffic, with partial estimates represented explicitly.
struct StepNodeCost {
    std::string_view layer_class;
    std::optional<std::uint64_t> bytes_read, bytes_written, flops;
};

struct StepReportContext {
    std::uint32_t position;
    std::string_view recipe, state_after_profile, roof_source;
    std::span<const std::string_view> layer_classes;
};

class StepTimingSeries {
public:
    explicit StepTimingSeries(std::size_t nodes) : node_us_(nodes) {}

    void record(std::span<const float> times, double graph_us, double eager_us) {
        if (times.size() != node_us_.size() || !valid(graph_us) || !valid(eager_us) ||
            !std::ranges::all_of(times, valid))
            throw std::invalid_argument("invalid step profile sample");
        for (std::size_t i = 0; i < times.size(); ++i) node_us_[i].push_back(times[i]);
        graph_wall_us_.push_back(graph_us);
        eager_wall_us_.push_back(eager_us);
    }

    [[nodiscard]] ProfileSampleObserver observer() {
        return {this, [](void* context, std::span<const float> times, double graph_us, double eager_us) {
            static_cast<StepTimingSeries*>(context)->record(times, graph_us, eager_us);
        }};
    }

    [[nodiscard]] const auto& node_us() const { return node_us_; }
    [[nodiscard]] const auto& graph_wall_us() const { return graph_wall_us_; }
    [[nodiscard]] const auto& eager_wall_us() const { return eager_wall_us_; }

private:
    static bool valid(double value) { return std::isfinite(value) && value >= 0; }

    std::vector<std::vector<float>> node_us_;
    std::vector<double> graph_wall_us_, eager_wall_us_;
};

namespace step_report_detail {

using receipt::detail::array;
using receipt::detail::nullable;
using receipt::detail::object;
using serve::Json;

struct Range {
    double median, min;
};

template<class T>
[[nodiscard]] Range range(std::vector<T> samples) {
    if (samples.empty()) throw std::invalid_argument("missing step profile samples");
    std::ranges::sort(samples);
    const auto mid = samples.size() / 2;
    const auto upper = static_cast<double>(samples[mid]);
    return {samples.size() % 2 ? upper : std::midpoint(static_cast<double>(samples[mid - 1]), upper),
            static_cast<double>(samples.front())};
}

[[nodiscard]] inline std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a)
        throw std::overflow_error("step profile traffic overflow");
    return a + b;
}

struct Total {
    std::uint64_t nodes = 0, byte_priced_nodes = 0, known_bytes_read = 0, known_bytes_written = 0;
    std::uint64_t flop_priced_nodes = 0, known_flops = 0;
    double summed_node_median_us = 0;

    void accumulate(const StepNodeCost& cost, double median_us) {
        ++nodes;
        if (cost.bytes_read && cost.bytes_written) ++byte_priced_nodes;
        known_bytes_read = add(known_bytes_read, cost.bytes_read.value_or(0));
        known_bytes_written = add(known_bytes_written, cost.bytes_written.value_or(0));
        if (cost.flops) ++flop_priced_nodes;
        known_flops = add(known_flops, cost.flops.value_or(0));
        summed_node_median_us += median_us;
    }

    [[nodiscard]] Json json() const {
        return object({{"nodes", nodes}, {"byte_priced_nodes", byte_priced_nodes},
            {"known_bytes_read", known_bytes_read}, {"known_bytes_written", known_bytes_written},
            {"known_bytes", add(known_bytes_read, known_bytes_written)}, {"flop_priced_nodes", flop_priced_nodes},
            {"known_flops", known_flops},
            {"summed_node_median_us", summed_node_median_us}});
    }
};

[[nodiscard]] inline Json totals(const std::map<std::string_view, Total>& groups) {
    auto result = Json::object();
    for (const auto& [name, total] : groups) result.set(std::string(name), total.json());
    return result;
}

[[nodiscard]] inline Json known(std::optional<std::uint64_t> value) {
    return value ? Json(*value) : Json("unknown");
}

}  // namespace step_report_detail

[[nodiscard]] inline StepNodeCost step_node_cost(std::string_view layer_class, const EntryCost& cost) {
    using step_report_detail::add;
    const auto read = add(cost.weight_bytes, cost.state_bytes);
    return {layer_class, read ? std::optional{read} : std::nullopt,
            cost.bytes_written ? std::optional{cost.bytes_written} : std::nullopt,
            cost.macs ? std::optional{add(cost.macs, cost.macs)} : std::nullopt};
}

// Use the X6 wire envelope so the nested receipt stays valid under the closed R1
// schema. Throughput is modelled logical bytes per event interval, without a
// claim about actual DRAM transactions or graph overlap.
[[nodiscard]] inline serve::Json step_profile_report(const Program& program, const StepProfile& profile,
    std::span<const StepNodeCost> costs, const StepTimingSeries& samples, double roof_gbs,
    const StepReportContext& context, receipt::Receipt receipt) {
    using namespace step_report_detail;
    const auto count = program.launches.size();
    const auto repeats = profile.repeats;
    if (count == 0 || profile.launches.size() != count || costs.size() != count ||
        samples.node_us().size() != count || repeats == 0 || samples.graph_wall_us().size() != repeats ||
        samples.eager_wall_us().size() != repeats || !std::isfinite(roof_gbs) || roof_gbs <= 0)
        throw std::invalid_argument("incomplete step profile report");

    auto result = serve::parse_json_unique(write_step_profile(profile, std::move(receipt)));
    auto launches = *result.find("launches");
    auto& rows = launches.as_array();
    std::map<std::string_view, Total> by_family, by_class;
    for (const auto layer_class : context.layer_classes) by_class.try_emplace(layer_class);
    Total total;
    for (std::size_t i = 0; i < count; ++i) {
        const auto& cost = costs[i];
        const auto& node = program.launches[i];
        if (cost.layer_class.empty() || samples.node_us()[i].size() != repeats || profile.launches[i].launch != i)
            throw std::invalid_argument("invalid step profile node");
        const auto time = range(samples.node_us()[i]);
        const auto bytes = cost.bytes_read && cost.bytes_written
            ? std::optional{add(*cost.bytes_read, *cost.bytes_written)} : std::nullopt;
        const auto gbs = bytes && time.median > 0
            ? std::optional{static_cast<double>(*bytes) / time.median / 1000} : std::nullopt;
        auto& row = rows[i];
        row.set("grid", object({
            {"blocks", array(node.grid.blocks, [](auto n) { return std::uint64_t{n}; })},
            {"threads", array(node.grid.threads, [](auto n) { return std::uint64_t{n}; })},
            {"shared_bytes", std::uint64_t{node.grid.shared_bytes}}}));
        row.set("layer", static_cast<std::int64_t>(node.layer));
        row.set("layer_class", cost.layer_class);
        row.set("gpu_us_median", time.median);
        row.set("gpu_us_min", time.min);
        if (profile.launches[i].key.empty()) row.set("key", "unknown");
        row.set("modelled_bytes_read", known(cost.bytes_read));
        row.set("modelled_bytes_written", known(cost.bytes_written));
        row.set("modelled_bytes", known(bytes));
        row.set("modelled_flops", known(cost.flops));
        row.set("achieved_gbs", nullable(gbs));
        row.set("byte_roof_pct", nullable(gbs, [roof_gbs](double rate) { return 100 * rate / roof_gbs; }));
        by_family[profile.launches[i].family].accumulate(cost, time.median);
        by_class[cost.layer_class].accumulate(cost, time.median);
        total.accumulate(cost, time.median);
    }
    const auto graph = range(samples.graph_wall_us());
    const auto eager = range(samples.eager_wall_us());
    result.set("launches", std::move(launches));
    result.set("position", std::uint64_t{context.position});
    result.set("recipe", context.recipe);
    result.set("state_after_profile", context.state_after_profile);
    result.set("dram_roof_gbs", roof_gbs);
    result.set("dram_roof_source", context.roof_source);
    result.set("work_model", "planned logical weights and activations, projection MACs counted as two FLOPs");
    result.set("timing_overhead", "eager host dispatch and per-node CUDA events");
    result.set("warm_residency", "weights, prefix and installed graph");
    result.set("totals_by_family", totals(by_family));
    result.set("totals_by_layer_class", totals(by_class));
    result.set("totals", total.json());
    result.set("timing_comparison", object({{"graph_wall_us_median", graph.median}, {"graph_wall_us_min", graph.min},
        {"eager_wall_us_median", eager.median}, {"eager_wall_us_min", eager.min},
        {"summed_node_median_minus_graph_wall_us", total.summed_node_median_us - graph.median},
        {"eager_wall_minus_graph_wall_us", eager.median - graph.median}}));
    return result;
}

}  // namespace strix::exec
