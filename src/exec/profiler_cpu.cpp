#include "strix/exec/profiler.hpp"
#include "strix/receipt/writer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

namespace strix::exec {
namespace {

using receipt::detail::array;
using receipt::detail::nullable;
using receipt::detail::object;

template<class Enum>
std::uint64_t ordinal(Enum value) { return static_cast<std::uint64_t>(value); }

std::string catalogue_key(const KernelKey& key) {
    return object({{"op", ordinal(key.op)}, {"form", key.form}, {"weight", key.weight.name},
        {"weight_revision", std::uint64_t{key.weight.revision}}, {"act", ordinal(key.act)},
        {"rows_class", ordinal(key.rows)}, {"dim_class", std::uint64_t{key.dim_class}},
        {"prologue", std::uint64_t{key.prologue}}, {"epilogue", std::uint64_t{key.epilogue}},
        {"math", ordinal(key.math)}}).dump();
}

std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) throw std::overflow_error("step profile byte count");
    return a + b;
}

bool nonnegative(double value) { return std::isfinite(value) && value >= 0; }

void append_stages(LaunchSample& sample, const Program& program, std::span<const StageBinding> bindings) {
    for (const auto& stage : bindings) {
        const auto& id = program.stages.at(stage.instance).id;
        if (std::ranges::find(sample.stages, id) == sample.stages.end()) sample.stages.push_back(id);
    }
}

}  // namespace

StepProfile make_step_profile(const Program& program, const ProfileOptions& options, ProfileTimings timings) {
    const auto count = program.launches.size();
    const bool measured = !timings.launch_ms.empty();
    if (count == 0 || count > no_launch || options.work.size() != count || options.repeats == 0 ||
        !nonnegative(options.byte_gbps) || (measured && (timings.launch_ms.size() != count ||
        !nonnegative(timings.wall_ms) || !nonnegative(timings.host_ms) ||
        (timings.cold_wall_ms && !nonnegative(*timings.cold_wall_ms)))))
        throw std::invalid_argument("invalid step profile plan or timings");

    StepProfile result;
    result.key = program.key;
    result.program = profile::hex_identity(program.identity);
    if (program.predicted.launches && nonnegative(program.predicted.us)) result.predicted_ms = program.predicted.us / 1000;
    if (measured) {
        result.repeats = options.repeats;
        result.cold_wall_ms = timings.cold_wall_ms;
        result.wall_ms = timings.wall_ms;
        result.host_ms = timings.host_ms;
        result.busy_ms = 0;
        result.exchange_ms = 0;
        result.device_status = timings.device_status;
    }
    result.launches.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto& bound = program.launches[i];
        const auto& work = options.work[i];
        if (!nonnegative(work.peak_tops) || (work.entry && work.entry->identity != bound.identity))
            throw std::invalid_argument("invalid step profile entry or operand peak");
        auto& sample = result.launches.emplace_back();
        sample.launch = static_cast<std::uint32_t>(i);
        sample.entry = bound.identity.empty() ? bound.entry : bound.identity;
        sample.family = work.entry ? work.entry->key.form : std::string_view(bound.entry);
        if (sample.entry.empty()) throw std::invalid_argument("step profile node needs an entry identity");
        if (work.entry) sample.key = catalogue_key(work.entry->key);
        append_stages(sample, program, bound.prologue);
        append_stages(sample, program, bound.stages);
        append_stages(sample, program, bound.epilogue);
        sample.weight_bytes = work.weight_bytes;
        sample.state_bytes = work.state_bytes;
        sample.ops = work.ops;
        sample.exchange = work.exchange;
        const auto bytes = add(work.weight_bytes, work.state_bytes);
        result.bytes_per_step = add(result.bytes_per_step, bytes);
        if (work.exchange) ++result.exchanges;
        if (!measured) continue;

        const auto ms = timings.launch_ms[i];
        if (!nonnegative(ms)) throw std::invalid_argument("invalid step profile node time");
        sample.ms = ms;
        *result.busy_ms += ms;
        if (work.exchange) *result.exchange_ms += ms;
        if (ms > 0 && options.byte_gbps > 0)
            sample.byte_roof_pct = static_cast<double>(bytes) / 1e4 / options.byte_gbps / ms;
        if (ms > 0 && work.peak_tops > 0)
            sample.compute_roof_pct = static_cast<double>(work.ops) / 1e7 / work.peak_tops / ms;
    }
    if (options.byte_gbps > 0) result.byte_floor_ms = static_cast<double>(result.bytes_per_step) / 1e6 / options.byte_gbps;
    return result;
}

receipt::Receipt profile_receipt(const StepProfile& profile, receipt::Receipt receipt) {
    if (std::ranges::find(receipt.identity.programs, profile.program) == receipt.identity.programs.end())
        receipt.identity.programs.push_back(profile.program);
    if (!receipt.step) receipt.step.emplace();
    receipt.step->push_back({profile.program, profile.predicted_ms, profile.wall_ms, profile.launches.size(),
        profile.bytes_per_step, profile.exchanges, 0, 0, profile.device_status});

    std::map<std::pair<std::string, std::string>, receipt::ProgramStep> by_key;
    for (const auto& sample : profile.launches) {
        auto& row = by_key[{sample.entry, sample.key}];
        ++row.launches;
        if (sample.exchange) ++row.exchanges;
        row.bytes_read = add(row.bytes_read, add(sample.weight_bytes, sample.state_bytes));
        if (sample.ms) row.wall_ms = row.wall_ms.value_or(0) + *sample.ms;
    }
    for (auto& [key, row] : by_key) {
        row.program = profile.program + "/entry/" + key.first;
        if (!key.second.empty()) row.program += "/" + key.second;
        receipt.step->push_back(std::move(row));
    }
    return receipt;
}

std::string write_step_profile(const StepProfile& profile, receipt::Receipt receipt) {
    const auto& key = profile.key;
    const auto program_key = object({{"phase", ordinal(key.phase)}, {"rows", std::uint64_t{key.rows}},
        {"pos", ordinal(key.pos)}, {"tree", key.tree}, {"sampled", key.sampled}, {"tap", key.tap},
        {"graph_id", std::uint64_t{key.graph_id}}, {"tier", ordinal(key.tier)},
        {"ctx_bucket", std::uint64_t{key.ctx_bucket}}});
    const auto launches = array(profile.launches, [](const LaunchSample& sample) {
        return object({{"launch", std::uint64_t{sample.launch}}, {"entry", sample.entry}, {"family", sample.family},
            {"key", sample.key.empty() ? serve::Json() : serve::parse_json_unique(sample.key)},
            {"stages", array(sample.stages)}, {"ms", nullable(sample.ms)}, {"weight_bytes", sample.weight_bytes},
            {"state_bytes", sample.state_bytes}, {"ops", sample.ops}, {"exchange", sample.exchange},
            {"byte_roof_pct", nullable(sample.byte_roof_pct)},
            {"compute_roof_pct", nullable(sample.compute_roof_pct)}});
    });
    std::optional<double> interval_minus_wall;
    if (profile.busy_ms && profile.wall_ms) interval_minus_wall = *profile.busy_ms - *profile.wall_ms;
    return object({{"schema", "strix-step-profile/v1"}, {"program", profile.program}, {"program_key", program_key},
        {"timing", profile.repeats ? "eager-cuda-event-intervals" : "unmeasured"},
        {"repeats", std::uint64_t{profile.repeats}}, {"cold_wall_ms", nullable(profile.cold_wall_ms)},
        {"wall_ms", nullable(profile.wall_ms)},
        {"busy_ms", nullable(profile.busy_ms)}, {"gap_ms", serve::Json()}, {"host_ms", nullable(profile.host_ms)},
        {"eager_minus_production_ms", nullable(interval_minus_wall)}, {"exchange_ms", nullable(profile.exchange_ms)},
        {"bytes_per_step", profile.bytes_per_step}, {"byte_floor_ms", nullable(profile.byte_floor_ms)},
        {"launches", launches}, {"receipt", serve::parse_json_unique(receipt::write_json(profile_receipt(profile, std::move(receipt))))}}).dump();
}

std::string write_step_table(const StepProfile& profile) {
    std::vector<const LaunchSample*> ordered;
    ordered.reserve(profile.launches.size());
    for (const auto& sample : profile.launches) ordered.push_back(&sample);
    std::ranges::stable_sort(ordered, std::greater{}, &LaunchSample::ms);
    std::ostringstream out;
    out << "launch gpu_ms family entry stages\n";
    for (const auto* sample : ordered) {
        out << sample->launch << ' ';
        if (sample->ms) out << *sample->ms;
        else out << '-';
        out << ' ' << sample->family << ' ' << sample->entry;
        for (const auto& stage : sample->stages) out << ' ' << stage;
        out << '\n';
    }
    return out.str();
}

}  // namespace strix::exec
