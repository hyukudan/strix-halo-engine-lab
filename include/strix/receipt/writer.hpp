#pragma once

#include "strix/receipt/validator.hpp"
#include "strix/serve/json.hpp"

#include <functional>
#include <initializer_list>

namespace strix::receipt {
namespace detail {

using serve::Json;

inline Json object(std::initializer_list<Json::Member> members) { return Json(Json::Object(members)); }

template<class Rows, class Convert = std::identity>
Json array(const Rows& rows, Convert convert = {}) {
    Json::Array elements;
    elements.reserve(rows.size());
    for (const auto& row : rows) elements.emplace_back(convert(row));
    return Json(std::move(elements));
}

template<class T, class Convert = std::identity>
Json nullable(const std::optional<T>& value, Convert convert = {}) {
    return value ? Json(convert(*value)) : Json();
}

inline const char* name(Tier tier) {
    switch (tier) {
    case Tier::exact: return "EXACT";
    case Tier::base: return "BASE";
    case Tier::fast: return "FAST";
    case Tier::turbo: return "TURBO";
    }
    throw std::invalid_argument("resolution.tier");
}

inline const char* name(Reason reason) {
    switch (reason) {
    case Reason::explicit_value: return "explicit";
    case Reason::gate: return "gate";
    case Reason::regime: return "regime";
    case Reason::default_value: return "default";
    }
    throw std::invalid_argument("resolution.levers.reason");
}

inline Json encode_identity(const Identity& id) {
    const auto numerical = nullable(id.numerical, [](const auto& value) {
        return serve::parse_json_unique(value.json());
    });
    auto digests = Json::object();
    for (const auto& [key, value] : id.digests) digests.set(key, value);
    return object({{"engine", id.engine}, {"numerical", numerical}, {"declaration_hash", id.declaration_hash},
        {"catalogue_build_id", id.catalogue_build_id}, {"chain_table_id", id.chain_table_id},
        {"programs", array(id.programs)}, {"numeric_contracts", array(id.numeric_contracts)},
        {"route_fingerprint", id.route_fingerprint}, {"implementation", id.implementation},
        {"build", id.build}, {"kernel", id.kernel}, {"device", id.device}, {"topology", id.topology},
        {"command", id.command}, {"digests", std::move(digests)}});
}

inline Json encode_resolution(const Resolution& row) {
    return object({{"tier", name(row.tier)}, {"description", row.description},
        {"levers", array(row.levers, [](const Lever& lever) {
            return object({{"name", lever.name}, {"value", lever.value}, {"reason", name(lever.reason)},
                {"off_switch", lever.off_switch}});
        })}});
}

inline Json encode_load(const Load& row) {
    return object({{"tensors", array(row.tensors, [](const Tensor& tensor) {
            return object({{"name", tensor.name}, {"source", tensor.source}, {"format", tensor.format},
                {"bytes", tensor.bytes}, {"crc", std::uint64_t{tensor.crc}}});
        })}, {"images", array(row.images, [](const Image& image) {
            return object({{"name", image.name}, {"source", image.source}, {"lifetime", image.lifetime},
                {"crc", std::uint64_t{image.crc}}});
        })}, {"folds", array(row.folds)}, {"residency_plan", array(row.residency_plan)}});
}

inline Json encode_plan(const Plan& row) {
    return object({{"programs", array(row.programs, [](const ProgramPlan& plan) {
            return object({{"program", plan.program}, {"launches", plan.launches}, {"packs", plan.packs},
                {"sidecars", plan.sidecars}});
        })}, {"declined_fusions", array(row.declined_fusions, [](const DeclinedFusion& decline) {
            return object({{"program", decline.program}, {"fusion", decline.fusion},
                {"missing_catalogue_key", decline.missing_catalogue_key}});
        })}, {"split_plan", array(row.split_plan)}});
}

inline Json encode_step(const std::vector<ProgramStep>& rows) {
    return array(rows, [](const ProgramStep& row) {
        return object({{"program", row.program}, {"predicted_ms", nullable(row.predicted_ms)},
            {"wall_ms", nullable(row.wall_ms)}, {"launches", row.launches}, {"bytes_read", row.bytes_read},
            {"exchanges", row.exchanges}, {"guards", row.guards}, {"guard_failures", row.guard_failures},
            {"device_status", std::uint64_t{row.device_status}}});
    });
}

inline Json encode_speculation(const Speculation& row) {
    return object({{"cycles", row.cycles}, {"proposed", row.proposed}, {"accepted", row.accepted},
        {"rows_per_verify", array(row.rows_per_verify)}, {"drafter_ms", row.drafter_ms},
        {"controller_state", row.controller_state}});
}

inline Json encode_serving(const std::vector<Lease>& rows) {
    return array(rows, [](const Lease& row) {
        return object({{"lease", row.lease}, {"tokens", row.tokens}, {"prefix_hit_length", row.prefix_hit_length},
            {"rounds", row.rounds}, {"queue_ms", row.queue_ms}, {"finish_reason", row.finish_reason}});
    });
}

inline Json encode_devices(const std::vector<Device>& rows) {
    return array(rows, [](const Device& row) {
        return object({{"profile_id", row.profile_id}, {"power_state", row.power_state},
            {"bytes_per_second", nullable(row.bytes_per_second)}, {"clock_mhz", nullable(row.clock_mhz)},
            {"memory_clock_mhz", nullable(row.memory_clock_mhz)},
            {"links", array(row.links, [](const LinkCost& link) {
                return object({{"peer", link.peer}, {"latency_ms", link.latency_ms},
                    {"bytes_per_second", link.bytes_per_second}});
            })}, {"telemetry", array(row.telemetry, [](const Telemetry& sample) {
                return object({{"elapsed_ms", sample.elapsed_ms}, {"clock_mhz", nullable(sample.clock_mhz)},
                    {"memory_clock_mhz", nullable(sample.memory_clock_mhz)}, {"power_w", nullable(sample.power_w)}});
            })}});
    });
}

inline Json encode_quality(const std::vector<Gate>& rows) {
    return array(rows, [](const Gate& row) { return object({{"lever", row.lever}, {"record_id", row.record_id}}); });
}

}  // namespace detail

[[nodiscard]] inline std::string write_json(const Receipt& receipt) {
    using namespace detail;
    validate(receipt);
    auto output = object({{"schema", schema_id}, {"identity", encode_identity(receipt.identity)},
        {"resolution", nullable(receipt.resolution, encode_resolution)}, {"load", nullable(receipt.load, encode_load)},
        {"plan", nullable(receipt.plan, encode_plan)}, {"step", nullable(receipt.step, encode_step)},
        {"speculation", nullable(receipt.speculation, encode_speculation)}, {"serving", nullable(receipt.serving, encode_serving)},
        {"devices", nullable(receipt.devices, encode_devices)}, {"quality", nullable(receipt.quality, encode_quality)}}).dump();
    // The strict reader owns UTF-8 validation, including evidence strings from external assets.
    (void)serve::parse_json_unique(output);
    return output;
}

}  // namespace strix::receipt
