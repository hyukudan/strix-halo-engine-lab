#pragma once

#include "strix/receipt/validator.hpp"
#include "strix/serve/json.hpp"

namespace strix::receipt {
namespace detail {
inline const serve::Json& member(const serve::Json& object, std::string_view key) {
    const auto* value = object.find(key);
    if (!value) throw std::invalid_argument("missing legacy receipt field: " + std::string(key));
    return *value;
}
}  // namespace detail

// Project the identity and execution provenance of today's engine and route receipts.
// The route producer supplies a route fingerprint independently of the numerical identity.
[[nodiscard]] inline Receipt map_legacy_identity(std::string_view text) {
    using detail::member;
    constexpr std::string_view witness = "strix-engine-witness ";
    constexpr std::string_view route = "strix-route-receipt ";
    const auto prefix = text.starts_with(witness) ? witness : (text.starts_with(route) ? route : std::string_view{});
    text.remove_prefix(prefix.size());
    const auto source = serve::parse_json_unique(text);
    const auto& schema = member(source, "schema").as_string();
    Receipt out;
    auto& id = out.identity;
    if (schema == "strix-route-receipt-v1" && prefix != witness) {
        if (const auto* engine = source.find("engine")) id.engine = engine->as_string();
        id.route_fingerprint = member(source, "fingerprint").as_string();
        detail::require(detail::is_hex_id(id.route_fingerprint), "legacy route fingerprint");
    } else if (schema == "strix-engine-receipt/v1" && prefix != route) {
        id.engine = member(source, "engine").as_string();
        const auto& numerical = member(source, "identity");
        const auto field = [&](std::string_view key) { return member(numerical, key).as_string(); };
        id.numerical = profile::NumericalIdentity{field("arithmetic_contract"), field("model_sha256"),
            field("pack_hash"), field("table_hash"), field("state_format"), field("prefill_lineage"), field("reduction_plan")};
        detail::require(field("id") == profile::hex_identity(id.numerical->id()), "legacy numerical identity id mismatch");
        id.implementation = member(source, "implementation").as_string();
        id.build = member(source, "build").as_string();
        id.kernel = member(source, "kernel").as_string();
        id.device = member(source, "device").as_string();
        id.topology = member(source, "topology").as_string();
        id.command = member(source, "command").as_string();
        for (const auto& [key, value] : member(source, "digests").as_object()) id.digests.emplace_back(key, value.as_string());
    } else {
        throw std::invalid_argument("unsupported legacy receipt schema or framing");
    }
    return out;
}

}  // namespace strix::receipt
