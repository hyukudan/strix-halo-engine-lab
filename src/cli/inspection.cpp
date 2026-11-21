#include "detail.hpp"

#include "strix/deploy/minimax_layercut.hpp"
#include "strix/device/intel_enumerator.hpp"
#include "gemma4/engine/deployment.hpp"
#include "minimax/engine/deployment.hpp"
#if STRIX_CLI_QWEN_DESCRIPTION
#include "strix/qwen_model_description.hpp"
#endif

#include <type_traits>

namespace strix::cli {
namespace {
using serve::Json;
using model_description::Value;

Json metadata_json(const Value& value) {
    return std::visit([](const auto& field) -> Json {
        using T = std::decay_t<decltype(field)>;
        if constexpr (std::is_same_v<T, Value::Table>) {
            Json object = Json::object();
            for (const auto& [key, child] : field) object.set(key, metadata_json(child));
            return object;
        } else if constexpr (std::is_same_v<T, Value::Array>) {
            Json array = Json::array();
            for (const auto& child : field) array.push(metadata_json(child));
            return array;
        } else {
            return Json(field);
        }
    }, value.data);
}

Json strings_json(const std::vector<std::string>& values) {
    Json result = Json::array();
    for (const auto& value : values) result.push(value);
    return result;
}
} // namespace

Json describe(const model_registry::Entry& entry, const std::filesystem::path& path) {
    auto description = model_description::load(path.string());
    const auto* name = description.find("name");
    const auto* parsed_name = name ? std::get_if<std::string>(&name->data) : nullptr;
    if (!parsed_name || *parsed_name != entry.name)
        throw std::invalid_argument("description name disagrees with the registry for " + entry.name);
#if STRIX_CLI_QWEN_DESCRIPTION
    if (entry.family == "qwen" && entry.name == "qwen3.8-flash-next")
        (void)qwen_description::inspect(description);
#endif
    return metadata_json(Value{std::move(description.fields)});
}

Json devices() {
    const auto path = device::read_environment().devices_path;
    Json declared = Json::array();
    if (!path.empty()) {
        const auto declaration = device::load_declaration(path);
        const auto* host = device::host_unit(declaration);
        for (const auto& entry : declaration.devices) {
            Json row = Json::Object{{"id", entry.id}, {"kind", device::kind_name(entry.kind)},
                {"vendor", entry.kind == device::Kind::cpu ? Json() : Json(device::vendor_name(entry.vendor))},
                {"name", entry.name}, {"arch", entry.arch}, {"pci", entry.pci}, {"uuid", entry.uuid},
                {"memory_bytes", entry.memory_bytes}, {"roles", strings_json(entry.roles)}, {"profile", entry.profile},
                {"excluded", entry.excluded}, {"reason", entry.reason}, {"planned", entry.planned},
                {"count", static_cast<std::uint64_t>(entry.count)}};
            if (entry.kind == device::Kind::gpu && !entry.excluded)
                row.set("budget", serve::parse_json(device::resolve_budget(entry, host, 0).json()));
            declared.push(std::move(row));
        }
    }

    Json discovered = Json::array();
    for (const auto& fact : deploy::probe_devices()) {
        discovered.push(Json::Object{{"vendor", "amd"}, {"arch", fact.gfx_target},
            {"memory_bytes", fact.mem_total_bytes}, {"source", fact.source}});
    }
    for (const auto& listing : {device::query_nvidia_smi(), device::query_xpu_smi()}) {
        if (!listing) continue;
        for (const auto& visible : *listing) {
            discovered.push(Json::Object{{"vendor", device::vendor_name(visible.vendor)}, {"name", visible.name},
                {"arch", visible.arch}, {"pci", visible.pci}, {"uuid", visible.uuid},
                {"memory_bytes", visible.memory_bytes}});
        }
    }
    return Json::Object{{"schema", "strix-cli-devices-v1"}, {"declaration", path},
        {"declared", declared}, {"discovered", discovered}};
}

Json plan(const model_registry::Entry& entry, const Request& request) {
    Json budgets = Json::array();
    deploy::Inventory inventory;
    if (!request.devices.empty()) inventory.declaration = device::load_declaration(request.devices);
    const auto* host = device::host_unit(inventory.declaration);
    std::vector<deploy::DeviceFacts> facts;
    std::vector<const device::Descriptor*> active;
    for (const auto& descriptor : inventory.declaration.devices) {
        if (descriptor.kind != device::Kind::gpu || descriptor.excluded) continue;
        active.push_back(&descriptor);
        const auto budget = device::resolve_budget(descriptor, host, 0);
        budgets.push(serve::parse_json(budget.json()));
        deploy::DeviceFacts fact;
        fact.gfx_target = descriptor.arch;
        fact.mem_total_bytes = budget.device_bytes ? budget.device_bytes : descriptor.memory_bytes;
        fact.mem_free_bytes = fact.mem_total_bytes;
        fact.unified = descriptor.attach.kind == "integrated";
        fact.source = "declared planning ceiling";
        if (fact.mem_total_bytes) inventory.facts.emplace(descriptor.id, fact);
        if (descriptor.count > 256U - facts.size())
            throw std::invalid_argument("declared inventory exceeds 256 devices");
        facts.insert(facts.end(), descriptor.count, fact);
    }
    Json output = Json::Object{{"schema", "strix-cli-plan-v1"}, {"kind", "plan"}, {"measurement", false},
        {"model", entry.name}, {"context", static_cast<std::uint64_t>(request.context)},
        {"status", "refused"}, {"device_budgets", budgets}};
    deploy::RouteRequest route_request;
    route_request.spec.context = request.context;
    const auto record = [&](const deploy::Plan& planned) {
        output.set("deployment", serve::parse_json(planned.to_json()));
        output.set("status", planned.refused ? "refused" : "planned");
        output.set("reason", planned.refused ? planned.refusal : "fits the declared planning ceilings; runtime admission still required");
    };
    if (entry.family != "minimax" && entry.family != "gemma4") {
        output.set("reason", "no asset-only deployment provider is bound for family " + entry.family);
    } else if (request.model == entry.id || request.model == entry.name) {
        output.set("reason", "supply the registered model asset path to read its weight headers");
    } else if (facts.empty()) {
        output.set("reason", "supply a --devices declaration with device memory budgets");
    } else if (entry.family == "minimax") {
        const auto provider = minimax::engine::FootprintProvider(minimax::engine::deployment_model(request.model));
        const auto route = deploy::plan_route(route_request, inventory, deploy::MinimaxLayercutProvider(provider), {});
        output.set("route", serve::parse_json(route.to_json()));
        if (active.size() == 1 && active[0]->count == 2 && !active[0]->interconnect.empty()) {
            route_request.spec.link = active[0]->interconnect;
        } else if (active.size() == 2 && facts.size() == 2) {
            for (const auto& link : inventory.declaration.links) {
                if ((link.a == active[0]->id && link.b == active[1]->id) ||
                    (link.b == active[0]->id && link.a == active[1]->id))
                    route_request.spec.link = link.link;
            }
        }
        record(deploy::resolve(route_request.spec, facts, provider));
    } else {
        const auto provider = gemma4::engine::FootprintProvider(gemma4::engine::deployment_model(request.model));
        record(deploy::resolve(route_request.spec, facts, provider));
    }
    return output;
}

} // namespace strix::cli
