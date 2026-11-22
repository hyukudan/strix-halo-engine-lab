#include "strix/profile/catalog_levers.hpp"

#include <algorithm>
#include <cmath>

namespace strix::profile {
namespace {

constexpr std::uint32_t kMinClassRows = 800;

std::string lever_option(std::string_view name) {
    return "--lever-" + std::string(name);
}

GateVerdict gate_verdict(const CatalogLever& row, const CatalogInputs& inputs,
                         std::span<const GateRecord> gates) {
    const GateRecord* match = nullptr;
    for (const auto& gate : gates) {
        if (gate.lever != row.name || gate.model != inputs.model || gate.gate_set != inputs.gate_set) continue;
        if (match) throw std::invalid_argument("duplicate gate record for " + row.name);
        match = &gate;
    }
    if (!match) return GateVerdict::unresolved;
    if (match->verdict != GateVerdict::pass) return match->verdict;

    const ClassVerdict* window = nullptr;
    for (const auto& candidate : match->classes) {
        if (candidate.content_class != inputs.content_class || candidate.band != inputs.band) continue;
        if (window) throw std::invalid_argument("duplicate content class and band for " + row.name);
        window = &candidate;
    }
    if (!window) return GateVerdict::unresolved;
    if (window->verdict == GateVerdict::pass && window->rows < kMinClassRows) return GateVerdict::unresolved;
    return window->verdict;
}

const RegimeRecord* measured_regime(const CatalogLevers& table, const CatalogLever& row,
                                   const lever::Inputs& inputs, std::span<const RegimeRecord> regimes) {
    for (const auto& record : regimes) {
        if (record.lever != row.name || record.device != table.device || record.receipt.empty() ||
            !std::isfinite(record.gain) || record.gain <= 0) continue;
        const lever::Decl decl{.name = row.name, .exactness = row.exactness, .default_on = true, .regime = record.regime};
        if (lever::automatic(decl, inputs, true).on) return &record;
    }
    return nullptr;
}

CatalogDecision choose(const CatalogLevers& table, const CatalogLever& row, Name tier,
                       const CatalogInputs& inputs, std::span<const GateRecord> gates,
                       std::span<const RegimeRecord> regimes) {
    if (row.exactness == lever::Exactness::bitwise) return {row, true, CatalogReason::reference, {}};
    const std::string* raw = nullptr;
    for (const auto& [key, value] : inputs.options.cli)
        if (key == row.name) raw = &value;
    const std::string option = lever_option(row.name);
    const lever::Decl decl{.name = row.name, .env = option, .exactness = row.exactness, .default_on = true};
    if (raw && raw->empty()) throw std::invalid_argument(option + " requires 0 or 1");
    const auto requested = lever::select(decl, inputs.regime, raw ? raw->c_str() : nullptr);
    if (std::ranges::find(inputs.force_off, row.name) != inputs.force_off.end())
        return {row, false, CatalogReason::model_off, {}};
    if (raw && !requested.on) return {row, false, CatalogReason::explicit_switch, {}};
    if (tier == Name::standard || (row.tier == Name::turbo && tier != Name::turbo))
        return {row, false, CatalogReason::tier, {}};

    const auto verdict = row.tier == Name::fast ? gate_verdict(row, inputs, gates) : GateVerdict::pass;
    const auto* regime = measured_regime(table, row, inputs.regime, regimes);
    if (verdict == GateVerdict::pass && regime)
        return {row, true, raw ? CatalogReason::explicit_switch : CatalogReason::regime, std::string(regime->receipt)};
    const bool provisional = std::ranges::find(inputs.provisional, row.name) != inputs.provisional.end();
    if (raw && row.tier == Name::fast && provisional &&
        (verdict == GateVerdict::pass || verdict == GateVerdict::unresolved))
        return {row, true, CatalogReason::explicit_switch, {}, true};
    return {row, false, verdict == GateVerdict::pass ? CatalogReason::regime : CatalogReason::gate, {}};
}

const char* reason_name(CatalogReason reason) {
    switch (reason) {
    case CatalogReason::reference: return "reference";
    case CatalogReason::tier: return "tier";
    case CatalogReason::gate: return "gate";
    case CatalogReason::regime: return "regime";
    case CatalogReason::explicit_switch: return "explicit";
    case CatalogReason::model_off: return "model-off";
    }
    return "unknown";
}

} // namespace

CatalogLevers levers_of(const exec::Catalog& catalog, const device::Profile& device,
                        std::span<const CatalogSite> sites, std::span<const std::string_view> turbo_entries) {
    CatalogLevers table{std::string(device.name), {}};
    const auto arch = device::arch_mask(device.arch);
    for (const auto& site : sites) {
        for (const auto* entry : catalog.find(site.key, arch)) {
            const bool bitwise = entry->exact == lever::Exactness::bitwise;
            if (bitwise && entry->identity != site.reference_identity) continue;
            if (site.site.empty()) throw std::invalid_argument("catalogue lever requires a site class");
            const bool turbo = std::ranges::find(turbo_entries, entry->identity) != turbo_entries.end();
            CatalogLever row{std::string(site.site) + "/" + std::string(entry->identity), std::string(entry->identity),
                             entry->exact, entry->reference, bitwise ? Name::standard : turbo ? Name::turbo : Name::fast};
            const auto previous = std::ranges::find(table.rows, row.name, &CatalogLever::name);
            if (previous == table.rows.end()) {
                table.rows.push_back(std::move(row));
            } else if (previous->exactness != row.exactness || previous->reference != row.reference) {
                throw std::invalid_argument("conflicting catalogue entries for " + row.name);
            }
        }
    }
    return table;
}

CatalogResolution resolve(const CatalogLevers& table, const CatalogInputs& inputs,
                          std::span<const GateRecord> gates, std::span<const RegimeRecord> regimes) {
    const auto tier = inputs.options.profile.value_or(Name::standard);
    if (tier == Name::quality) throw std::invalid_argument("catalogue profile requires exact, fast or turbo");
    CatalogResolution result{tier, {}};
    result.entries.reserve(table.rows.size());
    for (const auto& row : table.rows)
        result.entries.push_back(choose(table, row, tier, inputs, gates, regimes));
    return result;
}

std::string CatalogResolution::describe() const {
    std::string out = "{\"profile\":" + json_string(tier == Name::standard ? "exact" : name(tier)) + ",\"levers\":{";
    const char* comma = "";
    for (const auto& entry : entries) {
        out += comma + json_string(entry.lever.name) + ":{\"entry\":" + json_string(entry.lever.identity) +
               ",\"tier\":" + json_string(entry.lever.tier == Name::standard ? "exact" : name(entry.lever.tier)) +
               ",\"on\":" + (entry.on ? "true" : "false") +
               ",\"reason\":" + json_string(reason_name(entry.reason));
        if (entry.lever.exactness == lever::Exactness::bit_changing) {
            out += ",\"switch\":" + json_string(lever_option(entry.lever.name)) + ",\"off\":\"0\"";
        } else {
            out += ",\"reference\":" + json_string(entry.lever.reference == exec::ReferenceClass::base ?
                                                    "base" : "host-bitwise");
        }
        if (!entry.receipt.empty()) out += ",\"receipt\":" + json_string(entry.receipt);
        if (entry.provisional) out += ",\"provisional\":true";
        out += "}";
        comma = ",";
    }
    return out + "}}";
}

std::vector<CatalogOption> options_table(const CatalogLevers& table, std::span<const std::string_view> request_options,
                                         std::span<const std::string_view> device_options) {
    std::vector<CatalogOption> options;
    const auto append = [&options](std::string_view option, OptionSource source, std::string_view row = {}) {
        if (!option.starts_with("--") || option.size() == 2 || option.find('=') != std::string_view::npos ||
            std::ranges::find(options, option, &CatalogOption::option) != options.end())
            throw std::invalid_argument("invalid or duplicate profile option: " + std::string(option));
        options.push_back({std::string(option), source, std::string(row)});
    };
    for (const auto option : {"--profile", "--allow-pending"})
        append(option, OptionSource::request);
    for (const auto option : {turbo::kWeightsOption, turbo::kBitsOption, turbo::kPackOption})
        append(option, OptionSource::request);
    for (const auto option : request_options) append(option, OptionSource::request);
    for (const auto& row : table.rows)
        if (row.exactness == lever::Exactness::bit_changing)
            append(lever_option(row.name), OptionSource::lever, row.name);
    for (const auto option : device_options) append(option, OptionSource::device);
    return options;
}

void take_catalog_options(std::span<const CatalogOption> options, std::vector<std::string>& args, Inputs& inputs) {
    std::vector<Lever> rows;
    for (const auto& option : options) {
        if (option.source != OptionSource::lever) continue;
        rows.push_back({.name = option.lever, .env = {}, .option = option.option, .value = {}, .builtin = {},
                        .target = {}, .off = "0", .source = {}});
    }
    take_options(Engine{.tag = "catalogue", .allow_env = {}, .table = rows}, args, inputs);
}

std::vector<EffectiveCell> effective_cells(const CatalogResolution& resolution, std::span<const CatalogBinding> bindings) {
    std::vector<EffectiveCell> cells;
    for (const auto& binding : bindings) {
        const auto decision = std::ranges::find_if(resolution.entries,
            [&binding](const CatalogDecision& entry) { return entry.lever.name == binding.lever; });
        if (decision == resolution.entries.end() ||
            std::ranges::find(cells, binding.legacy_row, &EffectiveCell::row) != cells.end())
            throw std::invalid_argument("invalid legacy binding for " + std::string(binding.lever));
        cells.push_back({binding.legacy_row, decision->on ? binding.on : binding.off, decision->provisional});
    }
    return cells;
}

} // namespace strix::profile
