#include "strix/profile/multi_device.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <variant>

namespace strix::profile::multi_device {
namespace {
using model_description::Value;
using serve::Json;

[[noreturn]] void refuse(const std::string& message) { throw std::invalid_argument("multi_device: " + message); }

const Value::Table& table(const Value& v, std::string_view path) {
    if (const auto* t = std::get_if<Value::Table>(&v.data)) return *t;
    refuse(std::string(path) + " must be a table");
}

const std::string& text(const Value& v, std::string_view path) {
    const auto* s = std::get_if<std::string>(&v.data);
    if (!s || s->empty()) refuse(std::string(path) + " must be a nonempty string");
    return *s;
}

std::int64_t integer(const Value& v, const std::string& path, std::int64_t lo, std::int64_t hi) {
    const auto* i = std::get_if<std::int64_t>(&v.data);
    if (!i || *i < lo || *i > hi)
        refuse(path + " must be an integer in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
    return *i;
}

const std::string& one_of(const Value& v, std::string_view path, std::initializer_list<std::string_view> allowed) {
    const auto& s = text(v, path);
    if (std::ranges::find(allowed, std::string_view(s)) == allowed.end()) {
        std::string list;
        for (auto w : allowed) list += (list.empty() ? "" : ", ") + std::string(w);
        refuse(std::string(path) + " must be one of: " + list + " (got '" + s + "')");
    }
    return s;
}

std::vector<std::string> words(const Value& v, const std::string& path) {
    const auto* a = std::get_if<Value::Array>(&v.data);
    if (!a || a->empty()) refuse(path + " must be a nonempty array of strings");
    std::vector<std::string> out;
    for (const auto& item : *a) {
        out.push_back(text(item, path));
        if (std::ranges::count(out, out.back()) > 1) refuse(path + " repeats '" + out.back() + "'");
    }
    return out;
}

Json value_json(const Value& v) {
    return std::visit([](const auto& field) -> Json {
        using T = std::decay_t<decltype(field)>;
        if constexpr (std::is_same_v<T, Value::Array>) {
            Json array = Json::array();
            for (const auto& child : field) array.push(value_json(child));
            return array;
        } else if constexpr (std::is_same_v<T, Value::Table>) {
            Json object = Json::object();
            for (const auto& [key, child] : field) object.set(key, value_json(child));
            return object;
        } else {
            return Json(field);
        }
    }, v.data);
}

// "auto" or a positive integer (rows, cut layer).
void auto_or_count(const Value& v, const std::string& path) {
    if (std::holds_alternative<std::int64_t>(v.data)) (void)integer(v, path, 1, std::numeric_limits<std::int32_t>::max());
    else (void)text(v, path);   // "auto" or a validated map spelling owned by the binding route
}

const std::initializer_list<std::string_view> kWires{"reference", "bf16-exact", "fp16", "fp8-e4m3-g64-v1"};

int tier_rank(Name tier) noexcept { return tier == Name::turbo ? 2 : tier == Name::fast ? 1 : 0; }

Name parse_min_tier(const Value& v, const std::string& path) {
    const auto& s = one_of(v, path, {"exact", "fast", "turbo"});
    return s == "exact" ? Name::standard : s == "fast" ? Name::fast : Name::turbo;
}

Form form_word(const Value& v, const std::string& path) {
    const auto& s = text(v, path);
    const auto form = parse_form(s);
    if (!form) refuse(path + " names an unknown form '" + s + "' (single, replica, layers, tensor, speculation)");
    return *form;
}

FormSpec parse_form_spec(Form form, const Value& v, const std::string& path) {
    FormSpec spec;
    spec.form = form;
    spec.ranks = form == Form::single ? 1 : 2;
    for (const auto& [key, item] : table(v, path)) {
        const auto at = path + "." + key;
        const auto allowed = [&](std::initializer_list<Form> forms) {
            if (std::ranges::find(forms, form) == forms.end()) refuse("unknown key " + at);
        };
        if (key == "ranks") {
            allowed({Form::replica, Form::layers, Form::tensor, Form::speculation});
            spec.ranks = static_cast<std::uint32_t>(integer(item, at, 2, 64));
        } else if (key == "objective") {
            allowed({Form::replica});
            (void)one_of(item, at, {"throughput", "latency"});
        } else if (key == "axis") {
            allowed({Form::layers});
            (void)one_of(item, at, {"layer", "visit"});
        } else if (key == "cut" ) {
            allowed({Form::layers});
            auto_or_count(item, at);
        } else if (key == "window_rows" || key == "rows_per_message") {
            allowed(key == "window_rows" ? std::initializer_list<Form>{Form::layers}
                                         : std::initializer_list<Form>{Form::tensor, Form::speculation});
            auto_or_count(item, at);
        } else if (key == "components") {
            allowed({Form::layers, Form::tensor});
            for (const auto& c : words(item, at))
                if (c != "trunk" && c != "attention" && c != "ffn" && c != "moe" && c != "head" && c != "drafter")
                    refuse(at + " has unknown component '" + c + "'");
        } else if (key == "state") {
            allowed({Form::layers});
            (void)one_of(item, at, {"relay-if-needed", "stage-owned"});
        } else if (key == "stages") {
            allowed({Form::tensor});
            (void)text(item, at);
        } else if (key == "expert_partition") {
            allowed({Form::tensor});
            (void)one_of(item, at, {"intermediate", "expert_ids", "slots"});
        } else if (key == "expert_shares") {
            allowed({Form::tensor});
            if (const auto* list = std::get_if<Value::Array>(&item.data)) {
                double sum = 0;
                for (const auto& share : *list) {
                    const auto* d = std::get_if<double>(&share.data);
                    if (!d || *d <= 0) refuse(at + " must be \"auto\" or positive fractions");
                    sum += *d;
                }
                if (list->size() < 2 || sum < 0.999 || sum > 1.001) refuse(at + " fractions must sum to 1");
            } else if (text(item, at) != "auto") refuse(at + " must be \"auto\" or an array of fractions");
        } else if (key == "wire") {
            allowed({Form::layers, Form::tensor, Form::speculation});
            spec.wire = one_of(item, at, kWires);
        } else if (key == "draft_placement") {
            allowed({Form::tensor, Form::speculation});
            spec.draft_placement = one_of(item, at, {"none", "leader", "split"});
        } else if (key == "draft_rank") {
            allowed({Form::tensor, Form::speculation});
            spec.draft_rank = static_cast<std::uint32_t>(integer(item, at, 0, 63));
        } else {
            refuse("unknown key " + at);
        }
        spec.fields.emplace(key, value_json(item));
    }
    if (spec.draft_rank >= spec.ranks) refuse(path + ".draft_rank must name one of the form's ranks");
    if (spec.form == Form::speculation && spec.draft_placement == "none")
        refuse(path + ".draft_placement must be leader or split for the speculation form");
    return spec;
}

Regime parse_regime(const std::string& name, const Value& v, const std::string& path) {
    Regime r;
    r.name = name;
    for (const auto& [key, item] : table(v, path)) {
        const auto at = path + "." + key;
        constexpr auto kMax = std::numeric_limits<std::int64_t>::max();
        if (key == "phase") {
            const auto& p = one_of(item, at, {"request", "prefill", "decode"});
            r.phase = p == "request" ? Phase::request : p == "prefill" ? Phase::prefill : Phase::decode;
        } else if (key == "min_prompt_rows") r.min_prompt_rows = static_cast<std::uint64_t>(integer(item, at, 0, kMax));
        else if (key == "max_prompt_rows") r.max_prompt_rows = static_cast<std::uint64_t>(integer(item, at, 0, kMax));
        else if (key == "min_ready_requests") r.min_ready_requests = static_cast<std::uint64_t>(integer(item, at, 1, kMax));
        else if (key == "max_ready_requests") r.max_ready_requests = static_cast<std::uint64_t>(integer(item, at, 1, kMax));
        else if (key == "min_tier") r.min_tier = parse_min_tier(item, at);
        else if (key == "prefer") {
            for (const auto& w : words(item, at)) {
                const auto form = parse_form(w);
                if (!form) refuse(at + " names an unknown form '" + w + "'");
                r.prefer.push_back(*form);
            }
        } else if (key == "wire") r.wire = one_of(item, at, kWires);
        else if (key == "lever") r.lever = text(item, at);
        else if (key == "enabled") {
            const auto* b = std::get_if<bool>(&item.data);
            if (!b) refuse(at + " must be a boolean");
            r.enabled = *b;
        } else refuse("unknown key " + at);
    }
    if (r.prefer.empty()) refuse(path + ".prefer is required");
    if (r.min_prompt_rows && r.max_prompt_rows && *r.min_prompt_rows > *r.max_prompt_rows)
        refuse(path + " has min_prompt_rows > max_prompt_rows");
    if (r.min_ready_requests && r.max_ready_requests && *r.min_ready_requests > *r.max_ready_requests)
        refuse(path + " has min_ready_requests > max_ready_requests");
    if (!r.wire.empty() && !exact_wire(r.wire) && r.min_tier == Name::standard)
        refuse(path + ".wire '" + r.wire + "' changes bits: the regime needs min_tier = \"fast\" or \"turbo\"");
    return r;
}

// A regime may name only forms the model declares, and its wire must bind to forms that carry one: a disabled
// regime is data awaiting a receipt, so it obeys the same rules (enabling it changes nothing else).
void validate_regime(const Policy& policy, const Regime& r, const std::string& path) {
    for (auto form : r.prefer) {
        if (!policy.forms.contains(form))
            refuse(path + ".prefer names '" + std::string(form_name(form)) +
                   "', which the model does not declare under multi_device.forms");
        const bool carries_wire = form != Form::single && form != Form::replica;
        if (!r.wire.empty() && !carries_wire)
            refuse(path + ".wire '" + r.wire + "' cannot apply to '" + std::string(form_name(form)) +
                   "' (it runs the reference wire): list only layers / tensor / speculation forms, or drop the wire");
        if (r.wire.empty() && carries_wire) {
            const auto& wire = policy.forms.at(form).wire;
            if (!exact_wire(wire) && r.min_tier == Name::standard)
                refuse(path + " prefers '" + std::string(form_name(form)) + "' whose wire '" + wire +
                       "' changes bits: the regime needs min_tier = \"fast\" or \"turbo\"");
        }
    }
}

bool rows_match(const std::optional<std::uint64_t>& value, const std::optional<std::uint64_t>& lo,
                const std::optional<std::uint64_t>& hi) {
    if (!lo && !hi) return true;
    if (!value) return false;
    return (!lo || *value >= *lo) && (!hi || *value <= *hi);
}

// Empty = the form can run for this query; otherwise the reason it cannot.
std::string blocked(const Policy& policy, Form form, const Query& q, std::string_view wire) {
    if (form == Form::single) return {};
    const auto it = policy.forms.find(form);
    if (it == policy.forms.end()) return std::string(form_name(form)) + " is not declared by the model";
    if (q.devices < it->second.ranks)
        return std::string(form_name(form)) + " needs " + std::to_string(it->second.ranks) + " devices, " +
               std::to_string(q.devices) + " available";
    if (!exact_wire(wire) && tier_rank(q.tier) == 0)
        return std::string(form_name(form)) + " wire '" + std::string(wire) + "' changes bits and the tier is exact";
    return {};
}

Decision decided(const Policy& policy, Form form, const Query& q, std::string wire, std::string rule, std::string reason) {
    Decision d;
    d.form = form;
    d.ranks = form == Form::single ? 1 : form == Form::replica ? q.devices : policy.forms.at(form).ranks;
    d.wire = form == Form::single || form == Form::replica ? "reference" : std::move(wire);
    d.rule = std::move(rule);
    d.reason = std::move(reason);
    return d;
}

std::string form_wire(const Policy& policy, Form form) {
    const auto it = policy.forms.find(form);
    return it == policy.forms.end() ? std::string("reference") : it->second.wire;
}
} // namespace

std::string_view form_name(Form form) noexcept {
    switch (form) {
    case Form::single: return "single";
    case Form::replica: return "replica";
    case Form::layers: return "layers";
    case Form::tensor: return "tensor";
    case Form::speculation: return "speculation";
    }
    return "single";
}

std::optional<Form> parse_form(std::string_view word) noexcept {
    for (auto f : {Form::single, Form::replica, Form::layers, Form::tensor, Form::speculation})
        if (form_name(f) == word) return f;
    return std::nullopt;
}

std::string_view phase_name(Phase phase) noexcept {
    return phase == Phase::prefill ? "prefill" : phase == Phase::decode ? "decode" : "request";
}

bool exact_wire(std::string_view wire) noexcept { return wire == "reference" || wire == "bf16-exact"; }

Policy parse_policy(const model_description::ModelDescription& description) {
    Policy policy;
    policy.forms.emplace(Form::single, FormSpec{});
    const auto* section = description.find("multi_device");
    if (!section) return policy;
    std::optional<Form> declared_default, declared_fallback;
    std::vector<std::string> order;
    const Value::Table* regimes = nullptr;
    bool placement_keys = false;   // a section that only carries the model's legacy inventory declares no placement policy
    for (const auto& [key, item] : table(*section, "multi_device")) {
        const auto at = "multi_device." + key;
        placement_keys |= key != "legacy_env";
        if (key == "version") policy.version = integer(item, at, 1, 1);
        else if (key == "default") declared_default = form_word(item, at);
        else if (key == "fallback") declared_fallback = form_word(item, at);
        else if (key == "order") order = words(item, at);
        else if (key == "legacy_env") {
            for (const auto& [env, target] : table(item, at)) {
                const auto path = text(target, at + "." + env);
                if (!env.starts_with("STRIX_") || !path.starts_with("forms."))
                    refuse(at + "." + env + " must map a STRIX_* switch to a forms.* key");
                policy.legacy.push_back({env, path});
            }
            continue;
        }
        else if (key == "forms") {
            for (const auto& [name, spec] : table(item, at)) {
                const auto form = parse_form(name);
                if (!form || *form == Form::single) refuse(at + "." + name + " is not a declarable form");
                policy.forms[*form] = parse_form_spec(*form, spec, at + "." + name);
            }
        } else if (key == "regimes") regimes = &table(item, at);
        else refuse("unknown key " + at);
    }
    if (!placement_keys) return policy;
    if (policy.version != 1) refuse("multi_device.version = 1 is required");
    policy.default_form = declared_default.value_or(Form::single);
    policy.fallback = declared_fallback.value_or(Form::single);
    for (const auto& [form, path] : {std::pair{policy.default_form, "default"}, std::pair{policy.fallback, "fallback"}})
        if (!policy.forms.contains(form)) refuse(std::string("multi_device.") + path + " names an undeclared form");
    if (policy.fallback != Form::single) refuse("multi_device.fallback must be \"single\" (it runs on one device)");
    const std::size_t count = regimes ? regimes->size() : 0;
    if (count > 1 && order.size() != count) refuse("multi_device.order must list every regime exactly once");
    if (count == 1 && order.empty()) order.push_back(regimes->begin()->first);
    if (!regimes && !order.empty()) refuse("multi_device.order without regimes");
    for (const auto& name : order) {
        const auto it = regimes->find(name);
        if (it == regimes->end()) refuse("multi_device.order names an unknown regime '" + name + "'");
        auto regime = parse_regime(name, it->second, "multi_device.regimes." + name);
        validate_regime(policy, regime, "multi_device.regimes." + name);
        policy.regimes.push_back(std::move(regime));
    }
    return policy;
}

Decision select(const Policy& policy, const Query& q) {
    if (q.devices == 0) refuse("--device-count must be at least 1");
    if (q.placement) {
        const auto wire = form_wire(policy, *q.placement);
        if (auto why = blocked(policy, *q.placement, q, wire); !why.empty()) refuse("--placement " + why);
        return decided(policy, *q.placement, q, wire, "explicit", "--placement");
    }
    std::string skipped;
    for (const auto& r : policy.regimes) {
        if (!r.enabled) continue;
        if (r.phase != Phase::request && r.phase != q.phase) continue;
        if (tier_rank(q.tier) < tier_rank(r.min_tier)) continue;
        if (!rows_match(q.prompt_rows, r.min_prompt_rows, r.max_prompt_rows)) continue;
        if (!rows_match(q.ready_requests, r.min_ready_requests, r.max_ready_requests)) continue;
        for (auto form : r.prefer) {
            const auto wire = r.wire.empty() ? form_wire(policy, form) : r.wire;
            if (auto why = blocked(policy, form, q, wire); !why.empty()) {
                skipped += (skipped.empty() ? "" : "; ") + why;
                continue;
            }
            return decided(policy, form, q, wire, "regime:" + r.name, skipped.empty() ? "matched" : skipped);
        }
    }
    const auto default_wire = form_wire(policy, policy.default_form);
    auto why = blocked(policy, policy.default_form, q, default_wire);
    if (why.empty())
        return decided(policy, policy.default_form, q, default_wire, policy.version ? "default" : "absent",
                       policy.version ? "no regime matched" : "model declares no [multi_device]");
    return decided(policy, policy.fallback, q, form_wire(policy, policy.fallback), "fallback", why);
}

std::vector<LegacyOverride> legacy_overrides(const Policy& policy, const Lookup& env) {
    std::vector<LegacyOverride> out;
    for (const auto& row : policy.legacy)
        if (const char* value = environment(env, row.env.c_str()); value && *value)
            out.push_back({row.env, value, "multi_device." + row.key});
    return out;
}

std::string_view executed_by(Form form) noexcept {
    return form == Form::single ? "engine" : form == Form::replica ? "router" : "none";
}

void admit(const Decision& d, Executor executor) {
    const auto what = "multi_device form '" + std::string(form_name(d.form)) + "' (" + d.rule + ": " + d.reason + ")";
    if (executor == Executor::engine) {
        if (d.form == Form::single) return;
        if (d.form == Form::replica)
            throw std::invalid_argument(what + " runs behind strix-router, which holds the endpoints; this launch runs "
                                        "one engine: start each worker with --device-count 1 and strix-router with the "
                                        "worker endpoints");
        throw std::invalid_argument(what + " is not executed by any engine yet; a run / serve launch executes single");
    }
    if (d.form == Form::replica) return;
    throw std::invalid_argument(what + " is not a replica placement; strix-router executes only replica");
}

Decision resolve(const model_description::ModelDescription& description, const Query& q, const Lookup& env) {
    const auto policy = parse_policy(description);
    auto d = select(policy, q);
    d.legacy = legacy_overrides(policy, env);
    return d;
}

serve::Json Decision::json(const Policy& policy, const Query& q) const {
    Json declared = Json::array();
    for (const auto& entry : policy.forms) declared.push(form_name(entry.first));
    Json out = Json::Object{{"version", policy.version}, {"form", form_name(form)},
        {"ranks", static_cast<std::int64_t>(ranks)}, {"devices", static_cast<std::int64_t>(q.devices)},
        {"phase", phase_name(q.phase)}, {"wire", wire},
        {"wire_status", exact_wire(wire) ? "exact" : "report-only: changes bits and no executor resolves a qualified lever yet"}, {"rule", rule}, {"reason", reason}, {"declared", declared},
        {"executed_by", executed_by(form)}};
    if (const auto it = policy.forms.find(form); it != policy.forms.end() && !it->second.fields.empty()) {
        // The form block is the form's declared defaults; a regime's wire replaces the form's, so the effective wire
        // is reported once (top-level `wire`) and `settings.wire` agrees with it; the declared one stays visible.
        Json fields = Json::object();
        for (const auto& [key, value] : it->second.fields) fields.set(key, value);
        if (it->second.fields.contains("wire") && it->second.wire != wire) {
            fields.set("wire", wire);
            fields.set("declared_wire", it->second.wire);
        }
        out.set("settings", std::move(fields));
    }
    Json legacy_json = Json::array();
    for (const auto& o : legacy)
        legacy_json.push(Json::Object{{"env", o.env}, {"value", o.value}, {"read_by", "engine"}});
    out.set("legacy_env", std::move(legacy_json));
    return out;
}

} // namespace strix::profile::multi_device
