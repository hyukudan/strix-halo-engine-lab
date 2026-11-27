#include "strix/profile/all_on.hpp"

#include "strix/env_flags.hpp"
#include "strix/profile/model_preset.hpp"
#include "strix/serve/json.hpp"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <ostream>
#include <set>
#include <stdexcept>

#include "strix/profile/all_on_data.hpp"

namespace strix::profile::all_on {
namespace {
using serve::Json;

#ifdef _WIN32
void put_env(const std::string& key, const char* value) { (void)_putenv_s(key.c_str(), value); }
#else
void put_env(const std::string& key, const char* value) {
    if (*value) (void)::setenv(key.c_str(), value, 1);
    else (void)::unsetenv(key.c_str());
}
#endif

std::optional<std::string> env_value(std::string_view key) {
    const char* value = strix::envcfg::lookup_unrecorded(std::string(key).c_str());
    if (value == nullptr || *value == '\0') return std::nullopt;
    return std::string(value);
}

bool env_set(std::string_view key) { return env_value(key).has_value(); }

const std::string& text(const Json& row, std::string_view key) {
    const auto* item = row.find(key);
    if (item == nullptr || !item->is_string()) throw std::invalid_argument("all_on catalogue: missing string " + std::string(key));
    return item->as_string();
}

std::string text_or_empty(const Json& row, std::string_view key) {
    const auto* item = row.find(key);
    return item != nullptr && item->is_string() ? item->as_string() : std::string{};
}

std::vector<std::string> strings(const Json& row, std::string_view key) {
    std::vector<std::string> out;
    if (const auto* item = row.find(key)) {
        if (!item->is_array()) throw std::invalid_argument("all_on catalogue: " + std::string(key) + " must be an array");
        for (const auto& entry : item->as_array()) out.push_back(entry.as_string());
    }
    return out;
}

Class parse_class(const std::string& word) {
    if (word == "exact") return Class::exact;
    if (word == "fast") return Class::fast;
    if (word == "turbo") return Class::turbo;
    if (word == "kv") return Class::kv;
    if (word == "unclassified") return Class::unclassified;
    throw std::invalid_argument("all_on catalogue: unknown class " + word);
}

Gate parse_gate(const std::string& word) {
    if (word == "none") return Gate::none;
    if (word == "pass") return Gate::pass;
    if (word == "pending") return Gate::pending;
    if (word == "fail") return Gate::fail;
    throw std::invalid_argument("all_on catalogue: unknown gate " + word);
}

Phase parse_phase(std::string_view word) {
    if (word == "any") return Phase::any;
    if (word == "prefill") return Phase::prefill;
    if (word == "decode") return Phase::decode;
    throw std::invalid_argument("all_on: unknown phase " + std::string(word));
}

std::string_view phase_word(Phase phase) {
    return phase == Phase::prefill ? "prefill" : phase == Phase::decode ? "decode" : "any";
}

std::string_view tier_word(Tier tier) { return tier == Tier::exact ? "exact" : tier == Tier::fast ? "fast" : "turbo"; }

bool contains(const std::vector<std::string>& list, std::string_view item) {
    return std::ranges::find(list, item) != list.end();
}

bool engine_matches(const Lever& lever, const std::vector<std::string>& engines) {
    if (contains(lever.engines, "*")) return true;
    return std::ranges::any_of(engines, [&](const std::string& engine) { return contains(lever.engines, engine); });
}

// Why a lever does not apply in the band: empty when it does; "-" when the lever belongs to the other phase.
std::string band_defect(const Lever& lever, const Band& band) {
    if (lever.phase != Phase::any && band.phase != Phase::any && lever.phase != band.phase) return "-";
    if (!lever.tokens) return {};
    const std::string range = std::to_string(lever.tokens->first) + ".." + std::to_string(lever.tokens->second);
    if (!band.tokens) return "band " + range + " tokens needs --all-on-band " + std::string(phase_word(lever.phase)) + ":N";
    if (*band.tokens < lever.tokens->first || *band.tokens > lever.tokens->second)
        return "outside the qualified band " + range + " tokens";
    return {};
}

std::string_view gate_word(Gate gate) {
    return gate == Gate::pass ? "pass" : gate == Gate::pending ? "pending" : gate == Gate::fail ? "FAIL" : "none";
}
}  // namespace

std::vector<Lever> parse_catalogue(std::string_view json) {
    const Json root = serve::parse_json(json);
    if (!root.is_array()) throw std::invalid_argument("all_on catalogue must be an array");
    std::vector<Lever> out;
    std::set<std::string> seen;
    for (const auto& row : root.as_array()) {
        Lever lever;
        lever.row = text(row, "row");
        lever.name = text(row, "lever");
        if (!seen.insert(lever.name).second) throw std::invalid_argument("all_on catalogue: duplicate lever " + lever.name);
        lever.cls = parse_class(text(row, "class"));
        lever.gate = parse_gate(text(row, "gate"));
        lever.phase = parse_phase(text(row, "phase"));
        lever.engines = strings(row, "engines");
        lever.backends = strings(row, "backends");
        if (const auto* tokens = row.find("tokens")) {
            if (!tokens->is_array() || tokens->size() != 2U) throw std::invalid_argument("all_on catalogue: tokens needs [min, max]");
            lever.tokens = std::pair{static_cast<std::uint64_t>(tokens->as_array()[0].as_int()),
                                     static_cast<std::uint64_t>(tokens->as_array()[1].as_int())};
        }
        lever.env = text_or_empty(row, "env");
        lever.value = text_or_empty(row, "value");
        lever.args = strings(row, "args");
        lever.incompatible_with = strings(row, "incompatible_with");
        lever.requires_levers = strings(row, "requires");
        lever.default_tiers = strings(row, "default_tiers");
        lever.quality = text_or_empty(row, "quality");
        lever.reason = text_or_empty(row, "reason");
        if (const auto identity = text_or_empty(row, "identity"); identity == "pending") lever.identity_pending = true;
        else if (!identity.empty()) throw std::invalid_argument("all_on catalogue: " + lever.name + ": identity must be pending");
        const bool applicable = lever.cls != Class::unclassified && lever.cls != Class::kv;
        if (applicable && (lever.env.empty() == lever.args.empty() || lever.backends.empty()))
            throw std::invalid_argument("all_on catalogue: " + lever.name + " needs backends and one of env and args");
        out.push_back(std::move(lever));
    }
    for (const auto& lever : out) {
        for (const auto* list : {&lever.incompatible_with, &lever.requires_levers})
            for (const auto& other : *list)
                if (!seen.contains(other)) throw std::invalid_argument("all_on catalogue: " + lever.name + " names unknown lever " + other);
        // Incompatibility is a symmetric relation: each side names the other.
        for (const auto& other : lever.incompatible_with) {
            const auto peer = std::ranges::find(out, other, &Lever::name);
            if (!contains(peer->incompatible_with, lever.name))
                throw std::invalid_argument("all_on catalogue: " + lever.name + " is incompatible with " + other + " but " +
                                            other + " does not name it");
        }
    }
    return out;
}

const std::vector<Lever>& catalogue() {
    static const std::vector<Lever> levers = parse_catalogue(detail::catalogue_json);
    return levers;
}

Band parse_band(std::string_view word) {
    Band band;
    if (word.empty() || word == "any") return band;
    const auto colon = word.find(':');
    band.phase = parse_phase(word.substr(0, colon));
    if (colon != std::string_view::npos) {
        const auto number = word.substr(colon + 1U);
        std::uint64_t tokens = 0;
        const auto parsed = std::from_chars(number.data(), number.data() + number.size(), tokens);
        if (number.empty() || parsed.ec != std::errc{} || parsed.ptr != number.data() + number.size() || band.phase == Phase::any)
            throw std::invalid_argument("--all-on-band requires any, prefill[:TOKENS] or decode[:TOKENS] (got '" + std::string(word) + "')");
        band.tokens = tokens;
    }
    return band;
}

std::string Line::text() const {
    std::string out = "lever " + lever + " source=" + source + marks;
    if (!reason.empty()) out += " reason=" + reason;
    return out;
}

std::string Plan::text() const {
    std::string out;
    for (const auto& line : lines) out += line.text() + '\n';
    return out;
}

Plan resolve(std::span<const Lever> levers, const Query& query) {
    enum class State : std::uint8_t { skip, refused, def, explicit_on, explicit_off, all_on };
    struct Slot {
        const Lever* lever = nullptr;
        State state = State::skip;
        std::string reason, marks;
    };
    std::vector<Slot> slots;
    const auto rank = [](Class cls) { return cls == Class::turbo ? 2 : cls == Class::fast ? 1 : 0; };
    const int tier_rank = query.tier == Tier::turbo ? 2 : query.tier == Tier::fast ? 1 : 0;
    for (const auto& lever : levers) {
        if (!engine_matches(lever, query.engines)) continue;
        Slot slot;
        slot.lever = &lever;
        if (lever.cls == Class::kv) {
            // KV precision is the preset's toggle: reported, never switched.
            if (contains(lever.default_tiers, tier_word(query.tier))) slot.state = State::def;
            else {
                slot.state = State::refused;
                slot.reason = "KV precision is a separate toggle, off in " + std::string(tier_word(query.tier));
            }
        } else if (lever.cls == Class::unclassified) {
            slot.state = State::refused;
            slot.reason = "unclassified" + (lever.reason.empty() ? std::string{} : ": " + lever.reason);
        } else if (!contains(lever.backends, query.backend)) {
            continue;
        } else if (rank(lever.cls) > tier_rank) {
            continue;   // a lever of a looser tier is not part of this tier
        } else if (const auto defect = band_defect(lever, query.band); defect == "-") {
            continue;
        } else if (!defect.empty()) {
            slot.state = State::refused;
            slot.reason = defect;
        } else if (contains(lever.default_tiers, tier_word(query.tier))) {
            slot.state = State::def;
        } else if (const auto value = query.env_value && !lever.env.empty() ? query.env_value(lever.env) : std::nullopt) {
            const bool off = *value == "0";
            slot.state = off ? State::explicit_off : State::explicit_on;
            slot.reason = off ? "explicit setting kept: off" : "explicit setting kept";
        } else if (lever.cls == Class::fast && lever.gate != Gate::pass && query.tier == Tier::fast) {
            slot.state = State::refused;
            slot.reason = "gate v2 " + std::string(gate_word(lever.gate)) + " for fast";
        } else {
            slot.state = State::all_on;
            if (lever.identity_pending) slot.marks += " identity=pending";
            // A lever that needs more than FAST says its gate and its quality.
            if (query.tier == Tier::turbo && lever.cls == Class::fast && lever.gate != Gate::pass)
                slot.marks += " gate=" + std::string(gate_word(lever.gate));
            if (query.tier == Tier::turbo && (lever.cls == Class::turbo || (lever.cls == Class::fast && lever.gate != Gate::pass)))
                slot.reason = "quality: " + (lever.quality.empty() ? std::string("not stated") : lever.quality);
        }
        slots.push_back(std::move(slot));
    }
    const auto find = [&](const std::string& name) -> Slot* {
        for (auto& slot : slots) if (slot.lever->name == name) return &slot;
        return nullptr;
    };
    const auto settled = [](const Slot* slot) {
        return slot != nullptr && (slot->state == State::def || slot->state == State::explicit_on);
    };
    const auto on = [&](const Slot* slot) { return settled(slot) || (slot != nullptr && slot->state == State::all_on); };
    // Incompatible pairs (symmetric): a default or explicit peer always wins wherever its row is; between two all-on
    // levers the earlier catalogue row keeps its place and the later one is refused with the reason.
    for (auto& slot : slots) {
        if (slot.state != State::all_on) continue;
        for (const auto& other : slot.lever->incompatible_with) {
            const Slot* peer = find(other);
            if (settled(peer) || (on(peer) && peer < &slot)) {
                slot.state = State::refused;
                slot.reason = "incompatible with " + other;
                slot.marks.clear();
                break;
            }
        }
    }
    for (bool changed = true; changed;) {
        changed = false;
        for (auto& slot : slots) {
            if (slot.state != State::all_on) continue;
            for (const auto& need : slot.lever->requires_levers) {
                if (!on(find(need))) {
                    slot.state = State::refused;
                    slot.reason = "requires " + need;
                    slot.marks.clear();
                    changed = true;
                    break;
                }
            }
        }
    }
    Plan plan;
    for (const auto& slot : slots) {
        const char* source = slot.state == State::all_on ? "all-on"
                             : slot.state == State::def ? "default"
                             : slot.state == State::explicit_on || slot.state == State::explicit_off ? "explicit"
                                                                                                       : "refused";
        Line line{slot.lever->name, source, slot.reason, slot.marks};
        plan.lines.push_back(std::move(line));
        if (slot.state != State::all_on) continue;
        if (!slot.lever->env.empty()) plan.env.emplace_back(slot.lever->env, slot.lever->value);
        plan.args.insert(plan.args.end(), slot.lever->args.begin(), slot.lever->args.end());
    }
    return plan;
}

Tier tier_of(Name name) {
    return name == Name::turbo ? Tier::turbo : name == Name::fast ? Tier::fast : Tier::exact;
}

std::vector<std::string> engines_of(const model_description::ModelDescription* description) {
    std::vector<std::string> out;
    if (description == nullptr) return out;
    if (const auto* list = description->find("catalogue.engines")) {
        const auto* array = std::get_if<model_description::Value::Array>(&list->data);
        if (array == nullptr) throw std::invalid_argument("catalogue.engines must be an array of strings");
        for (const auto& item : *array) {
            const auto* word = std::get_if<std::string>(&item.data);
            if (word == nullptr) throw std::invalid_argument("catalogue.engines must be an array of strings");
            out.push_back(*word);
        }
    }
    return out;
}

bool env_requested() {
    const char* value = strix::envcfg::text(kEnv);
    if (value == nullptr || *value == '\0' || std::string_view(value) == "0") return false;
    if (std::string_view(value) == "1") return true;
    throw std::invalid_argument(std::string(kEnv) + " must be 0 or 1");
}

Scope::Scope(Scope&& other) noexcept : plan_(std::move(other.plan_)), set_(std::move(other.set_)) { other.set_.clear(); }

Scope& Scope::operator=(Scope&& other) noexcept {
    if (this != &other) {
        release();
        plan_ = std::move(other.plan_);
        set_ = std::move(other.set_);
        other.set_.clear();
    }
    return *this;
}

Scope::~Scope() { release(); }

void Scope::release() noexcept {
    for (const auto& key : set_) put_env(key, "");
    set_.clear();
}

Scope begin(const model_description::ModelDescription* description, std::optional<Name> tier, std::string_view backend,
            std::string_view band, std::ostream& out, bool apply, std::vector<std::string>& args) {
    Scope scope;
    if (env_set(kAppliedEnv)) return scope;
    Name resolved = tier.value_or(Name::standard);
    if (description != nullptr) resolved = resolve_model_preset(*description, tier, backend).requested;
    Query query;
    query.tier = tier_of(resolved);
    query.backend = backend.empty() ? "hip" : std::string(backend);
    query.engines = engines_of(description);
    const char* band_env = band.empty() ? strix::envcfg::text(kBandEnv) : nullptr;
    query.band = parse_band(band.empty() ? std::string_view(band_env ? band_env : "") : band);
    query.env_value = env_value;
    scope.plan_ = resolve(catalogue(), query);
    out << "all-on: tier=" << tier_word(query.tier) << " backend=" << query.backend << " levers=" << scope.plan_.lines.size() << '\n'
        << scope.plan_.text();
    args.insert(args.end(), scope.plan_.args.begin(), scope.plan_.args.end());
    if (!apply) return scope;
    for (const auto& [key, value] : scope.plan_.env) {
        if (env_set(key)) continue;
        put_env(key, value.c_str());
        scope.set_.push_back(key);
    }
    put_env(kAppliedEnv, "1");
    scope.set_.emplace_back(kAppliedEnv);
    return scope;
}

}  // namespace strix::profile::all_on
