#pragma once
#include <array>
#include <cstdlib>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "strix/serve/json.hpp"

namespace strix::qwen::profile {
enum class Name : unsigned { standard, fast, quality };
enum class Kind { exact, bit_changing, pack };
struct Lever {
    std::string_view name, env, option;
    std::array<std::string_view, 3> value;
    std::string_view builtin;
    Kind kind;
    std::array<bool, 3> ready, runnable;
    std::string_view target, off, source;
    // The existing CLI parser owns validation and typed precedence for this row.
    bool native_option = false;
    // Selected by this profile ahead of qualification; still requires runnable code.
    std::array<bool, 3> provisional{};
};
#include "strix/qwen_profile_table.hpp"
inline std::span<const Lever> levers() { return table; }
inline Name parse(std::string_view value) {
    if (value == "default") return Name::standard;
    if (value == "fast") return Name::fast;
    if (value == "quality") return Name::quality;
    throw std::invalid_argument("profile must be default, fast or quality");
}
inline const char* name(Name value) {
    switch (value) {
    case Name::standard: return "default";
    case Name::fast: return "fast";
    case Name::quality: return "quality";
    }
    return "unknown";
}
enum class Status { set, explicit_env, explicit_cli, builtin, pending, unqualified, off, provisional };
inline const char* status_name(Status value) {
    switch (value) {
    case Status::set: return "profile";
    case Status::explicit_env: return "explicit-env";
    case Status::explicit_cli: return "explicit-cli";
    case Status::builtin: return "builtin";
    case Status::pending: return "pending";
    case Status::unqualified: return "unqualified";
    case Status::off: return "off";
    case Status::provisional: return "provisional";
    }
    return "unknown";
}
inline const char* kind_name(Kind value) {
    switch (value) {
    case Kind::exact: return "exact";
    case Kind::bit_changing: return "bit-changing";
    case Kind::pack: return "pack";
    }
    return "unknown";
}
struct Entry {
    const Lever* lever;
    std::string value;
    Status status;
    std::string target;
    bool cli_override = false;
    std::string reason;
};
struct Resolution {
    Name profile = Name::standard;
    std::vector<Entry> entries;
    const Entry* find(std::string_view key) const {
        for (const auto& entry : entries) if (entry.lever->name == key) return &entry;
        return nullptr;
    }
    std::string qualification() const {
        std::string provisional, unqualified;
        for (const auto& entry : entries) {
            auto* rows = entry.status == Status::provisional ? &provisional :
                entry.status == Status::unqualified ? &unqualified : nullptr;
            if (rows) *rows += (rows->empty() ? "" : ",") + std::string(entry.lever->name);
        }
        std::string out;
        if (!provisional.empty()) out = "qualification in progress: " + provisional;
        if (!unqualified.empty()) out += (out.empty() ? "" : "; ") + std::string("unqualified: ") + unqualified;
        return out.empty() ? "qualified" : out;
    }
    std::vector<std::string> notices() const {
        std::vector<std::string> out;
        if (profile == Name::quality) out.emplace_back("quality: no quality rows for this model, using default");
        for (const auto& entry : entries) if (entry.status == Status::provisional)
            out.push_back(std::string("profile ") + name(profile) + ": " + std::string(entry.lever->name) +
                " is bit-changing; qualification in progress");
        for (const auto& entry : entries) if (entry.status == Status::pending)
            out.push_back(std::string("profile ") + name(profile) + ": " + std::string(entry.lever->name) +
                " pending " + entry.target + " (" + entry.value + " runs)" +
                (entry.reason.empty() ? "" : ": " + entry.reason));
        return out;
    }
    std::string json() const {
        auto out = serve::Json::object(), rows = serve::Json::object();
        auto pending = serve::Json::array(), unqualified = serve::Json::array();
        out.set("profile", serve::Json(name(profile)));
        out.set("qualification", serve::Json(qualification()));
        if (profile == Name::quality) out.set("note", serve::Json("quality: no quality rows for this model, using default"));
        for (const auto& entry : entries) {
            const auto& lever = *entry.lever;
            auto row = serve::Json::object();
            row.set("value", serve::Json(entry.value));
            row.set("status", serve::Json(status_name(entry.status)));
            row.set("switch", serve::Json(std::string(lever.env.empty() ? lever.option : lever.env)));
            row.set("kind", serve::Json(kind_name(lever.kind)));
            row.set("source", serve::Json(std::string(lever.source)));
            if (!entry.reason.empty()) row.set("reason", serve::Json(entry.reason));
            rows.set(std::string(lever.name), std::move(row));
            if (entry.status == Status::pending) {
                auto item = serve::Json::object();
                item.set("lever", serve::Json(std::string(lever.name)));
                item.set("target", serve::Json(entry.target));
                item.set("runs", serve::Json(entry.value));
                if (!entry.reason.empty()) item.set("reason", serve::Json(entry.reason));
                pending.push(std::move(item));
            }
            if (entry.status == Status::unqualified) unqualified.push(serve::Json(std::string(lever.name)));
        }
        out.set("levers", std::move(rows));
        out.set("pending", std::move(pending));
        out.set("unqualified", std::move(unqualified));
        return out.dump();
    }
};
using Lookup = std::function<const char*(const char*)>;
using Pairs = std::vector<std::pair<std::string, std::string>>;
struct Inputs {
    Pairs cli;
    Lookup env;
    bool allow_pending = false;
};
inline const char* environment(const Lookup& env, const std::string& key) {
    return env ? env(key.c_str()) : std::getenv(key.c_str());
}
// Export provenance survives a future re-exec. A changed inherited value
// is explicit again; matching profile exports retain their original status.
inline std::string marker_key(const Lever& lever) { return "STRIX_QWEN_PROFILE_OWNED_" + std::string(lever.name); }
inline std::string marker(Name profile, Status status, std::string_view value) {
    return std::string(name(profile)) + ":" + status_name(status) + ":" + std::string(value);
}
inline Resolution resolve(Name profile, const Inputs& inputs = {}, std::span<const Lever> rows = levers()) {
    Resolution out{profile, {}};
    const char* allow = environment(inputs.env, "STRIX_QWEN_PROFILE_ALLOW_PENDING");
    const bool allow_pending = inputs.allow_pending || (allow && std::string_view(allow) == "1");
    const auto column = static_cast<unsigned>(profile);
    for (const auto& lever : rows) {
        const std::string* cli = nullptr;
        for (const auto& [key, value] : inputs.cli) if (key == lever.name) cli = &value;
        const char* env = lever.env.empty() ? nullptr : environment(inputs.env, std::string(lever.env));
        const auto wanted = lever.value[column];
        bool inherited = false;
        if (env && !cli && !wanted.empty() && std::string_view(env) == wanted) {
            const char* owned = environment(inputs.env, marker_key(lever));
            const auto status = lever.provisional[column] ? Status::provisional :
                lever.ready[column] ? Status::set : Status::unqualified;
            inherited = owned && std::string_view(owned) == marker(profile, status, env) &&
                (lever.ready[column] || ((lever.provisional[column] || allow_pending) && lever.runnable[column]));
        }
        Entry entry{&lever, std::string(lever.builtin), Status::builtin, {}, false, {}};
        if (cli || (env && !inherited)) {
            entry.cli_override = cli != nullptr;
            entry.value = cli ? *cli : env;
            entry.status = cli ? Status::explicit_cli : Status::explicit_env;
            if ((!lever.off.empty() && entry.value == lever.off) || entry.value == "off") entry.status = Status::off;
            bool runnable = false;
            for (bool available : lever.runnable) runnable |= available;
            if (!runnable && entry.status != Status::off && entry.value != lever.builtin)
                throw std::invalid_argument(std::string(lever.env.empty() ? lever.option : lever.env) + "=" + entry.value +
                    " is not in this build (" + std::string(lever.source) + ")");
        } else if (wanted.empty()) continue;
        else if (!lever.ready[column] && !((lever.provisional[column] || allow_pending) && lever.runnable[column])) {
            entry.status = Status::pending;
            entry.target = std::string(lever.target.empty() ? wanted : lever.target);
        } else {
            entry.value = std::string(wanted);
            entry.status = lever.provisional[column] ? Status::provisional : !lever.ready[column] ? Status::unqualified :
                wanted == lever.builtin ? Status::builtin : Status::set;
        }
        out.entries.push_back(std::move(entry));
    }
    return out;
}
using Export = std::function<bool(const char*, const char*)>;
inline bool export_environment(const char* key, const char* value) {
#ifdef _WIN32
    return _putenv_s(key, value) == 0;
#else
    return ::setenv(key, value, 1) == 0;
#endif
}
// Apply immediately after parsing, before any switch readers. Emit notices
// once at startup; if a re-exec is added, emit them in its final process image.
inline void apply(const Resolution& resolution, const Export& put = export_environment) {
    for (const auto& entry : resolution.entries) {
        const auto& lever = *entry.lever;
        // Native CLI overrides already live in Options; exporting them would
        // change the historical environment receipt even with profile=default.
        if (lever.env.empty() || (entry.status != Status::set && entry.status != Status::unqualified && entry.status != Status::provisional &&
            !(entry.cli_override && !lever.native_option))) continue;
        if (!put(std::string(lever.env).c_str(), entry.value.c_str()) ||
            !put(marker_key(lever).c_str(), marker(resolution.profile, entry.status, entry.value).c_str()))
            throw std::runtime_error("cannot export profile switch " + std::string(lever.env));
    }
}
// Profile policy has no numerical identity bytes of its own. Only applied
// arithmetic switches belong in session/prefix identities.
inline bool metadata_environment(std::string_view key) {
    return key == "STRIX_QWEN_PROFILE_ALLOW_PENDING" || key.starts_with("STRIX_QWEN_PROFILE_OWNED_");
}
inline bool policy_option(std::string_view key) {
    return key == "--profile" || key == "--allow-pending";
}
inline bool take_option(int argc, char** argv, int& i, std::optional<Name>& profile, Inputs& inputs) {
    const std::string_view arg = argv[i];
    const auto equal = arg.find('=');
    const auto key = arg.substr(0, equal);
    if (arg == "--allow-pending") { inputs.allow_pending = true; return true; }
    const Lever* lever = nullptr;
    for (const auto& row : levers()) if (!row.native_option && key == row.option) lever = &row;
    if (key != "--profile" && !lever) return false;
    const auto value = equal != std::string_view::npos ? arg.substr(equal + 1U) :
        (i + 1 < argc ? std::string_view(argv[++i]) : std::string_view{});
    if (value.empty() || value.starts_with("--")) throw std::invalid_argument(std::string(key) + " requires a value");
    if (key == "--profile") {
        if (profile) throw std::invalid_argument("duplicate --profile");
        profile = parse(value);
    } else inputs.cli.emplace_back(lever->name, value);
    return true;
}
} // namespace strix::qwen::profile
