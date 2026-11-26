#pragma once
// KV precision toggle shared by every engine: one CLI option `--kv native|fp32|fp16|bf16|fp8|int8` and one serve-config
// key `kv = "..."` (kServeConfigOption expands the key to the same flag, so the binary's parser sees one spelling).
//
// Resolution (resolve()):
//   exact / fast / quality   the model's native KV precision (Caps::native).
//   turbo                    fp8 when the engine has an fp8 KV path, else the native precision with a note saying so.
//   explicit --kv X          wins over the tier (CLI > tier default, as every lever). A non-native value is recorded as
//                            explicit and bit-changing; under exact the receipt says the run is not the exact reference.
//   unsupported X            refused by name, with the precisions the engine does implement (Caps::unavailable gives the
//                            engine's own hint). Never replaced by another precision.
// No other input exists: no slot count, memory pressure or context length selects the precision, so it never switches
// silently. Every receipt prints Resolution::receipt() (text) or Resolution::json() (object); identity() ("kv=fp8") is the
// field cache keys and rank agreement carry.
//
// Precision names what the attention reads; the storage form identities are strix/kv_storage.hpp's (int8 =
// kv-int8-tok-v1, fp8 = kv-e4m3-tok-v1 unless the engine names its own form, e.g. a model-defined FP8 ring).
// Header-only, host-only.
#include "strix/engine_profile.hpp"
#include "strix/kv_storage.hpp"
#include "strix/serve/serve_config.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::session::kv_precision {

enum class Precision : std::uint8_t { fp32 = 0, fp16 = 1, bf16 = 2, fp8 = 3, int8 = 4 };
inline constexpr std::array<std::string_view, 5> kWords{"fp32", "fp16", "bf16", "fp8", "int8"};
inline constexpr std::string_view kOption = "--kv";
inline constexpr serve::config_file::Option kServeConfigOption{"kv", "--kv", serve::config_file::Kind::value};

[[nodiscard]] constexpr std::string_view word(Precision p) noexcept { return kWords[static_cast<std::size_t>(p)]; }
[[nodiscard]] constexpr unsigned bit(Precision p) noexcept { return 1U << static_cast<unsigned>(p); }

// The KV word of a value; `native` is handled by the request (Request::native).
[[nodiscard]] constexpr std::optional<Precision> parse(std::string_view value) noexcept {
    for (std::size_t i = 0; i < kWords.size(); ++i)
        if (kWords[i] == value) return static_cast<Precision>(i);
    return std::nullopt;
}

// Default storage form identity of a precision (receipt only; an engine with its own form names it in Caps).
[[nodiscard]] constexpr std::string_view default_form(Precision p) noexcept {
    switch (p) {
        case Precision::fp32: return "f32";
        case Precision::fp16: return "f16";
        case Precision::bf16: return kv::name(kv::Form::bf16);
        case Precision::fp8: return kv::name(kv::Form::e4m3_token);
        case Precision::int8: return kv::name(kv::Form::int8_token);
    }
    return "?";
}

// What one engine (backend + model family) implements. The native precision is the model's; `supported` lists the
// other precisions this build has KV writers and readers for (the native one is always supported).
struct Caps {
    std::string_view engine;                 // refusal / receipt name, e.g. "qwen-hip"
    Precision native = Precision::bf16;
    unsigned supported = 0;                  // mask of bit(Precision) beyond the native one
    std::string_view native_form{};          // form identity of the native KV (empty: default_form(native))
    std::string_view fp8_form{};             // form identity of the fp8 path (empty: kv-e4m3-tok-v1)
    std::string_view unavailable{};          // engine hint appended to a refusal / turbo note (e.g. "use --kv int8")
    [[nodiscard]] constexpr bool supports(Precision p) const noexcept { return p == native || (supported & bit(p)) != 0U; }
    [[nodiscard]] constexpr std::string_view form(Precision p) const noexcept {
        if (p == native && !native_form.empty()) return native_form;
        if (p == Precision::fp8 && !fp8_form.empty()) return fp8_form;
        return default_form(p);
    }
    [[nodiscard]] std::string supported_text() const {
        std::string out;
        for (std::size_t i = 0; i < kWords.size(); ++i)
            if (supports(static_cast<Precision>(i))) out += (out.empty() ? "" : "|") + std::string(kWords[i]);
        return out;
    }
};

// What the command line (or the serve-config key, expanded to the same flag) asked for.
struct Request {
    std::optional<Precision> value;  // explicit precision
    bool native = false;             // explicit `--kv native`
    std::string spelling;            // the option that set it ("--kv", or an engine's legacy spelling)
    [[nodiscard]] bool given() const noexcept { return value.has_value() || native; }
};

// Records one value. Throws on an unknown word or a second setting (two spellings of one setting included).
inline void set(Request& r, std::string_view value, std::string_view spelling = kOption) {
    if (r.given())
        throw std::invalid_argument(std::string(spelling) + ": the KV precision is already set by " + r.spelling);
    if (value == "native") r.native = true;
    else if (const auto p = parse(value)) r.value = *p;
    else throw std::invalid_argument(std::string(spelling) + " expects native|fp32|fp16|bf16|fp8|int8 (got '" +
                                     std::string(value) + "')");
    r.spelling = std::string(spelling);
}

// `--kv VALUE` or `--kv=VALUE` at argv[i]: consumes it (advancing i over the value) and returns true.
inline bool take(int argc, const char* const* argv, int& i, Request& r) {
    const std::string_view a = argv[i];
    if (a == kOption) {
        if (i + 1 >= argc) throw std::invalid_argument("--kv requires a value (native|fp32|fp16|bf16|fp8|int8)");
        set(r, argv[++i]);
        return true;
    }
    if (a.size() > kOption.size() && a.starts_with(kOption) && a[kOption.size()] == '=') {
        set(r, a.substr(kOption.size() + 1U));
        return true;
    }
    return false;
}

// Removes every --kv option from argv in place (argc shrinks; argv[argc] stays nullptr when it was), for an engine whose
// own parser runs afterwards. Stops at "--" (the arguments after it are not this binary's).
inline Request extract(int& argc, char** argv) {
    Request r;
    int out = 1;
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--") {
            while (i < argc) argv[out++] = argv[i++];
            break;
        }
        if (take(argc, argv, i, r)) continue;
        argv[out++] = argv[i];
    }
    if (out < argc) argv[out] = nullptr;
    argc = out;
    return r;
}

// The product tier named by `--profile WORD` / `--profile=WORD` (the last one), for an engine whose own parser resolves
// the profile later; a bare `--profile` (timing in some engines) or no option gives exact.
inline profile::Name scan_tier(int argc, const char* const* argv) {
    profile::Name out = profile::Name::standard;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a == "--") break;
        if (a == "--profile" && i + 1 < argc && profile::is_name(argv[i + 1])) out = profile::parse(argv[++i]);
        else if (a.starts_with("--profile=") && profile::is_name(a.substr(10))) out = profile::parse(a.substr(10));
    }
    return out;
}

// lever: an engine's per-phase lever (its profile cell or its own option) selected the precision; the engine reports it.
enum class Source : std::uint8_t { native = 0, tier = 1, explicit_option = 2, lever = 3 };
[[nodiscard]] constexpr std::string_view source_name(Source s) noexcept {
    switch (s) {
        case Source::native: return "native";
        case Source::tier: return "turbo-default";
        case Source::explicit_option: return "explicit";
        case Source::lever: return "lever";
    }
    return "?";
}

struct Resolution {
    std::string engine;
    profile::Name tier = profile::Name::standard;
    Precision precision = Precision::bf16;
    Precision native = Precision::bf16;
    Source source = Source::native;
    std::string form;   // storage form identity of `precision`
    std::string note;   // turbo default not available, or an explicit non-native value under exact
    [[nodiscard]] bool changes_output() const noexcept { return precision != native; }
    [[nodiscard]] std::string_view word() const noexcept { return kv_precision::word(precision); }
    // Cache-key / rank-agreement field.
    [[nodiscard]] std::string identity() const { return "kv=" + std::string(word()); }
    // One receipt line: "kv=fp8 native=bf16 source=turbo-default form=kv-e4m3-tok-v1 tier=turbo[ note=...]".
    [[nodiscard]] std::string receipt() const {
        std::string out = identity() + " native=" + std::string(kv_precision::word(native)) + " source=" +
                          std::string(source_name(source)) + " form=" + form + " tier=" + profile::name(tier);
        if (!note.empty()) out += " note=\"" + note + "\"";
        return out;
    }
    [[nodiscard]] std::string json() const {
        std::string out = "{\"precision\":" + profile::json_string(word()) + ",\"native\":" +
                          profile::json_string(kv_precision::word(native)) + ",\"source\":" +
                          profile::json_string(source_name(source)) + ",\"form\":" + profile::json_string(form) +
                          ",\"tier\":" + profile::json_string(profile::name(tier)) +
                          ",\"changes_output\":" + (changes_output() ? "true" : "false");
        if (!note.empty()) out += ",\"note\":" + profile::json_string(note);
        return out + "}";
    }
};

enum class TierDefault { fp8_if_supported, resolved_by_levers };
enum class ExactOverride { report_identity_change, require_native };

// Throws std::invalid_argument naming the engine, the request and what the engine implements.
inline Resolution resolve(profile::Name tier, const Request& request, const Caps& caps,
                          TierDefault policy = TierDefault::fp8_if_supported,
                          ExactOverride override = ExactOverride::report_identity_change) {
    Resolution r;
    r.engine = std::string(caps.engine);
    r.tier = tier;
    r.native = caps.native;
    r.precision = caps.native;
    if (request.value) {
        const Precision want = *request.value;
        if (!caps.supports(want)) {
            std::string message = std::string(caps.engine) + ": " + request.spelling + " " +
                                  std::string(kv_precision::word(want)) + " refused: this engine has no " +
                                  std::string(kv_precision::word(want)) + " KV path (supported: " + caps.supported_text() +
                                  "; native " + std::string(kv_precision::word(caps.native)) + ")";
            if (!caps.unavailable.empty()) message += "; " + std::string(caps.unavailable);
            throw std::invalid_argument(message);
        }
        if (tier == profile::Name::standard && override == ExactOverride::require_native && want != caps.native)
            throw std::invalid_argument(std::string(caps.engine) + ": " + request.spelling + " " +
                                        std::string(kv_precision::word(want)) + " refused: exact requires native KV");
        r.precision = want;
        r.source = Source::explicit_option;
        if (r.changes_output() && tier == profile::Name::standard)
            r.note = "explicit non-native KV under exact: not the exact reference";
    } else if (!request.native && tier == profile::Name::turbo && policy == TierDefault::fp8_if_supported) {
        if (caps.supports(Precision::fp8)) {
            r.precision = Precision::fp8;
            r.source = caps.native == Precision::fp8 ? Source::native : Source::tier;
        } else {
            r.note = "turbo default fp8 unavailable on this engine: native kept";
            if (!caps.unavailable.empty()) r.note += " (" + std::string(caps.unavailable) + ")";
        }
    }
    r.form = std::string(caps.form(r.precision));
    return r;
}

// The receipt line every engine prints once per run: "strix-kv: {json}".
[[nodiscard]] inline std::string receipt_line(const Resolution& r) { return "strix-kv: " + r.json(); }

// For a main() whose own parser runs afterwards (an engine whose KV precision is only its native one, or that applies
// the resolution itself): takes --kv out of the argument vector, resolves it with the tier named by --profile, prints
// receipt_line() on `receipt` and returns the resolution. On a refusal it prints the one-line reason and exits with 2.
inline Resolution bind_main(int& argc, char** argv, const Caps& caps, std::FILE* receipt = stderr) {
    try {
        const auto request = extract(argc, argv);
        auto r = resolve(scan_tier(argc, argv), request, caps);
        if (receipt) {
            std::fprintf(receipt, "%s\n", receipt_line(r).c_str());
            std::fflush(receipt);
        }
        return r;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        std::fflush(stderr);
        std::exit(2);
    }
}

}  // namespace strix::session::kv_precision
