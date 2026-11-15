#pragma once
// ONE engine-option parser for every CUDA Qwen entry point (serve, prefill service, gate tools). Host-only.
// Each app keeps only its own non-engine options and calls take() for the rest; resolve() then turns the request into a
// Config for its entry, and names every value that differs between entries (serve_differences) with its reason.
#include "strix/cuda/qwen_prefill.hpp"
#include "strix/profiles/qwen_cuda.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix::cuda::qwen_options {
namespace qp = ::strix::cuda::qwen_prefill;
namespace cp = ::strix::qwen::cuda_profile;
namespace pr = ::strix::qwen::profile;

enum class Entry { serve, prefill_service, cli };
inline const char* entry_name(Entry e) {
    return e == Entry::serve ? "serve" : e == Entry::prefill_service ? "prefill-service" : "cli";
}

struct Request {
    qp::Config engine;                     // common engine options as typed; entry roles are applied by resolve()
    std::optional<pr::Name> profile;       // --profile (default | exact | fast | quality)
    pr::Inputs inputs;                     // lever options (cli) and the environment lookup (empty: process env)
    bool print_effective_config = false;   // --print-effective-config
    std::optional<bool> release_by_layer;  // typed explicitly
};

struct Difference {
    std::string name, value, reason;
};
struct Resolved {
    Entry entry = Entry::cli;
    qp::Config config;
    pr::Resolution resolution;
    cp::Choice choice;
    std::string contract;                  // arithmetic_contract
    std::vector<Difference> differences;   // named entry-role differences, each with its reason
};

inline std::uint64_t number(std::string_view option, const std::string& text, std::uint64_t max = UINT32_MAX) {
    std::size_t end = 0;
    if (text.empty() || text[0] == '-' || text[0] == '+') throw std::invalid_argument(std::string(option) + " requires a nonnegative integer");
    unsigned long long n = 0;
    try { n = std::stoull(text, &end); } catch (const std::exception&) { end = 0; }
    if (end != text.size() || n > max) throw std::invalid_argument(std::string(option) + " value out of range");
    return n;
}

// Consumes one common engine / profile option at argv[i] (advancing i over its value). false: not a common option.
inline bool take(int argc, char** argv, int& i, Request& r) {
    const std::string a = argv[i];
    const auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
        return argv[++i];
    };
    auto& c = r.engine;
    if (a == "--print-effective-config") r.print_effective_config = true;
    else if (a == "--model") c.model_gguf = next();
    else if (a == "--mtp-sidecar") c.mtp_sidecar_gguf = next();
    else if (a == "--draft-vocab") c.draft_vocab = next();
    else if (a == "--tables") c.tables_dir = next();
    else if (a == "--rope-table") c.rope_table = next();
    else if (a == "--max-positions") c.max_positions = static_cast<std::uint32_t>(number(a, next()));
    else if (a == "--block-schedule") {
        const auto v = next();
        if (v != "default" && v != "tail-large") throw std::invalid_argument("invalid block schedule");
        c.tail_large = v == "tail-large";
    } else if (a == "--final-block-rows") {
        const auto v = next();
        const auto n = number(a, v, qp::kMaxBlockRows);
        c.final_block_rows = static_cast<std::uint32_t>(n);
    } else if (a == "--blocks") {
        c.block_schedule.clear();
        const auto v = next();
        for (std::size_t at = 0; at < v.size();) {
            const auto comma = v.find(',', at);
            c.block_schedule.push_back(static_cast<std::uint32_t>(number(a, v.substr(at, comma - at))));
            at = comma == std::string::npos ? v.size() : comma + 1;
        }
    } else if (a == "--block-rows") {
        const auto rows = static_cast<std::uint32_t>(number(a, next()));
        if (rows < 1024 || rows % 16) throw std::invalid_argument("--block-rows must be a multiple of 16 and >= 1024");
        c.block_schedule = {rows};
    } else if (a == "--release-by-layer") r.release_by_layer = true;
    else if (a == "--no-release-by-layer") r.release_by_layer = false;
    else return cp::take_option(argc, argv, i, r.profile, r.inputs);
    return true;
}

// Maps a lever Resolution onto the engine Config: the only place that does.
inline void apply(const cp::Choice& choice, qp::Config& config) {
    config.forms = {choice.hc_mma, choice.mtp_mma, choice.qsa_mma};
    config.moe_route_bf16 = choice.moe_route_bf16;  // explicit: the resolver already read the environment
}

inline Resolved resolve(const Request& r, Entry entry, std::string_view default_model = {}) {
    Resolved out;
    out.entry = entry;
    out.config = r.engine;
    auto& c = out.config;
    out.resolution = cp::resolve(r.profile.value_or(pr::Name::standard), r.inputs);
    out.choice = cp::choose(out.resolution);
    apply(out.choice, c);
    out.contract = cp::arithmetic_contract(out.choice);
    const auto name = [&](std::string n, std::string v, std::string why) { out.differences.push_back({std::move(n), std::move(v), std::move(why)}); };
    if (entry == Entry::serve) {
        if (r.release_by_layer.value_or(false))
            throw std::invalid_argument("--release-by-layer: serve keeps the state on the device (no pieces to release)");
        c.decode = true;
        c.export_state = false;
        c.release_by_layer = false;
        name("decode", "1", "serve runs decode/verify steps (LM head + final mix uploaded, step() enabled)");
        name("export_state", "0", "a decode session never hands its state off; no D2H staging");
        name("release_by_layer", "0", "nothing is exported, so there are no pieces to release per layer");
    } else if (r.release_by_layer) c.release_by_layer = *r.release_by_layer;
    if (entry != Entry::serve && c.model_gguf.empty() && !default_model.empty()) {
        c.model_gguf = std::string(default_model);
        name("model_gguf", c.model_gguf, "prefill service default file name when --model is absent");
    }
    if (c.model_gguf.empty() || c.tables_dir.empty() || c.rope_table.empty())
        throw std::invalid_argument("--model, --tables and --rope-table are required");
    if (c.block_schedule.empty() ||
        c.final_block_rows > *std::max_element(c.block_schedule.begin(), c.block_schedule.end()))
        throw std::invalid_argument("invalid final block rows");
    return out;
}

inline std::string effective_config_json(const Resolved& r) {
    using J = ::strix::serve::Json;
    const auto& c = r.config;
    auto out = J::object();
    out.set("entry", J(entry_name(r.entry)));
    out.set("profile", J(pr::name(r.resolution.profile)));
    out.set("arithmetic_contract", J(r.contract));
    auto forms = J::object();
    forms.set("hc_mma", J(c.forms.hc_mma));
    forms.set("mtp_mma", J(c.forms.mtp_mma));
    forms.set("qsa_mma", J(c.forms.qsa_mma));
    forms.set("moe_route_bf16", J(c.moe_route_bf16.value_or(false)));
    out.set("forms", std::move(forms));
    out.set("form_identities", J(cp::form_identities(r.choice)));
    out.set("resolution", ::strix::serve::parse_json(r.resolution.json()));
    auto notices = J::array();
    for (const auto& n : r.resolution.notices()) notices.push(J(n));
    out.set("notices", std::move(notices));
    auto engine = J::object();
    engine.set("max_positions", J(static_cast<std::uint64_t>(c.max_positions)));
    engine.set("tail_large", J(c.tail_large));
    engine.set("final_block_rows", J(static_cast<std::uint64_t>(c.final_block_rows)));
    auto blocks = J::array();
    for (const auto b : c.block_schedule) blocks.push(J(static_cast<std::uint64_t>(b)));
    engine.set("block_schedule", std::move(blocks));
    engine.set("decode", J(c.decode));
    engine.set("export_state", J(c.export_state));
    engine.set("release_by_layer", J(c.release_by_layer));
    engine.set("mtp_sidecar", J(!c.mtp_sidecar_gguf.empty()));
    engine.set("draft_vocab", J(!c.draft_vocab.empty()));
    out.set("engine", std::move(engine));
    auto diffs = J::array();
    for (const auto& d : r.differences) {
        auto row = J::object();
        row.set("name", J(d.name));
        row.set("value", J(d.value));
        row.set("reason", J(d.reason));
        diffs.push(std::move(row));
    }
    out.set("serve_differences", std::move(diffs));
    return out.dump();
}

inline constexpr std::uint64_t fnv64(std::string_view s) {
    std::uint64_t h = 14695981039346656037ULL;
    for (const char ch : s) { h ^= static_cast<unsigned char>(ch); h *= 1099511628211ULL; }
    return h;
}
inline cp::Choice choice_of(const qp::Config& c) {
    return {c.moe_route_bf16.value_or(false), c.forms.hc_mma, c.forms.mtp_mma, c.forms.qsa_mma};
}
// Hand-off producer id: the exact CUDA path keeps its historical id bytes; a FAST arithmetic folds its contract in, so an
// exact-identity node refuses a FAST state (different producer id).
inline std::uint64_t handoff_numerical_id(std::string_view exact_name, const cp::Choice& choice) {
    if (!choice.fast()) return fnv64(exact_name);
    return fnv64(std::string(exact_name) + "|" + cp::arithmetic_contract(choice) + "|" + cp::form_identities(choice));
}
}  // namespace strix::cuda::qwen_options
