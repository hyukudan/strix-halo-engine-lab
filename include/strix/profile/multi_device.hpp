#pragma once

#include "strix/engine_profile.hpp"
#include "strix/model_description.hpp"
#include "strix/serve/json.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

// Multi-device placement as model DATA: the [multi_device] section of a model TOML, resolved once by the shared
// profile resolver for the CLI, the server and the replica router (one function, one effective config).
//
//   [multi_device]
//   version = 1
//   default = "replica"          # form used when no regime matches; "single" when absent
//   fallback = "single"          # form used when the default cannot run (too few devices, tier); "single" when absent
//   order = ["short", "long"]    # regime evaluation order; required when there is more than one regime
//
//   [multi_device.forms.replica]    ranks (minimum devices), objective
//   [multi_device.forms.layers]     ranks, axis, cut, components, window_rows, state, wire
//   [multi_device.forms.tensor]     ranks, stages, components, expert_partition, expert_shares, wire,
//                                   rows_per_message, draft_placement, draft_rank
//   [multi_device.forms.speculation] ranks, draft_placement, draft_rank, wire, rows_per_message
//
//   [multi_device.regimes.NAME]     phase, min/max_prompt_rows, min/max_ready_requests, min_tier, prefer,
//                                   wire, lever, enabled
//
// A model without the section resolves `single`: nothing a launcher passes to an engine changes. Forms that are not
// declared are unavailable. The wire `reference` keeps every site's native exact contract (BF16 joins stay BF16, FP32
// partials stay FP32); narrower wires (fp16, fp8-*) are FAST / TURBO data and never resolve under EXACT. A regime
// whose `enabled = false` is kept as data (awaiting its receipt) and never selected.
namespace strix::profile::multi_device {

enum class Form { single, replica, layers, tensor, speculation };
enum class Phase { request, prefill, decode };

[[nodiscard]] std::string_view form_name(Form) noexcept;
[[nodiscard]] std::optional<Form> parse_form(std::string_view) noexcept;
[[nodiscard]] std::string_view phase_name(Phase) noexcept;
// True for a wire that keeps the reference bits (reference, bf16-exact).
[[nodiscard]] bool exact_wire(std::string_view wire) noexcept;

struct FormSpec {
    Form form = Form::single;
    std::uint32_t ranks = 1;              // single: 1; replica: minimum devices; layers/tensor/speculation: ranks used
    std::string wire = "reference";
    std::uint32_t draft_rank = 0;
    std::string draft_placement = "none";
    std::map<std::string, serve::Json, std::less<>> fields;   // every validated key, for the effective config
};

struct Regime {
    std::string name;
    Phase phase = Phase::request;
    std::optional<std::uint64_t> min_prompt_rows, max_prompt_rows, min_ready_requests, max_ready_requests;
    Name min_tier = Name::standard;
    std::vector<Form> prefer;
    std::string wire;                     // "" = the form's wire
    std::string lever;                    // nonempty: requires a matching measured band for this regime
    bool enabled = true;
    bool measured = false;
};

struct LegacySwitch {
    std::string env, key;
};

struct Policy {
    std::int64_t version = 0;             // 0 = the model declares no [multi_device] section
    Form default_form = Form::single;
    Form fallback = Form::single;
    std::map<Form, FormSpec> forms;       // always holds single
    std::vector<Regime> regimes;          // evaluation order
    std::vector<LegacySwitch> legacy;     // [multi_device.legacy_env]: this model's engine-read switches
};

// Throws std::invalid_argument naming the offending key.
[[nodiscard]] Policy parse_policy(const model_description::ModelDescription&);

// The deployment's measured comparison is scoped to a single pair, link and layout.
struct BandReceipt {
    std::string id, model, lever, pair, link, layout, device;
    std::uint64_t min_context = 0, max_context = 0;
    std::uint64_t min_prompt_rows = 0, max_prompt_rows = 0;
    std::uint64_t min_decode_tokens = 0, max_decode_tokens = 0;
    double speedup = 0, single_speedup = 0, noise = 0;
};
[[nodiscard]] BandReceipt load_band_receipt(const std::string& path);

struct Execution {
    std::string model, pair, link, layout, device;
    std::uint64_t context = 0, decode_tokens = 0;
    const BandReceipt* receipt = nullptr;
};

struct Query {
    Name tier = Name::standard;
    Phase phase = Phase::request;
    std::optional<std::uint64_t> prompt_rows;   // unknown: a regime with a row bound does not match
    std::uint64_t ready_requests = 1;
    std::uint32_t devices = 1;                  // whole-model capable devices available to this launch
    Execution execution;
    bool rank_executor = false;   // native binding executes both layer and tensor pair forms
    std::optional<Form> placement;              // explicit --placement: wins or refuses, never silently replaced
};

struct LegacyOverride {
    std::string env, value, key;
};

struct Decision {
    Form form = Form::single;
    std::uint32_t ranks = 1;
    std::string wire = "reference";
    std::string rule = "absent";                // absent | explicit | regime:NAME | default | fallback
    std::string reason;
    std::string ladder_reference;
    std::vector<LegacyOverride> legacy;
    [[nodiscard]] serve::Json json(const Policy&, const Query&) const;
};

// Explicit placement > first matching enabled regime (in `order`) whose preferred form can run > default > fallback
// > single. An explicit placement that cannot run throws std::invalid_argument with the reason.
[[nodiscard]] Decision select(const Policy&, const Query&);

// The model's legacy inventory: its own engine-read switches, each with the [multi_device] key planned to replace it
// (declared by the model under [multi_device.legacy_env]; no model's names live in this code). They still act inside the
// engines that read them (no kit breaks) and no resolver reads the planned keys yet, so the resolver only lists the set
// ones of the selected model in the effective JSON (env, value, read by the engine); it names no key until it is read.
[[nodiscard]] std::vector<LegacyOverride> legacy_overrides(const Policy&, const Lookup& env = {});

// Inspection and execution are distinct. Inspection (--print-effective-config, --dry-run) reports any selected form
// (Decision::json names who executes it). A launch is admitted only for the form its entry point executes:
//   engine  one engine process (strix run / serve, strix-serve, a native backend launcher) executes `single`;
//   router  strix-router, which holds the whole-model endpoints, executes `replica`.
// Anything else throws std::invalid_argument with the rule and reason (a replica launch outside the router is told
// to start one worker per endpoint and the router). No executed form carries a bit-changing wire.
enum class Executor { engine, router };
[[nodiscard]] std::string_view executed_by(Form) noexcept;   // "engine" | "router" | "none"
void admit(const Decision&, Executor);

// One call for every launcher: parse + select + legacy report.
[[nodiscard]] Decision resolve(const model_description::ModelDescription&, const Query&, const Lookup& env = {});

} // namespace strix::profile::multi_device
