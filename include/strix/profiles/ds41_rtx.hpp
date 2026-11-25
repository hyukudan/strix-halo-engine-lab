#pragma once

#include "strix/engine_profile.hpp"
#include "strix/session/kv_precision.hpp"

#include <array>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// DS RTX product profiles: one switch, --profile default|fast|quality (STRIX_DS41_PROFILE), selects every lever.
//   default  today's replica: every exact lever that is the replica's default (lean state, q8-r1, dsrtxk / fuse kernels,
//            auto eviction, controller window + draft-keep + draft-vocab auto, wb-defer, CE pacing, wb-stream, the file
//            tier, residency auto seed / save, warmup graphs in serve). No --profile = default = today, byte for byte.
//            The layer-major exact prompt path (PRO-qualified) is wanted in every profile but is pending: not in this build.
//   fast     default + every further exact lever and every bit-changing lever that has passed its quality qualification. Today none has:
//            the fast-only rows are pending (listed in the receipt and --print-config, never silently used).
//   quality  = default for DS on the RTX (no higher-precision expert pack exists for the replica).
// The prefill service rows (App::prefill_service; the service takes --profile / --allow-pending through the same resolve()
// as the replica, one word for both apps): host_promote
// (lazy host promotion of file-tier expert layers, exact) on, --baseline / STRIX_DS41_RTX_BASELINE=1 turn it off.
// One table row per lever (ds41_rtx_profile.cpp): a value per profile, ready (qualified and present) and runnable (the
// implementation is in this build) per profile, kind exact | bit-changing, the CLI options and the environment switch
// that override it, and the source (owner commit, or what a pending column waits for).
// Precedence: explicit CLI (the last of the lever's own options and --baseline) > explicit environment
// (STRIX_DS41_RTX_BASELINE=1 over the lever's own variable, as the replica always did) > profile > builtin.
// A pending column is refused (the builtin form runs, the row is listed as pending); --allow-pending
// (STRIX_DS41_PROFILE_ALLOW_PENDING=1) runs a pending column whose implementation is in this build, for benchmarks only,
// and stamps the receipt "unqualified: <rows>".
namespace strix::ds41::rtx::profile {
// KV precision (--kv, strix/session/kv_precision.hpp): the model defines its KV numerics (FP8 sliding-window ring, FP4
// compressed cache; strix/ds41_attention.hpp). fp8 is native in every tier; no other precision has a path.
inline constexpr ::strix::session::kv_precision::Caps kKvCaps{"deepseek41-rtx", ::strix::session::kv_precision::Precision::fp8, 0U,
    "ds41-fp8-ring+fp4-compressed (model-defined)", {}, "the KV format is defined by the model"};

using strix::profile::Name;
using strix::profile::name;
using strix::profile::parse;
enum class App : unsigned { replica = 1, prefill_service = 2 };
using strix::profile::Kind;
using strix::profile::kind_name;

// Typed adapter: RTX has multiple options per row, app masks and ordered baseline
// statuses. These differ from the common single-option/export schema.
struct Option {
    std::string_view flag;   // e.g. "--no-wb-stream"
    std::string_view value;  // what the flag sets the lever to; "*" = the option's argument
};

struct Lever {
    std::string_view name;                  // receipt key
    std::string_view env{};                 // the lever's own environment switch (empty: none)
    std::array<Option, 6> options{};        // CLI options that set the lever explicitly (unused entries empty)
    std::array<std::string_view, 3> value;  // per profile (default, fast, quality); empty: the profile leaves it alone
    std::string_view builtin;               // this build with no profile, option or variable
    std::array<bool, 3> ready{};            // qualified and present; false: pending
    std::array<bool, 3> runnable{};         // a pending column's implementation exists in this build
    Kind kind = Kind::exact;
    unsigned apps = 0;                      // mask of App values the lever concerns
    bool baseline = true;                   // --baseline / STRIX_DS41_RTX_BASELINE=1 turn it off
    std::string_view source;                // owner commit / branch; for a pending column, what it waits for
    strix::profile::Identities identity{}; // declared identity per value (strix::profile::Lever::identity)
};
[[nodiscard]] std::span<const Lever> levers();
// The same rows as strix::profile::Lever (first option as the CLI switch, a three-column value padded with an empty turbo
// cell: turbo is refused by this adapter): lets the shared fast-set / table checks read this table. Built once.
[[nodiscard]] std::span<const strix::profile::Lever> shared_rows();

enum class Status { profile, explicit_cli, explicit_env, baseline_cli, baseline_env, pending, unqualified };
[[nodiscard]] const char* status_name(Status status);

struct Entry {
    const Lever* lever = nullptr;
    std::string value;   // the value the resolution asks for ("on", "off", an option's argument, ...)
    Status status = Status::profile;
    std::string wanted;  // pending / unqualified: the profile's value
    std::string ladder{};
    std::string ladder_reference{};  // the reference its timing rows must name (LadderRow::over)
};

struct Resolution {
    Name name = Name::standard;
    App app = App::replica;
    bool allow_pending = false;
    std::vector<Entry> entries;
    [[nodiscard]] const Entry* find(std::string_view lever) const;
    // true when the resolved value of `lever` is "on" (absent lever: false)
    [[nodiscard]] bool on(std::string_view lever) const;
    // "qualified", or "unqualified: <rows>" when --allow-pending ran pending columns
    [[nodiscard]] std::string qualification() const;
    [[nodiscard]] std::vector<std::string> pending() const;
    // {"profile","app","qualification","allow_pending","levers":{name:{value,status,switch,kind,source[,wanted][,effective]}},
    //  "pending":[...],"unqualified":[...]}; `effective` (optional) gives the app's resolved value of a lever ("" = omit)
    [[nodiscard]] std::string json(const std::function<std::string(std::string_view)>& effective = {}) const;
    // Host profile provenance only. The execution owner supplies the actual numerical
    // contract/assets after admission; unknown fields stay empty. Never auto-emitted.
    [[nodiscard]] strix::profile::ExecutionReceipt host_receipt(strix::profile::NumericalIdentity identity = {}) const;
};

[[nodiscard]] const char* app_name(App app);

using strix::profile::Lookup;
struct Inputs {
    std::vector<std::string> args;  // the launch options (argv[2..])
    Lookup env;                     // environment lookup; empty: the process environment
    strix::profile::Regime regime{};
    std::span<const strix::profile::LadderRow> ladder{};
};
// --profile NAME (else STRIX_DS41_PROFILE, else default) and --allow-pending (or STRIX_DS41_PROFILE_ALLOW_PENDING=1) out
// of the options; throws std::invalid_argument on a bad name or a missing value.
struct Selection {
    Name name = Name::standard;
    bool allow_pending = false;
};
[[nodiscard]] Selection select(const Inputs& inputs);
// Throws only through select(); a pending column never refuses the run (it is refused as a lever and listed).
[[nodiscard]] Resolution resolve(App app, const Inputs& inputs, std::span<const Lever> table = levers());
[[nodiscard]] Resolution resolve(Name profile, bool allow_pending, App app, const Inputs& inputs,
                                 std::span<const Lever> table = levers());

}  // namespace strix::ds41::rtx::profile
