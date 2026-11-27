#pragma once

// --all-on (env STRIX_ALL_ON=1): turn on every lever of the catalogue that is qualified for the resolved tier in the
// current regime band and backend but is still an explicit option. One resolver for the strix launcher and strix-serve.
//
//   EXACT  only exact levers (bit-exact stays bit-exact)
//   FAST   exact levers + lossy levers whose gate v2 verdict is pass
//   TURBO  everything above + lossy levers whose gate is pending or failed for FAST and the turbo levers; each such
//          line carries its stated quality
// KV precision is never switched by this option: it stays the tier preset's separate toggle (OFF in EXACT and FAST,
// ON in TURBO). A lever without a complete classification stays off.
//
// The lever data is the "all_on" object of the rows of docs/optimizations.json, exported to data/profile/all_on.json
// by tools/ci/export_all_on.py and embedded here. The models name which catalogue engines they use in
// models/*.toml ([catalogue] engines = [...]). Nothing here names a model.
#include "strix/engine_profile.hpp"
#include "strix/model_description.hpp"

#include <cstdint>
#include <functional>
#include <iosfwd>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace strix::profile::all_on {

inline constexpr char kEnv[] = "STRIX_ALL_ON";
inline constexpr char kBandEnv[] = "STRIX_ALL_ON_BAND";
inline constexpr char kAppliedEnv[] = "STRIX_ALL_ON_APPLIED";  // set by the first shell so a second one does not repeat it

enum class Tier : std::uint8_t { exact, fast, turbo };
enum class Class : std::uint8_t { exact, fast, turbo, kv, unclassified };
enum class Gate : std::uint8_t { none, pass, pending, fail };
enum class Phase : std::uint8_t { any, prefill, decode };

struct Lever {
    std::string row;    // catalogue row name
    std::string name;   // lever key
    Class cls = Class::unclassified;
    Gate gate = Gate::none;
    std::vector<std::string> engines, backends;   // catalogue engines ("*": every engine) and backends
    Phase phase = Phase::any;
    std::optional<std::pair<std::uint64_t, std::uint64_t>> tokens;   // prompt / context band, inclusive
    std::string env, value;                       // one environment switch ...
    std::vector<std::string> args;                // ... or engine arguments
    std::vector<std::string> incompatible_with, requires_levers, default_tiers;
    std::string quality, reason;
};

[[nodiscard]] std::vector<Lever> parse_catalogue(std::string_view json);
[[nodiscard]] const std::vector<Lever>& catalogue();   // the embedded export

struct Band {
    Phase phase = Phase::any;
    std::optional<std::uint64_t> tokens;
};
// "any" | "prefill" | "decode" | "prefill:8192" | "decode:4096" (std::invalid_argument otherwise)
[[nodiscard]] Band parse_band(std::string_view);

struct Query {
    Tier tier = Tier::exact;
    std::string backend;                    // "hip" | "cuda"
    std::vector<std::string> engines;       // catalogue engines of the model
    Band band;
    std::function<bool(std::string_view)> env_is_set;   // an explicit environment value always wins
};

struct Line {
    std::string lever, source, reason;      // source: default | all-on | refused
    [[nodiscard]] std::string text() const;  // "lever NAME source=SOURCE[ reason=TEXT]"
};

struct Plan {
    std::vector<Line> lines;
    std::vector<std::pair<std::string, std::string>> env;   // exports of the all-on levers
    std::vector<std::string> args;                          // engine arguments of the all-on levers
    [[nodiscard]] std::string text() const;                 // the lines, one per line, newline terminated
};

[[nodiscard]] Plan resolve(std::span<const Lever>, const Query&);

[[nodiscard]] Tier tier_of(Name);
[[nodiscard]] std::vector<std::string> engines_of(const model_description::ModelDescription*);
// STRIX_ALL_ON: unset or 0 off, 1 on, anything else std::invalid_argument.
[[nodiscard]] bool env_requested();

// The shell binding shared by the strix launcher and strix-serve: resolves the plan for the model, prints it to `out`
// and, when `apply`, exports the environment (never overwriting a set variable) for the scope's lifetime and appends
// the lever arguments to `args`. A second call under the same process tree prints and applies nothing.
class Scope {
public:
    Scope() = default;
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    Scope(Scope&&) noexcept;
    Scope& operator=(Scope&&) noexcept;
    ~Scope();
    [[nodiscard]] const Plan& plan() const { return plan_; }
private:
    friend Scope begin(const model_description::ModelDescription*, std::optional<Name>, std::string_view,
                       std::string_view, std::ostream&, bool, std::vector<std::string>&);
    void release() noexcept;
    Plan plan_;
    std::vector<std::string> set_;   // variables this scope exported
};
[[nodiscard]] Scope begin(const model_description::ModelDescription*, std::optional<Name> tier, std::string_view backend,
                          std::string_view band, std::ostream& out, bool apply, std::vector<std::string>& args);

}  // namespace strix::profile::all_on
