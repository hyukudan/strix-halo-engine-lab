#pragma once
// The engine side of --all-on (strix/profile/all_on.hpp), header only so an engine library needs no resolver link.
// The first shell lists the switches it exported in STRIX_ALL_ON_LEVERS ("LEVER:ENV,LEVER:ENV"; an argument lever as
// "LEVER:--flag", its first argument). An engine whose own admission gates one of them calls engaged() with the switch
// (environment name or argument) and the result: when the switch was exported by --all-on and the engine
// does not engage it, one line "all-on: lever LEVER source=all-on applied=no reason=REASON" goes to `out`, so the
// printed plan never claims a lever the engine did not run. A switch the user set by hand is not in the list.
#include "strix/env_flags.hpp"

#include <ostream>
#include <string_view>

namespace strix::profile::all_on {

inline constexpr char kLeversEnv[] = "STRIX_ALL_ON_LEVERS";

// Returns `on`.
inline bool engaged(std::string_view env, bool on, std::string_view reason, std::ostream& out) {
    if (on) return true;
    const char* list = strix::envcfg::text(kLeversEnv);
    for (std::string_view rest(list != nullptr ? list : ""); !rest.empty();) {
        const auto comma = rest.find(',');
        const auto item = rest.substr(0, comma);
        const auto colon = item.find(':');
        if (colon != std::string_view::npos && item.substr(colon + 1) == env) {
            out << "all-on: lever " << item.substr(0, colon) << " source=all-on applied=no reason=" << reason << '\n';
            return false;
        }
        if (comma == std::string_view::npos) break;
        rest.remove_prefix(comma + 1);
    }
    return false;
}

}  // namespace strix::profile::all_on
