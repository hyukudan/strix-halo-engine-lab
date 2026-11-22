#pragma once
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace strix {
enum class KnownWindowOutputPolicy : std::uint8_t { all_rows=0, terminal_row=1, state_only=2 };
// A segmented forward can return one terminal head per member, or just state.
[[nodiscard]] constexpr bool known_prefill_group_supports_output(KnownWindowOutputPolicy policy,
    bool logits) noexcept {
    return policy == KnownWindowOutputPolicy::terminal_row ||
        (policy == KnownWindowOutputPolicy::state_only && !logits);
}
inline void validate_known_window_output_policy(KnownWindowOutputPolicy policy) {
    if(policy!=KnownWindowOutputPolicy::all_rows&&policy!=KnownWindowOutputPolicy::terminal_row&&
        policy!=KnownWindowOutputPolicy::state_only)
        throw std::invalid_argument("unknown known-window output policy");
}
[[nodiscard]] inline std::string_view known_window_output_policy_name(KnownWindowOutputPolicy policy) {
    validate_known_window_output_policy(policy);
    switch(policy){case KnownWindowOutputPolicy::all_rows:return "all-rows";
        case KnownWindowOutputPolicy::terminal_row:return "terminal-row";
        case KnownWindowOutputPolicy::state_only:return "state-only";}
    throw std::invalid_argument("unknown known-window output policy");
}
[[nodiscard]] inline bool known_window_produces_logits(KnownWindowOutputPolicy policy,
    std::uint32_t row,std::uint32_t rows) {
    validate_known_window_output_policy(policy);
    if(!rows||row>=rows)throw std::invalid_argument("known-window output row is outside its geometry");
    return policy==KnownWindowOutputPolicy::all_rows||
        (policy==KnownWindowOutputPolicy::terminal_row&&row==rows-1U);
}
} // namespace strix
