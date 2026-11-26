#pragma once

namespace strix::profile {

enum class EmptyOverride { present, unset };

struct MemberOverride {
    const char* value;
    bool explicit_value;
};

// A user setting wins over a proposed bundle member. Delay the proposal so the
// caller reads the master switch and checks admission only for an unset member.
template <class Propose>
constexpr MemberOverride resolve_member_override(const char* raw, EmptyOverride empty, const Propose& propose) {
    if (raw != nullptr && (empty == EmptyOverride::present || *raw != '\0')) return {raw, true};
    return {propose(), false};
}

} // namespace strix::profile
