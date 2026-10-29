#include "strix/ds41_rtx_schedule.hpp"
#undef NDEBUG
#include <cassert>
#include <cstdint>
#include <iostream>

int main() {
    using namespace strix::ds41::rtx;
    constexpr std::uint32_t gate = 3041280, down = 3870720;
    constexpr auto p = ce_pacing(gate, down);
    static_assert(p.background_limit == 9953280);
    static_assert(p.outstanding_threshold == 6082561);
    // Any permitted part, including after a partial victim was resurrected,
    // stays under the background bound. Both threshold endpoints matter.
    for (auto outstanding : {0U, gate, 2U * gate, p.outstanding_threshold - 1, p.outstanding_threshold}) {
        for (auto part : {gate, down}) {
            if (outstanding < p.outstanding_threshold)
                assert(std::uint64_t{outstanding} + part <= p.background_limit);
        }
    }
    // A full victim can finish without a forced wait (all three parts).
    std::uint64_t outstanding = 0;
    for (auto part : {gate, gate, down}) {
        assert(outstanding < p.outstanding_threshold);
        outstanding += part;
    }
    assert(outstanding == p.background_limit);
    assert(outstanding >= p.outstanding_threshold);
    std::cout << "PASS CE pacing: background bound and partial-victim progress\n";
}
