// Host tests of the shared pair schedule plan (src/xport/ahead_plan.hpp).
#include "xport/ahead_plan.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}
} // namespace

int main() {
    using strix::xport::AheadPlan;
    using strix::xport::ahead_banks;
    using strix::xport::ahead_fill_ms;

    // the incumbent: one bank, one unit ahead (DS41 early first-chunk q_b), or none
    constexpr AheadPlan one{1U, 6U, 1U};
    static_assert(one.ahead() == 1U && one.bank(5U) == 0U && one.produced_ahead(0U) && !one.produced_ahead(1U));
    constexpr AheadPlan off{1U, 6U, 0U};
    static_assert(off.ahead() == 0U && !off.produced_ahead(0U));
    constexpr AheadPlan zero_banks{0U, 3U, 2U};
    static_assert(zero_banks.bank_count() == 1U && zero_banks.ahead() == 1U);

    // ahead never exceeds banks, units or the limit
    expect(AheadPlan{4U, 6U, 9U}.ahead() == 4U, "ahead bounded by banks");
    expect(AheadPlan{8U, 3U, 9U}.ahead() == 3U, "ahead bounded by units");
    expect(AheadPlan{8U, 6U, 2U}.ahead() == 2U, "ahead bounded by the limit (task parking)");

    // slot reuse: every unit's bank was last written by a unit already consumed when the unit is produced in order
    // (units ahead are produced before any consumption, so their banks must all differ)
    for (std::uint32_t banks = 1U; banks <= 8U; ++banks) {
        for (std::uint32_t units = 0U; units <= 20U; ++units) {
            for (std::uint32_t limit = 0U; limit <= 9U; ++limit) {
                const AheadPlan p{banks, units, limit};
                std::vector<int> last(banks, -1);  // last unit written per bank
                std::vector<bool> consumed(units, false);
                bool ok = true;
                const auto produce = [&](std::uint32_t u) {
                    const std::uint32_t b = p.bank(u);
                    if (last[b] >= 0 && !consumed[static_cast<std::size_t>(last[b])]) ok = false;
                    if ((last[b] >= 0) != p.reuses_bank(u) && u < banks) ok = false;
                    last[b] = static_cast<int>(u);
                };
                for (std::uint32_t u = 0U; u < p.ahead(); ++u) produce(u);  // before the wait
                for (std::uint32_t u = 0U; u < units; ++u) {                 // the consumer loop
                    if (!p.produced_ahead(u)) produce(u);
                    if (last[p.bank(u)] != static_cast<int>(u)) ok = false;   // unit u's output still in its bank
                    consumed[u] = true;
                }
                expect(ok, "bank reuse only after consumption");
            }
        }
    }

    // banks from the lever and the memory allowance
    constexpr std::size_t kAny = static_cast<std::size_t>(-1);
    expect(ahead_banks(0U, 6U, 128U, kAny) == 1U, "off keeps one bank");
    expect(ahead_banks(1U, 6U, 128U, kAny) == 1U, "incumbent one bank");
    expect(ahead_banks(6U, 6U, 128U, kAny) == 6U, "all units");
    expect(ahead_banks(9U, 6U, 128U, kAny) == 6U, "capped by units");
    expect(ahead_banks(6U, 6U, 128U, 300U) == 3U, "capped by the allowance: 1 + 300/128");
    expect(ahead_banks(6U, 6U, 128U, 0U) == 1U, "no allowance: one bank");
    expect(ahead_banks(6U, 0U, 128U, kAny) == 1U, "no units: one bank");

    // fill estimate: bounded by the gap, never a sum of waits
    expect(ahead_fill_ms(7.0, 5U, 0.69) > 3.44 && ahead_fill_ms(7.0, 5U, 0.69) < 3.46, "work-bound fill");
    expect(ahead_fill_ms(2.0, 5U, 0.69) == 2.0, "gap-bound fill");
    expect(ahead_fill_ms(0.0, 5U, 0.69) == 0.0, "no gap, no fill");

    if (failures != 0) return EXIT_FAILURE;
    std::puts("xport ahead plan tests: ok");
    return EXIT_SUCCESS;
}
