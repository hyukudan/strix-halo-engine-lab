#include "inkling2_mini_schedule.hpp"
#include <array>
#include <cassert>

int main() {
    constexpr auto schedule = strix::inkling2::mini_schedule();
    static_assert(schedule.size() == 48);
    static_assert(schedule[0].physical_layer == 0);
    static_assert(schedule[5].physical_layer == 5);
    static_assert(schedule[6].physical_layer == 6 && schedule[6].pass == 0);
    static_assert(schedule[17].physical_layer == 17 && schedule[17].pass == 0);
    static_assert(schedule[18].physical_layer == 6 && schedule[18].pass == 1);
    static_assert(schedule[30].physical_layer == 6 && schedule[30].pass == 2);
    static_assert(schedule[41].physical_layer == 17 && schedule[41].pass == 2);
    static_assert(schedule[42].physical_layer == 18);
    static_assert(schedule[47].physical_layer == 23);
    std::array<bool, 48> seen{};
    for (const auto& visit : schedule) {
        assert(visit.logical_layer < 48);
        assert(strix::inkling2::state_slot(visit.physical_layer, visit.pass) == visit.logical_layer);
        assert(!seen[visit.logical_layer]);
        seen[visit.logical_layer] = true;
    }
    for (bool value : seen) assert(value);
    return 0;
}
