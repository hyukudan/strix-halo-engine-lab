#include "inkling2/inkling2_mini_schedule.hpp"
#include "inkling2/inkling2_mini_state.hpp"
#include <array>
#include <stdexcept>

namespace {

using namespace strix::inkling2;

struct ForwardFixture {
    int hidden = 0;
    int residual_blocks = 0;
    int entry_hidden = 0;
    int entry_blocks = 0;
    int boundary_hidden = 0;
    std::array<std::array<int, 2>, 2> previous_banks{};
    std::array<unsigned, kLogicalVisits> state_updates{};

    void save_core_entry() {
        entry_hidden = hidden;
        entry_blocks = residual_blocks;
    }
    void capture_previous_pass(unsigned pass) {
        previous_banks.at(pass) = {boundary_hidden, hidden};
    }
    void restore_core_entry() {
        hidden = entry_hidden;
        residual_blocks = entry_blocks;
    }
    void run_layer(LayerVisit visit) {
        if (visit.physical_layer == kCoreFirst && visit.pass != 0) {
            // Model an update sourced by the two captured states.
            const auto& bank = previous_banks.at(visit.pass - 1);
            hidden += bank[0] + bank[1];
        }
        if (captures_message_boundary(visit)) boundary_hidden = hidden;
        if (starts_residual_block(visit)) ++residual_blocks;
        ++state_updates.at(visit.logical_layer);
        ++hidden;
    }
};

}  // namespace

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
    static_assert(reference_state_slot(schedule[18]) == 24);
    static_assert(reference_state_slot(schedule[30]) == 36);
    static_assert(reference_state_slot(schedule[42]) == 18);
    std::array<bool, 48> seen{};
    std::array<bool, 48> reference_seen{};
    std::array<unsigned, 24> visits{};
    for (const auto& visit : schedule) {
        if (visit.logical_layer >= 48 || visit.physical_layer >= 24) return 1;
        if (strix::inkling2::state_slot(visit.physical_layer, visit.pass) != visit.logical_layer) return 2;
        if (seen[visit.logical_layer]) return 3;
        seen[visit.logical_layer] = true;
        const unsigned reference_slot = reference_state_slot(visit);
        if (reference_slot >= reference_seen.size() || reference_seen[reference_slot]) return 7;
        reference_seen[reference_slot] = true;
        ++visits[visit.physical_layer];
    }
    for (bool value : seen) if (!value) return 4;
    for (unsigned layer = 0; layer < 24; ++layer)
        if (visits[layer] != (strix::inkling2::is_core(layer) ? 3u : 1u)) return 5;
    const auto rejects = [](unsigned layer, unsigned pass) {
        try {
            (void)strix::inkling2::state_slot(layer, pass);
            return false;
        } catch (const std::out_of_range&) {
            return true;
        }
    };
    if (!rejects(24, 0) || !rejects(6, 3) || !rejects(0, 1) || !rejects(23, 1)) return 6;
    ForwardFixture forward;
    MiniLayers::run(forward);
    if (forward.previous_banks != std::array{std::array{12, 18}, std::array{42, 48}}) return 8;
    if (forward.hidden != 114 || forward.residual_blocks != 4) return 9;
    for (unsigned updates : forward.state_updates) if (updates != 1) return 10;
    unsigned full_visits = 0;
    std::uint64_t cache_bytes = 0;
    for (const auto visit : schedule) {
        full_visits += uses_sliding_attention(visit) ? 0U : 1U;
        cache_bytes += visit_state_plan(visit, kMaxContext, 1).kv_bytes();
    }
    if (full_visits != 8 || cache_bytes != (32ULL << 30U) + (160ULL << 20U)) return 11;
    constexpr auto chunk_plan = visit_state_plan(schedule[6], kMaxContext, 8192);
    static_assert(chunk_plan.kv.ring == 16384);
    static_assert(chunk_plan.convolution_widths == std::array<unsigned, 4>{1024, 1024, 2304, 2304});
    static_assert(visit_state_plan(schedule[6], 512, 512).kv.ring == 0);
    for (unsigned key = 1; key < kSlidingWindow - 1U + 8192U; ++key)
        if (strix::attn::cache_rows::slot(chunk_plan.kv, key) ==
            strix::attn::cache_rows::slot(chunk_plan.kv, 0)) return 12;
    return 0;
}
