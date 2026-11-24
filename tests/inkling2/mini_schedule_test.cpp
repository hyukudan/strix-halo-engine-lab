#include "inkling2/inkling2_mini_prefill_rows.hpp"
#include "inkling2/inkling2_mini_schedule.hpp"
#include "inkling2/inkling2_mini_state.hpp"
#include "strix/session/prompt_windows.hpp"
#include <array>
#include <optional>
#include <span>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

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

// Chunk sizes prefill() runs for a prompt of `prompt` rows, nominal chunk `chunk` and an offered tail (the remainder).
std::vector<std::size_t> mini_prefill_chunks(std::size_t prompt, unsigned chunk, unsigned offered, bool admitted) {
    std::vector<std::size_t> sizes;
    const std::vector<unsigned> tokens(prompt);
    strix::session::prompt_windows::append_chunks(std::span<const unsigned>(tokens), chunk,
        [&](std::span<const unsigned> rows, bool) { sizes.push_back(rows.size()); },
        mini_prefill_tail_rows(chunk, offered, admitted));
    return sizes;
}

int prefill_tail_gate() {
    // A tail of at most the key-split bound takes the small-row attention (and, at one row, the packed R1 MoE) in its
    // own chunk, so it must not be absorbed: tails 1 and 16 keep the greedy schedule, tail 17 is absorbed.
    static_assert(mini_prefill_tail_rows(8192U, 1U, true) == 0U);
    static_assert(mini_prefill_tail_rows(8192U, 16U, true) == 0U);
    static_assert(mini_prefill_tail_rows(8192U, 17U, true) == 17U);
    static_assert(mini_prefill_tail_rows(8192U, 1024U, true) == 1024U);
    // The tail must start a 16-position wave tile in both schedules.
    static_assert(mini_prefill_tail_rows(8200U, 40U, true) == 0U);
    static_assert(mini_prefill_tail_rows(8208U, 40U, true) == 40U);
    // The enlarged chunk must pass the device admission, else the nominal chunk runs (65,535 + 1 and 65,472 + 100).
    static_assert(mini_prefill_tail_rows(65535U, 1U, true) == 0U);
    static_assert(mini_prefill_tail_rows(65472U, 100U, false) == 0U);
    static_assert(mini_prefill_tail_rows(0U, 17U, true) == 0U);
    if (mini_prefill_chunks(8193U, 8192U, 1U, true) != std::vector<std::size_t>{8192U, 1U}) return 30;
    if (mini_prefill_chunks(8208U, 8192U, 16U, true) != std::vector<std::size_t>{8192U, 16U}) return 31;
    if (mini_prefill_chunks(8209U, 8192U, 17U, true) != std::vector<std::size_t>{8209U}) return 32;
    if (mini_prefill_chunks(16401U, 8192U, 17U, true) != std::vector<std::size_t>{8192U, 8209U}) return 33;
    if (mini_prefill_chunks(65572U, 65472U, 100U, false) != std::vector<std::size_t>{65472U, 100U}) return 34;
    return 0;
}

// Entry-point defaults, explicit precedence and bounds of the prefill rows.
int prefill_rows_gate() {
    using strix::inkling2::MiniPrefillEntry;
    using strix::inkling2::mini_prefill_rows;
    constexpr auto cli = MiniPrefillEntry::cli;
    constexpr auto http = MiniPrefillEntry::http;
    static_assert(mini_prefill_rows(cli, std::nullopt, 0U) == 8192U);
    static_assert(mini_prefill_rows(http, std::nullopt, 0U) == 128U);
    static_assert(mini_prefill_rows(cli, 4096U, 0U) == 4096U && mini_prefill_rows(http, 1024U, 0U) == 1024U);
    static_assert(mini_prefill_rows(cli, std::nullopt, 300U) == 300U);   // chunk never exceeds the prompt
    static_assert(mini_prefill_rows(cli, 512U, 300U) == 300U && mini_prefill_rows(cli, 256U, 300U) == 256U);
    static_assert(mini_prefill_rows(http, std::nullopt, 64U) == 64U);    // slice never exceeds the context
    static_assert(mini_prefill_rows(http, 65535U, 8192U) == 8192U);
    static_assert(strix::inkling2::kMiniCliPrefillFallbackRows == 2048U);

    return 0;
}

}  // namespace

int main() {
    if (const auto failed = prefill_tail_gate()) return failed;
    if (const auto failed = prefill_rows_gate()) return failed;
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
    static_assert(valid_mini_visit(schedule[18]));
    static_assert(!valid_mini_visit({6, 2, 18}));
    static_assert(!valid_mini_visit({7, 1, 18}));
    static_assert(!valid_mini_visit({6, 1, 48}));
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
    for (const auto visit : schedule) {
        full_visits += uses_sliding_attention(visit) ? 0U : 1U;
    }
    constexpr MiniSessionStatePlan decode_session(kMaxContext, 1);
    static_assert(decode_session.kv_bytes() == (32ULL << 30U) + (160ULL << 20U));
    if (full_visits != 8) return 11;

    constexpr auto chunk_plan = visit_state_plan(schedule[6], kMaxContext, 8192);
    static_assert(chunk_plan.kv.ring == 16384);
    static_assert(chunk_plan.convolution_widths == std::array<unsigned, 4>{1024, 1024, 2304, 2304});
    constexpr MiniSessionStatePlan short_session(512, 512);
    static_assert(short_session.visit(schedule[6]).kv.ring == 0);
    static_assert(short_session.visit(schedule[6]).kv_rows == 512);
    static_assert(short_session.visit(schedule[6]).window == kSlidingWindow);
    constexpr MiniSessionStatePlan contiguous_session(2048, 1024);
    static_assert(contiguous_session.visit(schedule[6]).kv.ring == 0);
    static_assert(contiguous_session.visit(schedule[6]).window == kSlidingWindow);

    constexpr unsigned max_chunk = 2048;
    constexpr MiniSessionStatePlan session(kMaxContext, max_chunk);
    static_assert(session.kv_bytes() == (32ULL << 30U) + (640ULL << 20U));
    static_assert(session.convolution_bytes() == 3833856);
    static_assert(session.visit(schedule[6]).kv.ring == 4096);
    static_assert(session.visit(schedule[6]).window == kSlidingWindow);
    static_assert(session.visit(schedule[5]).kv.ring == 0);
    static_assert(session.visit(schedule[5]).window == 0);
    static_assert(session.visit(schedule[5]).kv_rows == kMaxContext);
    static_assert(session.admits(0, max_chunk) && session.admits(kMaxContext - 1, 1));
    static_assert(!session.admits(kMaxContext, 1) && !session.admits(0, 0));
    static_assert(!session.admits(0, max_chunk + 1) && !session.admits(kMaxContext - 1, 2));

    std::uint64_t next_kv = 0;
    std::uint64_t next_conv = 0;
    for (const auto visit : schedule) {
        const auto& state = session.visit(visit);
        if (state.kv_offset != next_kv) return 12;
        next_kv += state.kv_elements();
        for (unsigned stream = 0; stream < state.convolution_offsets.size(); ++stream) {
            if (state.convolution_offsets[stream] != next_conv) return 13;
            next_conv += state.convolution_elements(stream);
        }
    }
    if (next_kv != session.kv_elements_per_plane() || next_conv != session.convolution_elements()) return 14;

    // This repeat keeps a slot layout derived from max_chunk when decode starts.
    // The prefill chunk straddles the physical wrap after appending all its rows.
    constexpr auto state = session.visit(schedule[18]);
    constexpr unsigned pad = 64;
    constexpr unsigned prompt = state.kv.ring - pad;
    constexpr unsigned oldest = prompt - state.window + 1U;
    constexpr unsigned decode = prompt + max_chunk;
    static_assert(session.admits(prompt, max_chunk) && session.admits(decode, 1));
    static_assert(state.kv_offset != session.visit(schedule[6]).kv_offset);
    static_assert(state.kv_offset != session.visit(schedule[30]).kv_offset);
    std::array<unsigned, state.kv.ring> tokens{};
    const auto slot = [&](unsigned position) {
        return strix::attn::cache_rows::row_offset(state.kv, position) / state.kv.width;
    };
    for (unsigned key = oldest; key < prompt; ++key) tokens[slot(key)] = key;
    for (unsigned key = prompt; key < decode; ++key) tokens[slot(key)] = key;
    for (unsigned key = oldest; key < decode; ++key) if (tokens[slot(key)] != key) return 15;
    tokens[slot(decode)] = decode;
    for (unsigned key = decode - state.window + 1U; key <= decode; ++key)
        if (tokens[slot(key)] != key) return 16;

    constexpr MiniSessionStatePlan small_session(2, 1);
    std::vector<std::uint16_t> keys(small_session.kv_elements_per_plane());
    std::vector<std::uint16_t> values(small_session.kv_elements_per_plane());
    const auto small_visit = schedule[18];
    const auto slice = mini_visit_kv_cache(small_session, small_visit, {keys, values});
    const auto offset = small_session.visit(small_visit).kv_offset;
    if (slice.slots != 2 || slice.wrapped || slice.keys != keys.data() + offset ||
        slice.values != values.data() + offset ||
        mini_visit_kv_cache(small_session, {6, 0, 18}, {keys, values}).slots != 0 ||
        mini_visit_kv_cache(small_session, small_visit, {keys, keys}).slots != 0 ||
        mini_visit_kv_cache(small_session, small_visit, {keys, {}}).slots != 0) return 19;

    try {
        const MiniSessionStatePlan invalid(512, 513);
        (void)invalid;
        return 17;
    } catch (const std::invalid_argument&) {
    }
    try {
        (void)session.visit({6, 0, 18});
        return 18;
    } catch (const std::invalid_argument&) {
    }
    return 0;
}
