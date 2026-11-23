#include "inkling2/inkling2_mini_schedule.hpp"
#include "inkling2/inkling2_mini_state.hpp"
#include "strix/serve/cohort_decode.hpp"
#include "strix/serve/serial_generation_cli.hpp"
#include "strix/session/prompt_windows.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <utility>
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

}  // namespace

// One resolver for CLI and server: request wins, context clamps, the tail capacity is chunk / divisor.
static_assert(mini_resolve_prefill_chunk(0U, 36864U, true) == 8192U);
static_assert(mini_resolve_prefill_chunk(0U, 36864U, false) == 2048U);
static_assert(mini_resolve_prefill_chunk(0U, 4096U, true) == 4096U);
static_assert(mini_resolve_prefill_chunk(128U, 36864U, true) == 128U);
static_assert(mini_prefill_tail_capacity(8192U, 8U) == 1024U && mini_prefill_tail_capacity(2048U, 0U) == 0U);
static_assert(strix::serve::prompt_slice_rows(8192U, 8192U, 8235U, 1024U) == 8235U);   // 8192 + 43 -> one pass
static_assert(strix::serve::prompt_slice_rows(8192U, 8192U, 9300U, 1024U) == 8192U);   // remainder > tail
static_assert(strix::serve::prompt_slice_rows(8192U, 8192U, 9216U, 1024U) == 9216U);   // remainder == tail
static_assert(strix::serve::prompt_slice_rows(8192U, 8192U, 8235U, 0U) == 8192U);      // greedy chunks
static_assert(strix::serve::prompt_slice_rows(4096U, 8192U, 8235U, 1024U) == 4096U);   // planner granted a part
static_assert(strix::serve::prompt_slice_rows(8192U, 8192U, 100U, 1024U) == 100U);

// Served prefill = CLI prefill, chunk by chunk. CLI (serial_generation_cli): chunk = min(chunk, prompt), the decoder
// admits the requested remainder at construction and each prefill call absorbs its actual remainder (call_tail_rows).
// Server: the lane prefills its WHOLE prompt (FixedKvCohort emits-first); the cohort cuts one unit per step and lets the
// final slice carry up to the decoder's admitted capacity (prompt_slice_rows); the decoder splits each call by the same
// per-call rule. The device admits every enlarged chunk here; the rule is the decoder's own (mini_call_tail_rows).
namespace {
using Slices = std::vector<std::pair<std::size_t, std::size_t>>;

unsigned admitted_capacity(unsigned chunk, unsigned tail) {
    return tail == 0U ? 0U : mini_prefill_tail_rows(chunk, tail, true);  // MiniResidentDecoder::admitted_tail_rows
}

void device_chunks(std::size_t start, std::size_t rows, unsigned chunk, unsigned capacity, Slices& out) {
    static const std::vector<unsigned> ids(32768U);
    std::size_t at = start;
    strix::session::prompt_windows::append_chunks(std::span<const unsigned>(ids).first(rows), chunk, [&](auto part, bool) {
        out.emplace_back(at, part.size());
        at += part.size();
    }, mini_call_tail_rows(rows, chunk, capacity, [](unsigned) { return true; }));
}

Slices cli_chunks(unsigned prompt, unsigned requested, unsigned divisor) {
    const unsigned chunk = std::min(requested != 0U ? requested : mini_prefill_schedule(true).chunk, prompt);
    Slices out;
    device_chunks(0U, prompt, chunk, admitted_capacity(chunk, strix::serve::prefill_tail_rows(prompt, chunk, divisor)), out);
    return out;
}

Slices served_chunks(unsigned prompt, unsigned requested, unsigned context, unsigned divisor) {
    const unsigned chunk = mini_resolve_prefill_chunk(requested, context, true);
    const unsigned capacity = admitted_capacity(chunk, mini_prefill_tail_capacity(chunk, divisor));
    Slices out;
    std::uint64_t read = 0;
    while (read < prompt) {
        const std::uint64_t left = prompt - read;
        const auto rows = strix::serve::prompt_slice_rows(std::min<std::uint64_t>(chunk, left), chunk, left, capacity);
        device_chunks(read, rows, chunk, capacity, out);
        read += rows;
    }
    return out;
}

int served_equals_cli() {
    struct Case { unsigned requested, context, divisor, prompts; };
    const Case cases[] = {{0U, 36864U, 8U, 20000U}, {0U, 36864U, 0U, 20000U}, {0U, 12000U, 8U, 12000U},
                          {0U, 4096U, 8U, 4096U}, {1024U, 36864U, 8U, 20000U}, {1000U, 36864U, 8U, 5000U}};
    unsigned compared = 0;
    for (const auto& c : cases)
        for (unsigned prompt = 1U; prompt <= c.prompts; ++prompt, ++compared)
            if (served_chunks(prompt, c.requested, c.context, c.divisor) != cli_chunks(prompt, c.requested, c.divisor)) {
                std::fprintf(stderr, "FAIL served != CLI chunks: prompt %u chunk %u context %u divisor %u\n", prompt,
                             c.requested, c.context, c.divisor);
                return 1;
            }
    // The review cases: a 1..16-row remainder runs as its own chunk in both, 17 and more ride in the final chunk.
    const auto expect = [](unsigned prompt, Slices want) { return served_chunks(prompt, 0U, 36864U, 8U) == want; };
    if (!expect(8197U, {{0U, 8192U}, {8192U, 5U}}) || !expect(8208U, {{0U, 8192U}, {8192U, 16U}}) ||
        !expect(8209U, {{0U, 8209U}}) || !expect(8236U, {{0U, 8236U}}) || !expect(9216U, {{0U, 9216U}}) ||
        !expect(9217U, {{0U, 8192U}, {8192U, 1025U}}) || !expect(16389U, {{0U, 8192U}, {8192U, 8192U}, {16384U, 5U}}) ||
        !expect(1U, {{0U, 1U}})) {
        std::fprintf(stderr, "FAIL review remainder cases\n");
        return 1;
    }
    std::printf("served == CLI prefill chunks for %u (prompt, schedule) cases\n", compared);
    return 0;
}
}  // namespace

int main() {
    if (const auto failed = prefill_tail_gate()) return failed;
    if (const auto failed = served_equals_cli()) return failed;
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
