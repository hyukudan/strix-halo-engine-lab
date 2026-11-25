#include "inkling2/inkling2_mini_resident_route.hpp"
#include "inkling2/inkling2_mini_schedule.hpp"
#include "inkling2/inkling2_mini_state.hpp"
#include "strix/serve/cohort_decode.hpp"
#include "strix/serve/prompt_piece.hpp"
#include "strix/serve/serial_generation_cli.hpp"
#include "strix/session/prompt_windows.hpp"
#include <array>
#include <span>
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

using namespace strix::inkling2;
using strix::profile::Name;

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

// Route forms of every prompt row: 1 = one-row chunk (packed R1 MoE body with packed decode), 2 = short chunk (small-row
// attention, at most the key-split bound), 3 = long chunk; with the row's offset in its chunk modulo the wave tile.
struct RowForm {
    unsigned kind = 0, phase = 0;
    bool operator==(const RowForm&) const = default;
};
constexpr bool admits_rows(unsigned rows) { return rows <= 65535U; }
void record_call(std::vector<RowForm>& rows, std::size_t at, std::size_t call, unsigned chunk, unsigned capacity) {
    const std::vector<unsigned> tokens(call);
    const auto tail = mini_call_tail_rows(call, chunk, capacity, admits_rows);
    strix::session::prompt_windows::append_chunks(std::span<const unsigned>(tokens), chunk,
        [&](std::span<const unsigned> part, bool) {
            const unsigned kind = part.size() == 1U ? 1U : part.size() <= strix::attn_core::dense::kKeySplitMaxRows ? 2U : 3U;
            for (std::size_t r = 0; r < part.size(); ++r)
                rows.at(at + r) = {kind, static_cast<unsigned>(r % kMiniPrefillWaveRows)};
            at += part.size();
        }, tail);
}
// The command line: ONE prefill() call over the prompt with the decoder built for it (serial_generation_cli: chunk
// clamped to the prompt, offered tail = the actual remainder).
std::vector<RowForm> cli_forms(std::size_t prompt, MiniPrefillSchedule schedule) {
    std::vector<RowForm> rows(prompt);
    const auto chunk = static_cast<unsigned>(std::min<std::size_t>(schedule.chunk, prompt));
    const auto offered = strix::serve::prefill_tail_rows(static_cast<std::uint32_t>(prompt), chunk, schedule.tail_divisor);
    record_call(rows, 0U, prompt, chunk, mini_admitted_tail_capacity(chunk, offered, admits_rows));
    return rows;
}
// The server: the cohort's regions (prompt_slice_rows with the decoder's tail capacity), each region one call when no
// other lane decodes, else pieces of kMiniServeDecodingRows cut inside each of the region's chunks by the decoder's
// piece rule (prompt_call_chunk = mini_call_first_chunk); every call runs the decoder's per-call chunk rule.
std::vector<RowForm> served_forms(std::size_t prompt, MiniPrefillSchedule schedule, unsigned context, bool decoding,
                                  bool hip) {
    std::vector<RowForm> rows(prompt);
    const auto chunk = std::min(schedule.chunk, context);
    const auto capacity = mini_admitted_tail_capacity(chunk, mini_prefill_tail_capacity(chunk, schedule.tail_divisor, context),
                                                      admits_rows);
    const auto rule = mini_prefill_piece_rule(hip);
    const auto call_tail = [&](std::uint64_t call) { return mini_call_tail_rows(call, chunk, capacity, admits_rows); };
    std::size_t at = 0;
    while (at < prompt) {
        const auto region = strix::serve::prompt_slice_rows(chunk, chunk, prompt - at, capacity);
        if (!decoding || !rule.cuttable()) {
            record_call(rows, at, region, chunk, capacity);
            at += region;
            continue;
        }
        const auto first = mini_call_first_chunk(region, chunk, call_tail(region));
        for (const auto part : {first, region - first}) {
            for (std::uint64_t left = part; left != 0U;) {
                const auto piece = strix::serve::prompt_piece_rows(left, kMiniServeDecodingRows, chunk, rule);
                record_call(rows, at, piece, chunk, capacity);
                at += piece;
                left -= piece;
            }
        }
    }
    return rows;
}

int served_prefill_gate() {
    const std::size_t lengths[] = {1U, 2U, 16U, 17U, 18U, 33U, 127U, 128U, 129U, 144U, 145U, 255U, 2047U, 2048U, 2049U,
                                   2064U, 2065U, 2080U, 4113U, 8191U, 8192U, 8193U, 8208U, 8209U, 8300U, 9216U, 9217U,
                                   16384U, 16401U, 20000U};
    int failed = 0;
    for (const bool hip : {true, false}) {
        for (const unsigned divisor : {mini_prefill_schedule(hip, Name::standard).tail_divisor, 0U}) {
            const MiniPrefillSchedule schedule{mini_prefill_schedule(hip, Name::standard).chunk, divisor};
            for (const auto prompt : lengths) {
                const auto cli = cli_forms(prompt, schedule);
                for (const bool decoding : {false, true})
                    if (served_forms(prompt, schedule, 65536U, decoding, hip) != cli) {
                        std::fprintf(stderr, "FAIL served prefill forms != CLI: hip=%d divisor=%u prompt=%zu decoding=%d\n",
                                     hip ? 1 : 0, divisor, prompt, decoding ? 1 : 0);
                        failed = 40;
                    }
            }
        }
    }
    // The resolver of both entry points takes the declared rows per backend and profile (mini_cli_prefill_rows): HIP
    // 8192 rows with a chunk / 8 tail in every profile, CUDA greedy chunks of 8192 rows in EXACT and 2048 otherwise;
    // cuts on HIP.
    for (const auto tier : {Name::standard, Name::quality, Name::fast, Name::turbo}) {
        const auto hip = mini_prefill_schedule(true, tier), cuda = mini_prefill_schedule(false, tier);
        const unsigned cuda_rows = tier == Name::standard ? 8192U : 2048U;
        if (hip.chunk != 8192U || hip.tail_divisor != 8U || cuda.chunk != cuda_rows || cuda.tail_divisor != 0U ||
            hip.chunk != mini_cli_prefill_rows(MiniPrefillBackend::hip, tier) ||
            cuda.chunk != mini_cli_prefill_rows(MiniPrefillBackend::cuda, tier)) {
            std::fprintf(stderr, "FAIL schedule != declared prefill rows (profile %u)\n", static_cast<unsigned>(tier));
            failed = 41;
        }
    }
    static_assert(mini_prefill_piece_rule(true).cuttable() && !mini_prefill_piece_rule(false).cuttable());
    static_assert(mini_prefill_tail_capacity(8192U, 8U, 8192U) == 0U && mini_prefill_tail_capacity(8192U, 8U, 9000U) == 808U);
    static_assert(mini_call_first_chunk(8193U, 8192U, 0U) == 8192U && mini_call_first_chunk(8209U, 8192U, 17U) == 8209U);
    return failed;
}

}  // namespace

int main() {
    if (const auto failed = prefill_tail_gate()) return failed;
    if (const auto failed = served_prefill_gate()) return failed;
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

    // HIP verify lanes (verify rows lever; attn_core sequence_lane_caches): a sliding-window ring sized for a verify
    // window's chunk holds window - 1 + rows, so storing every row of the window before its flash reuses no slot a row
    // still reads.
    for (unsigned rows = 2U; rows <= kMiniMaxVerifyRows; ++rows) {
        const auto verify_plan = visit_state_plan(schedule[6], kMaxContext, rows);
        if (verify_plan.window != kSlidingWindow || verify_plan.kv.ring == 0U ||
            verify_plan.kv.ring < kSlidingWindow - 1U + rows)
            return 35;
    }

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
