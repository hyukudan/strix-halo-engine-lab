#include "inkling2/inkling2_mini_resident_route.hpp"
#include "inkling2/inkling2_mini_schedule.hpp"
#include "inkling2/inkling2_mini_state.hpp"
#include "strix/serve/cohort_decode.hpp"
#include "strix/serve/prompt_piece.hpp"
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
// Served prefill data (models/inkling2-mini.toml [serve_prefill]): the gap only where a device class has a receipt.
static_assert(mini_serve_prefill(true, "gfx1151").gap_ms == 1000.0 && mini_serve_prefill(true, "gfx1151").pass_fixed_ms == 240.0);
// HIP classes without a receipt: main's busy regime (share 1, 256-row pieces); gfx1151: the gap, share 0 and 512-row
// floor pieces (a shorter pass, e.g. a streaming request's own ~280-row prompt, under-prices a row: n5b).
static_assert(mini_serve_prefill(true, "gfx1151").decoding_rows == 512U &&
              mini_serve_prefill(true, "gfx1100").decoding_rows == 256U &&
              mini_resolve_decoding_rows(0U, 8192U, true, "gfx1151") == 512U &&
              mini_resolve_decoding_rows(0U, 512U, true, "gfx1151") == 0U &&
              mini_resolve_decoding_rows(384U, 8192U, true, "gfx1151") == 384U);
static_assert(mini_serve_prefill(true, "gfx1100").gap_ms == 0.0 && mini_serve_prefill(true).decoding_rows == 256U &&
              mini_serve_prefill(true).decode_share == 1.0 && mini_serve_prefill(true).chunk == 0U &&
              mini_serve_prefill(true, "gfx1151").decode_share == 0.0);
static_assert(mini_serve_prefill(false).chunk == 128U && mini_serve_prefill(false).decoding_rows == 0U &&
              mini_serve_prefill(false).decode_share == 1.0 && mini_serve_prefill(false, "gfx1151").gap_ms == 0.0);
static_assert(mini_device_class("gfx1151:sramecc-:xnack-|AMD Radeon|cu40") == "gfx1151" &&
              mini_device_class("sm_120|NVIDIA RTX|cu188") == "sm_120" && mini_device_class("").empty());
static_assert(mini_resolve_serve_chunk(0U, 36864U, true) == 8192U && mini_resolve_serve_chunk(0U, 36864U, false) == 128U &&
              mini_resolve_serve_chunk(2048U, 36864U, false) == 2048U && mini_resolve_serve_chunk(0U, 4096U, true) == 4096U);
static_assert(mini_resolve_decoding_rows(0U, 8192U, true) == 256U && mini_resolve_decoding_rows(0U, 128U, true) == 0U &&
              mini_resolve_decoding_rows(32U, 8192U, true) == 32U && mini_resolve_decoding_rows(0U, 128U, false) == 0U);
static_assert(mini_prefill_tail_capacity(8192U, 8U) == 1024U && mini_prefill_tail_capacity(2048U, 0U) == 0U);
static_assert(strix::serve::prompt_slice_rows(8192U, 8192U, 8235U, 1024U) == 8235U);   // 8192 + 43 -> one pass
static_assert(strix::serve::prompt_slice_rows(8192U, 8192U, 9300U, 1024U) == 8192U);   // remainder > tail
static_assert(strix::serve::prompt_slice_rows(8192U, 8192U, 9216U, 1024U) == 9216U);   // remainder == tail
static_assert(strix::serve::prompt_slice_rows(8192U, 8192U, 8235U, 0U) == 8192U);      // greedy chunks
static_assert(strix::serve::prompt_slice_rows(4096U, 8192U, 8235U, 1024U) == 4096U);   // planner granted a part
static_assert(strix::serve::prompt_slice_rows(8192U, 8192U, 100U, 1024U) == 100U);

// Variant served-cli-prefill: served prefill = CLI prefill, chunk by chunk (a simulation of the prefill partitioning
// and of the route selection, not of the kernels). CLI (serial_generation_cli): chunk = min(chunk, prompt), the decoder
// admits the requested remainder at construction and each prefill call absorbs its actual remainder (call_tail_rows).
// Server: the lane prefills its WHOLE prompt (FixedKvCohort emits-first); the cohort cuts one unit per step and lets the
// final slice carry up to the decoder's admitted capacity (prompt_slice_rows); the decoder splits each call by the same
// per-call rule. The device admits every enlarged chunk here; the rule is the decoder's own (mini_call_tail_rows).
namespace {
using Slices = std::vector<std::pair<std::size_t, std::size_t>>;

unsigned admitted_capacity(unsigned chunk, unsigned tail) {  // MiniResidentDecoder::admitted_tail_rows
    return mini_admitted_tail_capacity(chunk, tail, [](unsigned) { return true; });
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
    const unsigned capacity = admitted_capacity(chunk, mini_prefill_tail_capacity(chunk, divisor, context));
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

// Served bits == CLI bits needs equal ROUTES as well as equal chunks: every prefill chunk takes its MoE body from
// mini_resident_route (the resident backend's only route choice). The server runs the grouped decoder; the CLI runs
// packed (its default) or grouped (--grouped-decode). A prefill chunk's route must not depend on that decode mode, or
// a 1-row chunk (a 1-token prompt, chunk + 1 rows such as 8193 or 4097 at chunk 4096) prefills on the packed body in
// the CLI and on the grouped body in the server (the review r9 P1 divergence). Decode rows follow the declared decode
// route: the server's first decode row equals the CLI's under --grouped-decode only.
int served_route_equals_cli() {
    using strix::inkling2::MiniAppendKind;
    using strix::inkling2::mini_resident_route;
    const auto same = [](const MiniResidentRoute& a, const MiniResidentRoute& b) {
        return a.attention_rows == b.attention_rows && a.packed_moe == b.packed_moe && a.step_graph == b.step_graph;
    };
    unsigned compared = 0;
    for (const unsigned requested : {0U, 4096U, 1000U})
        for (unsigned prompt = 1U; prompt <= 20000U; ++prompt) {
            const Slices served = served_chunks(prompt, requested, 36864U, 8U);
            const Slices cli = cli_chunks(prompt, requested, 8U);
            if (served != cli) {
                std::fprintf(stderr, "FAIL route test: served != CLI chunks at prompt %u chunk %u\n", prompt, requested);
                return 1;
            }
            for (const auto& [at, rows] : served)
                for (const bool cli_packed : {true, false}) {
                    const auto r = static_cast<unsigned>(rows);
                    if (!same(mini_resident_route(MiniAppendKind::prefill, r, false, false),
                              mini_resident_route(MiniAppendKind::prefill, r, cli_packed, false))) {
                        std::fprintf(stderr, "FAIL served prefill route != CLI (%s decode) at prompt %u chunk %u: rows "
                                             "[%zu, +%zu)\n", cli_packed ? "packed" : "grouped", prompt, requested,
                                     at, rows);
                        return 1;
                    }
                    ++compared;
                }
        }
    // Decode rows follow the decoder's decode mode only: the server's grouped-only decoder takes the grouped body (as
    // the CLI under --grouped-decode), a packed decoder (the CLI default) the packed one for decode / verify rows.
    if (mini_resident_route(MiniAppendKind::decode, 1U, false, false).packed_moe ||
        !mini_resident_route(MiniAppendKind::decode, 1U, true, false).packed_moe ||
        !mini_resident_route(MiniAppendKind::verify, 3U, true, false).packed_moe) {
        std::fprintf(stderr, "FAIL decode routes: grouped serve, packed only for decode / verify rows\n");
        return 1;
    }
    std::printf("served == CLI prefill routes for %u (prompt, chunk, CLI decode mode) chunks\n", compared);
    return 0;
}

// Default served-incumbent == main's served bits (a simulation of the prefill partitioning; bits per row by the
// qualified piece rule). Main (d3a16318e): the server prefilled the prompt minus its final token in greedy 128-row
// calls of one chunk each (decoder tail 0) and advanced the final token by the grouped decode row. A row's bits are
// its chunk class (at most kKeySplitMaxRows rows: small-row attention; one row) and its 16-row wave tile counted from
// its chunk start. served-incumbent runs chunks of C (a multiple of 128) with main's short tail as its own chunk,
// alone (one call per region) and beside a decoding lane (pieces of the actual chunk under any grant,
// serve::prompt_piece_within with the decoder's cut rule, then the region's second chunk as its own call).
struct RowBits {
    bool small = false, one = false;
    std::uint32_t tile = 0, tile_end = 0;
    bool operator==(const RowBits&) const = default;
};
void record_chunk(std::vector<RowBits>& bits, std::size_t at, std::size_t rows) {
    for (std::size_t p = at; p < at + rows; ++p) {
        const std::size_t tile = at + (p - at) / kMiniPrefillWaveRows * kMiniPrefillWaveRows;
        bits[p] = {rows <= strix::attn_core::dense::kKeySplitMaxRows, rows == 1U, static_cast<std::uint32_t>(tile),
                   static_cast<std::uint32_t>(std::min<std::size_t>(tile + kMiniPrefillWaveRows, at + rows))};
    }
}
// MiniResidentDecoder::prefill_slot of one call: chunks of `chunk` over the call minus its short tail, then the tail.
void incumbent_call(std::vector<RowBits>& bits, std::size_t at, std::size_t rows, unsigned chunk, unsigned form) {
    const std::size_t head = rows - mini_incumbent_short_tail(rows, form);
    for (std::size_t done = 0; done < head;) {
        const std::size_t c = std::min<std::size_t>(chunk, head - done);
        record_chunk(bits, at + done, c);
        done += c;
    }
    if (head != rows) record_chunk(bits, at + head, rows - head);
}

int incumbent_equals_main() {
    using strix::serve::prompt_piece_within;
    const unsigned form = kMiniIncumbentFormRows;
    const std::uint64_t grants[] = {128U, 300U, 1000U, 2050U, 5000U};
    unsigned compared = 0;
    const auto check_prompt = [&](unsigned prompt, unsigned chunk) {
        const std::size_t n = prompt - 1U;  // the final token is the decode row in both
        std::vector<RowBits> main_bits(n), alone(n), busy(n);
        for (std::size_t at = 0; at < n; at += form) record_chunk(main_bits, at, std::min<std::size_t>(form, n - at));
        const auto rule = mini_serve_piece_rule(true, MiniServedIdentity::incumbent, form, chunk);
        std::size_t g = 0;
        for (std::size_t read = 0; read < n;) {
            const std::size_t region = std::min<std::size_t>(chunk, n - read);  // tail 0: no absorption
            incumbent_call(alone, read, region, chunk, form);
            const auto first = static_cast<std::size_t>(mini_incumbent_first_chunk(region, chunk, form));
            if (region != first &&
                static_cast<std::size_t>(mini_incumbent_first_chunk(region - first, chunk, form)) != region - first)
                return false;  // the shared scheduler refuses a call of more than two chunks
            for (std::size_t r = read; r < read + first;) {
                const auto piece = static_cast<std::size_t>(
                    prompt_piece_within(read + first - r, grants[g++ % std::size(grants)], rule).rows);
                incumbent_call(busy, r, piece, chunk, form);
                r += piece;
            }
            if (region != first) incumbent_call(busy, read + first, region - first, chunk, form);
            read += region;
        }
        ++compared;
        return alone == main_bits && busy == main_bits;
    };
    for (const unsigned chunk : {128U, 1024U, 2048U, 4096U})
        for (unsigned prompt = 1U; prompt <= 4300U; ++prompt)
            if (!check_prompt(prompt, chunk)) {
                std::fprintf(stderr, "FAIL served-incumbent != main served forms: prompt %u chunk %u\n", prompt, chunk);
                return 1;
            }
    for (const unsigned prompt : {1U, 17U, 4101U, 8192U, 8193U, 8209U, 8292U, 9216U, 16389U, 32768U})
        if (!check_prompt(prompt, 8192U)) {
            std::fprintf(stderr, "FAIL served-incumbent != main served forms: prompt %u chunk 8192\n", prompt);
            return 1;
        }
    for (unsigned prompt = 8150U; prompt <= 8360U; ++prompt)
        if (!check_prompt(prompt, 8192U)) {
            std::fprintf(stderr, "FAIL served-incumbent != main served forms: prompt %u chunk 8192\n", prompt);
            return 1;
        }
    // the incumbent's first id comes from the decode row (FixedKvCohort prefill_slot route), the variant's from prefill
    if (mini_served_identity(MiniServedIdentity::incumbent).emits_first ||
        !mini_served_identity(MiniServedIdentity::cli_prefill).emits_first ||
        mini_resolve_served_identity("") != MiniServedIdentity::incumbent ||
        mini_resolve_served_identity("served-cli-prefill") != MiniServedIdentity::cli_prefill) {
        std::fprintf(stderr, "FAIL served identity registry\n");
        return 1;
    }
    bool refused = false;
    try {
        (void)mini_resolve_served_identity("cli");
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    if (!refused) {
        std::fprintf(stderr, "FAIL an unknown served identity was accepted\n");
        return 1;
    }
    // Main's short tail of the prompt minus its final token: P = 4101 (4096 + 4) and 8209 (8192 + 16) run it as a short
    // chunk in main and in served-incumbent, not in the CLI chunking (served-cli-prefill).
    if (mini_incumbent_short_tail(4100U, form) != 4U || mini_incumbent_short_tail(8208U, form) != 16U ||
        mini_incumbent_short_tail(8191U, form) != 0U || mini_incumbent_short_tail(16U, form) != 0U) {
        std::fprintf(stderr, "FAIL incumbent short tail\n");
        return 1;
    }
    std::printf("served-incumbent == main served forms for %u (prompt, chunk) cases, alone and in pieces\n", compared);
    return 0;
}

// Slots (review P2): main's implicit cap, explicit --slots never silently fewer, a default chunk that costs slots the
// incumbent chunk admits refused with a message.
static_assert(mini_serve_slot_cap(0U, 8192U) == 128U && mini_serve_slot_cap(0U, 64U) == 64U &&
              mini_serve_slot_cap(512U, 8192U) == 512U && mini_serve_slot_cap(512U, 300U) == 300U);
static_assert(mini_row_arena_bytes(8192U, 2304U, 16U) - mini_row_arena_bytes(128U, 2304U, 16U) ==
              std::uint64_t{8064U} * 2304U * 18U * 4U);
int slot_refusals() {
    const bool ok = mini_slot_refusal(4U, 4U, 0U, false, 8192U, 128U, 1000U).empty() &&
                    mini_slot_refusal(4U, 4U, 4U, false, 8192U, 128U, 1000U).empty() &&
                    mini_slot_refusal(3U, 4U, 4U, false, 8192U, 128U, 1000U).find("--slots 4 do not fit") !=
                        std::string::npos &&
                    mini_slot_refusal(3U, 4U, 0U, false, 8192U, 128U, 1000U).find("--prefill-rows 128") !=
                        std::string::npos &&
                    mini_slot_refusal(3U, 4U, 0U, true, 8192U, 128U, 1000U).empty() &&   // chunk requested
                    mini_slot_refusal(3U, 4U, 0U, false, 128U, 128U, 1000U).empty() &&   // the incumbent chunk
                    mini_slot_refusal(0U, 4U, 0U, true, 128U, 128U, 1000U).find("no session slots") != std::string::npos;
    if (!ok) {
        std::fprintf(stderr, "FAIL slot refusals\n");
        return 1;
    }
    std::printf("slot sizing: explicit slots and default-chunk slot costs refused with a message\n");
    return 0;
}

// Route forms of every row (inkling2_mini_schedule.hpp): short chunk (key-split policy, the one-row packed body) or
// not, and the 16-position wave tile [start, end) counted from the row's chunk start. Equal forms = equal bits.
struct Form {
    bool short_chunk;
    std::size_t tile_start, tile_end;
    bool operator==(const Form&) const = default;
};
std::vector<Form> forms(const Slices& chunks, std::size_t prompt) {
    std::vector<Form> out(prompt, Form{false, 0U, 0U});
    for (const auto& [at, rows] : chunks)
        for (std::size_t p = at; p < at + rows; ++p) {
            const std::size_t tile = at + (p - at) / kMiniPrefillWaveRows * kMiniPrefillWaveRows;
            out[p] = {rows <= kMiniPieceShortRows, tile, std::min<std::size_t>(tile + kMiniPrefillWaveRows, at + rows)};
        }
    return out;
}

// Served with another lane decoding on the steps `busy(step)` says: the cohort's piece rule (serve::prompt_piece_rows,
// CohortDecoder::piece_rows) over the decoder's call chunks (mini_call_first_chunk), with the HIP schedule's pieces.
template <class Busy>
Slices served_pieces(unsigned prompt, unsigned context, unsigned divisor, unsigned pieces, Busy&& busy) {
    const unsigned chunk = mini_resolve_prefill_chunk(0U, context, true);
    const unsigned capacity = admitted_capacity(chunk, mini_prefill_tail_capacity(chunk, divisor, context));
    const strix::serve::PromptPieceRule rule{kMiniPieceAlignRows, kMiniPieceShortRows};
    const auto admits = [](unsigned) { return true; };
    Slices out;
    std::uint64_t read = 0, region_start = 0, region_end = 0, chunk_end = 0;
    for (unsigned step = 0; read < prompt; ++step) {
        const std::uint64_t left = prompt - read;
        if (read >= region_end) {
            const auto region = strix::serve::prompt_slice_rows(chunk, chunk, left, capacity);
            region_start = read;
            region_end = read + region;
            chunk_end = read + mini_call_first_chunk(region, chunk,
                                                     region <= chunk ? 0U : mini_call_tail_rows(region, chunk, capacity, admits));
        } else if (read >= chunk_end) {
            chunk_end = region_end;
        }
        const std::uint64_t want = std::min<std::uint64_t>(busy(step) ? pieces : chunk, left);
        const std::uint64_t rows = read == region_start && want >= std::min<std::uint64_t>(left, chunk)
            ? region_end - read : strix::serve::prompt_piece_rows(chunk_end - read, want, chunk, rule);
        device_chunks(read, rows, chunk, capacity, out);
        read += rows;
    }
    return out;
}

// Bounded pieces keep the bits: for prompt lengths up to 20000 rows (all to 2000, then +-40 rows around every 512) (HIP chunk 8192, tail 1024, pieces of 1024 and
// 512) and three patterns of a decoding lane (always, every other step, from the third step), the served rows take the
// route forms of the CLI's chunks. And the review's admission edge: chunk 60000, prompt 60017, a capacity whose full
// enlargement (67500 rows) the device refuses: the decoder keeps the largest admitted capacity and the 17-row tail rides
// in the final chunk as in the CLI (one device chunk in both).
int pieces_keep_forms() {
    unsigned compared = 0;
    for (const unsigned pieces : {1024U, 512U})
        for (unsigned prompt = 1U; prompt <= 20000U; ++prompt) {
            if (prompt > 2000U && prompt % 512U >= 40U && prompt % 512U <= 472U) continue;  // every piece/chunk edge kept
            const auto cli = forms(cli_chunks(prompt, 0U, 8U), prompt);
            const bool ok = forms(served_pieces(prompt, 36864U, 8U, pieces, [](unsigned) { return true; }), prompt) == cli &&
                            forms(served_pieces(prompt, 36864U, 8U, pieces, [](unsigned s) { return s % 2U == 1U; }), prompt) == cli &&
                            forms(served_pieces(prompt, 36864U, 8U, pieces, [](unsigned s) { return s >= 2U; }), prompt) == cli;
            compared += 3U;
            if (!ok) {
                std::fprintf(stderr, "FAIL bounded pieces change the route forms: prompt %u pieces %u\n", prompt, pieces);
                return 1;
            }
        }
    const auto device = [](unsigned rows) { return rows <= 65535U; };
    const unsigned kept = mini_admitted_tail_capacity(60000U, 7500U, device);
    const unsigned cli_capacity = mini_admitted_tail_capacity(60000U, strix::serve::prefill_tail_rows(60017U, 60000U, 8U), device);
    if (mini_prefill_tail_rows(60000U, 7500U, device(67500U)) != 0U || kept != 5535U || cli_capacity != 17U ||
        mini_call_tail_rows(60017U, 60000U, kept, device) != 17U || mini_call_tail_rows(60017U, 60000U, cli_capacity, device) != 17U ||
        mini_call_first_chunk(60017U, 60000U, 17U) != 60017U || mini_prefill_tail_capacity(8192U, 8U, 8192U) != 0U ||
        mini_prefill_tail_capacity(8192U, 8U, 8700U) != 508U || mini_prefill_tail_capacity(8192U, 8U, 36864U) != 1024U) {
        std::fprintf(stderr, "FAIL tail capacity admission / context cap\n");
        return 1;
    }
    std::printf("bounded pieces keep the CLI route forms for %u (prompt, pieces, pattern) cases\n", compared);
    return 0;
}
}  // namespace

int main() {
    if (const auto failed = prefill_tail_gate()) return failed;
    if (const auto failed = served_equals_cli()) return failed;
    if (const auto failed = served_route_equals_cli()) return failed;
    if (const auto failed = incumbent_equals_main()) return failed;
    if (const auto failed = slot_refusals()) return failed;
    if (const auto failed = pieces_keep_forms()) return failed;
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
