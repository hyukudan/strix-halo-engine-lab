// Expert-slot merge inside the producing launch (strix/moe_slot_merge.hpp): for every arrival order of the
// (row tile, slot) blocks, the ticket protocol yields bit-for-bit the incumbent one-CTA serial tail (merged
// accumulation, committed output, status and first-error record), and the ticket words fit the status pad.
#include "strix/moe_slot_merge.hpp"
#include "strix/numeric/bf16_bits.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

namespace sm = strix::moe_slot_merge;

namespace {

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

struct Step {
    std::uint32_t hidden = 0, top_k = 0, experts = 0;
    std::vector<std::uint32_t> routes;
    std::vector<float> weights, rawdots; // rawdots [slot][hidden]
    std::uint32_t status = 0;            // status after the down rows (entry bits | invalid routes)
    bool bf16 = true;
};
struct Result {
    std::vector<float> accumulation;
    std::vector<std::uint16_t> output;
    std::uint32_t status = 0, record_bits = 0, record_domain = 0;
    std::vector<std::uint32_t> tickets;
};

constexpr std::uint32_t kDomain = 2U;

float merge_one(const Step& s, std::uint32_t out, bool& written) {
    float current = 0.0F;
    written = true;
    for (std::uint32_t slot = 0; slot < s.top_k; ++slot) {
        if (s.routes[slot] >= s.experts) {
            written = false;
            break;
        }
        current = std::fma(s.weights[slot], s.rawdots[static_cast<std::size_t>(slot) * s.hidden + out], current);
    }
    return current;
}
void publish(Result& r, std::uint32_t final_status) {
    if (final_status != 0U && r.record_domain == 0U) {
        r.record_bits = final_status;
        r.record_domain = kDomain;
    }
}

// The incumbent q4k_a8_serial_tail_kernel, one CTA after the whole down launch.
Result serial_tail(const Step& s, const Result& seed) {
    Result r = seed;
    const std::uint32_t initial = s.status;
    bool nonfinite = false;
    if (initial == 0U)
        for (std::uint32_t out = 0; out < s.hidden; ++out) {
            bool written = true;
            const float current = merge_one(s, out, written);
            if (written) r.accumulation[out] = current;
            const float value = written ? current : r.accumulation[out];
            if (sm::fails_commit(std::bit_cast<std::uint32_t>(value), s.bf16)) nonfinite = true;
        }
    const std::uint32_t final_status = initial | (nonfinite ? sm::nonfinite_status : 0U);
    if (final_status == 0U)
        for (std::uint32_t out = 0; out < s.hidden; ++out)
            r.output[out] = sm::bf16_rne(std::bit_cast<std::uint32_t>(r.accumulation[out]));
    if (nonfinite) r.status |= sm::nonfinite_status;
    publish(r, final_status);
    return r;
}

// The in-launch fold: blocks (tile, slot) arrive in `order`; tickets as in slot_merge_epilogue.
Result folded(const Step& s, const Result& seed, const std::vector<std::uint32_t>& order) {
    Result r = seed;
    const std::uint32_t tiles = sm::tiles(s.hidden);
    for (const std::uint32_t block : order) {
        const std::uint32_t tile = block / s.top_k;
        if (!sm::last_arrival(r.tickets[tile]++, s.top_k)) continue;
        // last arrival of this tile: merge its rows in slot order
        if (s.status == 0U)
            for (std::uint32_t row = 0; row < sm::tile_rows; ++row) {
                const std::uint32_t out = tile * sm::tile_rows + row;
                bool written = true;
                const float current = merge_one(s, out, written);
                if (written) r.accumulation[out] = current;
                const float value = written ? current : r.accumulation[out];
                if (sm::fails_commit(std::bit_cast<std::uint32_t>(value), s.bf16)) r.tickets[sm::nonfinite_word(s.hidden)] |= 1U;
            }
        r.tickets[tile] = 0U;
        if (!sm::last_arrival(r.tickets[sm::tile_done_word(s.hidden)]++, tiles)) continue;
        const std::uint32_t flagged = r.tickets[sm::nonfinite_word(s.hidden)];
        const std::uint32_t final_status = s.status | (flagged != 0U ? sm::nonfinite_status : 0U);
        if (final_status == 0U)
            for (std::uint32_t out = 0; out < s.hidden; ++out)
                r.output[out] = sm::bf16_rne(std::bit_cast<std::uint32_t>(r.accumulation[out]));
        if (flagged != 0U) r.status |= sm::nonfinite_status;
        publish(r, final_status);
        r.tickets[sm::tile_done_word(s.hidden)] = 0U;
        r.tickets[sm::nonfinite_word(s.hidden)] = 0U;
    }
    return r;
}

bool same(const Result& a, const Result& b) {
    if (a.status != b.status || a.record_bits != b.record_bits || a.record_domain != b.record_domain) return false;
    for (std::size_t i = 0; i < a.accumulation.size(); ++i)
        if (std::bit_cast<std::uint32_t>(a.accumulation[i]) != std::bit_cast<std::uint32_t>(b.accumulation[i])) return false;
    return a.output == b.output;
}

void run_case(std::mt19937& rng, std::uint32_t hidden, std::uint32_t top_k, int mode, const char* name) {
    Step s;
    s.hidden = hidden;
    s.top_k = top_k;
    s.experts = 512U;
    std::uniform_real_distribution<float> value(-4.0F, 4.0F);
    std::uniform_int_distribution<std::uint32_t> expert(0U, s.experts - 1U);
    for (std::uint32_t k = 0; k < top_k; ++k) {
        s.routes.push_back(expert(rng));
        s.weights.push_back(value(rng) * 0.25F);
    }
    s.rawdots.resize(static_cast<std::size_t>(top_k) * hidden);
    for (auto& v : s.rawdots) v = value(rng) * std::ldexp(1.0F, static_cast<int>(rng() % 40U) - 20);
    if (mode == 1) s.rawdots[hidden / 3U] = std::numeric_limits<float>::infinity();       // nonfinite -> status 8
    if (mode == 2) s.status = 2U;                                                         // entry error: no merge
    if (mode == 3) { s.rawdots[5] = 3.3e38F; s.weights[0] = 1.0F; }                       // rounds to bf16 inf
    if (mode == 4) { s.routes[top_k - 1U] = s.experts; s.status = 1U; }                   // invalid route
    if (mode == 5) s.routes[top_k / 2U] = s.experts;                                      // invalid route, clean status
    Result seed;
    seed.accumulation.assign(hidden, 0.0F);
    seed.output.assign(hidden, 0x5555U);
    seed.status = s.status;
    seed.tickets.assign(sm::ticket_words(hidden), 0U);
    const Result reference = serial_tail(s, seed);
    std::vector<std::uint32_t> order(static_cast<std::size_t>(sm::tiles(hidden)) * top_k);
    for (std::uint32_t i = 0; i < order.size(); ++i) order[i] = i;
    for (int trial = 0; trial < 64; ++trial) {
        if (trial == 1) std::reverse(order.begin(), order.end());
        if (trial > 1) std::shuffle(order.begin(), order.end(), rng);
        const Result got = folded(s, seed, order);
        expect(same(reference, got), name);
        expect(std::all_of(got.tickets.begin(), got.tickets.end(), [](std::uint32_t w) { return w == 0U; }),
            "tickets return to zero");
    }
}

} // namespace

int main() {
    std::mt19937 rng(20261118U);
    // Ticket words fit the status pad for every admitted hidden size (status is 256-aligned, codes at +256).
    for (std::uint32_t hidden = 256U; hidden <= 2560U; hidden += 256U)
        expect(4U + 4U * sm::ticket_words(hidden) <= 256U, "ticket words fit the status pad");
    expect(sm::last_arrival(9U, 10U) && !sm::last_arrival(8U, 10U), "last arrival");
    expect(sm::fails_commit(0x7f7f8000U, true) && !sm::fails_commit(0x7f7f8000U, false), "bf16 commit threshold");
    expect(sm::fails_commit(0xff800000U, false) && sm::fails_commit(0x7fc00000U, false), "inf / nan fail");
    expect(sm::bf16_rne(0x3f808000U) == 0x3f80U && sm::bf16_rne(0x3f818000U) == 0x3f82U, "bf16 ties to even");
    {
        // The shared BF16 activation codec equals the launchers' former private rounding for every high half and
        // the rounding-critical low halves (ties, carries into the exponent, NaN payloads), and so does the slot merge.
        namespace bb = strix::numeric::bf16_bits;
        const std::uint32_t lows[] = {0x0000U, 0x0001U, 0x7fffU, 0x8000U, 0x8001U, 0xffffU, 0x1234U, 0xc000U};
        bool same = true;
        for (std::uint32_t high = 0; high < 0x10000U; ++high)
            for (const auto low : lows) {
                const std::uint32_t bits = (high << 16U) | low;
                const auto former = static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
                const float value = std::bit_cast<float>(bits);
                const std::uint32_t widened = static_cast<std::uint32_t>(former) << 16U;
                same = same && bb::rne(bits) == former && bb::float_to_bf16(value) == former &&
                       sm::bf16_rne(bits) == former && std::bit_cast<std::uint32_t>(bb::bf16_to_float(former)) == widened &&
                       std::bit_cast<std::uint32_t>(bb::round_bf16(value)) == widened;
            }
        expect(same, "shared bf16 codec equals the former launcher rounding");
    }
    const std::uint32_t shapes[][2] = {{2560U, 10U}, {2048U, 8U}, {256U, 1U}, {1024U, 4U}, {2560U, 1U}};
    for (const auto& shape : shapes)
        for (int mode = 0; mode < 6; ++mode) {
            static const char* names[] = {"clean step bitwise", "nonfinite -> status 8, no commit",
                "entry status -> no merge", "bf16 overflow fails commit", "invalid route", "invalid route, clean status"};
            run_case(rng, shape[0], shape[1], mode, names[mode]);
        }
    if (failures != 0) {
        std::fprintf(stderr, "moe_slot_merge_tests: %d failures\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("moe_slot_merge_tests: PASS\n");
    return EXIT_SUCCESS;
}
