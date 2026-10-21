// DSpark drafter, CPU tests: scalar formats, declared operations against double-precision oracles,
// the greedy verify protocol and a speculative decode simulation whose tokens must equal plain
// greedy decoding for every window policy.

#include "strix/ds41_dspark.hpp"
#include "ds41_dspark_model.hpp"
#include "strix/ds41_dspark_driver.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace strix::ds41::dspark;
int failures = 0;
void expect(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

void test_f16() {
    std::size_t bad = 0;
    for (std::uint32_t b = 0; b < 65536U; ++b) {
        const auto bits = static_cast<std::uint16_t>(b);
        const std::uint32_t e = (b >> 10U) & 31U, m = b & 1023U;
        double v = e == 0U ? std::ldexp(static_cast<double>(m), -24) : std::ldexp(1.0 + static_cast<double>(m) / 1024.0, static_cast<int>(e) - 15);
        if ((b & 0x8000U) != 0U) v = -v;
        const float got = f16_value(bits);
        if (e == 31U) bad += m == 0U ? !std::isinf(got) : !std::isnan(got);
        else bad += static_cast<double>(got) != v || std::signbit(got) != ((b & 0x8000U) != 0U);
    }
    expect(bad == 0U, "F16 decode is exact for every code (" + std::to_string(bad) + " differ)");
}

void test_e4m3() {
    std::size_t bad = 0;
    for (std::uint32_t c = 0; c < 256U; ++c) {
        const auto code = static_cast<std::uint8_t>(c);
        if ((c & 0x7fU) == 0x7fU) {
            bad += !std::isnan(decode_e4m3(code));
            continue;
        }
        const auto back = encode_e4m3(decode_e4m3(code));
        bad += back != code && !(c == 0x80U && back == 0x80U);
    }
    expect(bad == 0U, "E4M3 encode(decode(c)) == c for every finite code");
    expect(encode_e4m3(1000.0F) == 0x7eU && encode_e4m3(-1000.0F) == 0xfeU, "E4M3 saturates to +-448");
    expect(encode_e4m3(1.0625F) == 0x38U, "E4M3 tie 1.0625 rounds to even (1.0)");
    expect(encode_e4m3(1.1875F) == 0x3aU, "E4M3 tie 1.1875 rounds to even (1.25)");
    expect(encode_e4m3(0.0009765625F) == 0x00U && encode_e4m3(0.0029296875F) == 0x02U, "E4M3 subnormal ties to even");
    // FP8 scale: 2^ceil(log2(amax / 448)).
    expect(scale_value(fp8_scale_code(448.0F)) == 1.0F && scale_value(fp8_scale_code(449.0F)) == 2.0F, "FP8 scale is a ceiling power of two");
    expect(scale_value(fp8_scale_code(0.0F)) == std::ldexp(1.0F, -22), "FP8 scale floors amax at 1e-4");
    // Entries: every value within half an E4M3 step of the scaled input.
    std::mt19937 random(5);
    std::normal_distribution<float> normal(0.0F, 3.0F);
    std::vector<float> x(head_dim);
    for (auto& v : x) v = normal(random);
    x[7] = 0.0F; x[40] = 1e-30F;
    std::vector<std::uint8_t> entry(entry_bytes);
    fp8_entry(x, entry);
    double worst = 0.0;
    for (std::uint32_t i = 0; i < head_dim; ++i) {
        const double s = scale_value(entry[head_dim + i / 32U]);
        worst = std::max(worst, std::fabs(static_cast<double>(entry_value(entry, i)) - x[i]) / (s * 448.0));
    }
    expect(worst <= 1.0 / 16.0 + 1e-9, "FP8 entry error within half an E4M3 step of the group range");
}

void test_wave_sum() {
    std::mt19937 random(3);
    std::uniform_real_distribution<float> u(-1.0F, 1.0F);
    std::vector<float> a(20480);
    for (auto& v : a) v = u(random);
    double exact = 0.0;
    for (const float v : a) exact += v;
    const float got = wave_sum(20480U, [&](std::uint32_t i) { return a[i]; });
    expect(std::fabs(got - exact) < 1e-3, "wave_sum close to the double sum");
    // The order is the declared one: lane partial sums then the butterfly.
    float lanes[32] = {};
    for (std::uint32_t i = 0; i < 20480U; ++i) lanes[i % 32U] += a[i];
    for (std::uint32_t d = 16; d; d /= 2U) {
        float next[32];
        for (std::uint32_t l = 0; l < 32U; ++l) next[l] = lanes[l] + lanes[l ^ d];
        std::copy(next, next + 32, lanes);
    }
    expect(std::bit_cast<std::uint32_t>(got) == std::bit_cast<std::uint32_t>(lanes[0]), "wave_sum is the declared order");
}

void test_sinkhorn() {
    std::mt19937 random(11);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> scale{0.7F, 1.3F, 2.1F}, base(mix_count);
    for (auto& b : base) b = normal(random);
    double worst = 0.0, worst_oracle = 0.0;
    for (int trial = 0; trial < 200; ++trial) {
        std::array<float, mix_count> m{};
        for (auto& v : m) v = 2.0F * normal(random);
        const auto out = sinkhorn_split(m, scale, base);
        // Double-precision oracle of the same algorithm.
        double c[4][4], pre[4], post[4];
        for (int j = 0; j < 4; ++j) {
            pre[j] = 1.0 / (1.0 + std::exp(-(static_cast<double>(m[j]) * scale[0] + base[j]))) + 1e-6;
            post[j] = 2.0 / (1.0 + std::exp(-(static_cast<double>(m[4 + j]) * scale[1] + base[4 + j])));
            for (int k = 0; k < 4; ++k) c[j][k] = static_cast<double>(m[8 + 4 * j + k]) * scale[2] + base[8 + 4 * j + k];
        }
        for (int j = 0; j < 4; ++j) {
            double mx = c[j][0], sum = 0.0;
            for (int k = 1; k < 4; ++k) mx = std::max(mx, c[j][k]);
            for (int k = 0; k < 4; ++k) sum += c[j][k] = std::exp(c[j][k] - mx);
            for (int k = 0; k < 4; ++k) c[j][k] = c[j][k] / sum + 1e-6;
        }
        for (int it = 0; it < 20; ++it) {
            if (it)
                for (int j = 0; j < 4; ++j) {
                    const double sum = c[j][0] + c[j][1] + c[j][2] + c[j][3];
                    for (int k = 0; k < 4; ++k) c[j][k] /= sum + 1e-6;
                }
            for (int k = 0; k < 4; ++k) {
                const double sum = c[0][k] + c[1][k] + c[2][k] + c[3][k];
                for (int j = 0; j < 4; ++j) c[j][k] /= sum + 1e-6;
            }
        }
        for (int j = 0; j < 4; ++j) {
            worst_oracle = std::max({worst_oracle, std::fabs(out[static_cast<std::size_t>(j)] - pre[j]),
                                     std::fabs(out[static_cast<std::size_t>(4 + j)] - post[j])});
            double column = 0.0;
            for (int k = 0; k < 4; ++k) {
                worst_oracle = std::max(worst_oracle, std::fabs(out[static_cast<std::size_t>(8 + 4 * j + k)] - c[j][k]));
                column += out[static_cast<std::size_t>(8 + 4 * k + j)];
            }
            worst = std::max(worst, std::fabs(column - 1.0)); // ends with a column pass
        }
    }
    expect(worst < 2e-3, "Sinkhorn comb columns sum to 1 within 2e-3 (" + std::to_string(worst) + ")");
    expect(worst_oracle < 1e-5, "Sinkhorn split within 1e-5 of the double oracle (" + std::to_string(worst_oracle) + ")");
}

void test_rope() {
    const auto row = rope_row(1000);
    double worst = 0.0;
    for (std::uint32_t i = 0; i < 32U; ++i) {
        const double a = 1000.0 / std::pow(10000.0, 2.0 * i / 64.0);
        worst = std::max({worst, std::fabs(row.c[i] - std::cos(a)), std::fabs(row.s[i] - std::sin(a))});
    }
    expect(worst < 1e-7, "RoPE table is the rounded double table");
    std::mt19937 random(2);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> x(2U * 512U), y;
    for (auto& v : x) v = normal(random);
    y = x;
    apply_rope(y, 2, 512, 77, false);
    double changed = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) changed = std::max(changed, static_cast<double>(std::fabs(x[i] - y[i])));
    apply_rope(y, 2, 512, 77, true);
    double back = 0.0, untouched = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        back = std::max(back, static_cast<double>(std::fabs(x[i] - y[i])));
        if (i % 512U < 448U) untouched = std::max(untouched, static_cast<double>(std::fabs(x[i] - y[i])));
    }
    expect(changed > 0.1 && back < 1e-5 && untouched == 0.0, "RoPE rotates only the last 64 dims and inverts");
}

// Block attention against a double-precision softmax-with-sink over the same decoded keys.
void test_block_attention() {
    std::mt19937 random(9);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (const std::uint64_t first : {0ULL, 5ULL, 140ULL}) {
        Ring ring;
        std::vector<float> v(head_dim);
        for (std::uint64_t p = 0; p < first; ++p) {
            for (auto& x : v) x = normal(random);
            fp8_entry(v, std::span<std::uint8_t>(ring.entries).subspan((p % window) * entry_bytes, entry_bytes));
        }
        ring.next = first;
        const std::uint32_t rows = block_size;
        std::vector<std::uint8_t> block(static_cast<std::size_t>(rows) * entry_bytes);
        for (std::uint32_t r = 0; r < rows; ++r) {
            for (auto& x : v) x = normal(random);
            fp8_entry(v, std::span<std::uint8_t>(block).subspan(static_cast<std::size_t>(r) * entry_bytes, entry_bytes));
        }
        std::vector<float> q(static_cast<std::size_t>(rows) * q_width), sinks(heads);
        for (auto& x : q) x = 3.0F * normal(random);
        for (auto& x : sinks) x = normal(random);
        const auto o = block_attention(q, rows, ring, block, sinks, first);
        // Oracle.
        const std::uint64_t oldest = first > window ? first - window : 0U;
        std::vector<const std::uint8_t*> keys;
        for (std::uint64_t p = oldest; p < first; ++p) keys.push_back(ring.entries.data() + (p % window) * entry_bytes);
        for (std::uint32_t r = 0; r < rows; ++r) keys.push_back(block.data() + static_cast<std::size_t>(r) * entry_bytes);
        std::vector<double> ref(o.size());
        for (std::uint32_t r = 0; r < rows; ++r)
            for (std::uint32_t h = 0; h < heads; ++h) {
                const float* qh = q.data() + (static_cast<std::size_t>(r) * heads + h) * head_dim;
                std::vector<double> s(keys.size());
                double mx = -1e300;
                for (std::size_t j = 0; j < keys.size(); ++j) {
                    double dot = 0.0;
                    for (std::uint32_t d = 0; d < head_dim; ++d)
                        dot += static_cast<double>(qh[d]) * entry_value(std::span<const std::uint8_t>(keys[j], entry_bytes), d);
                    s[j] = dot / std::sqrt(512.0);
                    mx = std::max(mx, s[j]);
                }
                double sum = std::exp(static_cast<double>(sinks[h]) - mx);
                for (auto& x : s) sum += x = std::exp(x - mx);
                for (std::uint32_t d = 0; d < head_dim; ++d) {
                    double acc = 0.0;
                    for (std::size_t j = 0; j < keys.size(); ++j)
                        acc += s[j] * entry_value(std::span<const std::uint8_t>(keys[j], entry_bytes), d);
                    ref[(static_cast<std::size_t>(r) * heads + h) * head_dim + d] = acc / sum;
                }
            }
        // Inverse RoPE of the oracle in double.
        for (std::uint32_t r = 0; r < rows; ++r)
            for (std::uint32_t h = 0; h < heads; ++h)
                for (std::uint32_t i = 0; i < 32U; ++i) {
                    const double a = static_cast<double>(first + r) / std::pow(10000.0, 2.0 * i / 64.0);
                    double& x0 = ref[(static_cast<std::size_t>(r) * heads + h) * head_dim + 448U + 2U * i];
                    double& x1 = ref[(static_cast<std::size_t>(r) * heads + h) * head_dim + 449U + 2U * i];
                    const double p = x0, b = x1;
                    x0 = p * std::cos(a) + b * std::sin(a);
                    x1 = b * std::cos(a) - p * std::sin(a);
                }
        double worst = 0.0, scale = 0.0;
        for (std::size_t i = 0; i < o.size(); ++i) {
            worst = std::max(worst, std::fabs(o[i] - ref[i]));
            scale = std::max(scale, std::fabs(ref[i]));
        }
        expect(worst <= 1e-5 * std::max(1.0, scale), "block attention (next " + std::to_string(first) + ") within 1e-5 of the double oracle (" +
                                                          std::to_string(worst) + ")");
    }
}

void test_verify() {
    const std::vector<std::uint32_t> d{4, 5, 6};
    expect(verify_greedy(d, std::vector<std::uint32_t>{4, 5, 6, 7}).accepted == 3U &&
               verify_greedy(d, std::vector<std::uint32_t>{4, 5, 6, 7}).next_token == 7U,
           "all drafts accepted gives the bonus token");
    const auto partial = verify_greedy(d, std::vector<std::uint32_t>{4, 9, 6, 7});
    expect(partial.accepted == 1U && partial.committed_rows == 2U && partial.next_token == 9U, "first mismatch stops acceptance");
    const auto none = verify_greedy(std::vector<std::uint32_t>{}, std::vector<std::uint32_t>{3});
    expect(none.accepted == 0U && none.next_token == 3U, "a one-row window is a plain target step");
    const std::array<float, 5> high{4, 4, 4, 4, 4}, low{-4, -4, -4, -4, -4};
    const std::array<float, 6> flat{1, 1, 1, 1, 1, 1}, steep{1.0F, 1.3F, 1.6F, 1.9F, 2.2F, 2.5F};
    expect(choose_drafts(high, steep, 0) == 5U, "confident drafts fill the window");
    expect(choose_drafts(low, steep, 0) == 0U, "unconfident drafts are not verified");
    expect(choose_drafts(low, steep, 2) == 2U && choose_drafts(high, flat, 0) == 5U, "the minimum is honoured; free confident drafts fill the window");
}

// Toy speculative decoding: the target is a deterministic function of the last three tokens; the
// drafter guesses with errors. Every policy must reproduce plain greedy decoding exactly.
std::uint32_t toy_target(const std::vector<std::uint32_t>& seq, std::size_t end) {
    const std::uint32_t a = seq[end - 1U], b = end > 1U ? seq[end - 2U] : 0U, c = end > 2U ? seq[end - 3U] : 0U;
    return (a * 2654435761U ^ b * 40503U ^ c * 97U) % 61U;
}
void test_speculative_simulation() {
    std::vector<std::uint32_t> greedy{1, 2, 3};
    while (greedy.size() < 400U) greedy.push_back(toy_target(greedy, greedy.size()));
    for (int policy = 0; policy < 4; ++policy) {
        std::mt19937 random(static_cast<unsigned>(policy) + 100U);
        std::uniform_real_distribution<float> u(0.0F, 1.0F);
        std::vector<std::uint32_t> seq{1, 2, 3, 0};
        seq.back() = toy_target(seq, 3); // first generated token (the prefill's output)
        std::uint64_t drafter_next = 3, windows = 0, appended = 0; // drafter holds rows for positions < next
        while (seq.size() < 400U) {
            const std::size_t p = seq.size() - 1U; // position of the last committed token t
            expect(drafter_next == p, "drafter rows end right before the last token");
            // Draft: block from t at p; guesses for p + 1 .. p + 5, correct with probability 0.7.
            std::vector<std::uint32_t> guess(seq), drafts;
            std::array<float, 5> confidence{};
            for (std::uint32_t j = 0; j < block_size; ++j) {
                std::uint32_t g = toy_target(guess, guess.size());
                if (u(random) > 0.7F) g = (g + 1U) % 61U;
                guess.push_back(g);
                drafts.push_back(g);
                confidence[j] = u(random) * 4.0F - 1.0F;
            }
            const std::array<float, 6> cost{1.0F, 1.2F, 1.4F, 1.6F, 1.8F, 2.0F};
            const std::uint32_t k = policy == 3 ? choose_drafts(confidence, cost, 0) : static_cast<std::uint32_t>(policy * 2 + 1);
            drafts.resize(std::min<std::uint32_t>(k, block_size));
            // Target window: rows t, d_1..d_k; target[j] = greedy token after row j.
            std::vector<std::uint32_t> window_seq(seq.begin(), seq.end()), target;
            for (std::size_t j = 0; j <= drafts.size(); ++j) {
                target.push_back(toy_target(window_seq, window_seq.size()));
                if (j < drafts.size()) window_seq.push_back(drafts[j]);
            }
            const auto v = verify_greedy(drafts, target);
            for (std::uint32_t j = 0; j < v.accepted; ++j) seq.push_back(drafts[j]);
            seq.push_back(v.next_token);
            drafter_next += v.committed_rows; // main rows of window rows 0..accepted
            appended += v.committed_rows;
            ++windows;
        }
        seq.resize(400U);
        expect(seq == greedy, "policy " + std::to_string(policy) + ": speculative tokens equal plain greedy");
        expect(appended + 3U == drafter_next, "drafter rows are appended exactly once");
        std::cout << "speculative simulation, policy " << policy << ": " << windows << " windows for 396 tokens, tokens identical\n";
    }
}

// The speculative driver with a toy target (its position-indexed history is the target state) and a
// toy drafter that guesses with errors: tokens equal generate_greedy, rows are appended once each.
struct ToyTarget final : SpeculativeTarget {
    std::vector<std::uint32_t> seq;
    std::uint32_t step(std::uint32_t token) override {
        seq.push_back(token);
        return toy_target(seq, seq.size());
    }
    void main_hidden(std::span<float> out) const override {
        std::fill(out.begin(), out.end(), 0.0F);
        out[0] = static_cast<float>(seq.size() - 1U); // the row's position
        out[1] = static_cast<float>(seq.back());
    }
};
struct ToyDrafter final : Drafter {
    std::vector<std::uint32_t> history; // tokens of the appended rows, in position order
    std::mt19937 random{7};
    bool rows_ok = true;
    void append(std::span<const float> rows, std::uint32_t count) override {
        for (std::uint32_t r = 0; r < count; ++r) {
            rows_ok = rows_ok && rows[static_cast<std::size_t>(r) * main_width] == static_cast<float>(history.size());
            history.push_back(static_cast<std::uint32_t>(rows[static_cast<std::size_t>(r) * main_width + 1U]));
        }
    }
    Draft draft(std::uint32_t token) override {
        Draft d;
        d.tokens[0] = token;
        std::vector<std::uint32_t> guess(history);
        guess.push_back(token);
        std::uniform_real_distribution<float> u(0.0F, 1.0F);
        for (std::uint32_t j = 0; j < block_size; ++j) {
            std::uint32_t g = toy_target(guess, guess.size());
            const bool wrong = u(random) > 0.75F;
            if (wrong) g = (g + 1U) % 61U;
            guess.push_back(g);
            d.tokens[j + 1U] = g;
            // An informative but noisy confidence (overlapping ranges), as the confidence head should be.
            d.confidence[j] = wrong ? u(random) * 3.0F - 2.0F : u(random) * 3.0F;
        }
        return d;
    }
};
void test_driver() {
    const std::vector<std::uint32_t> prompt{1, 2, 3, 4, 5};
    ToyTarget plain;
    const auto greedy = generate_greedy(plain, prompt, 300);
    for (const char* window : {"fixed:0", "fixed:1", "fixed:3", "fixed:5", "auto", "survival", "controller"}) {
        WindowPolicy policy;
        if (std::string(window) == "controller") policy.controller = std::make_shared<WindowController>();
        else if (std::string(window).rfind("auto", 0) == 0) policy.automatic = true;
        else if (std::string(window).rfind("survival", 0) == 0) policy.survival = std::make_shared<SurvivalController>(policy.cost);
        else policy.drafts = static_cast<std::uint32_t>(std::stoul(std::string(window).substr(6)));
        ToyTarget target;
        ToyDrafter drafter;
        SpeculativeStats stats;
        const auto tokens = generate_speculative(target, drafter, prompt, 300, policy, &stats);
        expect(tokens == greedy, std::string("driver, window ") + window + ": tokens equal target-only greedy");
        // Every committed row is appended once, in order; the target was fed exactly the committed tokens.
        expect(drafter.rows_ok && drafter.history.size() <= target.seq.size(), std::string("driver, window ") + window + ": rows appended in order");
        std::cout << "driver, window " << window << ": " << stats.windows << " windows, " << stats.accepted << " of " << stats.verified
                  << " drafts accepted, " << stats.target_steps << " target steps, tokens identical\n";
    }
}

// The window controller: measured costs with interpolation, the initial sweep, the gate against single steps,
// exploration every 16th decision, and the same decisions for two copies fed the same inputs (the two ranks).
void test_controller() {
    WindowController::Costs c;
    c.verify[1] = 0.050F;
    c.verify[6] = 0.088F;
    c.draft = 0.007F;
    WindowController w;
    w.set_costs(c);
    expect(std::abs(w.verify_cost(3) - (0.050 + 0.038 * 2.0 / 5.0)) < 1e-6, "controller: R3 interpolated between R1 and R6");
    c.verify[6] = 0.0F;
    c.verify[4] = 0.070F;
    w.set_costs(c);
    expect(std::abs(w.verify_cost(6) - (0.070 + 2.0 * (0.070 - 0.050) / 3.0)) < 1e-6, "controller: R6 extrapolated from R1 and R4");
    c.verify[4] = 0.0F;
    w.set_costs(c);
    expect(std::abs(w.verify_cost(2) - 0.050 * 1.15) < 1e-6, "controller: only R1 measured, +15 % per row");
    // Sweep: the first five windows verify 5, 4, 3, 2, 1 drafts whatever the drafts say.
    Draft d{};
    for (std::uint32_t k = 5; k >= 1; --k) {
        expect(w.draft_next(), "controller: the sweep drafts");
        expect(w.choose(d) == k, "controller: sweep order");
        w.observe(d, k, 0);  // every first draft wrong
    }
    // With every draft wrong so far, single steps win, except the exploration decision every 16th time.
    std::uint32_t drafted = 0;
    for (std::uint32_t i = 0; i < 64; ++i) drafted += w.draft_next();
    expect(drafted >= 3U && drafted <= 5U, "controller: low survival -> single steps, explored every 16th decision");
    // Good drafts: survival rises, the gate opens and long windows win.
    WindowController g, g2;
    c.verify[6] = 0.088F;
    g.set_costs(c), g2.set_costs(c);
    for (std::uint32_t k = 5; k >= 1; --k) {
        (void)g.draft_next(), (void)g2.draft_next();
        (void)g.choose(d), (void)g2.choose(d);
    }
    for (int i = 0; i < 40; ++i) {
        d.confidence.fill(3.0F);
        g.observe(d, 5, 5), g2.observe(d, 5, 5);
    }
    expect(g.draft_next(), "controller: high survival drafts");
    (void)g2.draft_next();
    d.confidence.fill(3.0F);
    expect(g.choose(d) == 5U, "controller: high survival verifies all five drafts");
    (void)g2.choose(d);
    bool same = true;
    for (int i = 0; i < 100; ++i) {
        d.confidence.fill(static_cast<float>(i % 7) - 2.0F);
        const bool a = g.draft_next(), b = g2.draft_next();
        (void)g.draft_next(), (void)g2.draft_next();
        same = same && a == b;
        const auto ka = g.choose(d), kb = g2.choose(d);
        same = same && ka == kb;
        g.observe(d, ka, i % 4), g2.observe(d, kb, i % 4);
    }
    expect(same, "controller: two copies fed the same inputs decide alike (ranks)");
    std::cout << "controller: costs, sweep, gate, exploration and rank symmetry ok\n";
}

// The host drafter on the real weights (STRIX_DS41_Q2_GGUF, STRIX_DS41_DSPARK_GGUF): the same inputs as
// the device fixture; with STRIX_DS41_DSPARK_DUMP the inputs and outputs are written for the
// independent float64 oracle.
void run_real_host(const std::string& q2, const std::string& dspark, const std::string& dir) {
    using namespace ds41_dspark_test;
    const auto t0 = std::chrono::steady_clock::now();
    const auto model = load_model({q2, dspark});
    const auto& w = model->w;
    const auto seed = normal_values(static_cast<std::size_t>(130) * main_width, 1.0F, 77);
    dump(dir, "main_hidden_seed.f32", seed.data(), seed.size() * 4U);
    State state;
    const auto main_x = main_rows(w, seed, 130);
    append(w, state, std::span<const float>(main_x).first(static_cast<std::size_t>(128) * hidden), 128);
    append(w, state, std::span<const float>(main_x).subspan(static_cast<std::size_t>(128) * hidden), 2);
    const auto t1 = std::chrono::steady_clock::now();
    const auto d1 = draft(w, state, 1000U, true);
    const auto t2 = std::chrono::steady_clock::now();
    dump(dir, "draft1_logits.f32", d1.logits.data(), d1.logits.size() * 4U);
    dump(dir, "draft1_tokens.u32", d1.tokens.data(), d1.tokens.size() * 4U);
    for (std::uint32_t s = 0; s < stages; ++s)
        dump(dir, "draft1_stage" + std::to_string(s) + ".f32", d1.stage_out[s].data(), d1.stage_out[s].size() * 4U);
    dump(dir, "draft1_confidence.f32", d1.confidence.data(), d1.confidence.size() * 4U);
    const auto accepted = normal_values(static_cast<std::size_t>(3) * main_width, 1.0F, 78);
    dump(dir, "main_hidden_accepted.f32", accepted.data(), accepted.size() * 4U);
    append(w, state, main_rows(w, accepted, 3), 3);
    const auto d2 = draft(w, state, d1.tokens[3], true);
    dump(dir, "draft2_logits.f32", d2.logits.data(), d2.logits.size() * 4U);
    dump(dir, "draft2_tokens.u32", d2.tokens.data(), d2.tokens.size() * 4U);
    for (std::uint32_t s = 0; s < stages; ++s)
        dump(dir, "draft2_stage" + std::to_string(s) + ".f32", d2.stage_out[s].data(), d2.stage_out[s].size() * 4U);
    std::cout << "host drafter on the real weights: load + 130 main rows + seeding "
              << std::chrono::duration<double>(t1 - t0).count() << " s, one host draft step " << std::chrono::duration<double>(t2 - t1).count()
              << " s\ndraft 1 tokens";
    for (const auto t : d1.tokens) std::cout << ' ' << t;
    std::cout << ", confidence";
    for (const auto c : d1.confidence) std::cout << ' ' << c;
    std::cout << "\ndraft 2 tokens";
    for (const auto t : d2.tokens) std::cout << ' ' << t;
    std::cout << '\n';
}

} // namespace

int main() {
    test_f16();
    test_e4m3();
    test_wave_sum();
    test_sinkhorn();
    test_rope();
    test_block_attention();
    test_verify();
    test_speculative_simulation();
    test_driver();
    test_controller();
    const char* q2 = std::getenv("STRIX_DS41_Q2_GGUF");
    const char* dspark = std::getenv("STRIX_DS41_DSPARK_GGUF");
    const char* dir = std::getenv("STRIX_DS41_DSPARK_DUMP");
    if (q2 && *q2 && dspark && *dspark) run_real_host(q2, dspark, dir ? dir : "");
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "ds41 dspark CPU tests passed\n";
    return 0;
}
