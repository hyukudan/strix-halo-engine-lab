// MiniMax DSpark drafter, CPU tests: declared helpers, the verify protocol and a speculative decode simulation (tokens
// identical to greedy for every window policy), and with STRIX_MM_DSPARK_DIR + STRIX_MM_DSPARK_CASE the host drafter on
// the real draft weights against the PyTorch float64 reference (ref/dspark_reference.py), writing host fixtures
// (host_y.f32, host.json) for the device test.
#include "../minimax_dspark.hpp"
#include "../minimax_ngram.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace strix::minimax::dspark;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

template <class T>
std::vector<T> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.seekg(0, std::ios::end);
    const auto n = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    std::vector<T> v(n / sizeof(T));
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
    return v;
}
template <class T>
void write_file(const std::string& path, const std::vector<T>& v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
}
// The few integers / arrays of a flat JSON written by the reference script.
std::vector<double> json_array(const std::string& text, const std::string& key) {
    const auto at = text.find("\"" + key + "\"");
    if (at == std::string::npos) throw std::runtime_error("no " + key);
    const auto b = text.find('[', at), e = text.find(']', b);
    std::vector<double> out;
    const char* p = text.c_str() + b + 1;
    const char* end = text.c_str() + e;
    while (p < end) {
        char* stop = nullptr;
        const double d = std::strtod(p, &stop);
        if (stop == p) { ++p; continue; }
        out.push_back(d);
        p = stop;
    }
    return out;
}
double json_number(const std::string& text, const std::string& key) {
    const auto at = text.find("\"" + key + "\"");
    if (at == std::string::npos) throw std::runtime_error("no " + key);
    return std::strtod(text.c_str() + text.find(':', at) + 1, nullptr);
}
std::string read_text(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

void test_helpers() {
    // E4M3fn: 0x38 = 1.0, 0x7e = 448, 0x01 = 2^-9, 0xb8 = -1.0, 0x7f NaN.
    check(e4m3_value(0x38) == 1.0F && e4m3_value(0x7e) == 448.0F && e4m3_value(0x01) == 0.001953125F &&
              e4m3_value(0xb8) == -1.0F && std::isnan(e4m3_value(0x7f)),
          "e4m3fn decode table points");
    bool all = true;
    for (int c = 0; c < 256; ++c) {
        if ((c & 0x7f) == 0x7f) continue;
        const int e = (c >> 3) & 15, m = c & 7;
        const double ref = (e == 0 ? m / 8.0 * std::ldexp(1.0, -6) : (1.0 + m / 8.0) * std::ldexp(1.0, e - 7)) * ((c & 0x80) ? -1 : 1);
        all &= static_cast<double>(e4m3_value(static_cast<std::uint8_t>(c))) == ref;
    }
    check(all, "e4m3fn decode == formula for all 254 finite codes");
    check(e8m0_value(127) == 1.0F && e8m0_value(0) == std::ldexp(1.0F, -127) && e8m0_value(254) == std::ldexp(1.0F, 127),
          "e8m0 scale 2^(c-127) incl. code 0 (subnormal)");
    check(bf16_bits(1.0F) == 0x3f80 && bf16_bits(1.00390625F) == 0x3f80 && bf16_bits(1.01171875F) == 0x3f82 &&
              bf16_value(bf16_bits(-3.5F)) == -3.5F,
          "bf16 round to nearest even");
    double worst = 0;
    for (float x = -87.0F; x < 88.0F; x += 0.0137F)  // normal results only (subnormals lose relative precision)
        worst = std::max(worst, std::fabs(static_cast<double>(declared_exp(x)) / std::exp(static_cast<double>(x)) - 1.0));
    check(worst < 4e-7, "declared_exp rel err " + std::to_string(worst));
    check(std::fabs(swiglu_oai(10.0F, 9.0F) - 7.0 / (1.0 + std::exp(-1.702 * 7.0)) * 8.0) < 1e-5 &&
              std::fabs(swiglu_oai(-2.0F, -9.0F) - (-2.0 / (1.0 + std::exp(3.404)) * -6.0)) < 1e-5,
          "swiglu_oai clamps (gate <= 7, up in [-7, 7])");
}

void test_protocol() {
    const std::vector<std::uint32_t> d = {5, 6, 7}, g1 = {5, 6, 9, 1}, g2 = {4, 6, 7, 8}, g3 = {5, 6, 7, 8};
    const Verdict a = verify_greedy(d, g1), b = verify_greedy(d, g2), c = verify_greedy(d, g3);
    check(a.accepted == 2 && a.emitted == std::vector<std::uint32_t>({5, 6, 9}), "verify: partial accept + bonus");
    check(b.accepted == 0 && b.emitted == std::vector<std::uint32_t>({4}), "verify: reject first -> 1 token");
    check(c.accepted == 3 && c.emitted == g3, "verify: full accept -> k + 1 tokens");
    const std::vector<std::uint32_t> none;
    check(verify_greedy(none, std::vector<std::uint32_t>{9}).emitted == std::vector<std::uint32_t>({9}), "verify: k = 0");
    // choose_window: high confidence and flat cost -> longest; low confidence -> 0.
    const std::vector<float> hi(7, 0.95F), lo(7, 0.05F);
    std::vector<double> cost = {1.0};
    for (int k = 1; k <= 7; ++k) cost.push_back(1.2 + 0.05 * k);
    check(choose_window(hi, cost, 0) == 7 && choose_window(lo, cost, 0) == 0 && choose_window(lo, cost, 2) == 2,
          "choose_window by expected tokens per cost");
}

// Speculative decoding against a toy target: next = f(prefix) (deterministic hash), drafter right with probability q
// per position. The emitted stream must equal plain greedy for every policy.
void test_survival() {
    // Depth 0 always accepted, depth 1 never: after learning, a cheap window stops at k = 1.
    SurvivalController sc(0);
    const std::vector<float> conf(7, 0.5F);
    for (int i = 0; i < 200; ++i) sc.observe(conf, 7, 1);
    std::vector<double> cost = {1.0};
    for (int k = 1; k <= 7; ++k) cost.push_back(1.1 + 0.1 * k);
    check(sc.choose(conf, cost) == 7, "survival controller: the first window is a full exploration");
    check(sc.probability(0, 0.5F) > 0.98 && sc.probability(1, 0.5F) < 0.02 && sc.choose(conf, cost) == 1,
          "survival controller learns per-depth acceptance and picks k = 1");
    SurvivalController all(0);
    for (int i = 0; i < 200; ++i) all.observe(conf, 7, 7);
    (void)all.choose(conf, cost);
    check(all.choose(conf, cost) == 7, "survival controller: everything accepted -> full window");
    SurvivalController ex(4);
    int fulls = 0;
    std::vector<std::uint32_t> ks;
    for (int i = 0; i < 8; ++i) ks.push_back(ex.choose(conf, cost));  // the prior alone picks k = 4 or 5 here
    for (std::size_t i = 1; i < ks.size(); ++i) fulls += ks[i] > ks[1] ? 1 : 0;
    check(ks[0] == 7 && ks[1] <= 5 && fulls == 2 && ks[3] == std::min<std::uint32_t>(7, ks[1] + 2),
          "survival controller: first window full, every 4th call best + 2");
    // Censoring: windows of 3 always fully accepted (depths 3-6 never observed) must still open the full window, since
    // each cell borrows its bin's pooled rate (mmbench B json: the per-cell prior alone kept k < 7).
    SurvivalController cens(0);
    (void)cens.choose(conf, cost);
    for (int i = 0; i < 10; ++i) cens.observe(conf, 3, 3);
    check(cens.choose(conf, cost) == 7, "survival controller: fully accepted short windows -> full window");
}

// Draft gate: a losing drafter pauses after 12 drafting cycles for 8, then 16 cycles; a winning one never pauses.
void test_draft_gate() {
    DraftGate lose;
    std::vector<int> pauses;
    int run = 0;
    for (int c = 0; c < 80; ++c) {
        const bool d = lose.draft();
        if (!d) ++run;
        if (d && run > 0) {
            pauses.push_back(run);
            run = 0;
        }
        const std::uint32_t p = d ? lose.proposal() : 0;  // decided on a drafting cycle, applied after it
        // drafting: 1 token in 74 ms; no draft: 1 token in 55 ms
        lose.observe(d, 1, d ? 0.074 : 0.055, 0.057);
        lose.pause(p);
    }
    check(pauses.size() >= 2 && pauses[0] == 8 && pauses[1] == 16, "draft gate: losing drafter pauses 8, then 16 cycles");
    DraftGate win;
    bool paused = false;
    for (int c = 0; c < 300; ++c) {
        const bool d = win.draft();
        paused = paused || !d;
        const std::uint32_t p = d ? win.proposal() : 0;
        win.observe(d, (c % 3 == 0) ? 1U : 5U, 0.12, 0.057);  // mean 3.7 tokens per 120 ms vs 2.1 plain
        win.pause(p);
    }
    check(!paused && win.paused_cycles() == 0, "draft gate: winning drafter never pauses");
}

void test_simulation() {
    const auto target = [](const std::vector<std::uint32_t>& seq) {
        std::uint64_t h = 1469598103934665603ULL;
        for (std::size_t i = seq.size() > 6 ? seq.size() - 6 : 0; i < seq.size(); ++i) h = (h ^ seq[i]) * 1099511628211ULL;
        return static_cast<std::uint32_t>(h % 50U);
    };
    std::vector<std::uint32_t> greedy = {1, 2, 3};
    while (greedy.size() < 3 + 300) greedy.push_back(target(greedy));
    bool all = true;
    std::mt19937 rng(7);
    for (int policy = 0; policy <= 9; ++policy) {
        for (double q : {0.0, 0.5, 0.9, 1.0}) {
            std::vector<std::uint32_t> seq = {1, 2, 3};
            while (seq.size() < greedy.size()) {
                const std::uint32_t k = policy <= 7 ? static_cast<std::uint32_t>(policy) : rng() % 8U;
                std::vector<std::uint32_t> drafts, probe = seq, gr;
                // drafter: guesses the target chain with error rate 1 - q (wrong token = +1)
                std::vector<std::uint32_t> guess = seq;
                for (std::uint32_t i = 0; i < k; ++i) {
                    std::uint32_t t = target(guess);
                    if (std::uniform_real_distribution<double>(0, 1)(rng) >= q) t = (t + 1U) % 50U;
                    drafts.push_back(t);
                    guess.push_back(t);
                }
                // verify window: target argmax after each row of [last, d_1..d_k] (exact causal prefix)
                for (std::uint32_t j = 0; j <= k; ++j) {
                    gr.push_back(target(probe));
                    if (j < k) probe.push_back(drafts[j]);
                }
                const Verdict v = verify_greedy(drafts, gr);
                for (auto t : v.emitted) seq.push_back(t);
            }
            seq.resize(greedy.size());
            all &= seq == greedy;
        }
    }
    check(all, "speculative simulation == greedy (policies fixed:0..7, random, two random; q 0/0.5/0.9/1)");
}

void test_copy_drafter() {
    // A 20-token passage, then its first 10 tokens again: the drafter must propose the next 10 (anchor 8).
    std::vector<std::uint32_t> text;
    for (std::uint32_t i = 0; i < 20; ++i) text.push_back(100 + i);
    CopyDrafter c(8);
    c.append(text);
    c.append(std::span<const std::uint32_t>(text.data(), 10));
    const auto p = c.propose(7);
    check(p == std::vector<std::uint32_t>({110, 111, 112, 113, 114, 115, 116}), "copy drafter: continues the earlier passage");
    // Two records share the anchor and then differ at the 3rd token: the draft stops at the divergence.
    CopyDrafter r(4);
    const std::vector<std::uint32_t> rec1 = {1, 2, 3, 4, 50, 51, 60, 9}, rec2 = {1, 2, 3, 4, 50, 51, 70, 9};
    r.append(rec1);
    r.append(rec2);
    r.append(std::span<const std::uint32_t>(rec1.data(), 4));
    check(r.propose(7) == std::vector<std::uint32_t>({50, 51}), "copy drafter: ambiguous anchor stops where records diverge");
    // In-chain continuation: a 14-token phrase seen before, the present ends with its first 13 tokens split between the
    // history (10) and pending drafts (3): match 13 >= 12 -> the 14th token then what followed it.
    CopyDrafter ch(8);
    std::vector<std::uint32_t> phrase;
    for (std::uint32_t i = 0; i < 20; ++i) phrase.push_back(500 + i);
    ch.append(phrase);
    ch.append(std::vector<std::uint32_t>{7, 7});
    ch.append(std::span<const std::uint32_t>(phrase.data(), 10));
    const std::vector<std::uint32_t> pend(phrase.begin() + 10, phrase.begin() + 13);
    check(ch.continuation(pend, 4, 12) == std::vector<std::uint32_t>({513, 514, 515, 516}), "copy continuation: history + drafts, match 13 >= 12");
    check(ch.continuation(std::span<const std::uint32_t>(pend.data(), 1), 4, 12).empty(), "copy continuation: match 11 < 12 -> none");
    // Self-repetition: "a b c a b c a b c" -> continues the cycle past the present.
    CopyDrafter rep(8);
    rep.append(std::vector<std::uint32_t>{9, 1, 2, 3, 1, 2, 3, 1, 2, 3});
    check(rep.continuation({}, 7, 5) == std::vector<std::uint32_t>({1, 2, 3, 1, 2, 3, 1}), "copy continuation: self-repetition through the proposal");
    CopyDrafter none(8);
    none.append(text);
    check(none.propose(7).empty(), "copy drafter: no earlier occurrence -> no draft");
}

void test_real() {
    const char* dir = std::getenv("STRIX_MM_DSPARK_DIR");
    const char* case_env = std::getenv("STRIX_MM_DSPARK_CASE");
    if (dir == nullptr || case_env == nullptr) {
        std::printf("SKIP real weights (set STRIX_MM_DSPARK_DIR and STRIX_MM_DSPARK_CASE)\n");
        return;
    }
    const auto file = DraftFile::open(dir);
    const Weights& w = file->weights();
    std::printf("draft %s: %.3f GB\n", dir, static_cast<double>(file->bytes()) / 1e9);
    std::string cases = case_env;
    for (std::size_t b = 0; b < cases.size();) {
        const std::size_t e = std::min(cases.find(',', b), cases.size());
        const std::string cs = cases.substr(b, e - b);
        b = e + 1;
        const std::string meta = read_text(cs + "/meta.json");
        const auto n = static_cast<std::size_t>(json_number(meta, "context"));
        const auto P = static_cast<std::uint64_t>(json_number(meta, "position"));
        const auto anchor = static_cast<std::uint32_t>(json_number(meta, "anchor"));
        const auto taps_v = read_file<float>(cs + "/taps.f32");
        const auto embed = read_file<float>(cs + "/embed.f32");
        const auto base = read_file<float>(cs + "/base.f32");
        Ring ring;
        // Append in uneven pieces (as committed verify windows would), oldest rows past the window skipped.
        const std::size_t start = n > context_keys ? n - context_keys : 0;
        for (std::size_t p = start; p < n;) {
            const std::size_t m = std::min<std::size_t>(n - p, 1 + (p % 8));
            append_context(w, ring, taps_v.data() + p * tap_width, m, p);
            p += m;
        }
        std::vector<float> y(static_cast<std::size_t>(block) * hidden);
        block_forward(w, ring, embed.data(), P, y.data());
        const Draft d = sample(w, anchor, base.data(), y.data());
        const auto ref_y = read_file<double>(cs + "/ref_y.f64");
        const std::string ref = read_text(cs + "/ref.json");
        double num = 0, den = 0, mx = 0;
        for (std::size_t i = 0; i < y.size(); ++i) {
            const double diff = static_cast<double>(y[i]) - ref_y[i];
            num += diff * diff;
            den += ref_y[i] * ref_y[i];
            mx = std::max(mx, std::fabs(diff));
        }
        const double rel = std::sqrt(num / den);
        // Gate: within twice the deviation of an FP32 PyTorch run of the same math (ref_fp32_y.f64, if present) from the
        // float64 reference; BF16 K/V storage makes the draft's FP32 sensitivity ~3e-4 (one-ulp flips near midpoints).
        double tol = 1e-4;
        if (std::ifstream(cs + "/ref_fp32_y.f64")) {
            const auto f32 = read_file<double>(cs + "/ref_fp32_y.f64");
            double n2 = 0;
            for (std::size_t i = 0; i < f32.size(); ++i) n2 += (f32[i] - ref_y[i]) * (f32[i] - ref_y[i]);
            tol = std::max(tol, 2.0 * std::sqrt(n2 / den));
        }
        const auto rd = json_array(ref, "drafts"), rc = json_array(ref, "confidence");
        bool same = rd.size() == block;
        double cmax = 0;
        for (std::uint32_t k = 0; k < block && same; ++k) {
            same &= static_cast<std::uint32_t>(rd[k]) == d.tokens[k];
            cmax = std::max(cmax, std::fabs(static_cast<double>(d.confidence[k]) - rc[k]));
        }
        char line[256];
        std::snprintf(line, sizeof line, "real %s: y rel_l2 %.3e (tol %.2e) max_abs %.3e, drafts %s, confidence max diff %.2e", cs.c_str(), rel,
                      tol, mx, same ? "identical" : "DIFFER", cmax);
        check(rel <= tol && same && cmax < 1e-3, line);
        std::printf("  drafts:");
        for (auto t : d.tokens) std::printf(" %u", t);
        std::printf("\n  confidence:");
        for (auto c : d.confidence) std::printf(" %.4f", c);
        std::printf("\n");
        write_file(cs + "/host_y.f32", y);
        // The v1 attention order (one chunk over every key) for the device test of the v1 kernel.
        std::vector<float> y1(y.size());
        block_forward(w, ring, embed.data(), P, y1.data(), 0xffffffffU);
        write_file(cs + "/host_y_v1.f32", y1);
        std::ofstream h(cs + "/host.json");
        h << "{\"drafts\": [";
        for (std::uint32_t k = 0; k < block; ++k) h << (k ? ", " : "") << d.tokens[k];
        h << "], \"confidence_bits\": [";
        for (std::uint32_t k = 0; k < block; ++k) h << (k ? ", " : "") << __builtin_bit_cast(std::uint32_t, d.confidence[k]);
        h << "]}\n";
    }
}

} // namespace

int main() {
    try {
        test_helpers();
        test_protocol();
        test_simulation();
        test_survival();
        test_draft_gate();
        test_copy_drafter();
        test_real();
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 2;
    }
    std::printf("%s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
