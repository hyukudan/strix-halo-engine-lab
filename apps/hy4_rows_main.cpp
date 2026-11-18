// strix-hy4-rows: Hy4-flash verify rows on one node (hy4_rows).
//   bench (default): plain steps at G = 8 positions (logits digests), rewind, then forward_rows(R) for R = 1, 2, 4, 8 over
//     the same tokens: every row's logits digest and greedy id must equal the plain step's (ROWS_BITWISE n/15); then the
//     median wall of verify(R) and of one plain step at that position.
//   --spec N: plain greedy N ids (timed), reset, the same prompt again, speculative decode N ids through the shared
//     strix::spec layer (copy proposer + Arbiter, rows <= --max-rows): SPEC ids_equal, tok/s both, accepted per pass.
// strix-hy4-rows PART.gguf... --ids-file F [--context N] [--prompt-rows N] [--reps K] [--spec N] [--max-rows R]
//   [--verify-ms a,b,...] [--gbps G]; --content LABEL=PATH (repeatable) instead of --ids-file: one spec run per content
#include "hy4/hip/hy4_hip_forward.hpp"
#include "hy4/hip/hy4_hip_rows.hpp"
#include "strix/spec/block_parallel.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace hb = strix::hy4::hipbind;

std::vector<std::uint32_t> read_ids(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    std::vector<std::uint32_t> ids;
    std::string token;
    for (const char ch : ss.str() + ",") {
        if (ch >= '0' && ch <= '9') token += ch;
        else if (!token.empty()) {
            ids.push_back(static_cast<std::uint32_t>(std::stoul(token)));
            token.clear();
        }
    }
    return ids;
}
std::vector<double> read_list(const std::string& text) {
    std::vector<double> v;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) v.push_back(std::stod(item));
    return v;
}
std::uint64_t fnv(const float* v, std::size_t n) {
    std::uint64_t h = 1469598103934665603ULL;
    for (std::size_t i = 0; i < n; ++i) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, v + i, sizeof bits);
        for (int b = 0; b < 4; ++b) {
            h ^= (bits >> (8 * b)) & 0xFFU;
            h *= 1099511628211ULL;
        }
    }
    return h;
}
std::uint32_t argmax(const float* v, std::size_t n) {  // first max, lowest token on ties (the device greedy rule)
    std::size_t best = 0;
    for (std::size_t i = 1; i < n; ++i)
        if (v[i] > v[best]) best = i;
    return static_cast<std::uint32_t>(best);
}
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2U];
}
// The prompt into a fresh forward: position by position (class BASE) or the batched prompt windows (--prompt-rows).
// Returns the anchor = greedy id after the last prompt token.
std::uint32_t prime(hb::Forward& f, const std::vector<std::uint32_t>& prompt, bool batched) {
    f.reset();
    if (batched) {
        const auto& logits = f.prefill(prompt);
        return argmax(logits.data(), logits.size());
    }
    std::uint32_t next = 0;
    for (const auto t : prompt) next = f.step_greedy(t);
    return next;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        hb::ForwardOptions o;
        o.context = 2048;
        o.verify_rows = hb::rows::kMaxVerifyRows;
        std::vector<std::pair<std::string, std::vector<std::uint32_t>>> contents;  // (label, prompt ids)
        std::uint32_t reps = 10, spec_n = 0, max_rows = hb::rows::kMaxVerifyRows;
        std::vector<double> verify_ms;
        double gbps = 207.0;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            const bool more = i + 1 < argc;
            if (a == "--ids-file" && more) contents.emplace_back("prompt", read_ids(argv[++i]));
            else if (a == "--content" && more) {  // LABEL=PATH
                const std::string v = argv[++i];
                const auto eq = v.find('=');
                if (eq == std::string::npos) throw std::invalid_argument("--content LABEL=PATH");
                contents.emplace_back(v.substr(0, eq), read_ids(v.substr(eq + 1U)));
            }
            else if (a == "--context" && more) o.context = static_cast<std::uint32_t>(std::stoul(argv[++i]));
            else if (a == "--prompt-rows" && more) o.prompt_rows = static_cast<std::uint32_t>(std::stoul(argv[++i]));
            else if (a == "--reps" && more) reps = static_cast<std::uint32_t>(std::stoul(argv[++i]));
            else if (a == "--spec" && more) spec_n = static_cast<std::uint32_t>(std::stoul(argv[++i]));
            else if (a == "--max-rows" && more) max_rows = static_cast<std::uint32_t>(std::stoul(argv[++i]));
            else if (a == "--verify-ms" && more) verify_ms = read_list(argv[++i]);
            else if (a == "--gbps" && more) gbps = std::stod(argv[++i]);
            else if (!a.empty() && a[0] != '-') o.parts.emplace_back(a);
            else throw std::invalid_argument("unknown argument " + a);
        }
        if (o.parts.empty() || contents.empty() || contents.front().second.empty())
            throw std::invalid_argument("usage: strix-hy4-rows PART.gguf... --ids-file F | --content L=F ... [...]");
        const auto& prompt = contents.front().second;
        const double t_load = now_ms();
        hb::Forward f(o);
        std::printf("load %.1f s, %u verify rows, prompt %zu ids (%s), context %u\n", (now_ms() - t_load) / 1000.0, o.verify_rows,
                    prompt.size(), o.prompt_rows ? "batched" : "position by position", o.context);
        const auto vocab = static_cast<std::size_t>(f.config().vocab);
        const bool batched = o.prompt_rows != 0U;
        const std::uint32_t anchor = prime(f, prompt, batched);
        if (spec_n == 0U) {
            // ---- bench: plain digests at G positions, rows of the same tokens
            constexpr std::uint32_t G = hb::rows::kMaxVerifyRows;
            std::vector<double> curve;
            std::vector<std::uint32_t> t{anchor};
            std::vector<std::uint64_t> plain_fnv;
            std::vector<float> state;
            for (std::uint32_t i = 0; i < G; ++i) {
                const auto& logits = f.step(t[i], state);
                plain_fnv.push_back(fnv(logits.data(), logits.size()));
                t.push_back(argmax(logits.data(), logits.size()));
            }
            for (std::uint32_t i = 0; i < G; ++i) f.rewind();
            std::uint32_t equal = 0, total = 0;
            for (const std::uint32_t R : {1U, 2U, 4U, 8U}) {
                f.forward_rows(std::span<const std::uint32_t>(t.data(), R), true);
                const auto& lg = f.rows_logits();
                const auto g = f.rows_greedy();
                std::uint32_t bits = 0, ids = 0;
                for (std::uint32_t r = 0; r < R; ++r) {
                    const auto h = fnv(lg.data() + r * vocab, vocab);
                    bits += h == plain_fnv[r] ? 1U : 0U;
                    ids += g[r] == t[r + 1U] ? 1U : 0U;
                    if (h != plain_fnv[r]) std::printf("  R%u row %u fnv %016llx plain %016llx\n", R, r, static_cast<unsigned long long>(h),
                                                       static_cast<unsigned long long>(plain_fnv[r]));
                }
                f.commit_rows(0);
                equal += bits;
                total += R;
                std::printf("ROWS R%u logits_bitwise %u/%u greedy %u/%u\n", R, bits, R, ids, R);
            }
            std::printf("ROWS_BITWISE %u/%u\n", equal, total);
            std::vector<double> step_ms;
            for (std::uint32_t k = 0; k < reps; ++k) {
                const double t0 = now_ms();
                (void)f.step_greedy(t[0]);
                step_ms.push_back(now_ms() - t0);
                f.rewind();
            }
            std::printf("step median %.2f ms (min %.2f)\n", median(step_ms), *std::min_element(step_ms.begin(), step_ms.end()));
            for (const std::uint32_t R : {1U, 2U, 4U, 8U}) {
                std::vector<double> ms;
                for (std::uint32_t k = 0; k < reps; ++k) {
                    const double t0 = now_ms();
                    f.forward_rows(std::span<const std::uint32_t>(t.data(), R), false);
                    ms.push_back(now_ms() - t0);
                    f.commit_rows(0);
                }
                std::printf("verify R%u median %.2f ms (min %.2f) = %.2f ms/row\n", R, median(ms), *std::min_element(ms.begin(), ms.end()),
                            median(ms) / R);
                curve.push_back(median(ms));
            }
            // Arbiter seeds R = 1..8 from the measured R = 1, 2, 4, 8 (linear between them): --verify-ms of the spec ticket
            const double s1 = curve[0], s2 = curve[1], s4 = curve[2], s8 = curve[3];
            std::printf("seed --verify-ms %.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n", s1, s2, (s2 + s4) / 2.0, s4, s4 + (s8 - s4) / 4.0,
                        s4 + (s8 - s4) / 2.0, s4 + 3.0 * (s8 - s4) / 4.0, s8);
            return equal == total ? 0 : 2;
        }
        // ---- spec, per content: plain greedy reference, then the speculative decode from the same prompt state
        int rc = 0;
        for (std::size_t ci = 0; ci < contents.size(); ++ci) {
            const auto& [label, ids] = contents[ci];
            const std::uint32_t first_id = ci == 0U ? anchor : prime(f, ids, batched);
            std::vector<std::uint32_t> plain{first_id};
            const double p0 = now_ms();
            while (plain.size() < spec_n) plain.push_back(f.step_greedy(plain.back()));
            const double plain_ms = now_ms() - p0;
            const double step_ms = plain_ms / static_cast<double>(spec_n - 1U);
            hb::rows::SpecOptions so;
            so.max_rows = max_rows;
            so.verify_ms = verify_ms.empty() ? strix::spec::moe_verify_seed_ms(strix::spec::hy4_verify_bytes(), step_ms, gbps, max_rows) : verify_ms;
            if (prime(f, ids, batched) != first_id) throw std::runtime_error("repeat prompt anchor differs (" + label + ")");
            const auto res = hb::rows::spec_decode(f, ids, first_id, spec_n, so);
            std::size_t first_bad = std::min(res.ids.size(), plain.size());
            for (std::size_t i = 0; i < first_bad; ++i)
                if (res.ids[i] != plain[i]) {
                    first_bad = i;
                    break;
                }
            const bool equal = res.ids == plain;
            const double gen = static_cast<double>(spec_n - 1U);
            std::printf("%s seeds", label.c_str());
            for (const double v : so.verify_ms) std::printf(" %.1f", v);
            std::printf(" ms; rows histogram");
            for (std::size_t r = 1; r < res.rows_hist.size(); ++r) std::printf(" R%zu:%llu", r, static_cast<unsigned long long>(res.rows_hist[r]));
            std::printf("\nSPEC %s ids_equal %d (first diff %zu) plain %.2f tok/s (%.2f ms/token) spec %.2f tok/s passes %zu E %.3f proposed %llu accepted %llu\n",
                        label.c_str(), equal ? 1 : 0, first_bad, 1000.0 * gen / plain_ms, step_ms, 1000.0 * gen / res.wall_ms, res.steps.size(),
                        res.steps.empty() ? 0.0 : gen / static_cast<double>(res.steps.size()), static_cast<unsigned long long>(res.proposed),
                        static_cast<unsigned long long>(res.accepted));
            std::fflush(stdout);
            if (!equal) rc = 3;
        }
        return rc;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strix-hy4-rows: %s\n", e.what());
        return 1;
    }
}
