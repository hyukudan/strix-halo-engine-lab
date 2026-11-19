// strix-hy4-rows: Hy4-flash verify rows on one node (hy4_rows).
//   bench (default): plain steps at G = 8 positions (logits digests), rewind, then forward_rows(R) for R = 1, 2, 4, 8 over
//     the same tokens: every row's logits digest and greedy id must equal the plain step's (ROWS_BITWISE n/15); then the
//     median wall of verify(R) and of one plain step at that position.
//   --spec N: plain greedy N ids (timed), reset, the same prompt again, speculative decode N ids through the shared
//     strix::spec layer (copy proposer + Arbiter, rows <= --max-rows): SPEC ids_equal, tok/s both, accepted per pass.
//   --mtp (with --prompt-rows): the pack's next-token block is loaded and drafts as the shared MtpChain (chain + hybrid
//     arms of the Arbiter, the copy source kept unless --mtp-copy 0); --chain-ms STEP,FLUSH seeds its walls, --survival
//     c1,c2,... its stated confidence (default: the host oracle's prose conditionals). MTP line: block calls / rows / wall.
//   --mtp-taps DIR (with --mtp, first content): prompt + --taps-gen N greedy ids through verify rows of 1 (bitwise the
//     step), the target hidden (output_norm) of every fed position written as a taps run of strix-hy4-dfly mtp; then the
//     oracle's acceptance protocol on the device block (teacher at every position, chains of --depth D at every greedy
//     anchor): MTPACC P(run >= k) / E(d), and nextn_top.txt / nextn_logits.f32 for strix-hy4-dfly mtp's device check.
// strix-hy4-rows PART.gguf... --ids-file F [--context N] [--prompt-rows N] [--reps K] [--spec N] [--max-rows R]
//   [--verify-ms a,b,...] [--gbps G] [--mtp [--mtp-copy 0|1] [--chain-ms S,F] [--survival c,...] [--mtp-taps DIR
//   [--taps-gen N] [--depth D]]]; --content LABEL=PATH (repeatable) instead of --ids-file: one spec run per content
#include "hy4/hip/hy4_hip_forward.hpp"
#include "hy4/hip/hy4_hip_rows.hpp"
#include "strix/spec/block_parallel.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <span>
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
std::uint32_t prime(hb::Forward& f, const std::vector<std::uint32_t>& prompt, bool batched,
                    hb::rows::MtpSource<hb::Forward>* mtp = nullptr) {
    f.reset();
    if (batched) {
        if (mtp) mtp->install(prompt);  // the prompt rows' hidden feed the next-token block's teacher pass
        const auto& logits = f.prefill(prompt);
        const auto anchor = argmax(logits.data(), logits.size());
        if (mtp) mtp->finish_prompt(anchor);
        return anchor;
    }
    if (mtp) throw std::invalid_argument("--mtp needs --prompt-rows (the prompt rows' hidden come from prefill)");
    std::uint32_t next = 0;
    for (const auto t : prompt) next = f.step_greedy(t);
    return next;
}

// --mtp-taps: a taps run of the device target (prompt + gen greedy ids) and the host oracle's acceptance protocol on the
// device next-token block (apps/hy4_dfly_main.cpp mtp: teacher step at every fed position p = token ids[p + 1] with the
// target hidden at p; at every greedy anchor a chain of up to `depth` drafts fed with its own argmax and its own
// shared_head_norm output, compared with the target's greedy ids; rejected positions are rewritten by the next teacher).
int mtp_taps(hb::Forward& f, const std::vector<std::uint32_t>& prompt, const std::string& dir, std::uint32_t gen, std::uint32_t depth) {
    const std::size_t h = f.config().hidden;
    const auto P0 = static_cast<std::uint32_t>(prompt.size());
    const std::uint32_t P = P0 + gen;  // fed positions
    float* hid = f.alloc_f32(std::size_t{P} * h);
    float* carry = f.alloc_f32(h);
    f.reset();
    f.set_prompt_hidden([&](const float* rows, std::uint32_t p0, std::uint32_t n) { f.copy_f32_async(hid + std::size_t{p0} * h, rows, std::size_t{n} * h); });
    const auto& logits = f.prefill(prompt);
    f.set_prompt_hidden({});
    std::vector<std::uint32_t> ids(prompt);
    ids.push_back(argmax(logits.data(), logits.size()));
    for (std::uint32_t g = 0; g < gen; ++g) {  // verify rows of 1 = the step, bitwise; its hidden rows stay readable
        const std::uint32_t tok = ids.back();
        f.forward_rows(std::span<const std::uint32_t>(&tok, 1), false);
        ids.push_back(f.rows_greedy()[0]);
        f.copy_f32_async(hid + std::size_t{P0 + g} * h, f.rows_hidden(), h);
        f.commit_rows(1);
    }
    std::vector<float> host(std::size_t{P} * h);
    f.read_f32(hid, host.size(), host.data());
    std::filesystem::create_directories(dir);
    {
        std::ofstream m(std::filesystem::path(dir) / "manifest.txt");
        m << "positions " << P << "\nprompt " << P0 << "\nhidden " << h << "\nvocab " << f.config().vocab << "\nsource device\n";
        std::ofstream t(std::filesystem::path(dir) / "ids.txt");
        for (const auto id : ids) t << id << '\n';
        std::ofstream b(std::filesystem::path(dir) / "hidden_post.f32", std::ios::binary);
        b.write(reinterpret_cast<const char*>(host.data()), static_cast<std::streamsize>(host.size() * sizeof(float)));
    }
    constexpr std::uint32_t kLogitRows = 8;
    const std::uint32_t logits_from = P > kLogitRows ? P - kLogitRows : 0U;
    std::vector<std::uint32_t> teacher(P, 0U);
    std::vector<float> lrows;
    std::vector<std::uint64_t> eligible(depth + 1U, 0U), reached(depth + 1U, 0U);
    double ms = 0.0;
    std::uint64_t calls = 0;
    const auto run = [&](std::uint32_t token, const float* hidden, std::uint32_t pos) {
        const double t0 = now_ms();
        const auto o = f.nextn_rows(&token, hidden, 1U, pos, true);
        std::uint32_t top = 0;
        f.read_ids(o.top, 1U, &top);
        f.copy_f32_async(carry, o.hidden_post, h);
        ms += now_ms() - t0;
        ++calls;
        return top;
    };
    for (std::uint32_t p = 0; p < P && p + 1U < ids.size(); ++p) {
        std::uint32_t cur = run(ids[p + 1U], hid + std::size_t{p} * h, p);  // teacher: the block's K/V at p
        teacher[p] = cur;
        if (p >= logits_from) {
            const auto& lg = f.nextn_logits();
            lrows.insert(lrows.end(), lg.begin(), lg.end());
        }
        if (p + 1U < P0 || p + 2U >= ids.size()) continue;  // anchors: the greedy region
        for (std::uint32_t k = 1; k <= depth && p + 1U + k < ids.size(); ++k) {
            ++eligible[k];
            if (cur != ids[p + 1U + k]) {
                for (std::uint32_t r = k + 1U; r <= depth && p + 1U + r < ids.size(); ++r) ++eligible[r];
                break;
            }
            ++reached[k];
            if (k == depth || p + 2U + k >= ids.size()) break;
            cur = run(cur, carry, p + k);  // chained draft: own argmax, own shared_head_norm output
        }
    }
    {
        std::ofstream t(std::filesystem::path(dir) / "nextn_top.txt");
        for (const auto id : teacher) t << id << '\n';
        std::ofstream b(std::filesystem::path(dir) / "nextn_logits.f32", std::ios::binary);
        b.write(reinterpret_cast<const char*>(lrows.data()), static_cast<std::streamsize>(lrows.size() * sizeof(float)));
        std::ofstream m(std::filesystem::path(dir) / "manifest.txt", std::ios::app);
        m << "logits_from " << logits_from << "\nlogits_rows " << (P - logits_from) << '\n';
    }
    f.free_f32(carry);
    f.free_f32(hid);
    std::printf("MTPACC device prompt %u gen %u P(run>=k)", P0, gen);
    double e = 1.0;
    std::string es;
    for (std::uint32_t k = 1; k <= depth; ++k) {
        const double pk = eligible[k] ? static_cast<double>(reached[k]) / static_cast<double>(eligible[k]) : 0.0;
        e += pk;
        std::printf(" %.3f", pk);
        es += " " + std::to_string(e).substr(0, 5);
    }
    std::printf(" | E(d=1..%u)%s | %llu block calls %.2f ms/call\n", depth, es.c_str(), static_cast<unsigned long long>(calls),
                calls ? ms / static_cast<double>(calls) : 0.0);
    std::fflush(stdout);
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        hb::ForwardOptions o;
        o.context = 2048;
        o.verify_rows = hb::rows::kMaxVerifyRows;
        std::vector<std::pair<std::string, std::vector<std::uint32_t>>> contents;  // (label, prompt ids)
        std::uint32_t reps = 10, spec_n = 0, max_rows = hb::rows::kMaxVerifyRows;
        bool union_ab = false;  // bench: rows check + strict ABAB verify timing, expert union off / on (one load)
        std::vector<double> verify_ms;
        double gbps = 207.0;
        bool mtp = false, mtp_copy = true;
        std::vector<double> chain_ms, survival;
        std::string taps_dir;
        std::uint32_t taps_gen = 128, depth = 6;
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
            else if (a == "--union-ab") union_ab = true;
            else if (a == "--mtp") mtp = true;
            else if (a == "--mtp-copy" && more) mtp_copy = std::stoul(argv[++i]) != 0UL;
            else if (a == "--chain-ms" && more) chain_ms = read_list(argv[++i]);
            else if (a == "--survival" && more) survival = read_list(argv[++i]);
            else if (a == "--mtp-taps" && more) taps_dir = argv[++i];
            else if (a == "--taps-gen" && more) taps_gen = static_cast<std::uint32_t>(std::stoul(argv[++i]));
            else if (a == "--depth" && more) depth = static_cast<std::uint32_t>(std::stoul(argv[++i]));
            else if (!a.empty() && a[0] != '-') o.parts.emplace_back(a);
            else throw std::invalid_argument("unknown argument " + a);
        }
        if (o.parts.empty() || contents.empty() || contents.front().second.empty())
            throw std::invalid_argument("usage: strix-hy4-rows PART.gguf... --ids-file F | --content L=F ... [...]");
        const auto& prompt = contents.front().second;
        if (mtp && o.prompt_rows == 0U) throw std::invalid_argument("--mtp needs --prompt-rows");
        if (!taps_dir.empty() && !mtp) throw std::invalid_argument("--mtp-taps needs --mtp");
        if (!chain_ms.empty() && chain_ms.size() != 2U) throw std::invalid_argument("--chain-ms STEP,FLUSH");
        o.nextn = mtp;
        const double t_load = now_ms();
        hb::Forward f(o);
        std::printf("load %.1f s, %u verify rows, prompt %zu ids (%s), context %u\n", (now_ms() - t_load) / 1000.0, o.verify_rows,
                    prompt.size(), o.prompt_rows ? "batched" : "position by position", o.context);
        const auto vocab = static_cast<std::size_t>(f.config().vocab);
        const bool batched = o.prompt_rows != 0U;
        if (!taps_dir.empty()) return mtp_taps(f, prompt, taps_dir, taps_gen, depth);
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
            const std::vector<int> arms = union_ab ? std::vector<int>{0, 1} : std::vector<int>{-1};
            for (const int arm : arms) {
            for (const std::uint32_t R : {1U, 2U, 4U, 8U}) {
                if (arm >= 0) f.set_rows_expert_union(arm == 1);
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
                std::printf("ROWS%s R%u logits_bitwise %u/%u greedy %u/%u\n", arm < 0 ? "" : arm == 1 ? " union" : " perrow", R, bits, R, ids,
                            R);
            }
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
                std::vector<double> ms, ms_off;  // union_ab: strict ABAB per rep (off, on); ms = the arm the seeds use (on)
                for (std::uint32_t k = 0; k < reps; ++k)
                    for (const int arm : arms) {
                        if (arm >= 0) f.set_rows_expert_union(arm == 1);
                        const double t0 = now_ms();
                        f.forward_rows(std::span<const std::uint32_t>(t.data(), R), false);
                        (arm == 0 ? ms_off : ms).push_back(now_ms() - t0);
                        f.commit_rows(0);
                    }
                if (union_ab)
                    std::printf("verify R%u perrow median %.2f ms union median %.2f ms (min %.2f / %.2f) delta %.2f ms\n", R, median(ms_off),
                                median(ms), *std::min_element(ms_off.begin(), ms_off.end()), *std::min_element(ms.begin(), ms.end()),
                                median(ms) - median(ms_off));
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
            std::unique_ptr<hb::rows::MtpSource<hb::Forward>> source;
            if (mtp) {
                source = std::make_unique<hb::rows::MtpSource<hb::Forward>>(f);
                so.copy = mtp_copy;
                if (!chain_ms.empty()) {
                    so.chain_step_ms = chain_ms[0];
                    so.chain_flush_ms = chain_ms[1];
                }
                // the host oracle's prose conditionals (hy4_recipe: P(run>=k) .714 .452 .197 .133 .034 .017)
                so.chain_survival = survival.empty() ? std::vector<double>{0.714, 0.633, 0.436, 0.675, 0.256, 0.5} : survival;
            }
            if (prime(f, ids, batched, source.get()) != first_id) throw std::runtime_error("repeat prompt anchor differs (" + label + ")");
            const auto res = hb::rows::spec_decode(f, ids, first_id, spec_n, so, source.get());
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
            if (mtp)
                std::printf("MTP %s block calls %llu rows %llu drafts %llu id reads %llu draft wall %.1f ms (%.2f ms/draft) chain %s seeds step %.1f flush %.1f
",
                            label.c_str(), static_cast<unsigned long long>(res.chain_calls), static_cast<unsigned long long>(res.chain_rows),
                            static_cast<unsigned long long>(res.chain_drafts), static_cast<unsigned long long>(res.chain_reads), res.chain_ms,
                            res.chain_drafts ? res.chain_ms / static_cast<double>(res.chain_drafts) : 0.0, so.copy ? "+copy+hybrid" : "only",
                            so.chain_step_ms, so.chain_flush_ms);
            std::fflush(stdout);
            if (!equal) rc = 3;
        }
        return rc;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strix-hy4-rows: %s\n", e.what());
        return 1;
    }
}
