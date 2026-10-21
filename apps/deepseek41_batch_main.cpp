#include "strix/deepseek41_batch_forward.hpp"
#include "strix/deepseek41_batch_schedule.hpp"
#include "strix/dist/communicator.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

// Several requests decoded together on the two-node deepseek41 runner (BatchForward), or both ranks in one process
// (--transport emulate). Gate and measurement tool:
//   strix-deepseek41-batch MODEL.gguf --rank R --transport usb4auto|tcp-listen:P|tcp-connect:H:P|emulate
//       --prompts a.ids,b.ids,... --n 1,2,4,8 --generate G [--mode batch|alone|both] [--prompt-rows 8]
//       [--fill K --fill-ids FILE] [--layers L] [--max-positions P] [--tp-attention] [--engram-direct]
//       [--check] [--gather] [--print-rows] [--sync-start] [--pool N] [--context-state FILE]
// Request i of a run reads prompt i (mod the list). --mode both runs every request alone first (one row per step in
// decode, its prompt in chunks of --prompt-rows), then all N together, and compares, per request and position, the
// fed token, the greedy token and (with --check) the FNV of this rank's logits slice (--gather: of the full logits):
// "GATE PASS" only if all are equal. --mode windows: verify windows of all N requests in one step (see windows_gate). --fill: a context of K tokens (cycled from FILE) read once into a template slot
// and copied into every request's slot before its prompt (decode measured after a long context). --pool N: the slots'
// KV caches come from a shared pool of N positions (BatchForwardOptions::pool_positions); every request reserves its
// fill + prompt + generated positions (the tokens must not change). --context-state FILE: every request starts from the
// attention state of FILE (a long prefilled context, strix-deepseek41-longctx --state-out) imported into its slot; the
// prompts must be the tokens that follow it.
namespace {

using strix::deepseek41::BatchForward;
using strix::deepseek41::BatchRow;
using strix::deepseek41::BatchSequence;

struct Args {
    std::string model, transport = "emulate", prompts, fill_ids, mode = "batch", n_list = "1,2,4,8";
    std::uint32_t rank = 0, layers = 40, generate = 32, max_positions = 4096, fill = 0, prompt_rows = 8;
    unsigned connect_timeout = 60;
    bool tp = false, engram_direct = false, check = false, gather = false, print_rows = false;
    bool sync_start = false;  // timing: every prompt is read before any request decodes (N rows in every decode step)
    std::uint32_t pool = 0;   // BatchForwardOptions::pool_positions
    std::string context_state;
    const strix::deepseek41::AttentionState* context = nullptr;  // loaded --context-state
};

std::uint64_t fnv(std::span<const float> v) {
    std::uint64_t h = 14695981039346656037ULL;
    for (const float f : v) {
        std::uint32_t b;
        std::memcpy(&b, &f, 4);
        for (int k = 0; k < 4; ++k, b >>= 8U) h = (h ^ (b & 0xffU)) * 1099511628211ULL;
    }
    return h;
}

std::vector<std::uint32_t> read_ids(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::invalid_argument("cannot read " + path);
    std::vector<std::uint32_t> ids;
    for (std::string t; std::getline(f, t, ',');) {
        const auto b = t.find_first_not_of(" \r\n\t");
        if (b != std::string::npos) ids.push_back(static_cast<std::uint32_t>(std::stoul(t.substr(b))));
    }
    if (ids.empty()) throw std::invalid_argument(path + " holds no token ids");
    return ids;
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, sep);)
        if (!t.empty()) out.push_back(t);
    return out;
}

std::unique_ptr<strix::dist::Transport> open_transport(const std::string& spec, std::uint32_t rank, unsigned timeout) {
    const std::chrono::seconds limit(timeout);
    const auto colon = spec.find(':');
    const std::string kind = spec.substr(0, colon), rest = colon == std::string::npos ? "" : spec.substr(colon + 1);
    if (kind == "tcp-listen") return strix::dist::tcp_listen(static_cast<std::uint16_t>(std::stoul(rest)), limit);
    if (kind == "tcp-connect") {
        const auto c2 = rest.rfind(':');
        return strix::dist::tcp_connect(rest.substr(0, c2), static_cast<std::uint16_t>(std::stoul(rest.substr(c2 + 1))), limit);
    }
    if (kind == "usb4auto") {
        strix::dist::Usb4AutoOptions o;
        o.rank = rank;
        o.timeout = limit;
        strix::dist::Usb4Topology topology;
        auto t = strix::dist::usb4_host_auto(o, &topology);
        std::printf("usb4 topology: %s\n", topology.receipt().c_str());
        return t;
    }
    throw std::invalid_argument("unknown or unavailable transport " + spec);
}

// One row's record: the request's position, the token fed, the greedy token and the logits hash.
struct RowRecord {
    std::uint32_t fed = 0, argmax = 0;
    std::uint64_t hash = 0;
};
using Trace = std::map<std::uint32_t, RowRecord>;  // by position, per request

struct RunResult {
    std::vector<Trace> traces;
    std::uint64_t steady_steps = 0, steady_rows = 0;
    double steady_seconds = 0.0, steady_exchange = 0.0, steady_engram = 0.0;
    double steady_union = 0.0;        // mean over steps of the mean union per layer (this rank)
    double steady_max_rows = 0.0;     // mean over steps of the mean max rows of one expert
    double steady_expert_bytes = 0.0; // per step
    std::uint64_t steps = 0;
    double seconds = 0.0;
};

// A fresh request in `slot` with room for `positions` (the KV pool's reservation, or a reset without a pool).
void claim(BatchForward& fwd, const Args& a, std::uint32_t slot, std::uint64_t positions) {
    const auto want = static_cast<std::uint32_t>(std::min<std::uint64_t>(positions, a.max_positions));
    if (!fwd.reserve(slot, want)) throw std::runtime_error("no room for slot " + std::to_string(slot) + " (" + std::to_string(want) + " positions)");
}

// Runs `requests` (prompt per request) together; slots 0..n-1, each starting from the template slot when filling.
RunResult run(BatchForward& fwd, const Args& a, const std::vector<std::vector<std::uint32_t>>& prompts, std::uint32_t n,
              std::uint32_t template_slot, bool filled, bool print) {
    std::vector<BatchSequence> seqs(n);
    RunResult res;
    res.traces.resize(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        claim(fwd, a, i, std::uint64_t{a.fill} + (a.context ? a.context->position : 0U) + prompts[i].size() + a.generate + 16U);
        if (a.context) fwd.import_state(i, *a.context);
        else if (filled) fwd.copy_slot(template_slot, i);
        seqs[i].slot = i;
        seqs[i].position = fwd.position(i);
        seqs[i].prompt = prompts[i];
    }
    std::vector<std::uint32_t> map_back(n);
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        std::vector<BatchSequence*> active;
        std::vector<std::uint32_t> index;
        for (std::uint32_t i = 0; i < n; ++i)
            if (seqs[i].generated.size() < a.generate) {
                active.push_back(&seqs[i]);
                index.push_back(i);
            }
        if (active.empty()) break;
        if (a.sync_start && std::any_of(active.begin(), active.end(), [](const BatchSequence* q) { return q->reading_prompt(); })) {
            std::vector<BatchSequence*> reading;
            std::vector<std::uint32_t> reading_index;
            for (std::size_t k = 0; k < active.size(); ++k)
                if (active[k]->reading_prompt()) reading.push_back(active[k]), reading_index.push_back(index[k]);
            active = std::move(reading);
            index = std::move(reading_index);
        }
        const auto plan = strix::deepseek41::plan_step(active, a.prompt_rows);
        std::vector<std::uint32_t> positions(plan.rows.size());
        for (std::size_t r = 0; r < plan.rows.size(); ++r) positions[r] = active[plan.owner[r]]->position;  // fixed up below
        {  // the position of every row: its sequence's position plus its rank inside the segment
            std::map<std::uint32_t, std::uint32_t> next;
            for (std::size_t r = 0; r < plan.rows.size(); ++r) {
                auto [it, fresh] = next.emplace(plan.owner[r], positions[r]);
                positions[r] = it->second++;
                (void)fresh;
            }
        }
        const auto& tokens = fwd.step(plan.rows);
        const auto& st = fwd.last_stats();
        bool decode_only = plan.rows.size() == n;
        for (std::size_t r = 0; r < plan.rows.size(); ++r) {
            const std::uint32_t req = index[plan.owner[r]];
            RowRecord rec{plan.rows[r].token, tokens[r], 0};
            if (a.check) rec.hash = fnv(fwd.row_logits(static_cast<std::uint32_t>(r)));
            res.traces[req][positions[r]] = rec;
            if (active[plan.owner[r]]->reading_prompt()) decode_only = false;
            if (print && a.print_rows)
                std::printf("req %u pos %u fed %u argmax %u logits_fnv %016llx\n", req, positions[r], rec.fed, rec.argmax,
                            static_cast<unsigned long long>(rec.hash));
        }
        strix::deepseek41::apply_step(active, plan, tokens);
        ++res.steps;
        if (decode_only && res.steps > 1U) {
            ++res.steady_steps;
            res.steady_rows += plan.rows.size();
            res.steady_seconds += st.seconds;
            res.steady_exchange += st.exchange_seconds;
            res.steady_engram += st.engram_seconds;
            double u = 0.0, mr = 0.0;
            for (std::size_t l = 0; l < st.union_experts.size(); ++l) {
                u += st.union_experts[l];
                mr += st.max_rows_per_expert[l];
            }
            res.steady_union += u / static_cast<double>(std::max<std::size_t>(st.union_experts.size(), 1U));
            res.steady_max_rows += mr / static_cast<double>(std::max<std::size_t>(st.union_experts.size(), 1U));
            res.steady_expert_bytes += static_cast<double>(st.expert_bytes);
        }
    }
    res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return res;
}

// Verify windows of N requests in one step (the DSpark protocol with an oracle drafter): every decoding request
// proposes a window of w = 8 / N rows, its last token and the next w - 1 tokens of its run alone, with the draft at a
// random depth replaced by a wrong token; the step verifies every window greedily, commit() keeps the accepted rows
// and the correction. Every request must end with exactly its tokens alone (and, with --check, the same logits at
// every kept position).
bool windows_gate(BatchForward& fwd, const Args& a, const std::vector<std::vector<std::uint32_t>>& prompts, std::uint32_t n,
                  const std::vector<Trace>& alone, bool print) {
    const std::uint32_t w = BatchForward::max_rows / n;
    std::vector<BatchSequence> seqs(n);
    std::vector<std::vector<std::uint32_t>> truth(n);  // the tokens each request generates alone
    for (std::uint32_t i = 0; i < n; ++i) {
        claim(fwd, a, i, prompts[i].size() + a.generate + 16U);
        seqs[i].slot = i;
        seqs[i].prompt = prompts[i];
        for (const auto& [pos, rec] : alone[i])
            if (pos + 1U >= prompts[i].size()) truth[i].push_back(rec.argmax);
    }
    std::uint64_t state = 0x9E3779B97F4A7C15ULL;
    const auto rnd = [&] {
        state ^= state << 13U;
        state ^= state >> 7U;
        state ^= state << 17U;
        return state;
    };
    std::vector<Trace> traces(n);
    std::uint64_t steps = 0, windows = 0, rows_total = 0, rejected = 0;
    for (;;) {
        std::vector<BatchRow> rows;
        std::vector<std::uint32_t> owner, depth;
        std::vector<std::vector<std::uint32_t>> drafts(n);
        bool any = false;
        for (std::uint32_t i = 0; i < n; ++i) {
            auto& q = seqs[i];
            if (q.generated.size() >= a.generate) continue;
            any = true;
            if (q.reading_prompt()) {  // prompt chunks as in the batch runs
                std::size_t take = std::min<std::size_t>({a.prompt_rows, BatchForward::max_rows - rows.size(), q.prompt.size() - q.fed});
                if (q.position == 0U) take = std::min<std::size_t>(take, 1U);
                if (take == 0U) continue;
                for (std::size_t k = 0; k < take; ++k) {
                    rows.push_back({q.slot, q.prompt[q.fed + k]});
                    owner.push_back(i);
                    depth.push_back(static_cast<std::uint32_t>(k));
                }
                continue;
            }
            // window: the last token, then drafts = the request's own next tokens, one of them wrong
            const std::size_t g = q.generated.size();
            const std::uint32_t room = static_cast<std::uint32_t>(std::min<std::size_t>({w, BatchForward::max_rows - rows.size(), a.generate - g}));
            if (room == 0U) continue;
            const std::uint32_t wrong = static_cast<std::uint32_t>(rnd() % (room + 1U));  // == room: no wrong draft
            rows.push_back({q.slot, q.generated.back()});
            owner.push_back(i);
            depth.push_back(0U);
            for (std::uint32_t d = 1; d < room; ++d) {
                std::uint32_t t = g + d - 1U < truth[i].size() ? truth[i][g + d - 1U] : 0U;
                if (d == wrong) t = (t + 1U) % fwd.model().config.vocab_size;
                drafts[i].push_back(t);
                rows.push_back({q.slot, t});
                owner.push_back(i);
                depth.push_back(d);
            }
            ++windows;
        }
        if (!any) break;
        const auto& tokens = fwd.step(rows);
        ++steps;
        rows_total += rows.size();
        // per request: the accepted prefix, the commit, the emitted tokens
        for (std::uint32_t i = 0; i < n; ++i) {
            std::vector<std::size_t> mine;
            for (std::size_t r = 0; r < rows.size(); ++r)
                if (owner[r] == i) mine.push_back(r);
            if (mine.empty()) continue;
            auto& q = seqs[i];
            const std::uint32_t base = q.position;
            if (q.reading_prompt()) {
                for (std::size_t k = 0; k < mine.size(); ++k) {
                    const auto r = mine[k];
                    RowRecord rec{rows[r].token, tokens[r], a.check ? fnv(fwd.row_logits(static_cast<std::uint32_t>(r))) : 0U};
                    traces[i][base + static_cast<std::uint32_t>(k)] = rec;
                }
                q.position += static_cast<std::uint32_t>(mine.size());
                q.fed += mine.size();
                if (!q.reading_prompt()) q.generated.push_back(tokens[mine.back()]);
                continue;
            }
            std::size_t kept = 1;  // row 0 always; row k is kept when row k-1's greedy token equals draft k
            while (kept < mine.size() && tokens[mine[kept - 1U]] == drafts[i][kept - 1U]) ++kept;
            rejected += mine.size() - kept;
            fwd.commit(q.slot, static_cast<std::uint32_t>(kept));
            for (std::size_t k = 0; k < kept; ++k) {
                const auto r = mine[k];
                RowRecord rec{rows[r].token, tokens[r], a.check ? fnv(fwd.row_logits(static_cast<std::uint32_t>(r))) : 0U};
                traces[i][base + static_cast<std::uint32_t>(k)] = rec;
                if (q.generated.size() < a.generate) q.generated.push_back(tokens[r]);
            }
            q.position += static_cast<std::uint32_t>(kept);
        }
    }
    std::uint64_t compared = 0, differ = 0;
    for (std::uint32_t i = 0; i < n; ++i)
        for (const auto& [pos, rec] : traces[i]) {
            const auto it = alone[i].find(pos);
            if (it == alone[i].end()) continue;  // past the alone run's last fed row
            ++compared;
            if (it->second.fed != rec.fed || it->second.argmax != rec.argmax || it->second.hash != rec.hash) {
                if (print && differ < 10U)
                    std::printf("  DIFF windows req %u pos %u: alone fed %u argmax %u, windows fed %u argmax %u\n", i, pos, it->second.fed,
                                it->second.argmax, rec.fed, rec.argmax);
                ++differ;
            }
        }
    for (std::uint32_t i = 0; i < n; ++i) {
        const std::vector<std::uint32_t> head(truth[i].begin(), truth[i].begin() + static_cast<std::ptrdiff_t>(std::min<std::size_t>(truth[i].size(), a.generate)));
        if (seqs[i].generated != head) ++differ;
    }
    if (print)
        std::printf("GATE windows N %u (w %u) %s: %llu steps, %llu windows, %llu rows (%llu rejected), %llu kept rows compared, %llu differ\n", n, w,
                    differ == 0U ? "PASS" : "FAIL", static_cast<unsigned long long>(steps), static_cast<unsigned long long>(windows),
                    static_cast<unsigned long long>(rows_total), static_cast<unsigned long long>(rejected), static_cast<unsigned long long>(compared),
                    static_cast<unsigned long long>(differ));
    return differ == 0U;
}

void run_rank(const Args& a, std::uint32_t rank, std::unique_ptr<strix::dist::Transport> transport, bool print) {
    std::unique_ptr<strix::dist::Communicator> comm;
    if (transport) comm = std::make_unique<strix::dist::Communicator>(strix::dist::Group{rank, 2U, 1U, 0x64733462ULL}, std::move(transport), 4U << 20U);
    std::vector<std::uint32_t> ns;
    for (const auto& s : split(a.n_list, ',')) ns.push_back(static_cast<std::uint32_t>(std::stoul(s)));
    const std::uint32_t n_max = *std::max_element(ns.begin(), ns.end());
    if (n_max == 0U || n_max > BatchForward::max_rows) throw std::invalid_argument("--n values must be in 1..8");
    std::vector<std::vector<std::uint32_t>> prompt_list;
    for (const auto& f : split(a.prompts, ',')) prompt_list.push_back(read_ids(f));
    if (prompt_list.empty()) throw std::invalid_argument("--prompts needs at least one ids file");
    std::vector<std::vector<std::uint32_t>> prompts(n_max);
    for (std::uint32_t i = 0; i < n_max; ++i) prompts[i] = prompt_list[i % prompt_list.size()];
    strix::deepseek41::BatchForwardOptions o;
    o.layers = a.layers;
    o.max_positions = a.max_positions;
    o.slots = n_max + (a.fill != 0U ? 1U : 0U);
    o.pool_positions = a.pool;
    o.rank = rank;
    o.world = comm ? 2U : 1U;
    o.communicator = comm.get();
    o.tp_attention = a.tp;
    o.engram_direct_io = a.engram_direct;
    o.slice_logits = a.check && !a.gather;
    o.gather_logits = a.check && a.gather;
    BatchForward fwd(a.model, o);
    if (print) {
        std::printf("rank %u of %u (%s): layers [0, %u), weights %.2f GiB in %.1f s; per step: fixed %.3f GB + %.1f MB per routed expert per layer;"
                    " state %.1f MiB per request (%u positions), workspace %.1f MiB, %u slots\n",
                    rank, o.world, comm ? comm->transport_name().c_str() : "one node", a.layers, static_cast<double>(fwd.weight_bytes()) / 1073741824.0,
                    fwd.load_seconds(), static_cast<double>(fwd.fixed_bytes_per_step()) / 1e9, static_cast<double>(fwd.expert_bytes()) / 1e6,
                    static_cast<double>(fwd.state_bytes_per_slot()) / 1048576.0, a.max_positions, static_cast<double>(fwd.workspace_bytes()) / 1048576.0,
                    o.slots);
        if (const auto ov = fwd.overlay_description(); !ov.empty()) std::printf("weights overlay: %s\n", ov.c_str());
        if (a.pool != 0U)
            std::printf("KV pool: %u positions over the slots, %.1f MiB of arenas (largest free reservation %llu)\n", a.pool,
                        static_cast<double>(fwd.pool_bytes()) / 1048576.0, static_cast<unsigned long long>(fwd.pool_free_positions()));
        std::fflush(stdout);
    }
    const std::uint32_t template_slot = n_max;
    if (a.fill != 0U) {
        const auto filler = read_ids(a.fill_ids);
        const auto f0 = std::chrono::steady_clock::now();
        claim(fwd, a, template_slot, std::uint64_t{a.fill} + 8U);
        std::vector<BatchRow> window{{template_slot, filler[0]}};
        fwd.step(window);  // the first row alone (no compressed entry before it)
        for (std::uint32_t i = 1; i < a.fill;) {
            window.clear();
            while (window.size() < BatchForward::max_rows && i < a.fill) window.push_back({template_slot, filler[i++ % filler.size()]});
            fwd.step(window);
        }
        if (print) {
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - f0).count();
            std::printf("filled %u context tokens in %.1f s (%.1f tok/s in steps of 8 rows)\n", a.fill, s, a.fill / s);
        }
    }
    bool all_pass = true;
    for (const auto n : ns) {
        std::vector<Trace> alone(n);
        if (a.mode == "alone" || a.mode == "both" || a.mode == "windows") {
            for (std::uint32_t i = 0; i < n; ++i) {
                const std::vector<std::vector<std::uint32_t>> one{prompts[i]};
                auto r = run(fwd, a, one, 1U, template_slot, a.fill != 0U, false);
                alone[i] = std::move(r.traces[0]);
                if (print && a.mode == "alone")
                    for (const auto& [pos, rec] : alone[i])
                        std::printf("alone req %u pos %u fed %u argmax %u logits_fnv %016llx\n", i, pos, rec.fed, rec.argmax,
                                    static_cast<unsigned long long>(rec.hash));
            }
        }
        if (a.mode == "alone") continue;
        if (a.mode == "windows") {
            if (a.fill != 0U) throw std::invalid_argument("--mode windows starts every request at position 0 (no --fill)");
            all_pass = windows_gate(fwd, a, prompts, n, alone, print) && all_pass;
            continue;
        }
        const auto r = run(fwd, a, prompts, n, template_slot, a.fill != 0U, print);
        if (print) {
            const double step_ms = r.steady_steps ? r.steady_seconds * 1e3 / static_cast<double>(r.steady_steps) : 0.0;
            const double agg = r.steady_seconds > 0.0 ? static_cast<double>(r.steady_rows) / r.steady_seconds : 0.0;
            const double steps = static_cast<double>(std::max<std::uint64_t>(r.steady_steps, 1U));
            const double fixed = static_cast<double>(fwd.fixed_bytes_per_step()), experts = r.steady_expert_bytes / steps;
            std::printf("N %u: %llu steps (%llu decode-only), %.1f s; decode %.2f ms/step, aggregate %.2f tok/s, per request %.2f tok/s;"
                        " exchange %.2f ms/step, engram %.2f ms/step; union %.2f experts/layer (this rank, max %.2f rows/expert);"
                        " bytes/step %.3f GB (fixed %.3f + experts %.3f), %.0f GB/s\n",
                        n, static_cast<unsigned long long>(r.steps), static_cast<unsigned long long>(r.steady_steps), r.seconds, step_ms, agg,
                        agg / n, r.steady_exchange * 1e3 / steps, r.steady_engram * 1e3 / steps, r.steady_union / steps, r.steady_max_rows / steps,
                        (fixed + experts) / 1e9, fixed / 1e9, experts / 1e9, step_ms > 0.0 ? (fixed + experts) / (step_ms * 1e-3) / 1e9 : 0.0);
            for (std::uint32_t i = 0; i < n; ++i) {
                std::string toks;
                std::uint32_t shown = 0;
                for (const auto& [pos, rec] : r.traces[i])
                    if (pos + 1U >= prompts[i].size() + (a.fill != 0U ? a.fill : 0U) && shown++ < 12U) toks += (toks.empty() ? "" : ",") + std::to_string(rec.argmax);
                std::printf("  req %u: prompt %zu tokens, generated %s...\n", i, prompts[i].size(), toks.c_str());
            }
        }
        if (a.mode == "both") {
            std::uint64_t compared = 0, differ = 0;
            for (std::uint32_t i = 0; i < n; ++i) {
                if (alone[i].size() != r.traces[i].size()) {
                    ++differ;
                    if (print) std::printf("  req %u: %zu rows alone, %zu batched\n", i, alone[i].size(), r.traces[i].size());
                }
                for (const auto& [pos, rec] : r.traces[i]) {
                    const auto it = alone[i].find(pos);
                    ++compared;
                    if (it == alone[i].end() || it->second.fed != rec.fed || it->second.argmax != rec.argmax || it->second.hash != rec.hash) {
                        if (print && differ < 10U)
                            std::printf("  DIFF req %u pos %u: alone fed %u argmax %u fnv %016llx, batched fed %u argmax %u fnv %016llx\n", i, pos,
                                        it == alone[i].end() ? 0U : it->second.fed, it == alone[i].end() ? 0U : it->second.argmax,
                                        static_cast<unsigned long long>(it == alone[i].end() ? 0ULL : it->second.hash), rec.fed, rec.argmax,
                                        static_cast<unsigned long long>(rec.hash));
                        ++differ;
                    }
                }
            }
            if (print)
                std::printf("GATE N %u %s: %llu rows compared (%s), %llu differ\n", n, differ == 0U ? "PASS" : "FAIL",
                            static_cast<unsigned long long>(compared), a.check ? (a.gather ? "tokens + full logits" : "tokens + this rank's logits slice") : "tokens",
                            static_cast<unsigned long long>(differ));
            all_pass = all_pass && differ == 0U;
        }
        std::fflush(stdout);
    }
    if (print && (a.mode == "both" || a.mode == "windows")) std::printf("GATE %s\n", all_pass ? "PASS" : "FAIL");
    if (comm && print) {
        const auto& s = comm->stats();
        std::printf("exchanges %llu, %.1f MB sent, %.2f s in exchange (stage %.2f, wire %.2f, check %.2f)\n", static_cast<unsigned long long>(s.exchanges),
                    static_cast<double>(s.bytes_sent) / 1e6, s.seconds, s.stage, s.wire, s.check);
    }
    if (!all_pass) throw std::runtime_error("gate failed");
}

}  // namespace

int main(int argc, char** argv) try {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        const auto next = [&] {
            if (i + 1 >= argc) throw std::invalid_argument(s + " needs a value");
            return std::string(argv[++i]);
        };
        const auto u32 = [&] { return static_cast<std::uint32_t>(std::stoul(next())); };
        if (s == "--rank") a.rank = u32();
        else if (s == "--transport") a.transport = next();
        else if (s == "--layers") a.layers = u32();
        else if (s == "--prompts") a.prompts = next();
        else if (s == "--n") a.n_list = next();
        else if (s == "--generate") a.generate = u32();
        else if (s == "--mode") a.mode = next();
        else if (s == "--prompt-rows") a.prompt_rows = u32();
        else if (s == "--max-positions") a.max_positions = u32();
        else if (s == "--fill") a.fill = u32();
        else if (s == "--fill-ids") a.fill_ids = next();
        else if (s == "--tp-attention") a.tp = true;
        else if (s == "--engram-direct") a.engram_direct = true;
        else if (s == "--check") a.check = true;
        else if (s == "--gather") a.gather = true;
        else if (s == "--print-rows") a.print_rows = true;
        else if (s == "--sync-start") a.sync_start = true;
        else if (s == "--pool") a.pool = u32();
        else if (s == "--context-state") a.context_state = next();
        else if (s == "--connect-timeout") a.connect_timeout = static_cast<unsigned>(std::stoul(next()));
        else if (a.model.empty()) a.model = s;
        else throw std::invalid_argument("unknown argument " + s);
    }
    if (a.model.empty() || a.prompts.empty()) throw std::invalid_argument("usage: strix-deepseek41-batch MODEL.gguf --prompts a.ids,b.ids --n 1,2,4,8 ...");
    if (a.mode != "batch" && a.mode != "alone" && a.mode != "both" && a.mode != "windows")
        throw std::invalid_argument("--mode is batch, alone, both or windows");
    if (a.prompt_rows == 0U || a.prompt_rows > BatchForward::max_rows) throw std::invalid_argument("--prompt-rows must be in 1..8");
    if (a.fill != 0U && a.fill_ids.empty()) throw std::invalid_argument("--fill needs --fill-ids");
    strix::deepseek41::AttentionState context;
    if (!a.context_state.empty()) {
        if (a.fill != 0U) throw std::invalid_argument("--context-state and --fill exclude each other");
        context = strix::deepseek41::load_attention_state(a.context_state);
        a.context = &context;
        std::printf("context state %s: %u positions, %.1f MB, digest %016llx\n", a.context_state.c_str(), context.position,
                    static_cast<double>(context.bytes()) / 1e6, static_cast<unsigned long long>(context.digest(true)));
    }
    if (a.transport == "single") {
        run_rank(a, 0U, nullptr, true);
        return 0;
    }
    if (a.transport == "emulate") {
        auto [t0, t1] = strix::dist::memory_pair();
        std::exception_ptr failure;
        std::thread peer([&, t = std::move(t1)]() mutable {
            try {
                run_rank(a, 1U, std::move(t), false);
            } catch (...) {
                failure = std::current_exception();
            }
        });
        try {
            run_rank(a, 0U, std::move(t0), true);
        } catch (...) {
            peer.join();
            throw;
        }
        peer.join();
        if (failure) std::rethrow_exception(failure);
        return 0;
    }
    ::alarm(a.connect_timeout);
    auto transport = open_transport(a.transport, a.rank, a.connect_timeout);
    ::alarm(0);
    std::printf("connected (%s)\n", transport->name().c_str());
    std::fflush(stdout);
    run_rank(a, a.rank, std::move(transport), true);
    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 2;
}
