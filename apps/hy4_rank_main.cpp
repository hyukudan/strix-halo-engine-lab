// strix-hy4-rank: Hy4-flash on one HIP node (src/hy4/hip Forward over shared entries). Correctness tool of the first
// device wave: feeds the prompt position by position, then greedy tokens, and prints the same per-position lines as
// strix-hy4-ref (pos / in / argmax / top-K / FNV) so a receipt compares line by line with the CPU oracle.
//   strix-hy4-rank PART1.gguf PART2.gguf --ids-file F [--gen N] [--layers D] [--context C] [--top K] [--timing]
//   --teacher F: feed these ids after the prompt instead of the device's own greedy ids (the oracle's ids: a
//                divergence then shows per position instead of cascading).
// Pair (two stages of the shared layer pipeline strix/pair/layer_pipeline.hpp, one link):
//   strix-hy4-rank PARTS --rank 0|1 [--peer HOST] [--transport usb4auto|tcp] --ids-file F --prompt-rows R [--gen N]
//                  [--cut K] [--repeat K] [--port P] [--context C]
//   Rank 0 runs layers [0, K) with the embedding, rank 1 [K, depth) with the head (K default hy4_hip_plan pair_cut);
//   each loads only its layers. The prompt crosses as the X rows of each window (pair::Pipe), decode as one X row per
//   token with the token relayed back: the same work as one node, so the ids and logits equal the one-node run.
#include "hy4/hip/hy4_hip_forward.hpp"
#include "hy4/hip/hy4_hip_plan.hpp"
#include "hy4/hip/hy4_hip_rows.hpp"
#include "hy4/hy4_speculation.hpp"
#include "strix/hy4/tp_layout.hpp"
#include "strix/hy4/placement_policy.hpp"
#include "strix/weight_pack.hpp"
#include "strix/env_flags.hpp"
#include "strix/pair/layer_pipeline.hpp"
#include "strix/pair/tp_inputs.hpp"
#include "xport/tp_gather_hip.hpp"
#include "strix/pair/link.hpp"
#include "strix/pair/link_flags.hpp"
#include "strix/pair/link_mux.hpp"
#include "strix/pair/stage_map.hpp"
#include "strix/profiles/hy4.hpp"
#include "strix/spec/block_parallel.hpp"
#include "strix/spec/drafting.hpp"
#include "strix/spec/pair_round.hpp"
#include "xport/hip_pipe_staging.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

// --window-costs ROWS:MS[,ROWS:MS...]: half-window ms per pair window width, measured on this deployment (the warm
// prefill ms / (windows + 1) of a pair run with --prompt-rows ROWS); the pair picks its width from these, nothing compiled in.
std::vector<strix::hy4::hipbind::PairWindowCost> parse_window_costs(const std::string& text) {
    std::vector<strix::hy4::hipbind::PairWindowCost> costs;
    std::stringstream in(text);
    std::string item;
    while (std::getline(in, item, ',')) {
        const auto colon = item.find(':');
        if (colon == std::string::npos) throw std::invalid_argument("--window-costs ROWS:MS[,ROWS:MS...]");
        costs.push_back({static_cast<std::uint32_t>(std::stoul(item.substr(0, colon))), std::stod(item.substr(colon + 1))});
    }
    return costs;
}

// --gate-ref: the BASE reference of a long teacher-forced document, in pieces (FAST gate v2 of the prompt windows past
// what one ticket can reference). forward_rows chunks of --verify-rows R (each row bitwise step(): class BASE) run from
// the state image --state-in (else position 0) over the --ids-file document up to --until; the logits rows of the
// --score A:B ranges ([A, B) positions) go to --dump-logits F with their positions in F.pos; state images are written at
// every --save-at P:FILE and at the stop (--state-out). --budget S stops at the first chunk boundary after S seconds of
// rows (the ticket wall). The route tap (STRIX_MOE_ROUTES_OUT / _IN / STRIX_MOE_ROUTE_NULL) sees the chunks' router
// calls like any run: a reference records, a null replays + nudges.
struct GateRef {
    bool on = false;
    std::string state_in, state_out;
    std::uint32_t until = 0;  // 0: the document's end
    double budget_s = 0.0;    // 0: no budget
    std::vector<std::pair<std::uint32_t, std::uint32_t>> score;
    std::vector<std::pair<std::uint32_t, std::string>> save_at;
};

std::pair<std::uint32_t, std::uint32_t> parse_range(const std::string& v) {
    const auto colon = v.find(':');
    if (colon == std::string::npos) throw std::invalid_argument("--score A:B (positions [A, B))");
    const auto a = static_cast<std::uint32_t>(std::stoul(v.substr(0, colon)));
    const auto b = static_cast<std::uint32_t>(std::stoul(v.substr(colon + 1)));
    if (b <= a) throw std::invalid_argument("--score A:B needs A < B");
    return {a, b};
}

int run_gate_ref(strix::hy4::hipbind::Forward& fwd, const std::vector<std::uint32_t>& doc, std::uint32_t rows, const GateRef& g,
                 const std::string& dump_path) {
    using clock = std::chrono::steady_clock;
    if (rows == 0U) throw std::invalid_argument("--gate-ref needs --verify-rows R (1..8)");
    if (!g.state_in.empty()) fwd.load_state(g.state_in);
    const std::uint32_t p0 = fwd.position();
    const auto& hist = fwd.token_history();
    if (p0 > doc.size() || hist.size() != p0 || !std::equal(hist.begin(), hist.end(), doc.begin()))
        throw std::runtime_error("--gate-ref: the state image's token history is not a prefix of the --ids-file document");
    const auto end = static_cast<std::uint32_t>(g.until != 0U ? std::min<std::size_t>(g.until, doc.size()) : doc.size());
    if (p0 > end) throw std::invalid_argument("--gate-ref: the image is past --until");
    auto saves = g.save_at;
    std::sort(saves.begin(), saves.end());
    std::ofstream dump, dpos;
    if (!dump_path.empty()) {
        dump.open(dump_path, std::ios::binary | std::ios::trunc);
        dpos.open(dump_path + ".pos", std::ios::trunc);
        if (!dump || !dpos) throw std::runtime_error("cannot open " + dump_path);
    }
    const auto scored = [&g](std::uint32_t q) {
        for (const auto& [a, b] : g.score)
            if (q >= a && q < b) return true;
        return false;
    };
    std::size_t si = 0, dumped = 0;
    std::uint64_t digest = 0xcbf29ce484222325ULL;
    const auto save_due = [&](std::uint32_t at) {
        for (; si < saves.size() && saves[si].first <= at; ++si) {
            if (saves[si].first < at) {
                std::cout << "gate-ref: save point " << saves[si].first << " is behind the start " << at << ", skipped\n";
                continue;
            }
            fwd.save_state(saves[si].second);
            std::cout << "gate-ref: state image at " << at << " -> " << saves[si].second << "\n" << std::flush;
        }
    };
    const auto t0 = clock::now();
    std::uint32_t p = p0;
    bool stopped = false;
    while (p < end) {
        save_due(p);
        if (g.budget_s > 0.0 && std::chrono::duration<double>(clock::now() - t0).count() > g.budget_s) {
            stopped = true;
            break;
        }
        std::uint32_t R = std::min(rows, end - p);
        if (si < saves.size()) R = std::min(R, saves[si].first - p);
        bool want = false;
        for (std::uint32_t r = 0; r < R && !want; ++r) want = scored(p + r);
        fwd.forward_rows(std::span<const std::uint32_t>(doc.data() + p, R), want);
        if (want) {
            const auto& L = fwd.rows_logits();
            const std::size_t V = L.size() / R;
            for (std::uint32_t r = 0; r < R; ++r) {
                if (!scored(p + r)) continue;
                const float* row = L.data() + std::size_t{r} * V;
                const auto* bytes = reinterpret_cast<const unsigned char*>(row);
                for (std::size_t b = 0; b < V * sizeof(float); ++b) digest = (digest ^ bytes[b]) * 0x100000001b3ULL;
                if (dump) {
                    dump.write(reinterpret_cast<const char*>(row), static_cast<std::streamsize>(V * sizeof(float)));
                    dpos << (p + r) << '\n';
                }
                ++dumped;
            }
        }
        fwd.commit_rows(R);
        p += R;
    }
    if (!stopped) save_due(p);
    if (!g.state_out.empty()) fwd.save_state(g.state_out);
    const double s = std::chrono::duration<double>(clock::now() - t0).count();
    std::cout << "gate-ref positions " << p0 << ".." << p << " of " << end << (stopped ? " STOPPED (budget)" : " done") << " rows=" << rows
              << " s=" << std::fixed << std::setprecision(1) << s << " ms_per_pos=" << std::setprecision(2)
              << (p > p0 ? 1000.0 * s / static_cast<double>(p - p0) : 0.0) << " scored_rows=" << dumped << " logits_fnv64 " << std::hex
              << digest << std::dec << (g.state_out.empty() ? std::string() : " state_out " + g.state_out) << "\n" << std::flush;
    return 0;
}

std::vector<std::uint32_t> read_ids(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::stringstream s;
    s << in.rdbuf();
    std::vector<std::uint32_t> ids;
    std::string token;
    for (const char ch : s.str() + ",") {
        if (ch >= '0' && ch <= '9') {
            token += ch;
        } else if (!token.empty()) {
            ids.push_back(static_cast<std::uint32_t>(std::stoul(token)));
            token.clear();
        }
    }
    return ids;
}

std::uint64_t fnv(const std::vector<float>& v) {
    std::uint64_t h = 1469598103934665603ULL;
    for (const float f : v) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &f, sizeof bits);
        for (int b = 0; b < 4; ++b) {
            h ^= (bits >> (8 * b)) & 0xFFU;
            h *= 1099511628211ULL;
        }
    }
    return h;
}

std::uint32_t greedy(const std::vector<float>& logits) {  // first max, lowest token on ties (the host rule of run())
    std::uint32_t best = 0;
    for (std::uint32_t i = 1; i < logits.size(); ++i)
        if (logits[i] > logits[best]) best = i;
    return best;
}

void print_top(std::size_t p, std::uint32_t token, const std::vector<float>& logits, std::uint32_t top) {
    std::vector<std::uint32_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0U);
    const auto k = std::min<std::size_t>(std::max<std::size_t>(top, 1U), order.size());
    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(k), order.end(),
                      [&logits](std::uint32_t a, std::uint32_t b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); });
    std::cout << "pos " << p << " in " << token << " argmax " << order.front() << " top";
    for (std::size_t j = 0; j < k; ++j) std::cout << ' ' << order[j] << ':' << std::setprecision(4) << logits[order[j]];
    std::cout << " fnv " << std::hex << fnv(logits) << std::dec << "\n";
}

// Two stages of the shared layer pipeline over one link (pair/link.hpp: HY4_RANK / HY4_PEER / HY4_TRANSPORT).
int run_pair(strix::hy4::hipbind::ForwardOptions o, const std::vector<std::uint32_t>& prompt, std::uint32_t gen, std::uint32_t cut,
             std::uint32_t repeat, std::uint16_t port, bool rows_explicit, bool pipe_guess,
             const std::vector<strix::hy4::hipbind::PairWindowCost>& window_costs,
             const strix::profile::multi_device::PlacementVote& vote) {
    namespace layers = strix::pair::layers;
    using clock = std::chrono::steady_clock;
    const auto ms_since = [](clock::time_point t) { return std::chrono::duration<double, std::milli>(clock::now() - t).count(); };
    const unsigned rank = strix::pair::env_rank("HY4");
    // The width changes no bit: with a calibration (--window-costs) the flow-shop picker chooses it (hy4_hip_plan).
    if (const std::uint32_t picked = strix::hy4::hipbind::pair_window_rows(prompt.size(), window_costs); !rows_explicit && picked != 0U) {
        o.prompt_rows = picked;
        std::cout << "pair window rows " << o.prompt_rows << " (from --window-costs, " << prompt.size() << " tokens)" << '\n';
    }
    if (o.prompt_rows == 0U) throw std::invalid_argument("the pair runs the prompt in windows: --prompt-rows R");
    const auto full = strix::hy4::Config::from_gguf(o.parts.front());
    const std::uint32_t depth = o.depth != 0U ? o.depth : full.layers;
    if (cut == 0U) cut = strix::hy4::hipbind::pair_cut(depth);
    layers::Placement place{{0U, cut, depth}, rank};
    place.check();
    o.first = place.first();
    o.last = place.last();
    const auto t_load = clock::now();
    strix::hy4::hipbind::Forward fwd(o);
    const auto& lr = fwd.load_report();
    std::cout << "hy4 rank " << rank << " ready layers [" << o.first << ", " << o.last << ") embed=" << int(place.embeds())
              << " head=" << int(place.head()) << " rows=" << o.prompt_rows << "\nload tensors=" << lr.tensors << " GiB=" << std::fixed
              << std::setprecision(3) << static_cast<double>(lr.bytes) / 1073741824.0 << " s=" << std::setprecision(2)
              << ms_since(t_load) / 1000.0 << "\n" << std::flush;
    auto link = strix::pair::open_link(rank, "HY4", port);
    strix::dist::Transport* t = link.get();
    // First exchange on the link: both ranks run the placement they decided (see run(): pair placement).
    strix::profile::multi_device::agree_placement(vote, [t](auto out, auto in) { t->exchange(out, in); });
    const std::size_t row_bytes = fwd.seam_row_bytes();
    auto stream = static_cast<hipStream_t>(fwd.stream_handle());
    const std::uint32_t W = o.prompt_rows;
    const std::uint64_t identity = layers::fnv(
        layers::fnv(14695981039346656037ULL, "hy4|" + std::to_string(std::filesystem::file_size(o.parts.front())) + "|" +
                                                 std::to_string(depth) + "|" + std::to_string(cut) + "|" + std::to_string(W)),
        std::span<const std::uint32_t>(prompt));
    const auto count = layers::windows(prompt.size(), W).size();
    const auto make_pipe = [&](unsigned role) {
        strix::pair::PipeOptions po;
        po.validate = layers::validator(identity, count, cut, W, row_bytes);
        po.profile_label = "hy4-pair-profile";
        return std::make_unique<strix::pair::Pipe>(
            role, count, [t](std::span<const std::byte> d) { t->send(d); }, [t](std::span<std::byte> d) { t->receive(d); },
            [t] { t->abort("hy4 pair prefill stopped"); }, std::move(po),
            [&] { return strix::pair::make_hip_pipe_staging(std::size_t{W} * row_bytes, [stream] { return stream; }); });
    };
    const auto barrier = [&] {  // both stages start each timed phase together (load skew stays out of the timing)
        std::uint64_t w = 0x52524142ULL;
        if (rank == 0U) {
            t->receive(std::as_writable_bytes(std::span(&w, 1)));
            t->send(std::as_bytes(std::span(&w, 1)));
        } else {
            t->send(std::as_bytes(std::span(&w, 1)));
            t->receive(std::as_writable_bytes(std::span(&w, 1)));
        }
    };
    std::vector<float> head_logits;
    layers::StageOps ops;
    ops.row_bytes = row_bytes;
    ops.forward = [&](std::span<const std::uint32_t> tok, const layers::Window& w, bool head) {
        const auto& l = fwd.prefill_window(tok, w.start, head);
        if (head) head_logits = l;
    };
    ops.seam = [&](strix::pair::PipeFrame& f, bool save, std::size_t bytes) { f.staging->copy(save, 0, fwd.seam_rows(), bytes); };
    ops.fence = [&] { fwd.sync(); };
    std::uint32_t first_token = 0;
    std::uint64_t digest0 = 0;
    for (std::uint32_t pass = 0; pass < repeat; ++pass) {
        if (pass != 0U) fwd.reset();
        fwd.sync();
        barrier();  // before the pipes: their progress threads read the same transport
        auto pin = rank == 1U ? make_pipe(1U) : nullptr;
        auto pout = rank == 0U ? make_pipe(0U) : nullptr;
        const auto t0 = clock::now();
        const auto st = layers::prefill(place, prompt, W, identity, pin.get(), pout.get(), ops);
        fwd.sync();
        const double wall = ms_since(t0);
        pin.reset();
        pout.reset();
        const std::uint64_t digest = place.head() ? fnv(head_logits) : 0U;
        if (pass == 0U) digest0 = digest;
        std::cout << "prefill tokens=" << prompt.size() << " rows=" << W << " stage " << rank << " run " << pass << " ("
                  << (pass == 0U ? "cold" : "warm") << ") ms=" << std::setprecision(1) << wall << " tok_s="
                  << 1000.0 * static_cast<double>(prompt.size()) / wall << " seam_wait_ms=" << st.wait_ms << " seal_ms=" << st.seal_ms
                  << " windows=" << st.window_ms.size() << (pass != 0U && digest != digest0 ? " REPEAT_DIGEST_DIFFERS" : "") << "\n";
        if (place.head()) {
            first_token = greedy(head_logits);
            print_top(prompt.size() - 1U, prompt.back(), head_logits, 5U);
            std::cout << "hy4 prefill run " << pass << ": " << prompt.size() << " tokens wall " << std::setprecision(1) << wall
                      << " ms logits_fnv64 " << std::hex << std::setw(16) << std::setfill('0') << digest << std::dec << std::setfill(' ')
                      << "\n" << std::flush;
            std::cerr << "prefill-wall-ms=" << wall << "\n";
        }
    }
    {  // link counter for the decode hop: round trips of one decode message (header + one row) down and a token back
        std::vector<std::byte> big(32U + row_bytes), small(32U);
        std::vector<double> rtt;
        for (int i = 0; i < 21; ++i) {
            if (rank == 0U) {
                const auto t1 = clock::now();
                t->send(big);
                t->receive(small);
                if (i != 0) rtt.push_back(ms_since(t1));
            } else {
                t->receive(big);
                t->send(small);
            }
        }
        if (!rtt.empty()) {
            std::sort(rtt.begin(), rtt.end());
            std::cout << "link ping " << big.size() << " B down + 32 B back: median " << std::setprecision(3) << rtt[rtt.size() / 2U]
                      << " ms min " << rtt.front() << " max " << rtt.back() << "\n";
        }
    }
    std::vector<float> state(row_bytes / sizeof(float));
    std::uint32_t next = 0;
    layers::DecodeOps d;
    d.row_bytes = row_bytes;
    std::vector<double> fms;  // decode counter: this stage's forward (enqueue to its own sync) per token
    d.forward = [&](std::uint32_t tok, bool head) {
        const auto f0 = clock::now();
        if (head) next = fwd.step_greedy(tok, state);
        else (void)fwd.step(tok, state);
        fms.push_back(ms_since(f0));
    };
    d.top = [&] { return next; };
    d.export_row = [&](std::span<std::byte> b) { std::memcpy(b.data(), state.data(), row_bytes); };
    d.import_row = [&](std::span<const std::byte> b) {
        state.resize(row_bytes / sizeof(float));
        std::memcpy(state.data(), b.data(), row_bytes);
    };
    // --pipe-guess copy: pipelined speculation on stage 0 (layer_pipeline.hpp DecodeOps::guess / rewind) with the shared
    // copy proposer over prompt + confirmed ids; a miss rewinds the guessed row (Forward::rewind). Bits unchanged.
    strix::spec::ContextProposer proposer;
    strix::spec::Proposal proposal;
    if (pipe_guess && rank == 0U) {
        proposer.commit(prompt);
        d.guess = [&](std::uint32_t last, std::uint32_t& guess) {
            const std::uint32_t one[1] = {last};
            proposer.commit(one);
            proposer.propose(proposal, {}, 1U);
            if (proposal.tokens.empty()) return false;
            guess = proposal.tokens.front();
            return true;
        };
        d.rewind = [&] { fwd.rewind(); };
    }
    layers::Link lk;
    lk.send = [t](std::span<const std::byte> b) { t->send(b); };
    lk.receive = [t](std::span<std::byte> b) { t->receive(b); };
    barrier();
    const auto dec = layers::decode(place, first_token, gen, rank == 1U ? &lk : nullptr, rank == 0U ? &lk : nullptr, d);
    std::cout << "ids";
    for (const auto id : dec.ids) std::cout << ' ' << id;
    const double steps = gen > 1U ? static_cast<double>(gen - 1U) : 0.0;
    std::cout << "\ntime gen_ms=" << std::setprecision(1) << dec.total_ms << " decode_tok_s=" << std::setprecision(2)
              << (dec.total_ms > 0 ? 1000.0 * steps / dec.total_ms : 0.0) << " stage " << rank << "\n";
    if (d.guess) std::cout << "pipe guess copy: guesses " << dec.guesses << " hits " << dec.hits << " of " << (gen > 1U ? gen - 1U : 0U) << " steps\n";
    if (!fms.empty()) {
        std::sort(fms.begin(), fms.end());
        std::cout << "decode stage " << rank << " forward ms median " << std::setprecision(2) << fms[fms.size() / 2U] << " min "
                  << fms.front() << " max " << fms.back() << " n " << fms.size() << "\n";
    }
    return 0;
}

// Interleaved virtual stages of the shared layer pipeline (--stage-map CUTS, e.g. 0,10,21,32,42; strix/pair/stage_map.hpp,
// layers::prefill / decode over a StageMap): stage s runs on rank s % 2. This rank builds one Forward per stage it owns
// (a layer range of the same pack: the weights and caches of those layers only, as a two-stage rank holds its half), so
// the pipe fills and drains in one stage (P / 2v) instead of one half-pass (P / 2). Both seam directions ride the one pair
// link through the shared LinkMux; the decode then runs on the raw link (2v hops per token). Bits: every layer sees the
// same windows in prompt order (the shared stage-map invariant), so the head logits equal the two-stage and the one-node
// prefill at the same width.
int run_pair_stages(strix::hy4::hipbind::ForwardOptions o, const std::vector<std::uint32_t>& prompt, std::uint32_t gen,
                    const std::string& map_text, std::uint32_t repeat, std::uint16_t port,
                    const strix::profile::multi_device::PlacementVote& vote) {
    namespace layers = strix::pair::layers;
    using Forward = strix::hy4::hipbind::Forward;
    using clock = std::chrono::steady_clock;
    const auto ms_since = [](clock::time_point t) { return std::chrono::duration<double, std::milli>(clock::now() - t).count(); };
    const unsigned rank = strix::pair::env_rank("HY4");
    if (o.prompt_rows == 0U) throw std::invalid_argument("the pair runs the prompt in windows: --prompt-rows R");
    const auto full = strix::hy4::Config::from_gguf(o.parts.front());
    const std::uint32_t depth = o.depth != 0U ? o.depth : full.layers;
    const auto map = strix::pair::StageMap::parse(map_text, depth);
    layers::check_stage_map(map, rank, true);
    struct Owned {
        std::uint32_t stage = 0, first = 0, last = 0;
        std::unique_ptr<Forward> fwd;
        double ms = 0;  // this pass: the forward calls of every chunk of the stage
    };
    std::vector<Owned> own;
    const auto t_load = clock::now();
    std::uint64_t load_bytes = 0;
    std::uint32_t load_tensors = 0;
    for (std::uint32_t s = 0; s < map.stages(); ++s) {
        if (map.owner(s) != rank) continue;
        auto so = o;
        so.first = map.cuts[s].layer;
        so.last = map.cuts[s + 1U].layer;
        own.push_back({s, so.first, so.last, std::make_unique<Forward>(so), 0.0});
        load_bytes += own.back().fwd->load_report().bytes;
        load_tensors += own.back().fwd->load_report().tensors;
    }
    std::cout << "hy4 rank " << rank << " stage map " << map.text() << " (v=" << map.virtual_stages() << ") stages";
    for (const auto& w : own) std::cout << ' ' << w.stage << ":[" << w.first << ',' << w.last << ')';
    std::cout << " rows=" << o.prompt_rows << "\nload tensors=" << load_tensors << " GiB=" << std::fixed << std::setprecision(3)
              << static_cast<double>(load_bytes) / 1073741824.0 << " s=" << std::setprecision(2) << ms_since(t_load) / 1000.0
              << "\n" << std::flush;
    const auto by_range = [&own](std::uint32_t a, std::uint32_t b) -> Owned& {
        for (auto& w : own)
            if (w.first == a && w.last == b) return w;
        throw std::logic_error("hy4 stage map: a range this rank does not own");
    };
    const auto by_stage = [&own](std::uint32_t s) -> Owned& {
        for (auto& w : own)
            if (w.stage == s) return w;
        throw std::logic_error("hy4 stage map: a stage this rank does not own");
    };
    auto link = strix::pair::open_link(rank, "HY4", port);
    strix::dist::Transport* t = link.get();
    // First exchange on the link: both ranks run the placement they decided (see run(): pair placement).
    strix::profile::multi_device::agree_placement(vote, [t](auto out, auto in) { t->exchange(out, in); });
    const std::size_t row_bytes = own.front().fwd->seam_row_bytes();
    const std::uint32_t W = o.prompt_rows;
    const std::uint64_t identity = layers::fnv(
        layers::fnv(14695981039346656037ULL, "hy4|" + std::to_string(std::filesystem::file_size(o.parts.front())) + "|" +
                                                 std::to_string(depth) + "|" + map.text() + "|" + std::to_string(W)),
        std::span<const std::uint32_t>(prompt));
    const auto chunks = layers::windows(prompt.size(), W).size();
    const auto barrier = [&] {  // both ranks start each timed phase together (load skew stays out of the timing)
        std::uint64_t w = 0x52524142ULL;
        if (rank == 0U) {
            t->receive(std::as_writable_bytes(std::span(&w, 1)));
            t->send(std::as_bytes(std::span(&w, 1)));
        } else {
            t->send(std::as_bytes(std::span(&w, 1)));
            t->receive(std::as_writable_bytes(std::span(&w, 1)));
        }
    };
    // The seam copies run on the stream of the stage that produces (save, after its range) or consumes (load, before its
    // range) the rows: prefill(map) calls seam(load) for step k before range(k), so `step` indexes the pipes' order.
    Owned* cur = &own.front();
    strix::pair::StagePipes* pipes_now = nullptr;
    std::size_t step = 0;
    std::vector<float> head_logits;
    layers::StageOps ops;
    ops.row_bytes = row_bytes;
    ops.range = [&](std::span<const std::uint32_t> tok, const layers::Window& w, std::uint32_t a, std::uint32_t b, bool head) {
        auto& st = by_range(a, b);
        cur = &st;
        const auto r0 = clock::now();
        const auto& l = st.fwd->prefill_window(tok, w.start, head);
        if (head) head_logits = l;
        st.ms += ms_since(r0);
        ++step;
    };
    ops.seam = [&](strix::pair::PipeFrame& f, bool save, std::size_t bytes) {
        if (!save) cur = &by_stage(pipes_now->order().at(step).stage);
        f.staging->copy(save, 0, cur->fwd->seam_rows(), bytes);
    };
    ops.fence = [&] { cur->fwd->sync(); };
    std::uint32_t first_token = 0;
    std::uint64_t digest0 = 0;
    const bool head_rank = rank == map.owner(map.stages() - 1U);
    for (std::uint32_t pass = 0; pass < repeat; ++pass) {
        for (auto& w : own) {
            if (pass != 0U) w.fwd->reset();
            w.fwd->sync();
            w.ms = 0;
        }
        barrier();  // before the pipes: their progress threads read the same transport
        strix::pair::LinkMux mux([t](std::span<const std::byte> d) { t->send(d); }, [t](std::span<std::byte> d) { t->receive(d); },
                                 [t](const std::string& why) { t->abort(why); }, 2U, std::size_t{16} << 20U);
        const auto channel = [&mux](std::uint32_t ch) {
            return strix::pair::PipeLink{[&mux, ch](std::span<const std::byte> d) { mux.send(ch, d); },
                                         [&mux, ch](std::span<std::byte> d) { mux.receive(ch, d); },
                                         [&mux] { mux.abort("hy4 stage pipe stopped"); }};
        };
        strix::pair::PipeOptions po;
        po.validate = layers::stage_validator(identity, map, W, row_bytes);
        po.profile_label = "hy4-stage-profile";
        auto pipes = std::make_unique<strix::pair::StagePipes>(
            rank, map, chunks, channel(0U), channel(1U), std::move(po), [&cur, W, row_bytes] {
                return strix::pair::make_hip_pipe_staging(std::size_t{W} * row_bytes,
                                                          [&cur] { return static_cast<hipStream_t>(cur->fwd->stream_handle()); });
            });
        pipes_now = pipes.get();
        step = 0;
        const auto t0 = clock::now();
        const auto st = layers::prefill(map, rank, prompt, W, identity, *pipes, ops);
        for (auto& w : own) w.fwd->sync();
        const double wall = ms_since(t0);
        pipes.reset();
        pipes_now = nullptr;
        mux.close();
        const std::uint64_t digest = head_rank ? fnv(head_logits) : 0U;
        if (pass == 0U) digest0 = digest;
        double busy = 0;
        for (const auto& w : own) busy += w.ms;
        std::cout << "prefill tokens=" << prompt.size() << " rows=" << W << " map " << map.text() << " rank " << rank << " run "
                  << pass << " (" << (pass == 0U ? "cold" : "warm") << ") ms=" << std::setprecision(1) << wall << " tok_s="
                  << 1000.0 * static_cast<double>(prompt.size()) / wall << " seam_wait_ms=" << st.wait_ms << " seal_ms="
                  << st.seal_ms << " busy_ms=" << busy << " windows=" << chunks
                  << (pass != 0U && digest != digest0 ? " REPEAT_DIGEST_DIFFERS" : "") << "\n";
        std::cout << "stage ms per window:";
        for (const auto& w : own)
            std::cout << ' ' << w.stage << ":[" << w.first << ',' << w.last << ")=" << w.ms / static_cast<double>(chunks);
        std::cout << "\n";
        if (head_rank) {
            first_token = greedy(head_logits);
            print_top(prompt.size() - 1U, prompt.back(), head_logits, 5U);
            std::cout << "hy4 prefill run " << pass << ": " << prompt.size() << " tokens wall " << std::setprecision(1) << wall
                      << " ms logits_fnv64 " << std::hex << std::setw(16) << std::setfill('0') << digest << std::dec << std::setfill(' ')
                      << "\n" << std::flush;
            std::cerr << "prefill-wall-ms=" << wall << "\n";
        }
    }
    // Greedy decode over the same map on the raw link: the row visits the 2v stages, the head rank relays the token.
    std::vector<float> state(row_bytes / sizeof(float));
    std::uint32_t next = 0;
    layers::DecodeOps d;
    d.row_bytes = row_bytes;
    d.range = [&](std::uint32_t tok, std::uint32_t a, std::uint32_t b, bool head) {
        auto& st = by_range(a, b);
        if (head) next = st.fwd->step_greedy(tok, state);
        else (void)st.fwd->step(tok, state);
    };
    d.top = [&] { return next; };
    d.export_row = [&](std::span<std::byte> b) { std::memcpy(b.data(), state.data(), row_bytes); };
    d.import_row = [&](std::span<const std::byte> b) {
        state.resize(row_bytes / sizeof(float));
        std::memcpy(state.data(), b.data(), row_bytes);
    };
    layers::Link lk;
    lk.send = [t](std::span<const std::byte> b) { t->send(b); };
    lk.receive = [t](std::span<std::byte> b) { t->receive(b); };
    barrier();
    const auto dec = layers::decode(map, rank, first_token, gen, lk, d);
    std::cout << "ids";
    for (const auto id : dec.ids) std::cout << ' ' << id;
    const double steps = gen > 1U ? static_cast<double>(gen - 1U) : 0.0;
    std::cout << "\ntime gen_ms=" << std::setprecision(1) << dec.total_ms << " decode_tok_s=" << std::setprecision(2)
              << (dec.total_ms > 0 ? 1000.0 * steps / dec.total_ms : 0.0) << " rank " << rank << "\n";
    return 0;
}

int run(int argc, char** argv) {
    const auto kept = strix::pair::strip_flags(argc, argv, "HY4", false);  // --rank / --peer / --transport -> HY4_*
    (void)kept;
    namespace md = strix::profile::multi_device;
    strix::hy4::hipbind::ForwardOptions o;
    std::vector<std::uint32_t> prompt, teacher;
    std::uint32_t gen = 0, top = 5, cut = 0, repeat = 1;
    std::uint16_t port = 29755;
    bool timing = false, rows_explicit = false, pipe_guess = false;
    std::string stage_map;  // --stage-map CUTS: interleaved virtual stages of the pair (run_pair_stages)
    // Exact tensor-parallel decode over 2 whole-model ranks (Forward::enable_tp; the shared gather link and planner, as the
    // Qwen dense TP binding): --tp-rank 0|1 [--tp-transport usb4auto|tcp-listen:PORT|tcp-connect:HOST:PORT]
    // --tp-rates GB/s,GB/s [--tp-trip-us U] [--tp-force]. Each rank loads the whole pack and prefills alone; the decode
    // steps run on both ranks with this rank's shares; the plan refuses the pair when it does not win (unless forced).
    int tp_rank = -1;
    std::string tp_spec = "usb4auto";
    std::vector<double> tp_rates;
    double tp_trip_us = 0.0;
    bool tp_force = false, measured_tp = false, pair_expected = false;
    std::string pair_receipt = strix::envcfg::text("STRIX_PLACEMENT_RECEIPT") ?
        strix::envcfg::text("STRIX_PLACEMENT_RECEIPT") : "";
    std::string pair_model = STRIX_HY4_MODEL_DESCRIPTION;
    std::optional<md::Form> pair_placement;
    std::vector<double> verify_seeds;
    GateRef gref;           // --gate-ref (long BASE reference in pieces)
    bool head_explicit = false;
    bool print_config = false;
    // Exact speculative decode (src/hy4/hy4_speculation.hpp): --mtp forces it, --no-mtp / STRIX_MTP=0 opt out; the
    // EXACT prompt runs as verify rows of --prompt-vrows R (1..8, 0 = one step per position; each row bitwise the step).
    bool top_explicit = false, no_mtp = false, want_mtp = false;
    std::uint32_t prompt_vrows = strix::hy4::hipbind::rows::kMaxVerifyRows;
    std::string dump_path;  // --dump-logits F: every host logits row (FP32, vocab floats per step) appended to F
    std::vector<strix::hy4::hipbind::PairWindowCost> window_costs;  // --window-costs ROWS:MS,... (pair calibration)
    // The tier word (--profile exact|fast|turbo, --allow-pending) through the shared resolver; the lever options
    // stay in args and are parsed below (an explicit option wins over the profile).
    std::vector<std::string> args(argv + 1, argv + argc);
    strix::profile::Inputs inputs;
    strix::profile::take_options(strix::hy4::profile::engine(), args, inputs);
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        const bool more = i + 1 < args.size();
        if (a == "--print-effective-config") print_config = true;
        else if (a == "--ids-file" && more) prompt = read_ids(args[++i]);
        else if (a == "--teacher" && more) teacher = read_ids(args[++i]);
        else if (a == "--gen" && more) gen = static_cast<std::uint32_t>(std::stoul(args[++i]));
        else if (a == "--layers" && more) o.depth = static_cast<std::uint32_t>(std::stoul(args[++i]));
        else if (a == "--context" && more) o.context = static_cast<std::uint32_t>(std::stoul(args[++i]));
        else if (a == "--top" && more) top = static_cast<std::uint32_t>(std::stoul(args[++i])), top_explicit = true;
        else if (strix::session::speculation_opt_out_argument(a)) no_mtp = true;
        else if (a == "--mtp") want_mtp = true;
        else if (a == "--prompt-vrows" && more) {
            prompt_vrows = static_cast<std::uint32_t>(std::stoul(args[++i]));
            if (prompt_vrows > strix::hy4::hipbind::rows::kMaxVerifyRows) throw std::invalid_argument("--prompt-vrows 0..8");
        }
        else if (a == "--prompt-rows" && more) o.prompt_rows = static_cast<std::uint32_t>(std::stoul(args[++i])), rows_explicit = true;
        else if (a == "--timing") timing = true;
        else if (a == "--gate-ref") gref.on = true;
        else if (a == "--head-q8" && more) o.head_q8 = args[++i] == "1", head_explicit = true;
        else if (a == "--verify-rows" && more) o.verify_rows = static_cast<std::uint32_t>(std::stoul(args[++i]));
        else if (a == "--state-in" && more) gref.state_in = args[++i];
        else if (a == "--state-out" && more) gref.state_out = args[++i];
        else if (a == "--until" && more) gref.until = static_cast<std::uint32_t>(std::stoul(args[++i]));
        else if (a == "--budget" && more) gref.budget_s = std::stod(args[++i]);
        else if (a == "--score" && more) gref.score.push_back(parse_range(args[++i]));
        else if (a == "--save-at" && more) {
            const std::string v = args[++i];
            const auto colon = v.find(':');
            if (colon == std::string::npos) throw std::invalid_argument("--save-at P:FILE");
            gref.save_at.emplace_back(static_cast<std::uint32_t>(std::stoul(v.substr(0, colon))), v.substr(colon + 1));
        }
        else if (a == "--dump-logits" && more) dump_path = args[++i];
        else if (a == "--window-costs" && more) window_costs = parse_window_costs(args[++i]);
        else if (a == "--cut" && more) cut = static_cast<std::uint32_t>(std::stoul(args[++i]));
        else if (a == "--stage-map" && more) stage_map = args[++i];
        else if (a == "--repeat" && more) repeat = static_cast<std::uint32_t>(std::stoul(args[++i]));
        else if (a == "--port" && more) port = static_cast<std::uint16_t>(std::stoul(args[++i]));
        else if (a == "--pair-expected") pair_expected = true;
        else if (a == "--pair-model" && more) pair_model = args[++i];
        else if (a == "--pair-receipt" && more) pair_receipt = args[++i];
        else if (a == "--pair-placement" && more) {
            pair_placement = md::parse_form(args[++i]);
            if (pair_placement != md::Form::layers && pair_placement != md::Form::tensor)
                throw std::invalid_argument("--pair-placement layers|tensor");
        }
        else if (a == "--verify-ms" && more) verify_seeds = strix::hy4::hipbind::rows::parse_verify_seeds(args[++i]);
        else if (a == "--tp-rank" && more) tp_rank = static_cast<int>(std::stoul(args[++i]));
        else if (a == "--tp-transport" && more) tp_spec = args[++i];
        else if (a == "--tp-trip-us" && more) tp_trip_us = std::stod(args[++i]);
        else if (a == "--tp-force") tp_force = true;
        else if (a == "--tp-rates" && more) {
            std::stringstream ss(args[++i]);
            for (std::string v; std::getline(ss, v, ',');) tp_rates.push_back(std::stod(v));
        }
        else if (a == "--pipe-guess" && more) {
            const std::string v = args[++i];
            if (v != "copy" && v != "off") throw std::invalid_argument("--pipe-guess copy|off");
            pipe_guess = v == "copy";
        }
        else if (!a.empty() && a[0] != '-') o.parts.emplace_back(a);
        else throw std::invalid_argument("unknown argument " + a);
    }
    if (o.parts.size() == 1U) {
        const auto parts = strix::weight_pack::gguf_parts(o.parts.front().string());
        o.parts.assign(parts.begin(), parts.end());
    }
    bool layer_pair = std::getenv("HY4_RANK") != nullptr;
    namespace hp = strix::hy4::profile;
    namespace hr = strix::hy4::hipbind::rows;
    hipDeviceProp_t prop{};
    int hip_device = 0;
    if (hipGetDevice(&hip_device) == hipSuccess) (void)hipGetDeviceProperties(&prop, hip_device);
    const auto arch = hp::device_of(prop.gcnArchName);
    if (layer_pair && tp_rank >= 0)
        throw std::invalid_argument("--tp-rank replaces --rank and the automatic pair planner");
    if ((pair_placement || pair_expected) && !layer_pair && tp_rank < 0)
        throw std::invalid_argument("--pair-placement needs --rank or a two-node topology");

    const bool stage_route = stage_map.empty() && cut == 0U && !pipe_guess && window_costs.empty();
    strix::hy4::SpeculationRun spec_run;
    if (!o.parts.empty() && (!print_config || (layer_pair && !pair_receipt.empty())))
        spec_run.block_in_pack = strix::hy4::Config::from_gguf(o.parts.front()).nextn > 0U;
    spec_run.whole_model = o.depth == 0U;
    spec_run.no_mtp = no_mtp;
    spec_run.environment = std::getenv(strix::session::speculation_environment);
    spec_run.explicit_mtp = want_mtp;
    spec_run.per_step_route = (top_explicit && top > 0U) || !teacher.empty() || !dump_path.empty() || timing || gref.on;
    const bool spec_room = prompt.size() + gen + hr::kMaxVerifyRows <= o.context;
    std::optional<strix::serve::Json> placement_config;
    // The decision both ranks must share; an explicit --tp-rank is the tensor form without a measured reference.
    auto pair_vote = md::placement_vote(tp_rank >= 0 ? md::Form::tensor : md::Form::layers, {});
    if (layer_pair) {
        const auto rank = strix::pair::env_rank("HY4");
        const auto* transport = std::getenv("HY4_TRANSPORT");
        const auto& topology = strix::device::process_topology(static_cast<int>(rank), transport ? transport : "usb4auto");
        const auto native = hp::resolve({arch, hp::Topology::one_node, prompt.size()}, inputs);
        const auto* width = native->find("prompt_rows");
        const auto rows = rows_explicit ? o.prompt_rows : width && strix::profile::applied(width->status) ?
            static_cast<std::uint32_t>(std::stoul(width->value)) : 0U;
        const bool compatible = stage_route && repeat == 1U && gen > 1U && spec_room && !spec_run.per_step_route &&
            strix::hy4::speculation(spec_run).enabled && rows != 0U && (o.verify_rows == 0U || o.verify_rows == hr::kMaxVerifyRows) &&
            tp_trip_us == 0.0 && !tp_force && verify_seeds.empty() && tp_rates.empty() &&
            strix::envcfg::flag("STRIX_TP_CHANNEL", true) && strix::envcfg::flag("STRIX_HY4_TP_JOIN_BF16", true) &&
            strix::envcfg::flag("STRIX_HY4_TP_DRAFT", true) && strix::envcfg::flag("STRIX_HY4_ROWS_EXPERT_UNION", true);
        std::optional<md::BandReceipt> receipt;
        if (!pair_receipt.empty()) receipt = md::load_band_receipt(pair_receipt);
        const auto layout = compatible ? strix::hy4::tensor_layout(native->resolution, rows) : std::string{};
        auto q = strix::hy4::pair_query(topology, native->resolution.profile, prompt.size(), o.context, gen,
                                         transport ? transport : "usb4auto", hp::device_name(arch), layout,
                                         receipt ? &*receipt : nullptr);
        q.placement = pair_placement;
        q.rank_executor = true;
        const auto description = strix::model_description::load(pair_model);
        const auto policy = md::parse_policy(description);
        const auto decision = strix::hy4::resolve_pair_placement(description, q);
        placement_config = decision.json(policy, q);
        pair_vote = md::placement_vote(decision.form, decision.ladder_reference);
        if (!print_config) std::cout << "pair placement " << placement_config->dump() << '\n';
        if (decision.form == md::Form::tensor && !print_config) {
            if (!stage_route || o.depth != 0 || rows == 0U)
                throw std::invalid_argument("tensor placement needs a whole model, windows and no stage overrides");
            if (tp_rates.empty()) for (const auto& node : topology.nodes) tp_rates.push_back(node.memory_gbytes_ps);
            (void)strix::pair::tp::rank_rates({}, tp_rates, 2U);
            layer_pair = false;
            tp_rank = static_cast<int>(rank);
            tp_spec = transport ? transport : "usb4auto";
            measured_tp = !decision.ladder_reference.empty();
        }
    }
    // A pair width picked from --window-costs is a prompt_rows setting like --prompt-rows: the resolver stamps it.
    // On the interleaved pair route (no two-stage option) the shared planner picks the width from the prompt
    // length and the measured pair cost (hy4_hip_plan.hpp pair_stage_plan); the width changes no bit. The pair always runs
    // in windows, so the pick does not wait for a profile width: a profile whose prompt_rows lever is pending leaves 0 here.
    if (layer_pair && !rows_explicit) {
        std::uint32_t picked = strix::hy4::hipbind::pair_window_rows(prompt.size(), window_costs);
        if (picked == 0U && stage_route && !o.parts.empty() && !prompt.empty()) {
            const std::uint32_t depth = o.depth != 0U ? o.depth : strix::hy4::Config::from_gguf(o.parts.front()).layers;
            picked = strix::hy4::hipbind::pair_stage_plan(prompt.size(), depth).choice.chunk;
        }
        if (picked != 0U && picked != o.prompt_rows) {
            o.prompt_rows = picked;
            inputs.cli.emplace_back("prompt_rows", std::to_string(picked));
        }
    }
    // The regime (device, topology, this prompt's length) selects the qualified FAST cells (strix/profiles/hy4.hpp).
    const hp::Regime regime{arch, layer_pair ? hp::Topology::pair : hp::Topology::one_node, prompt.size()};
    const auto resolved = hp::resolve(regime, inputs);
    const auto& profile = resolved->resolution;
    if (print_config) {
        auto config = strix::serve::parse_json(profile.json());
        if (placement_config) config.set("multi_device", std::move(*placement_config));
        std::cout << config.dump() << '\n';
        return 0;
    }
    if (const auto* e = resolved->find("prompt_rows"); e != nullptr && !rows_explicit && strix::profile::applied(e->status))
        o.prompt_rows = static_cast<std::uint32_t>(std::stoul(e->value));
    if (const auto* e = resolved->find("head_q8"); e != nullptr && !head_explicit && strix::profile::applied(e->status)) o.head_q8 = e->value == "1";
    strix::profile::apply(profile);  // environment levers the profile sets (pf_mha), before any forward reads them
    std::cout << profile.describe() << "regime " << hp::device_name(regime.device) << " / " << hp::topology_name(regime.topology)
              << " / prompt " << regime.prompt_tokens << "\n" << std::flush;
    for (const auto& line : profile.receipt_lines()) std::cerr << line << '\n';
    for (const auto& n : profile.notices()) std::cerr << n << '\n';
    for (const auto& e : profile.entries)
        if (!e.reason.empty()) std::cerr << e.lever->name << ": " << e.reason << '\n';
    if (o.parts.empty() || prompt.empty())
        throw std::invalid_argument("usage: strix-hy4-rank PART1.gguf [PART2.gguf] --ids-file F [--gen N] [--teacher F] [--layers D] "
                                    "[--context C] [--top K] [--timing] [--prompt-rows R] [--dump-logits F] [--profile exact|fast|turbo] "
                                    "[--rank 0|1 --cut K | --stage-map C0,C1,..,L|off --repeat K --port P --pipe-guess copy|off --window-costs ROWS:MS,...]");
    // Decode route, every tier: exact speculative decode where the pack ships its next-token block and this rank holds the
    // whole model (one node, or a tensor-parallel rank); the shared resolver words the decision (hy4_speculation.hpp).
    spec_run.whole_model = !layer_pair && spec_run.whole_model;
    const auto spec_decision = strix::hy4::speculation(spec_run);
    const bool spec_on = spec_decision.enabled && gen > 1U && spec_room;
    // EXACT prompt (no windows): verify rows of prompt_vrows positions, each row bitwise step() at its position (the BASE
    // reference form of the long gate, run_gate_ref); one-node whole model; per-position logits dumps keep the steps.
    const bool rows_prompt = o.prompt_rows == 0U && prompt_vrows > 0U && spec_run.whole_model && tp_rank < 0 && !gref.on &&
                             dump_path.empty() && !timing;
    if ((spec_on || rows_prompt) && o.verify_rows == 0U) o.verify_rows = hr::kMaxVerifyRows;
    if (rows_prompt) prompt_vrows = std::min(prompt_vrows, o.verify_rows);
    o.nextn = o.nextn || spec_on;
    std::cout << "decode " << (spec_on ? "speculative (next-token block drafts, verify rows emit the greedy ids)" : "plain steps")
              << " [" << spec_decision.source
              << (spec_decision.enabled && !spec_on ? (gen <= 1U ? ", one id" : ", context holds no verify pass") : "") << "]"
              << (rows_prompt ? "; prompt as verify rows of " + std::to_string(prompt_vrows) : std::string()) << "\n" << std::flush;
    // Pair default: the shared interleaved stage map with v from the shared planner at this width (strix/pair/stage_map.hpp
    // plan_stages over the measured pair cost; bitwise equal to the two-stage cut, measured faster); --stage-map off, --cut,
    // --pipe-guess or --window-costs keep the two-stage route.
    if (layer_pair && stage_route && o.prompt_rows != 0U) {
        const std::uint32_t depth = o.depth != 0U ? o.depth : strix::hy4::Config::from_gguf(o.parts.front()).layers;
        const auto plan = strix::hy4::hipbind::pair_stage_plan(prompt.size(), depth, o.prompt_rows);
        std::cout << "pair plan rows " << o.prompt_rows << " map " << plan.map.text() << " (" << plan.choice.reason << ")\n";
        if (plan.map.virtual_stages() > 1U) stage_map = plan.map.text();
    }
    if (stage_map == "off") stage_map.clear();
    if (layer_pair && !stage_map.empty()) {
        if (cut != 0U || pipe_guess || !window_costs.empty())
            throw std::invalid_argument("--stage-map replaces --cut; --pipe-guess and --window-costs are two-stage options");
        return run_pair_stages(o, prompt, gen, stage_map, std::max(repeat, 1U), port, pair_vote);
    }
    if (std::getenv("HY4_RANK") != nullptr)
        return run_pair(o, prompt, gen, cut, std::max(repeat, 1U), port, rows_explicit, pipe_guess, window_costs,
                        pair_vote);
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    strix::hy4::hipbind::Forward fwd(o);
    fwd.set_timing(timing);
    const auto& c = fwd.config();
    const auto& lr = fwd.load_report();
    std::cout << "config layers=" << c.layers << " hidden=" << c.hidden << " vocab=" << c.vocab << " context=" << o.context
              << "\nload tensors=" << lr.tensors << " GiB=" << std::fixed << std::setprecision(3)
              << static_cast<double>(lr.bytes) / 1073741824.0 << " s=" << std::setprecision(2) << lr.seconds << " GiB/s="
              << (lr.seconds > 0 ? static_cast<double>(lr.bytes) / 1073741824.0 / lr.seconds : 0.0) << "\n"
              << lr.queue_line << std::flush;
    if (gref.on) return run_gate_ref(fwd, prompt, o.verify_rows, gref, dump_path);
    std::unique_ptr<strix::xport::tp_gather::HipPairGather> tpg;
    if (tp_rank >= 0) {  // after the load: the peer may load slower; open_pair waits up to its timeout
        if (tp_rank > 1 || o.prompt_rows == 0U || tp_rates.size() != 2U)
            throw std::invalid_argument("--tp-rank 0|1 needs --prompt-rows (each rank prefills alone) and --tp-rates A,B (GB/s)");
        auto opened = strix::dist::open_pair(tp_spec, static_cast<std::uint32_t>(tp_rank));
        if (!opened.receipt.empty()) std::cout << "tp transport: " << opened.receipt << "\n";
        // Startup handshake on the host link before the gather's first collective: both ranks decided tensor on the
        // same measured reference (a rank with a stale or missing receipt refuses here instead of desynchronising).
        md::agree_placement(pair_vote, [&opened](auto out, auto in) { opened.host->exchange(out, in); });
        // largest gather: half the logits row (vocab / 2 FP32); the link probe sizes fit in 1 MiB; speculative decode: a
        // verify pass's MoE join at 8 rows (8 x (8 slot rows x 1536 + shared / 2) FP32) < 2 MiB
        // route: the shared policy (pair::select_route); the device-waited channel unless STRIX_TP_CHANNEL=0 (staged)
        tpg = std::make_unique<strix::xport::tp_gather::HipPairGather>(std::move(opened), static_cast<std::uint32_t>(tp_rank),
                                                                       spec_on ? std::size_t{2} << 20 : std::size_t{1} << 20,
                                                                       static_cast<hipStream_t>(fwd.stream_handle()),
                                                                       strix::envcfg::flag("STRIX_TP_CHANNEL", true));
    }
    if (measured_tp && tpg->link().route() != strix::pair::GatherRoute::channel)
        throw std::runtime_error("the pair did not open the measured channel route");
    std::ofstream dump;
    if (!dump_path.empty()) {
        dump.open(dump_path, std::ios::binary | std::ios::trunc);
        if (!dump) throw std::runtime_error("cannot open " + dump_path);
    }
    std::vector<std::uint32_t> generated;
    std::vector<float> state;
    std::uint32_t token = prompt.front();
    const std::size_t steps = prompt.size() + gen - (gen > 0 ? 1U : 0U);
    double prompt_ms = 0, gen_ms = 0;
    std::size_t p0 = 0;
    std::vector<float> prompt_logits;
    const std::vector<float> no_logits;
    // The next-token block's teacher pass rides on the prompt (MtpSource installed before it): this rank drafts (one node,
    // TP rank 0) or replays rank 0's draft rounds on its own block (TP rank 1 with STRIX_HY4_TP_DRAFT on, the default).
    const bool draft_tp = tp_rank >= 0 && strix::envcfg::flag("STRIX_HY4_TP_DRAFT", true);
    std::unique_ptr<hr::MtpSource<strix::hy4::hipbind::Forward>> mtp;
    if (spec_on && (tp_rank != 1 || draft_tp)) mtp = std::make_unique<hr::MtpSource<strix::hy4::hipbind::Forward>>(fwd);
    // EXACT prompt as verify rows (rows_prompt): forward_rows over prompt_vrows positions, committed; the last row's logits
    // are the prompt's; with the block the rows' hidden are staged (the block's flush reuses the verify workspace) and fed.
    const auto vrows_prompt = [&](bool feed) { return hr::prefill_verify_rows(fwd, prompt, prompt_vrows, feed); };
    // --repeat K (batched prompt): K - 1 earlier passes, each followed by reset() (bit-reproducing), so the last pass is
    // the warm figure and the first the cold one; their last-position digests must equal the final pass's.
    for (std::uint32_t r = 1; (o.prompt_rows != 0U || rows_prompt) && r < repeat; ++r) {
        const auto s0 = clock::now();
        const std::uint64_t digest = fnv(rows_prompt ? vrows_prompt(false) : fwd.prefill(prompt));
        const double ms = std::chrono::duration<double, std::milli>(clock::now() - s0).count();
        std::cout << "prefill run " << r << " of " << repeat << " tokens=" << prompt.size() << " ms=" << std::setprecision(1) << ms
                  << " tok_s=" << 1000.0 * static_cast<double>(prompt.size()) / ms << " logits_fnv64 " << std::hex << digest << std::dec
                  << "\n";
        fwd.reset();
    }
    if (rows_prompt) {  // EXACT prompt: same bits as the position-by-position steps; the last position printed like a step
        const auto s0 = clock::now();
        if (mtp) mtp->install(prompt);
        prompt_logits = vrows_prompt(mtp != nullptr);
        prompt_ms = std::chrono::duration<double, std::milli>(clock::now() - s0).count();
        std::cout << "prefill tokens=" << prompt.size() << " verify_rows=" << prompt_vrows << " ms=" << std::setprecision(1) << prompt_ms
                  << " tok_s=" << std::setprecision(1) << 1000.0 * static_cast<double>(prompt.size()) / prompt_ms
                  << " logits_fnv64 " << std::hex << fnv(prompt_logits) << std::dec << " [position-by-position bits]\n";
        p0 = prompt.size() - 1U;
    }
    if (o.prompt_rows != 0U) {  // batched prompt: one call; the last prompt position is printed like a step
        const auto s0 = clock::now();
        if (mtp) mtp->install(prompt);  // prefill() feeds every prompt row's hidden to the block's teacher pass
        prompt_logits = fwd.prefill(prompt);
        prompt_ms = std::chrono::duration<double, std::milli>(clock::now() - s0).count();
        const auto& t = fwd.times();
        std::cout << "prefill tokens=" << prompt.size() << " rows=" << o.prompt_rows << " ms=" << std::setprecision(1) << prompt_ms
                  << " tok_s=" << std::setprecision(1) << 1000.0 * static_cast<double>(prompt.size()) / prompt_ms << " [windows embed "
                  << t.embed << " engram " << t.engram << " attn " << t.attention << " ffn " << t.ffn << " head " << t.head << " total "
                  << t.total << "]\n";
        p0 = prompt.size() - 1U;
    }
    // tp placement, decided once from measurements: both ranks first run kTpCalib decode-form steps one node (each holds
    // the whole pack, so the ids are the split form's ids), the smaller of the last two is this rank's one-node pass,
    // agreed max over the ranks; the planner prices the stage model's residual as fixed work on every rank
    // (split::with_measured_single), the trip comes from the start-up probe of the route the decode runs. One node wins ->
    // both ranks keep decoding one node (the same ids); --tp-force runs the split anyway.
    constexpr std::size_t kTpCalib = 3U;
    std::vector<double> tp_calib_ms;
    bool tp_placed = !tpg, tp_on = false;
    double tp_ms = 0.0;
    std::size_t tp_steps = 0;
    const auto tp_place = [&] {
        namespace psp = strix::pair::split;
        auto stages = strix::hy4::tp::decode_stages(c, [&](const std::string& n) { return fwd.tensor_bytes(n); });
        std::vector<double> bpms;
        for (const double g : tp_rates) bpms.push_back(g * 1e6);  // GB/s -> bytes/ms
        const auto n = tp_calib_ms.size();
        const double one_ms = tpg->agree_max(std::min(tp_calib_ms[n - 1U], tp_calib_ms[n - 2U]));
        const auto cal = psp::with_measured_single(stages, psp::Pass{}, *std::min_element(bpms.begin(), bpms.end()), one_ms);
        // trip cost measured now over the gather form the decode runs, agreed by both ranks (identical bits)
        const auto lc = tpg->measure_link();
        const auto trip = strix::pair::tp::trip_cost(lc, tp_trip_us);
        const auto plan = psp::plan(stages, psp::Pass{}, psp::Costs{bpms, trip, 0.0, false});
        const bool wins = plan.tensor.admissible && plan.tensor.ms < plan.single.ms;
        std::cout << "tp calibration: one node pass " << std::setprecision(1) << one_ms << " ms measured (" << kTpCalib
                  << " one-node steps, agreed max), stage model " << cal.model_ms << " ms, fixed residual " << cal.fixed_ms << " ms\n"
                  << "tp link cost: " << lc.describe() << "; trip " << trip.latency_ms * 1000.0 << " us"
                  << (tp_trip_us > 0.0 ? " (--tp-trip-us)" : " (measured)") << ", " << strix::pair::gather_route_name(tpg->link().route())
                  << "\ntp placement: one rank " << plan.single.ms << " ms/pass, 2 ranks " << plan.tensor.ms << " ms/pass (compute "
                  << plan.tensor.compute_ms << " + " << plan.tensor.trips << " trips " << plan.tensor.link_ms << ") -> "
                  << (wins ? "decode on all ranks" : tp_force ? "group loses, forced" : "decode stays on one node (both ranks)") << "\n";
        if (!wins && !tp_force) return;
        fwd.enable_tp(static_cast<std::uint32_t>(tp_rank), {&tpg->link(), &strix::xport::tp_gather::HipPairGather::Link::fn});
        if (plan.tensor.trips != fwd.tp_gathers_per_step())
            throw std::logic_error("tp: planner trips differ from the step's gathers");
        tpg->link().reset_counters();
        tp_on = true;
        std::cout << fwd.tp_receipt() << "\n" << std::flush;
    };
    if (!verify_seeds.empty() && !spec_on) throw std::invalid_argument("--verify-ms requires speculative decode");
    if (spec_on) {  // exact speculative decode (strix-hy4-rows --mtp, the record route): the ids of plain greedy decode
        const auto anchor = static_cast<std::uint32_t>(std::max_element(prompt_logits.begin(), prompt_logits.end()) - prompt_logits.begin());
        if (mtp) mtp->finish_prompt(anchor);
        strix::spec::pair_round::Exchange pair_ex;
        double trip_ms = 0.0;
        if (tpg) {  // tensor-parallel pair: every verify pass (and draft round with the block on both ranks) on both ranks
            const auto rank = static_cast<std::uint32_t>(tp_rank);
            // trip cost measured over the gather form the passes run, agreed by both ranks (prices the seeds below)
            const auto lc = tpg->measure_link();
            trip_ms = strix::pair::tp::trip_cost(lc, tp_trip_us).latency_ms;
            fwd.enable_tp(rank, {&tpg->link(), &strix::xport::tp_gather::HipPairGather::Link::fn});
            std::cout << fwd.tp_receipt() << "\ntp link cost: " << lc.describe() << "\n";
            // the round message rides the same gather: [2][kSlot] bytes on the device, stream-ordered after the last pass
            constexpr std::size_t kSlot = 128;
            static_assert(strix::spec::pair_round::kMessageBytes <= kSlot);
            auto* slots = reinterpret_cast<std::uint8_t*>(fwd.alloc_f32(2U * kSlot / 4U));
            const auto stream = static_cast<hipStream_t>(fwd.stream_handle());
            auto* glink = &tpg->link();
            pair_ex = [stream, slots, glink, rank](const std::uint8_t* mine, std::uint8_t* peer) {
                constexpr std::size_t B = strix::spec::pair_round::kMessageBytes;
                const auto ok = [](hipError_t e, const char* what) {
                    if (e != hipSuccess) throw std::runtime_error(std::string("spec pair round: ") + what + ": " + hipGetErrorString(e));
                };
                ok(hipMemcpyAsync(slots + std::size_t{rank} * kSlot, mine, B, hipMemcpyHostToDevice, stream), "message upload");
                strix::xport::tp_gather::HipPairGather::Link::fn(glink, stream, slots, kSlot, kSlot);
                ok(hipMemcpyAsync(peer, slots + std::size_t{1U - rank} * kSlot, B, hipMemcpyDeviceToHost, stream), "message read");
                ok(hipStreamSynchronize(stream), "message sync");
            };
        }
        // the Arbiter's verify seeds from this rank's plain step (best of two, each rewound: the rows bench's pattern)
        double step_ms = 0.0;
        for (int k = 0; k < 2; ++k) {
            const auto s0 = clock::now();
            (void)fwd.step_greedy(anchor);
            const double ms = std::chrono::duration<double, std::milli>(clock::now() - s0).count();
            fwd.rewind();
            step_ms = k == 0 ? ms : std::min(step_ms, ms);
        }
        if (tp_rank == 1) {  // rank 1: rank 0's rounds until its end message (position checked)
            const auto s0 = clock::now();
            const auto fr = mtp ? hr::spec_follow(fwd, pair_ex, *mtp) : hr::spec_follow(fwd, pair_ex);
            gen_ms = std::chrono::duration<double, std::milli>(clock::now() - s0).count();
            tpg->finish();
            std::cout << "spec follow rounds " << fr.rounds << " passes " << fr.passes << " rows " << fr.rows << " committed " << fr.committed
                      << " drafts " << fr.drafts << (mtp ? " (block on both ranks, draft ids cross-checked)" : "") << "\nids (rank 0 emits)\n"
                      << std::setprecision(1) << "time prompt_ms=" << prompt_ms << " gen_ms=" << gen_ms
                      << " total_ms=" << std::chrono::duration<double, std::milli>(clock::now() - t0).count() << "\n";
            return 0;
        }
        hr::SpecOptions so;
        // Arbiter seeds: the shared routed-MoE pass model over this topology's plain step (one node = moe_verify_seed_ms;
        // TP: every rank reads half of every weight, plus the pass's gathers at the measured trip)
        hr::seed_spec_options(so, step_ms, tpg ? 2U : 1U, trip_ms, tpg ? fwd.tp_gathers_per_step() : 0U, verify_seeds);
        so.draft_tp = draft_tp;
        if (!verify_seeds.empty()) std::cout << "spec verify seeds explicit\n";
        const auto res = hr::spec_decode(fwd, prompt, anchor, gen, so, *mtp, tpg ? &pair_ex : nullptr);
        if (tpg) tpg->finish();
        generated = res.ids;
        gen_ms = res.wall_ms;
        const auto passes = std::max<std::size_t>(res.steps.size(), 1U);
        std::cout << "spec passes " << res.steps.size() << " E " << std::setprecision(3)
                  << static_cast<double>(gen - 1U) / static_cast<double>(passes) << " proposed " << res.proposed << " accepted "
                  << res.accepted << " rows";
        for (std::size_t r = 1; r < res.rows_hist.size(); ++r) std::cout << " R" << r << ':' << res.rows_hist[r];
        std::cout << std::setprecision(2) << "; plain step " << step_ms << " ms; block calls " << res.chain_calls << " drafts "
                  << res.chain_drafts << "; wall verify " << std::setprecision(1) << res.verify_ms << " draft " << res.chain_ms << " other "
                  << res.wall_ms - res.verify_ms - res.chain_ms << " ms" << (tpg ? (draft_tp ? " (tp, block tp)" : " (tp, block rank 0)") : "")
                  << "\nids";
        for (const auto id : generated) std::cout << ' ' << id;
        std::cout << "\n" << std::setprecision(1) << "time prompt_ms=" << prompt_ms << " gen_ms=" << gen_ms
                  << " total_ms=" << std::chrono::duration<double, std::milli>(clock::now() - t0).count() << " decode_tok_s="
                  << std::setprecision(2) << (gen_ms > 0 ? 1000.0 * (gen - 1U) / gen_ms : 0.0) << "\n";
        if (tpg) std::cout << fwd.tp_receipt() << "\n";
        return 0;
    }
    for (std::size_t p = p0; p < steps; ++p) {
        if (p < prompt.size()) token = prompt[p];
        if (!tp_placed && tp_calib_ms.size() >= kTpCalib) {
            tp_place();
            tp_placed = true;
        }
        const auto s0 = clock::now();
        const bool batched = (o.prompt_rows != 0U || rows_prompt) && p + 1U == prompt.size();
        const bool device = top == 0U && !batched;  // --top 0: the shared device greedy row, no logits on the host
        const std::uint32_t device_next = device ? fwd.step_greedy(token) : 0U;
        const auto& logits = device ? no_logits : batched ? prompt_logits : fwd.step(token, state);
        const double step_ms = batched ? 0.0 : std::chrono::duration<double, std::milli>(clock::now() - s0).count();
        if (dump && !logits.empty())
            dump.write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(logits.size() * sizeof(float)));
        (p < prompt.size() ? prompt_ms : gen_ms) += step_ms;
        if (!tp_placed && !batched) tp_calib_ms.push_back(step_ms);
        if (tp_on && !batched) {
            tp_ms += step_ms;
            ++tp_steps;
        }
        std::vector<std::uint32_t> order(logits.size());
        std::iota(order.begin(), order.end(), 0U);
        const auto k = std::min<std::size_t>(std::max<std::size_t>(top, 1U), order.size());
        std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(k), order.end(),
                          [&logits](std::uint32_t a, std::uint32_t b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); });
        const auto next = device ? device_next : order.front();
        std::cout << "pos " << p << " in " << token << " argmax " << next;
        if (!device) {
            std::cout << " top";
            for (std::size_t j = 0; j < k; ++j) std::cout << ' ' << order[j] << ':' << std::setprecision(4) << logits[order[j]];
            std::cout << " fnv " << std::hex << fnv(logits) << std::dec;
        }
        std::cout << " ms " << std::setprecision(1) << step_ms;
        if (timing) {
            const auto& t = fwd.times();
            std::cout << " [embed " << t.embed << " engram " << t.engram << " attn " << t.attention << " ffn " << t.ffn << " head "
                      << t.head << "]";
        }
        std::cout << "\n" << std::flush;
        if (p + 1U >= prompt.size() && generated.size() < gen) {
            generated.push_back(next);
            const auto g = generated.size() - 1U;
            token = g < teacher.size() ? teacher[g] : next;
        }
    }
    if (tpg) tpg->finish();  // the channel's last gathers are checked before any result is printed
    std::cout << "ids";
    for (const auto id : generated) std::cout << ' ' << id;
    const auto gen_steps = gen > 1U ? gen - 1U : 0U;
    std::cout << "\n" << std::setprecision(1) << "time prompt_ms=" << prompt_ms << " gen_ms=" << gen_ms
              << " total_ms=" << std::chrono::duration<double, std::milli>(clock::now() - t0).count() << " decode_tok_s="
              << std::setprecision(2) << (gen_ms > 0 ? 1000.0 * gen_steps / gen_ms : 0.0) << "\n";
    if (tpg)
        std::cout << fwd.tp_receipt() << "\ntp link: " << tpg->link().calls() << " gathers, host time in the exchange "
                  << std::setprecision(1) << tpg->link().link_ms() << " ms (" << std::setprecision(3)
                  << (tp_steps > 0 ? tpg->link().link_ms() / static_cast<double>(tp_steps) : 0.0) << " ms/step); split steps "
                  << tp_steps << " decode_tok_s=" << std::setprecision(2)
                  << (tp_ms > 0 ? 1000.0 * static_cast<double>(tp_steps) / tp_ms : 0.0) << "\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "strix-hy4-rank: " << error.what() << "\n";
        return 1;
    }
}
