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
#include "strix/pair/layer_pipeline.hpp"
#include "strix/pair/link.hpp"
#include "strix/pair/link_flags.hpp"
#include "strix/profiles/hy4.hpp"
#include "strix/spec/drafting.hpp"
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
             const std::vector<strix::hy4::hipbind::PairWindowCost>& window_costs) {
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

int run(int argc, char** argv) {
    const auto kept = strix::pair::strip_flags(argc, argv, "HY4", false);  // --rank / --peer / --transport -> HY4_*
    (void)kept;
    strix::hy4::hipbind::ForwardOptions o;
    std::vector<std::uint32_t> prompt, teacher;
    std::uint32_t gen = 0, top = 5, cut = 0, repeat = 1;
    std::uint16_t port = 29755;
    bool timing = false, rows_explicit = false, pipe_guess = false;
    GateRef gref;           // --gate-ref (long BASE reference in pieces)
    bool head_explicit = false;
    std::string dump_path;  // --dump-logits F: every host logits row (FP32, vocab floats per step) appended to F
    std::vector<strix::hy4::hipbind::PairWindowCost> window_costs;  // --window-costs ROWS:MS,... (pair calibration)
    // The tier word (--profile default|fast|quality, --allow-pending) through the shared resolver; the lever options
    // stay in args and are parsed below (an explicit option wins over the profile).
    std::vector<std::string> args(argv + 1, argv + argc);
    strix::profile::Inputs inputs;
    strix::profile::take_options(strix::hy4::profile::engine(), args, inputs);
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        const bool more = i + 1 < args.size();
        if (a == "--ids-file" && more) prompt = read_ids(args[++i]);
        else if (a == "--teacher" && more) teacher = read_ids(args[++i]);
        else if (a == "--gen" && more) gen = static_cast<std::uint32_t>(std::stoul(args[++i]));
        else if (a == "--layers" && more) o.depth = static_cast<std::uint32_t>(std::stoul(args[++i]));
        else if (a == "--context" && more) o.context = static_cast<std::uint32_t>(std::stoul(args[++i]));
        else if (a == "--top" && more) top = static_cast<std::uint32_t>(std::stoul(args[++i]));
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
        else if (a == "--repeat" && more) repeat = static_cast<std::uint32_t>(std::stoul(args[++i]));
        else if (a == "--port" && more) port = static_cast<std::uint16_t>(std::stoul(args[++i]));
        else if (a == "--pipe-guess" && more) {
            const std::string v = args[++i];
            if (v != "copy" && v != "off") throw std::invalid_argument("--pipe-guess copy|off");
            pipe_guess = v == "copy";
        }
        else if (!a.empty() && a[0] != '-') o.parts.emplace_back(a);
        else throw std::invalid_argument("unknown argument " + a);
    }
    // A pair width picked from --window-costs is a prompt_rows setting like --prompt-rows: the resolver stamps it.
    if (std::getenv("HY4_RANK") != nullptr && !rows_explicit) {
        if (const std::uint32_t picked = strix::hy4::hipbind::pair_window_rows(prompt.size(), window_costs); picked != 0U)
            inputs.cli.emplace_back("prompt_rows", std::to_string(picked));
    }
    // The regime (device, topology, this prompt's length) selects the qualified FAST cells (strix/profiles/hy4.hpp).
    namespace hp = strix::hy4::profile;
    hp::Regime regime;
    regime.topology = std::getenv("HY4_RANK") != nullptr ? hp::Topology::pair : hp::Topology::one_node;
    regime.prompt_tokens = prompt.size();
    {
        int dev = 0;
        hipDeviceProp_t prop{};
        if (hipGetDevice(&dev) == hipSuccess && hipGetDeviceProperties(&prop, dev) == hipSuccess) regime.device = hp::device_of(prop.gcnArchName);
    }
    const auto resolved = hp::resolve(regime, inputs);
    const auto& profile = resolved->resolution;
    if (const auto* e = resolved->find("prompt_rows"); e != nullptr && !rows_explicit && strix::profile::applied(e->status))
        o.prompt_rows = static_cast<std::uint32_t>(std::stoul(e->value));
    if (const auto* e = resolved->find("head_q8"); e != nullptr && !head_explicit && strix::profile::applied(e->status)) o.head_q8 = e->value == "1";
    strix::profile::apply(profile);  // environment levers the profile sets (pf_mha), before any forward reads them
    std::cout << profile.describe() << "regime " << hp::device_name(regime.device) << " / " << hp::topology_name(regime.topology)
              << " / prompt " << regime.prompt_tokens << "\n" << std::flush;
    for (const auto& n : profile.notices()) std::cerr << n << '\n';
    for (const auto& e : profile.entries)
        if (!e.reason.empty()) std::cerr << e.lever->name << ": " << e.reason << '\n';
    if (o.parts.empty() || prompt.empty())
        throw std::invalid_argument("usage: strix-hy4-rank PART1.gguf [PART2.gguf] --ids-file F [--gen N] [--teacher F] [--layers D] "
                                    "[--context C] [--top K] [--timing] [--prompt-rows R] [--dump-logits F] [--profile default|fast] "
                                    "[--rank 0|1 --cut K --repeat K --port P --pipe-guess copy|off --window-costs ROWS:MS,...]");
    if (std::getenv("HY4_RANK") != nullptr)
        return run_pair(o, prompt, gen, cut, std::max(repeat, 1U), port, rows_explicit, pipe_guess, window_costs);
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
    // --repeat K (batched prompt): K - 1 earlier passes, each followed by reset() (bit-reproducing), so the last pass is
    // the warm figure and the first the cold one; their last-position digests must equal the final pass's.
    for (std::uint32_t r = 1; o.prompt_rows != 0U && r < repeat; ++r) {
        const auto s0 = clock::now();
        const std::uint64_t digest = fnv(fwd.prefill(prompt));
        const double ms = std::chrono::duration<double, std::milli>(clock::now() - s0).count();
        std::cout << "prefill run " << r << " of " << repeat << " tokens=" << prompt.size() << " ms=" << std::setprecision(1) << ms
                  << " tok_s=" << 1000.0 * static_cast<double>(prompt.size()) / ms << " logits_fnv64 " << std::hex << digest << std::dec
                  << "\n";
        fwd.reset();
    }
    if (o.prompt_rows != 0U) {  // batched prompt: one call; the last prompt position is printed like a step
        const auto s0 = clock::now();
        prompt_logits = fwd.prefill(prompt);
        prompt_ms = std::chrono::duration<double, std::milli>(clock::now() - s0).count();
        const auto& t = fwd.times();
        std::cout << "prefill tokens=" << prompt.size() << " rows=" << o.prompt_rows << " ms=" << std::setprecision(1) << prompt_ms
                  << " tok_s=" << std::setprecision(1) << 1000.0 * static_cast<double>(prompt.size()) / prompt_ms << " [windows embed "
                  << t.embed << " engram " << t.engram << " attn " << t.attention << " ffn " << t.ffn << " head " << t.head << " total "
                  << t.total << "]\n";
        p0 = prompt.size() - 1U;
    }
    for (std::size_t p = p0; p < steps; ++p) {
        if (p < prompt.size()) token = prompt[p];
        const auto s0 = clock::now();
        const bool batched = o.prompt_rows != 0U && p + 1U == prompt.size();
        const bool device = top == 0U && !batched;  // --top 0: the shared device greedy row, no logits on the host
        const std::uint32_t device_next = device ? fwd.step_greedy(token) : 0U;
        const auto& logits = device ? no_logits : batched ? prompt_logits : fwd.step(token, state);
        const double step_ms = batched ? 0.0 : std::chrono::duration<double, std::milli>(clock::now() - s0).count();
        if (dump && !logits.empty())
            dump.write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(logits.size() * sizeof(float)));
        (p < prompt.size() ? prompt_ms : gen_ms) += step_ms;
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
    std::cout << "ids";
    for (const auto id : generated) std::cout << ' ' << id;
    const auto gen_steps = gen > 1U ? gen - 1U : 0U;
    std::cout << "\n" << std::setprecision(1) << "time prompt_ms=" << prompt_ms << " gen_ms=" << gen_ms
              << " total_ms=" << std::chrono::duration<double, std::milli>(clock::now() - t0).count() << " decode_tok_s="
              << std::setprecision(2) << (gen_ms > 0 ? 1000.0 * gen_steps / gen_ms : 0.0) << "\n";
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
