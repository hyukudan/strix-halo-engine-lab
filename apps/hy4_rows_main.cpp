// strix-hy4-rows: Hy4-flash verify rows on one node (hy4_rows).
//   bench (default): plain steps at G = 8 positions (logits digests), rewind, then forward_rows(R) for R = 1, 2, 4, 8 over
//     the same tokens: every row's logits digest and greedy id must equal the plain step's (ROWS_BITWISE n/15); then the
//     median wall of verify(R) and of one plain step at that position.
//   --spec N: plain greedy N ids (timed), reset, the same prompt again, speculative decode N ids through the shared
//     strix::spec layer (copy proposer + Arbiter, rows <= --max-rows): SPEC ids_equal, tok/s both, accepted per pass.
//   --mtp (with --prompt-rows): the pack's next-token block is loaded and drafts as the shared MtpChain (chain + hybrid
//     arms of the Arbiter, the copy source kept unless --mtp-copy 0); --chain-ms STEP,FLUSH seeds its walls, --survival
//     c1,c2,... its stated confidence (default: the host oracle's prose conditionals). MTP line: block calls / rows / wall.
//   --mtp-head full|q8|q4: the drafts' head (default q4; full = the BF16 head; q8 / q4 = the shared MTP draft head over a Q8_0 /
//     Q4_0 copy of output.weight). The taps run prints MTPHEAD: the three forms on the same block rows (agreement, ms).
//   --mtp-taps DIR (with --mtp, first content): prompt + --taps-gen N greedy ids through verify rows of 1 (bitwise the
//     step), the target hidden (output_norm) of every fed position written as a taps run of strix-hy4-dfly mtp; then the
//     oracle's acceptance protocol on the device block (teacher at every position, chains of --depth D at every greedy
//     anchor): MTPACC P(run >= k) / E(d), and nextn_top.txt / nextn_logits.f32 for strix-hy4-dfly mtp's device check.
// strix-hy4-rows PART.gguf... --ids-file F [--context N] [--prompt-rows N] [--reps K] [--spec N] [--max-rows R]
//   [--verify-ms a,b,...] [--gbps G] [--mtp [--mtp-head full|q8|q4] [--mtp-copy 0|1] [--chain-ms S,F] [--survival c,...] [--mtp-taps DIR
//   [--taps-gen N] [--depth D]]]; --content LABEL=PATH (repeatable) instead of --ids-file: one spec run per content
//   [--tp-rank 0|1 [--tp-transport usb4auto|tcp-listen:PORT|tcp-connect:HOST:PORT]]: the same run on a tensor-parallel
//     pair (each rank loads the whole pack): bench rows / plain steps on both ranks with their shares; --spec: rank 0
//     drafts and decides, rank 1 replays its verify passes (FOLLOW line), ids compared with plain greedy on rank 0.
//     [--prefill-route auto|stages|one|ring]: how the pair primes a prompt. stages: the layer-stage pipeline of the shared
//     StageMap (default_stage_map) over the pair link on each rank's whole-model forward, the caches re-laid out for the
//     TP decode (strix/pair/relayout.hpp), the drafter's teacher rows replayed, the tail as TP steps (one process, KV in
//     place; bits = the one-node prime). one: every rank primes the whole prompt one-node (the previous form). auto
//     (default): stages when relayout::stage_prime_wins (>= 2 full windows of >= 256 rows), else one. STAGES line: the split of the prime.
//     ring: the context-parallel window ring (strix/pair/window_ring.hpp: window k on rank k % 2, every layer, the peer's
//     cache rows of window k-1 before each layer; both ranks end with every row, no relayout; bits = the one-node prime;
//     needs two windows). RING line: run / wait / load / save split and the bytes moved.
#include "hy4/hip/hy4_hip_forward.hpp"
#include "hy4/hip/hy4_hip_rows.hpp"
#include "strix/spec/block_parallel.hpp"
#include "strix/dist/open_transport.hpp"
#include "strix/env_flags.hpp"
#include "strix/spec/pair_round.hpp"
#include "strix/pair/layer_pipeline.hpp"
#include "strix/pair/link_mux.hpp"
#include "strix/pair/relayout.hpp"
#include "strix/pair/stage_map.hpp"
#include "strix/pair/window_ring.hpp"
#include "xport/hip_pipe_staging.hpp"
#include "xport/tp_gather_hip.hpp"

#include <algorithm>
#include <array>
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
// --prefill-route on a TP pair: the gather's transport, this rank, the pack identity and the prompt rows.
struct StagePrime {
    strix::xport::tp_gather::HipPairGather* gather = nullptr;
    unsigned rank = 0;
    std::string route = "auto";  // auto | stages | one
    std::string pack;            // pack identity in the request digest (file size)
    std::uint32_t rows = 0;      // --prompt-rows (the windows)
};

void swap_words(strix::dist::Transport& t, unsigned rank, std::span<const std::byte> mine, std::span<std::byte> peer) {
    if (rank == 0U) {
        t.send(mine);
        t.receive(peer);
    } else {
        t.receive(peer);
        t.send(mine);
    }
}

// The end of a pair prime once every window's caches are on this rank: the drafter's teacher pass over the windows' rows
// (host `hidden`, prompt order, as prefill() feeds them), position Pw, then the tail as TP steps (or the last window's
// logits), the anchor.
struct PrimeEnd {
    std::uint32_t anchor = 0;
    double block_ms = 0, tail_ms = 0;
};
PrimeEnd prime_end(hb::Forward& f, const std::vector<std::uint32_t>& prompt, hb::rows::MtpSource<hb::Forward>* mtp, std::uint32_t Pw,
                   std::uint32_t W, const std::vector<strix::pair::layers::Window>& chunks, const std::vector<float>& hidden,
                   std::vector<float>& logits) {
    const double t0 = now_ms();
    const std::size_t h = f.config().hidden;
    const auto stream = static_cast<hipStream_t>(f.stream_handle());
    if (mtp) mtp->install(prompt);  // position 0: the teacher rows' first position
    f.seat(Pw);
    if (mtp) {  // the block's teacher pass: the windows' rows in prompt order, as prefill() feeds them
        float* dev = f.alloc_f32(std::size_t{W} * h);
        try {
            for (const auto& w : chunks) {
                if (hipMemcpyAsync(dev, hidden.data() + std::size_t{w.start} * h, std::size_t{w.rows} * h * sizeof(float), hipMemcpyHostToDevice,
                                   stream) != hipSuccess)
                    throw std::runtime_error("pair prime: teacher rows upload");
                f.feed_prompt_hidden(dev, w.start, w.rows);
            }
            f.sync();
        } catch (...) {
            f.free_f32(dev);
            throw;
        }
        f.free_f32(dev);
    }
    const double t_block = now_ms();
    if (Pw != prompt.size()) logits = f.prompt_tail(std::span<const std::uint32_t>(prompt).subspan(Pw));
    const auto anchor = argmax(logits.data(), logits.size());
    if (mtp) mtp->finish_prompt(anchor);
    return {anchor, t_block - t0, now_ms() - t_block};
}

// The pair prime as layer stages on this rank's whole-model forward (both ranks, same point, same prompt): the windows
// of the one-node prompt plan through the shared stage pipeline (StageMap default_stage_map, StagePipes over a LinkMux on
// the gather's transport), every layer's caches re-laid out to both ranks (strix/pair/relayout.hpp), the head rank's
// teacher rows to a drafting peer and replayed into the drafter (the same block calls in the same order as prefill()),
// the anchor logits from the last window's head (no tail) or the tail as TP steps. Bits = the one-node prime.
std::uint32_t prime_stages(hb::Forward& f, const std::vector<std::uint32_t>& prompt, hb::rows::MtpSource<hb::Forward>* mtp,
                           const StagePrime& sp) {
    namespace layers = strix::pair::layers;
    namespace relayout = strix::pair::relayout;
    const double t0 = now_ms();
    auto& t = sp.gather->host_transport();
    const unsigned rank = sp.rank;
    f.reset();
    const auto P = static_cast<std::uint32_t>(prompt.size());
    const std::uint32_t Pw = f.window_rows(P), W = sp.rows;
    if (Pw == 0U) throw std::invalid_argument("--prefill-route stages: the prompt has no full window");
    const std::size_t h = f.config().hidden;
    const auto chunks = layers::windows(Pw, W);
    const auto map = strix::pair::default_stage_map(f.config().layers, chunks.size());
    layers::check_stage_map(map, rank, true);
    const std::uint64_t identity = layers::fnv(
        layers::fnv(14695981039346656037ULL, "hy4-prime|" + sp.pack + "|" + map.text() + "|" + std::to_string(W)),
        std::span<const std::uint32_t>(prompt));
    // which ranks draft (their teacher pass needs the head rank's rows); doubles as the start barrier of the pipes
    std::array<std::uint64_t, 2> mine{identity, mtp ? 1ULL : 0ULL}, peer{};
    swap_words(t, rank, std::as_bytes(std::span(mine)), std::as_writable_bytes(std::span(peer)));
    if (peer[0] != identity) throw std::runtime_error("--prefill-route stages: the ranks prime different requests");
    const bool head_rank = rank == map.owner(map.stages() - 1U), no_tail = Pw == P;
    std::vector<float> hidden;
    const auto stream = static_cast<hipStream_t>(f.stream_handle());
    if (head_rank && (mtp || peer[1] != 0U)) {
        hidden.resize(std::size_t{Pw} * h);
        f.set_prompt_hidden([&hidden, h, stream](const float* rows, std::uint32_t p0, std::uint32_t n) {
            if (hipMemcpyAsync(hidden.data() + std::size_t{p0} * h, rows, std::size_t{n} * h * sizeof(float), hipMemcpyDeviceToHost, stream) !=
                    hipSuccess ||
                hipStreamSynchronize(stream) != hipSuccess)
                throw std::runtime_error("--prefill-route stages: teacher rows read");
        });
    }
    std::vector<float> logits;
    layers::StageOps ops;
    const std::size_t row_bytes = f.seam_row_bytes();
    ops.row_bytes = row_bytes;
    ops.range = [&](std::span<const std::uint32_t> tok, const layers::Window& w, std::uint32_t a, std::uint32_t b, bool head) {
        const auto& l = f.range_window(tok, w.start, a, b, head && no_tail);
        if (head && no_tail) logits = l;
    };
    ops.seam = [&f](strix::pair::PipeFrame& fr, bool save, std::size_t bytes) { fr.staging->copy(save, 0, f.range_seam(), bytes); };
    ops.fence = [&f] { f.sync(); };
    {
        strix::pair::LinkMux mux([&t](std::span<const std::byte> d) { t.send(d); }, [&t](std::span<std::byte> d) { t.receive(d); },
                                 [&t](const std::string& why) { t.abort(why); }, 2U, std::size_t{16} << 20U);
        const auto channel = [&mux](std::uint32_t ch) {
            return strix::pair::PipeLink{[&mux, ch](std::span<const std::byte> d) { mux.send(ch, d); },
                                         [&mux, ch](std::span<std::byte> d) { mux.receive(ch, d); },
                                         [&mux] { mux.abort("hy4 stage prime pipe stopped"); }};
        };
        strix::pair::PipeOptions po;
        po.validate = layers::stage_validator(identity, map, W, row_bytes);
        po.profile_label = "hy4-stage-prime";
        auto pipes = std::make_unique<strix::pair::StagePipes>(
            rank, map, chunks.size(), channel(0U), channel(1U), std::move(po), [stream, W, row_bytes] {
                return strix::pair::make_hip_pipe_staging(std::size_t{W} * row_bytes, [stream] { return stream; });
            });
        (void)layers::prefill(map, rank, std::span<const std::uint32_t>(prompt.data(), Pw), W, identity, *pipes, ops);
        f.sync();
        pipes.reset();
        mux.close();
    }
    f.set_prompt_hidden({});
    const double t_stages = now_ms();
    const auto plan = relayout::plan(map, rank, [&f, Pw](std::uint32_t l) { return f.cache_bytes(l, Pw); });
    relayout::Link lk{[&t](std::span<const std::byte> b) { t.send(b); }, [&t](std::span<std::byte> b) { t.receive(b); },
                      [&t](const std::string& why) { t.abort(why); }};
    relayout::Io io{[&f, Pw](std::uint32_t l, std::span<std::byte> out) { f.export_cache(l, Pw, out); },
                    [&f, Pw](std::uint32_t l, std::span<const std::byte> in) { f.import_cache(l, Pw, in); }};
    const auto rs = relayout::exchange(map, rank, identity, plan, lk, io);
    const double t_relayout = now_ms();
    // the head rank's teacher rows to a drafting peer, its anchor logits (no tail) to the peer
    if (head_rank && peer[1] != 0U) t.send(std::as_bytes(std::span(hidden)));
    if (!head_rank && mtp) {
        hidden.resize(std::size_t{Pw} * h);
        t.receive(std::as_writable_bytes(std::span(hidden)));
    }
    if (no_tail) {
        if (head_rank) t.send(std::as_bytes(std::span(logits)));
        else {
            logits.resize(f.config().vocab);
            t.receive(std::as_writable_bytes(std::span(logits)));
        }
    }
    const double t_hidden = now_ms();
    const auto end = prime_end(f, prompt, mtp, Pw, W, chunks, hidden, logits);
    const double t_end = now_ms();
    std::printf("STAGES rank %u map %s windows %zu rows %u tail %u: stages %.1f relayout %.1f (sent %.1f MB, received %.1f MB) "
                "teacher+logits %.1f block %.1f tail %.1f total %.1f ms\n",
                rank, map.text().c_str(), chunks.size(), W, P - Pw, t_stages - t0, t_relayout - t_stages, static_cast<double>(rs.sent) / 1e6,
                static_cast<double>(rs.received) / 1e6, t_hidden - t_relayout, end.block_ms, end.tail_ms, t_end - t0);
    return end.anchor;
}

// The pair prime as a context-parallel window ring (strix/pair/window_ring.hpp) on this rank's whole-model forward (both
// ranks, same point, same prompt): window k of the one-node prompt plan runs on rank k % 2 through every layer (resident
// range_window per layer); before layer L its rank loads the peer's layer-L cache rows of window k-1 (export_rows /
// import_rows: every slot window k-1 wrote), and the last window's rows cross once more, so both ranks end with every
// cache row: no relayout. Each window runs the one-node kernels on the one-node inputs: bits = the one-node prime at the
// same rows. Each rank's own windows' teacher rows go to a drafting peer; the last window's head logits (no tail) to the
// peer; the tail as TP steps.
std::uint32_t prime_ring(hb::Forward& f, const std::vector<std::uint32_t>& prompt, hb::rows::MtpSource<hb::Forward>* mtp,
                         const StagePrime& sp) {
    namespace layers = strix::pair::layers;
    namespace ring = strix::pair::ring;
    const double t0 = now_ms();
    auto& t = sp.gather->host_transport();
    const unsigned rank = sp.rank;
    f.reset();
    const auto P = static_cast<std::uint32_t>(prompt.size());
    const std::uint32_t Pw = f.window_rows(P), W = sp.rows;
    const auto chunks = layers::windows(Pw, W);
    if (chunks.size() < 2U) throw std::invalid_argument("--prefill-route ring: the prompt needs two full windows of --prompt-rows");
    const std::size_t h = f.config().hidden;
    const std::uint32_t L = f.config().layers;
    const std::uint64_t identity = layers::fnv(
        layers::fnv(14695981039346656037ULL, "hy4-ring|" + sp.pack + "|" + std::to_string(L) + "|" + std::to_string(W)),
        std::span<const std::uint32_t>(prompt));
    std::array<std::uint64_t, 2> mine{identity, mtp ? 1ULL : 0ULL}, peer{};
    swap_words(t, rank, std::as_bytes(std::span(mine)), std::as_writable_bytes(std::span(peer)));
    if (peer[0] != identity) throw std::runtime_error("--prefill-route ring: the ranks prime different requests");
    const bool no_tail = Pw == P;
    const std::size_t last = chunks.size() - 1U;
    std::vector<float> hidden;
    const auto stream = static_cast<hipStream_t>(f.stream_handle());
    if (mtp || peer[1] != 0U) {
        hidden.resize(std::size_t{Pw} * h);
        f.set_prompt_hidden([&hidden, h, stream](const float* rows, std::uint32_t p0, std::uint32_t n) {
            if (hipMemcpyAsync(hidden.data() + std::size_t{p0} * h, rows, std::size_t{n} * h * sizeof(float), hipMemcpyDeviceToHost, stream) !=
                    hipSuccess ||
                hipStreamSynchronize(stream) != hipSuccess)
                throw std::runtime_error("--prefill-route ring: teacher rows read");
        });
    }
    f.record_prompt(std::span<const std::uint32_t>(prompt.data(), Pw));
    std::vector<float> logits;
    ring::Ops ops;
    ops.run = [&](std::size_t k, const layers::Window& w, std::uint32_t l) {
        const bool head = no_tail && k == last && l + 1U == L;
        const auto& lg = f.range_window(std::span<const std::uint32_t>(prompt.data() + w.start, w.rows), w.start, l, l + 1U, head, true);
        if (head) logits = lg;
    };
    ops.bytes = [&f](const layers::Window& w, std::uint32_t l) { return static_cast<std::size_t>(f.rows_bytes(l, w.start, w.start + w.rows)); };
    ops.save = [&f](const layers::Window& w, std::uint32_t l, std::span<std::byte> out) { f.export_rows(l, w.start, w.start + w.rows, out); };
    ops.load = [&f](const layers::Window& w, std::uint32_t l, std::span<const std::byte> in) { f.import_rows(l, w.start, w.start + w.rows, in); };
    ring::Stats rs;
    {
        strix::pair::LinkMux mux([&t](std::span<const std::byte> d) { t.send(d); }, [&t](std::span<std::byte> d) { t.receive(d); },
                                 [&t](const std::string& why) { t.abort(why); }, 1U, std::size_t{16} << 20U);
        const ring::Link lk{[&mux](std::span<const std::byte> d) { mux.send(0U, d); }, [&mux](std::span<std::byte> d) { mux.receive(0U, d); },
                            [&mux](const std::string& why) { mux.abort(why); }};
        rs = ring::prefill(chunks, L, rank, identity, lk, ops);
        mux.close();
    }
    f.set_prompt_hidden({});
    const double t_ring = now_ms();
    // each rank's own windows' teacher rows to a drafting peer (rank 0 sends first: no crossing sends on the raw link)
    const auto send_own = [&] {
        for (std::size_t k = rank; k < chunks.size(); k += 2U)
            t.send(std::as_bytes(std::span(hidden.data() + std::size_t{chunks[k].start} * h, std::size_t{chunks[k].rows} * h)));
    };
    const auto receive_peer = [&] {
        for (std::size_t k = 1U - rank; k < chunks.size(); k += 2U)
            t.receive(std::as_writable_bytes(std::span(hidden.data() + std::size_t{chunks[k].start} * h, std::size_t{chunks[k].rows} * h)));
    };
    if (rank == 0U) {
        if (peer[1] != 0U) send_own();
        if (mtp) receive_peer();
    } else {
        if (mtp) receive_peer();
        if (peer[1] != 0U) send_own();
    }
    if (no_tail) {
        if (rank == ring::owner(last)) t.send(std::as_bytes(std::span(logits)));
        else {
            logits.resize(f.config().vocab);
            t.receive(std::as_writable_bytes(std::span(logits)));
        }
    }
    const double t_hidden = now_ms();
    const auto end = prime_end(f, prompt, mtp, Pw, W, chunks, hidden, logits);
    const double t_end = now_ms();
    std::printf("RING rank %u windows %zu (here %zu) rows %u tail %u: ring %.1f (run %.1f wait %.1f load %.1f save %.1f send-wait %.1f; "
                "sent %.1f MB, received %.1f MB) teacher+logits %.1f block %.1f tail %.1f total %.1f ms\n",
                rank, chunks.size(), rs.windows_here, W, P - Pw, t_ring - t0, rs.run_ms, rs.wait_ms, rs.load_ms, rs.save_ms, rs.send_wait_ms,
                static_cast<double>(rs.sent) / 1e6, static_cast<double>(rs.received) / 1e6, t_hidden - t_ring, end.block_ms, end.tail_ms,
                t_end - t0);
    return end.anchor;
}

// The prompt into a fresh forward: position by position (class BASE) or the batched prompt windows (--prompt-rows).
// Returns the anchor = greedy id after the last prompt token. sp (TP pair): --prefill-route (prime_stages).
std::uint32_t prime(hb::Forward& f, const std::vector<std::uint32_t>& prompt, bool batched,
                    hb::rows::MtpSource<hb::Forward>* mtp = nullptr, const StagePrime* sp = nullptr) {
    if (batched && sp && sp->route == "ring") return prime_ring(f, prompt, mtp, *sp);
    if (batched && sp && sp->route != "one") {
        std::uint64_t full = 0;
        for (const auto& w : strix::pair::layers::windows(f.window_rows(static_cast<std::uint32_t>(prompt.size())), sp->rows))
            full += w.rows == sp->rows ? 1U : 0U;
        if (sp->route == "stages" || strix::pair::relayout::stage_prime_wins(full, sp->rows)) return prime_stages(f, prompt, mtp, *sp);
    }
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
const char* head_name(hb::NextnHead head) { return head == hb::NextnHead::q8 ? "q8" : head == hb::NextnHead::q4 ? "q4" : "full"; }

int mtp_taps(hb::Forward& f, const std::vector<std::uint32_t>& prompt, const std::string& dir, std::uint32_t gen, std::uint32_t depth,
             hb::NextnHead head) {
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
    {  // rows form vs the per-row loop of the next-token block (Forward::nextn_rows) over the first fed positions: the
       // teacher's calls, bitwise (hidden_pre, hidden_post, the last row's greedy id); walls of the second run of each form.
       // The teacher loop below rewrites every block cache row from position 0.
        const std::uint32_t n = std::min<std::uint32_t>(hb::Forward::kNextnRows, P - 1U);
        std::vector<float> pre[2], post[2];
        std::uint32_t top[2]{};
        double wall[2]{};
        for (int form = 0; form < 2; ++form) {
            f.set_nextn_row_loop(form == 0);
            for (int rep = 0; rep < 2; ++rep) {
                const double t0 = now_ms();
                const auto o = f.nextn_rows(ids.data() + 1, hid, n, 0U, true);
                f.read_ids(o.top + (n - 1U), 1U, &top[form]);
                wall[form] = now_ms() - t0;
                pre[form].resize(std::size_t{n} * h);
                post[form].resize(std::size_t{n} * h);
                f.read_f32(o.hidden_pre, pre[form].size(), pre[form].data());
                f.read_f32(o.hidden_post, post[form].size(), post[form].data());
            }
        }
        f.set_nextn_row_loop(false);
        const auto same = [](const std::vector<float>& a, const std::vector<float>& b) {
            return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
        };
        std::printf("MTPROWS rows %u pre_equal %d post_equal %d top %u %u | row loop %.2f ms rows form %.2f ms (%.3f vs %.3f ms/row)\n", n,
                    same(pre[0], pre[1]) ? 1 : 0, same(post[0], post[1]) ? 1 : 0, top[0], top[1], wall[0], wall[1], wall[0] / n, wall[1] / n);
    }
    if (P >= 3U) {  // KV-only rows (outputs = 1, the flush's form) vs the per-row loop computing every row: the last row's
                    // pre / post / greedy id, then one chained step at position n over the n cache rows (its pre / post / id);
                    // walls of the second run of each form (the n-row call alone)
        const std::uint32_t n = std::min<std::uint32_t>(hb::Forward::kNextnRows, P - 2U);
        std::vector<float> pre[2], post[2], spre[2], spost[2];
        std::uint32_t top[2]{}, stop[2]{};
        double wall[2]{};
        for (int form = 0; form < 2; ++form) {
            f.set_nextn_row_loop(form == 0);
            for (int rep = 0; rep < 2; ++rep) {
                const double t0 = now_ms();
                const auto o = f.nextn_rows(ids.data() + 1, hid, n, 0U, true, 1U);
                f.read_ids(o.top + (n - 1U), 1U, &top[form]);
                wall[form] = now_ms() - t0;
                pre[form].resize(h);
                post[form].resize(h);
                f.read_f32(o.hidden_pre + std::size_t{n - 1U} * h, h, pre[form].data());
                f.read_f32(o.hidden_post + std::size_t{n - 1U} * h, h, post[form].data());
                f.copy_f32_async(carry, o.hidden_post + std::size_t{n - 1U} * h, h);
                const auto so = f.nextn_rows(&ids[n + 1U], carry, 1U, n, true);
                f.read_ids(so.top, 1U, &stop[form]);
                spre[form].resize(h);
                spost[form].resize(h);
                f.read_f32(so.hidden_pre, h, spre[form].data());
                f.read_f32(so.hidden_post, h, spost[form].data());
            }
        }
        f.set_nextn_row_loop(false);
        const auto same = [](const std::vector<float>& a, const std::vector<float>& b) {
            return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
        };
        std::printf("MTPKV rows %u last pre_equal %d post_equal %d top %u %u | step pre_equal %d post_equal %d top %u %u | "
                    "row loop %.2f ms kv-only %.2f ms\n",
                    n, same(pre[0], pre[1]) ? 1 : 0, same(post[0], post[1]) ? 1 : 0, top[0], top[1], same(spre[0], spre[1]) ? 1 : 0,
                    same(spost[0], spost[1]) ? 1 : 0, stop[0], stop[1], wall[0], wall[1]);
    }
    if (P >= 3U) {  // the drafts' head forms on the same block rows (one-row teacher calls at positions 0 .. m-1, input token ids[r + 1],
       // hidden row r; top predicts ids[r + 2]): top agreement with the full BF16 head, correct tops, and ms per call (the
       // second pass of each form; the first builds the q8 / q4 copy). Every form rewrites the same cache rows in order.
        const std::uint32_t m = P - 2U;
        constexpr hb::NextnHead forms[3]{hb::NextnHead::full, hb::NextnHead::q8, hb::NextnHead::q4};
        std::vector<std::uint32_t> tops[3];
        double ms[3]{};
        for (int k = 0; k < 3; ++k) {
            f.set_nextn_head(forms[k]);
            tops[k].assign(m, 0U);
            for (int rep = 0; rep < 2; ++rep) {
                const double t0 = now_ms();
                for (std::uint32_t r = 0; r < m; ++r) {
                    const auto o = f.nextn_rows(&ids[r + 1U], hid + std::size_t{r} * h, 1U, r, true);
                    f.read_ids(o.top, 1U, &tops[k][r]);
                }
                ms[k] = (now_ms() - t0) / m;
            }
        }
        f.set_nextn_head(head);
        std::uint32_t agree[3]{}, correct[3]{};
        for (int k = 0; k < 3; ++k)
            for (std::uint32_t r = 0; r < m; ++r) {
                agree[k] += tops[k][r] == tops[0][r] ? 1U : 0U;
                correct[k] += tops[k][r] == ids[r + 2U] ? 1U : 0U;
            }
        std::printf("MTPHEAD rows %u agree q8 %u q4 %u | correct full %u q8 %u q4 %u | ms/call full %.3f q8 %.3f q4 %.3f (head saves q8 %.3f q4 %.3f)\n",
                    m, agree[1], agree[2], correct[0], correct[1], correct[2], ms[0], ms[1], ms[2], ms[0] - ms[1], ms[0] - ms[2]);
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
        // tensor-parallel pair (Forward::enable_tp on both ranks, each holding the whole pack): rank 0 drafts and decides,
        // both ranks run every verify pass with their shares (strix/spec/pair_round.hpp); bench mode runs on both ranks
        int tp_rank = -1;
        std::string tp_spec = "usb4auto";
        std::string prefill_route = "auto";
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
            else if (a == "--tp-rank" && more) tp_rank = static_cast<int>(std::stoul(argv[++i]));
            else if (a == "--tp-transport" && more) tp_spec = argv[++i];
            else if (a == "--prefill-route" && more) {
                prefill_route = argv[++i];
                if (prefill_route != "auto" && prefill_route != "stages" && prefill_route != "one" && prefill_route != "ring")
                    throw std::invalid_argument("--prefill-route auto|stages|one|ring");
            }
            else if (a == "--mtp-head" && more) {
                const std::string v = argv[++i];
                if (v == "full") o.nextn_head = hb::NextnHead::full;
                else if (v == "q8") o.nextn_head = hb::NextnHead::q8;
                else if (v == "q4") o.nextn_head = hb::NextnHead::q4;
                else throw std::invalid_argument("--mtp-head full|q8|q4");
            }
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
        std::printf("load %.1f s, %u verify rows, prompt %zu ids (%s), context %u, draft head %s\n", (now_ms() - t_load) / 1000.0,
                    o.verify_rows, prompt.size(), o.prompt_rows ? "batched" : "position by position", o.context, head_name(o.nextn_head));
        const auto vocab = static_cast<std::size_t>(f.config().vocab);
        std::unique_ptr<strix::xport::tp_gather::HipPairGather> tpg;
        strix::spec::pair_round::Exchange pair_ex;
        if (tp_rank >= 0) {
            if (tp_rank > 1 || !taps_dir.empty()) throw std::invalid_argument("--tp-rank 0|1 (bench or --spec; not with --mtp-taps)");
            const auto rank = static_cast<std::uint32_t>(tp_rank);
            auto opened = strix::dist::open_pair(tp_spec, rank);
            if (!opened.receipt.empty()) std::printf("tp transport: %s\n", opened.receipt.c_str());
            // largest gather: a verify pass's MoE join at 8 rows (8 x (8 slot rows x 1536 + shared / 2) FP32) < 2 MiB
            const auto stream = static_cast<hipStream_t>(f.stream_handle());
            tpg = std::make_unique<strix::xport::tp_gather::HipPairGather>(std::move(opened), rank, std::size_t{2} << 20, stream,
                                                                           strix::envcfg::flag("STRIX_TP_CHANNEL", true));
            f.enable_tp(rank, {&tpg->link(), &strix::xport::tp_gather::HipPairGather::Link::fn});
            std::printf("%s\n", f.tp_receipt().c_str());
            // the round message rides the same gather: [2][kSlot] bytes on the device, stream-ordered after the last pass
            constexpr std::size_t kSlot = 128;
            static_assert(strix::spec::pair_round::kMessageBytes <= kSlot);
            auto* slots = reinterpret_cast<std::uint8_t*>(f.alloc_f32(2U * kSlot / 4U));
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
        const bool follower = tp_rank == 1;
        const bool batched = o.prompt_rows != 0U;
        StagePrime stage_prime;  // TP pair only: the prompt as layer stages + state re-layout (--prefill-route)
        const StagePrime* sp = nullptr;
        if (tpg) {
            stage_prime.gather = tpg.get();
            stage_prime.rank = static_cast<unsigned>(tp_rank);
            stage_prime.route = prefill_route;
            stage_prime.pack = std::to_string(std::filesystem::file_size(o.parts.front()));
            stage_prime.rows = o.prompt_rows;
            sp = &stage_prime;
            std::printf("prefill route %s (pair: layer stages + state re-layout when chosen)\n", prefill_route.c_str());
        }
        if (!taps_dir.empty()) return mtp_taps(f, prompt, taps_dir, taps_gen, depth, o.nextn_head);
        const double prime_b0 = now_ms();
        const std::uint32_t anchor = prime(f, prompt, batched, nullptr, sp);
        std::printf("PRIME first prompt %zu ids %.1f ms anchor %u\n", prompt.size(), now_ms() - prime_b0, anchor);
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
            {  // the plain steps after the prime: compared across prime routes (bits of the primed state)
                std::uint64_t all = 1469598103934665603ULL;
                for (const auto v : plain_fnv) all = (all ^ v) * 1099511628211ULL;
                std::printf("PLAIN_STEPS fnv %016llx ids", static_cast<unsigned long long>(all));
                for (std::uint32_t i = 1; i <= G; ++i) std::printf(" %u", t[i]);
                std::printf("\n");
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
            const std::uint32_t first_id = ci == 0U ? anchor : prime(f, ids, batched, nullptr, sp);
            std::vector<std::uint32_t> plain{first_id};
            const double p0 = now_ms();
            while (plain.size() < spec_n) plain.push_back(f.step_greedy(plain.back()));
            const double plain_ms = now_ms() - p0;
            const double step_ms = plain_ms / static_cast<double>(spec_n - 1U);
            hb::rows::SpecOptions so;
            so.max_rows = max_rows;
            so.verify_ms = verify_ms.empty() ? strix::spec::moe_verify_seed_ms(strix::spec::hy4_verify_bytes(), step_ms, gbps, max_rows) : verify_ms;
            std::unique_ptr<hb::rows::MtpSource<hb::Forward>> source;
            std::uint64_t plain_fnv = 1469598103934665603ULL;  // the plain ids (compared across one node / pair runs)
            for (const auto id : plain) plain_fnv = (plain_fnv ^ id) * 1099511628211ULL;
            // tensor-parallel drafter (both ranks; STRIX_HY4_TP_DRAFT=0: rank 0 alone, one-node block): both ranks must agree
            so.draft_tp = tp_rank >= 0 && mtp && strix::envcfg::flag("STRIX_HY4_TP_DRAFT", true);
            if (follower) {  // rank 1: the same prompt pass (with the block when the drafter runs on both), then rank 0's rounds
                std::unique_ptr<hb::rows::MtpSource<hb::Forward>> fsrc;
                if (so.draft_tp) fsrc = std::make_unique<hb::rows::MtpSource<hb::Forward>>(f);
                const double prime_t1 = now_ms();
                if (prime(f, ids, batched, fsrc.get(), sp) != first_id) throw std::runtime_error("repeat prompt anchor differs (" + label + ")");
                std::printf("PRIME %s prompt %zu ids %.1f ms (tp rank 1%s)\n", label.c_str(), ids.size(), now_ms() - prime_t1,
                            fsrc ? ", with the block" : "");
                const double t1 = now_ms();
                const auto fr = fsrc ? hb::rows::spec_follow(f, pair_ex, *fsrc) : hb::rows::spec_follow(f, pair_ex);
                const double wall = now_ms() - t1;
                std::printf("FOLLOW %s rounds %llu passes %llu rows %llu committed %llu drafts %llu wall %.1f ms (position checked%s)\n",
                            label.c_str(), static_cast<unsigned long long>(fr.rounds), static_cast<unsigned long long>(fr.passes),
                            static_cast<unsigned long long>(fr.rows), static_cast<unsigned long long>(fr.committed),
                            static_cast<unsigned long long>(fr.drafts), wall, fsrc ? ", draft ids cross-checked" : "");
                std::printf("PLAIN %s tp rank 1 %.2f tok/s (%.2f ms/token) ids_fnv %016llx\n", label.c_str(), 1000.0 / step_ms, step_ms,
                            static_cast<unsigned long long>(plain_fnv));
                std::fflush(stdout);
                continue;
            }
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
            const double prime_t0 = now_ms();  // the spec path's prompt pass (with the block: its teacher flushes when the ring fills)
            if (prime(f, ids, batched, source.get(), sp) != first_id) throw std::runtime_error("repeat prompt anchor differs (" + label + ")");
            std::printf("PRIME %s prompt %zu ids %.1f ms%s\n", label.c_str(), ids.size(), now_ms() - prime_t0, source ? " (with the block)" : "");
            const auto* pex = tp_rank == 0 ? &pair_ex : nullptr;
            const auto res = source ? hb::rows::spec_decode(f, ids, first_id, spec_n, so, *source, pex)
                                    : hb::rows::spec_decode(f, ids, first_id, spec_n, so, pex);
            std::printf("PLAIN %s %s %.2f tok/s (%.2f ms/token) ids_fnv %016llx\n", label.c_str(), tp_rank == 0 ? "tp rank 0" : "one node",
                        1000.0 / step_ms, step_ms, static_cast<unsigned long long>(plain_fnv));
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
                std::printf("MTP %s block calls %llu rows %llu drafts %llu id reads %llu draft wall %.1f ms (%.2f ms/draft) chain %s seeds step %.1f flush %.1f%s\n",
                            label.c_str(), static_cast<unsigned long long>(res.chain_calls), static_cast<unsigned long long>(res.chain_rows),
                            static_cast<unsigned long long>(res.chain_drafts), static_cast<unsigned long long>(res.chain_reads), res.chain_ms,
                            res.chain_drafts ? res.chain_ms / static_cast<double>(res.chain_drafts) : 0.0, so.copy ? "+copy+hybrid" : "only",
                            so.chain_step_ms, so.chain_flush_ms, so.draft_tp ? " block tp" : (tp_rank == 0 ? " block rank 0" : ""));
            if (mtp)  // where a pass's wall goes: verify passes, drafter (flush + chain), the rest (round message, host)
                std::printf("WALL %s total %.1f ms verify %.1f draft %.1f other %.1f (per pass %.2f / %.2f / %.2f)\n", label.c_str(), res.wall_ms,
                            res.verify_ms, res.chain_ms, res.wall_ms - res.verify_ms - res.chain_ms,
                            res.verify_ms / static_cast<double>(std::max<std::size_t>(res.steps.size(), 1U)),
                            res.chain_ms / static_cast<double>(std::max<std::size_t>(res.steps.size(), 1U)),
                            (res.wall_ms - res.verify_ms - res.chain_ms) / static_cast<double>(std::max<std::size_t>(res.steps.size(), 1U)));
            std::fflush(stdout);
            if (!equal) rc = 3;
        }
        return rc;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strix-hy4-rows: %s\n", e.what());
        return 1;
    }
}
