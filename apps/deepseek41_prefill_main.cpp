#include "strix/deepseek41_device_forward.hpp"
#include "strix/deepseek41_prefill.hpp"
#include "strix/dist/communicator.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

// deepseek41 prefill of a prompt, one node (world 1, every expert of the layer range resident) or one rank of two
// (expert halves, half sums exchanged per chunk), or both ranks in one process (--transport emulate).
//   strix-deepseek41-prefill MODEL.gguf --prompt FILE (comma-separated ids) [--rows N] [--layers L]
//       [--world 1 | --rank R --transport usb4-auto | tcp-listen:PORT | tcp-connect:HOST:PORT | emulate]
//       [--chunk C] [--window W] [--lanes N] [--moe-chunk R] [--no-tp] [--row-split | --no-row-split] [--stream-experts] [--timing] [--reference] [--repeat K] [--connect-timeout S] [--no-cone] [--warmup N] [--split-min-rows N] [--joint-max-rows N] [--state-timing] [--loads] [--loads-out CSV] [--stream-hashes-out FILE]
//       [--ttft-timing [--ttft-warmup R] [--ttft-decode G]]
// --ttft-timing: one "ttft" line per stage from the process start to the first token (exec to main, prompt read, transport,
// communicator, weights, barrier, each run's wall time on the host); --ttft-warmup R: one run of the prompt's first R rows
// before the measured runs (the first run of a process pays one-time costs); --ttft-decode G: after the runs, the state
// goes to a decode forward (export, prefill freed, decode weights, import) and G greedy steps from the prefill's next
// token are timed one by one (the first step pays the decode forward's one-time costs).
// --reference (world 1 or emulate): the decode forward fed the same tokens one by one; the last row's logits, the last
// row's stream after every layer and every layer's attention state must be equal bit for bit.
namespace {

// Both ranks here before either goes on: a 4-byte exchange, waited for.
void barrier(strix::dist::Communicator& comm, std::uint32_t round) {
    // one buffer and stream per thread for the process (emulate: one per rank thread), not one per call
    thread_local void* buffers = nullptr;
    thread_local hipStream_t stream = nullptr;
    if (buffers == nullptr && (hipMalloc(&buffers, 8) != hipSuccess || hipStreamCreate(&stream) != hipSuccess)) throw std::runtime_error("barrier buffers");
    (void)hipMemsetAsync(buffers, 0, 8, stream);
    comm.exchange(stream, buffers, static_cast<char*>(buffers) + 4, 4,
                  strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 2U, round});
    (void)hipStreamSynchronize(stream);
}

struct Args {
    std::string model, prompt_path, transport, loads_out, hashes_out;
    std::uint32_t rank = 0, world = 1, layers = 40, rows = 0, chunk = 1024, window = 8, lanes = 4, moe_chunk = 4096, repeat = 1, warmup = 0,
                  split_min_rows = strix::deepseek41::PrefillOptions{}.split_min_rows,
                  joint_max_rows = strix::deepseek41::PrefillOptions{}.joint_max_rows;
    unsigned connect_timeout = 60;
    bool timing = false, reference = false, cone = true, loads = false, tp = true, stream_experts = false, row_split = true, decode_reference = false,
         state_timing = false;
    bool ttft = false;  // --ttft-timing
    std::uint32_t ttft_warmup = 0, ttft_decode = 0;
};

// --ttft-timing: wall time of each stage (host clock), printed as "ttft <stage>: <ms since the previous line> ms (at <ms
// since main> ms)".
struct TtftClock {
    bool on = false;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now(), last = t0;
    static double ms(std::chrono::steady_clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }
    void line(const char* what) {
        const auto now = std::chrono::steady_clock::now();
        if (on) {
            std::printf("ttft %s: %.3f ms (at %.3f ms)\n", what, ms(now - last), ms(now - t0));
            std::fflush(stdout);
        }
        last = now;
    }
    // exec to main: the process start time of /proc/self/stat (clock ticks since boot) against /proc/uptime
    void exec_to_main() const {
        if (!on) return;
        std::ifstream st("/proc/self/stat"), up("/proc/uptime");
        const std::string s((std::istreambuf_iterator<char>(st)), std::istreambuf_iterator<char>());
        double uptime = 0.0;
        up >> uptime;
        const auto close = s.rfind(')');
        if (close == std::string::npos || uptime <= 0.0) return;
        std::istringstream rest(s.substr(close + 2));
        std::string field;
        const double tick = static_cast<double>(::sysconf(_SC_CLK_TCK));
        for (int i = 3; i <= 22 && rest >> field; ++i)
            if (i == 22) std::printf("ttft exec to main: %.0f ms (clock ticks of %.0f ms)\n", (uptime - std::stod(field) / tick) * 1e3, 1e3 / tick);
    }
};
TtftClock ttft_clock;

std::uint64_t fnv(const std::vector<float>& v) {
    std::uint64_t h = 14695981039346656037ULL;
    for (const float f : v) {
        std::uint32_t b;
        std::memcpy(&b, &f, 4);
        for (int k = 0; k < 4; ++k, b >>= 8U) h = (h ^ (b & 0xffU)) * 1099511628211ULL;
    }
    return h;
}

std::vector<std::uint32_t> read_prompt(const std::string& path, std::uint32_t rows) {
    std::ifstream in(path);
    if (!in) throw std::invalid_argument("cannot read " + path);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<std::uint32_t> ids;
    std::stringstream ss(text);
    for (std::string t; std::getline(ss, t, ',');)
        if (!t.empty()) ids.push_back(static_cast<std::uint32_t>(std::stoul(t)));
    if (rows != 0U) {
        if (rows > ids.size()) throw std::invalid_argument("prompt shorter than --rows");
        ids.resize(rows);
    }
    return ids;
}

std::unique_ptr<strix::dist::Transport> open_transport(const std::string& spec, std::uint32_t rank, unsigned timeout, std::string& receipt) {
    const std::chrono::seconds limit(timeout);
    const auto colon = spec.find(':');
    const std::string kind = spec.substr(0, colon), rest = colon == std::string::npos ? "" : spec.substr(colon + 1);
    if (kind == "tcp-listen") return strix::dist::tcp_listen(static_cast<std::uint16_t>(std::stoul(rest)), limit);
    if (kind == "tcp-connect") {
        const auto c2 = rest.rfind(':');
        return strix::dist::tcp_connect(rest.substr(0, c2), static_cast<std::uint16_t>(std::stoul(rest.substr(c2 + 1))), limit);
    }
    if (kind == "usb4-auto") {
        strix::dist::Usb4AutoOptions o;
        o.rank = rank;
        o.timeout = std::chrono::milliseconds(std::uint64_t{timeout} * 1000U);
        strix::dist::Usb4Topology topology;
        auto t = strix::dist::usb4_host_auto(o, &topology);
        receipt = topology.receipt();
        return t;
    }
    throw std::invalid_argument("unknown transport " + spec);
}

void print_result(const char* who, const strix::deepseek41::PrefillResult& r, std::size_t rows, bool timing, const std::string& loads_out = {}) {
    std::printf("%s: %zu rows in %.3f s = %.1f tok/s (engram read %.3f s, head %.3f s; %llu row-layers); next token %u, logits_fnv %016llx\n",
                who, rows, r.seconds, static_cast<double>(rows) / r.seconds, r.engram_seconds, r.head_seconds,
                static_cast<unsigned long long>(r.row_layers), r.next_token, static_cast<unsigned long long>(fnv(r.logits)));
    if (r.engram_wait_seconds > 0.0) std::printf("  engram read overlapped: waited %.3f s for it\n", r.engram_wait_seconds);
    if (r.expert_read_seconds > 0.0)
        std::printf("  streamed experts: %.3f s of file reads on the prefetch thread, %.3f s waited for them\n", r.expert_read_seconds,
                    r.expert_wait_seconds);
    if (!r.routing_loads.empty() && !loads_out.empty()) {  // one line per layer: the entries of every expert
        std::FILE* f = std::fopen(loads_out.c_str(), "w");
        if (f == nullptr) throw std::runtime_error("cannot write " + loads_out);
        for (const auto& e : r.routing_loads) {
            for (std::size_t i = 0; i < e.size(); ++i) std::fprintf(f, i ? ",%u" : "%u", e[i]);
            std::fputc('\n', f);
        }
        std::fclose(f);
    }
    if (!r.routing_loads.empty()) {  // entries per expert half of every layer's output rows, and the largest expert
        std::uint64_t all_low = 0, all_high = 0;
        for (std::size_t l = 0; l < r.routing_loads.size(); ++l) {
            const auto& e = r.routing_loads[l];
            std::uint64_t low = 0, high = 0, top = 0;
            for (std::size_t i = 0; i < e.size(); ++i) {
                (i < e.size() / 2U ? low : high) += e[i];
                top = std::max<std::uint64_t>(top, e[i]);
            }
            all_low += low;
            all_high += high;
            std::printf("  loads layer %2zu: low %7llu high %7llu (high/low %.3f), largest expert %llu\n", l, static_cast<unsigned long long>(low),
                        static_cast<unsigned long long>(high), low ? static_cast<double>(high) / static_cast<double>(low) : 0.0,
                        static_cast<unsigned long long>(top));
        }
        std::printf("  loads all layers: low %llu high %llu (high/low %.3f)\n", static_cast<unsigned long long>(all_low),
                    static_cast<unsigned long long>(all_high), all_low ? static_cast<double>(all_high) / static_cast<double>(all_low) : 0.0);
    }
    if (!timing) return;
    // attention and moe include their own exchanges; other = layer - attention - moe; the exchange columns are
    // inside those intervals (wall, and CPU blocked in finish()), not additional time
    double tot = 0, at = 0, mo = 0, ex = 0, bl = 0;
    std::printf("  (ms; attention and moe include their exchanges; exchange wall / blocked are inside them, do not add)\n");
    for (std::size_t l = 0; l < r.layers.size(); ++l) {
        const auto& t = r.layers[l];
        std::printf("  layer %2zu %8.2f  attention %8.2f  moe %8.2f  other %7.2f  | exchange wall %7.2f blocked %7.2f\n", l, t.seconds * 1e3,
                    t.attention * 1e3, t.moe * 1e3, (t.seconds - t.attention - t.moe) * 1e3, t.exchange * 1e3, t.exchange_blocked * 1e3);
        tot += t.seconds;
        at += t.attention;
        mo += t.moe;
        ex += t.exchange;
        bl += t.exchange_blocked;
    }
    std::printf("  sum      %8.2f  attention %8.2f  moe %8.2f  other %7.2f  | exchange wall %7.2f blocked %7.2f\n", tot * 1e3, at * 1e3,
                mo * 1e3, (tot - at - mo) * 1e3, ex * 1e3, bl * 1e3);
    const auto& ap = r.attention_parts;
    std::printf("  attention parts: projections %.2f  window K/V + compressor %.2f  q_b %.2f  windows %.2f  o_a + wo_b %.2f ms\n",
                ap[0] * 1e3, ap[1] * 1e3, ap[2] * 1e3, ap[3] * 1e3, ap[4] * 1e3);
}

strix::deepseek41::PrefillOptions options(const Args& a, std::size_t rows) {
    strix::deepseek41::PrefillOptions o;
    o.layers = a.layers;
    o.max_positions = static_cast<std::uint32_t>(rows);
    o.chunk_rows = a.chunk;
    o.window_rows = a.window;
    o.attention_lanes = a.lanes;
    o.moe_chunk_rows = a.moe_chunk;
    o.tp_attention = a.tp;
    o.row_split = a.row_split;
    o.stream_experts = a.stream_experts;
    o.stream_hashes = a.reference || !a.hashes_out.empty();
    o.timing = a.timing;
    o.cone = a.cone;
    o.routing_loads = a.loads;
    o.warmup_rows = a.warmup;
    o.split_min_rows = a.split_min_rows;
    o.joint_max_rows = a.joint_max_rows;
    return o;
}

// --ttft-timing: the optional warm-up run, then --repeat runs of the prompt, each from the barrier (two ranks) to the
// result on the host; returns the last run's result.
strix::deepseek41::PrefillResult ttft_runs(const Args& a, strix::deepseek41::DevicePrefill& pre, const std::vector<std::uint32_t>& prompt,
                                           strix::dist::Communicator* comm, bool print, const std::string& loads_out) {
    strix::deepseek41::PrefillResult r;
    const auto one = [&](std::span<const std::uint32_t> rows, std::uint32_t round, const char* label) {
        if (comm != nullptr) barrier(*comm, round);
        ttft_clock.line("barrier");
        const auto t0 = std::chrono::steady_clock::now();
        r = pre.run(rows);
        const double wall = TtftClock::ms(std::chrono::steady_clock::now() - t0);
        ttft_clock.line(label);
        if (print) {
            std::printf("ttft %s: %zu rows, wall %.3f ms (run %.3f ms, engram read %.3f ms, head %.3f ms), next token %u, logits_fnv %016llx\n", label,
                        rows.size(), wall, r.seconds * 1e3, r.engram_seconds * 1e3, r.head_seconds * 1e3, r.next_token,
                        static_cast<unsigned long long>(fnv(r.logits)));
            std::fflush(stdout);
        }
    };
    if (a.ttft_warmup != 0U)
        one(std::span<const std::uint32_t>(prompt).first(std::min<std::size_t>(a.ttft_warmup, prompt.size())), 900U, "warm-up run");
    for (std::uint32_t k = 0; k < a.repeat; ++k) {
        char label[32];
        std::snprintf(label, sizeof(label), "run %u", k);
        one(prompt, k, label);
        if (print) print_result(comm == nullptr ? "prefill" : "rank prefill", r, prompt.size(), a.timing, loads_out);
    }
    return r;
}

// --ttft-decode: the prefill's state into a decode forward (the prefill already freed), then G greedy steps, each timed.
void ttft_decode(const Args& a, const std::vector<std::uint32_t>& prompt, strix::deepseek41::AttentionState state, std::uint32_t next,
                 strix::dist::Communicator* comm, std::uint32_t rank, bool print) {
    strix::deepseek41::DeviceForwardOptions fo;
    fo.layers = a.layers;
    fo.max_positions = static_cast<std::uint32_t>(prompt.size() + a.ttft_decode + 8U);
    fo.rank = rank;
    fo.world = comm != nullptr ? 2U : 1U;
    fo.communicator = comm;
    fo.op_hashes = false;
    fo.gather_logits = true;
    fo.engram_direct_io = true;
    fo.tp_attention = comm != nullptr;
    strix::deepseek41::DeviceForward fwd(a.model, fo);
    ttft_clock.line("decode weights load");
    if (print)
        std::printf("ttft decode forward: device weights %.2f GiB loaded in %.1f s\n", static_cast<double>(fwd.weight_bytes()) / 1073741824.0,
                    fwd.load_seconds());
    state.tokens = prompt;
    state.next_token = next;
    fwd.import_state(state);
    ttft_clock.line("import state");
    if (comm != nullptr) barrier(*comm, 950U);
    ttft_clock.line("barrier");
    std::uint32_t token = next;
    for (std::uint32_t g = 0; g < a.ttft_decode; ++g) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto& logits = fwd.step(token);
        const double wall = TtftClock::ms(std::chrono::steady_clock::now() - t0);
        ttft_clock.line(g == 0U ? "first decode step" : "decode step");
        if (print) {
            std::printf("ttft decode step %u: pos %zu fed %u argmax %u, wall %.3f ms (step %.3f ms), logits_fnv %016llx\n", g, prompt.size() + g, token,
                        fwd.next_token(), wall, fwd.last_step_seconds() * 1e3, static_cast<unsigned long long>(fnv(logits)));
            std::fflush(stdout);
        }
        token = fwd.next_token();
    }
}

int run_world1(const Args& a, const std::vector<std::uint32_t>& prompt) {
    auto o = options(a, prompt.size());
    if (a.ttft) {
        strix::deepseek41::AttentionState state;
        std::uint32_t next = 0;
        {
            strix::deepseek41::DevicePrefill pre(a.model, o);
            ttft_clock.line("prefill weights load");
            std::printf("world 1: layers [0, %u), device weights %.2f GiB loaded in %.1f s\n", a.layers,
                        static_cast<double>(pre.weight_bytes()) / 1073741824.0, pre.load_seconds());
            next = ttft_runs(a, pre, prompt, nullptr, true, a.loads_out).next_token;
            if (a.ttft_decode == 0U) return 0;
            state = pre.export_state();
            ttft_clock.line("export state");
        }
        ttft_clock.line("free prefill");
        ttft_decode(a, prompt, std::move(state), next, nullptr, 0U, true);
        return 0;
    }
    strix::deepseek41::DevicePrefill pre(a.model, o);
    std::printf("world 1: layers [0, %u), device weights %.2f GiB loaded in %.1f s\n", a.layers,
                static_cast<double>(pre.weight_bytes()) / 1073741824.0, pre.load_seconds());
    if (const auto ov = pre.overlay_description(); !ov.empty()) std::printf("weights overlay: %s\n", ov.c_str());
    strix::deepseek41::PrefillResult r;
    for (std::uint32_t k = 0; k < a.repeat; ++k) {
        r = pre.run(prompt);
        print_result("prefill", r, prompt.size(), a.timing, a.loads_out);
    }
    if (!a.hashes_out.empty()) {  // one line per layer stream, then the attention state (runs compared with diff)
        std::FILE* f = std::fopen(a.hashes_out.c_str(), "w");
        if (f == nullptr) throw std::runtime_error("cannot write " + a.hashes_out);
        for (std::size_t l = 0; l < r.stream_hashes.size(); ++l)
            std::fprintf(f, "stream@%zu %016llx\n", l, static_cast<unsigned long long>(r.stream_hashes[l]));
        for (const auto& [name, h] : pre.state_hashes()) std::fprintf(f, "%s %016llx\n", name.c_str(), static_cast<unsigned long long>(h));
        std::fprintf(f, "logits %016llx\n", static_cast<unsigned long long>(fnv(r.logits)));
        std::fclose(f);
    }
    if (!a.reference) return 0;
    const auto ph = pre.state_hashes();
    strix::deepseek41::DeviceForwardOptions fo;
    fo.layers = a.layers;
    fo.max_positions = static_cast<std::uint32_t>(prompt.size() + 8U);
    fo.op_hashes = true;  // the per-layer stream hashes of the last step
    strix::deepseek41::DeviceForward fwd(a.model, fo);
    std::vector<float> logits;
    for (const auto t : prompt) logits = fwd.step(t);
    const auto fh = fwd.state_hashes();
    const auto& sh = fwd.last_layer_stream_hashes();
    int bad = 0;
    if (fnv(logits) != fnv(r.logits) || fwd.next_token() != r.next_token) {
        std::printf("MISMATCH last-row logits: decode %016llx token %u, prefill %016llx token %u\n", static_cast<unsigned long long>(fnv(logits)),
                    fwd.next_token(), static_cast<unsigned long long>(fnv(r.logits)), r.next_token);
        ++bad;
    }
    for (std::size_t l = 0; l < sh.size() && l < r.stream_hashes.size(); ++l)
        if (sh[l] != r.stream_hashes[l]) {
            std::printf("MISMATCH stream after layer %zu\n", l);
            ++bad;
        }
    if (fh.size() != ph.size()) {
        std::printf("MISMATCH state labels (%zu vs %zu)\n", fh.size(), ph.size());
        ++bad;
    }
    for (std::size_t i = 0; i < fh.size() && i < ph.size(); ++i)
        if (fh[i] != ph[i]) {
            std::printf("MISMATCH state %s (decode %016llx, prefill %s %016llx)\n", fh[i].first.c_str(), static_cast<unsigned long long>(fh[i].second),
                        ph[i].first.c_str(), static_cast<unsigned long long>(ph[i].second));
            ++bad;
        }
    std::printf("reference (decode forward, %zu steps): %s; %zu state hashes, %zu layer streams compared\n", prompt.size(),
                bad ? "DIFFERENT" : "bitwise EQUAL", fh.size(), sh.size());
    return bad ? 1 : 0;
}

void run_rank(const Args& a, const std::vector<std::uint32_t>& prompt, std::uint32_t rank, std::unique_ptr<strix::dist::Transport> t,
              bool print, strix::deepseek41::PrefillResult* out, std::vector<std::pair<std::string, std::uint64_t>>* out_state = nullptr) {
    strix::dist::Communicator comm(strix::dist::Group{rank, 2U, 1U, 0x64733450ULL}, std::move(t), std::size_t{22} << 20U);
    ttft_clock.line("communicator");
    auto o = options(a, prompt.size());
    o.rank = rank;
    o.world = 2;
    o.communicator = &comm;
    const bool decode_reference = a.reference && (a.transport != "emulate" || a.decode_reference);
    o.stream_hashes = decode_reference || out_state != nullptr;
    strix::deepseek41::PrefillResult r;
    std::vector<std::pair<std::string, std::uint64_t>> state;
    if (a.ttft && !decode_reference && out_state == nullptr) {
        strix::deepseek41::AttentionState st;
        {
            strix::deepseek41::DevicePrefill pre(a.model, o);
            ttft_clock.line("prefill weights load");
            if (print)
                std::printf("rank %u of 2 (%s): layers [0, %u), device weights %.2f GiB loaded in %.1f s\n", rank, comm.transport_name().c_str(),
                            a.layers, static_cast<double>(pre.weight_bytes()) / 1073741824.0, pre.load_seconds());
            if (const auto ov = pre.overlay_description(); print && !ov.empty()) std::printf("weights overlay: %s\n", ov.c_str());
            r = ttft_runs(a, pre, prompt, &comm, print, rank == 0U ? a.loads_out : std::string{});
            if (a.ttft_decode != 0U) {
                st = pre.export_state();
                ttft_clock.line("export state");
            }
        }
        if (out) *out = r;
        if (a.ttft_decode == 0U) return;
        ttft_clock.line("free prefill");
        ttft_decode(a, prompt, std::move(st), r.next_token, &comm, rank, print);
        return;
    }
    {
        strix::deepseek41::DevicePrefill pre(a.model, o);
        if (print)
            std::printf("rank %u of 2 (%s): layers [0, %u), device weights %.2f GiB loaded in %.1f s\n", rank, comm.transport_name().c_str(),
                        a.layers, static_cast<double>(pre.weight_bytes()) / 1073741824.0, pre.load_seconds());
        if (const auto ov = pre.overlay_description(); print && !ov.empty()) std::printf("weights overlay: %s\n", ov.c_str());
        for (std::uint32_t k = 0; k < a.repeat; ++k) {
            barrier(comm, k);  // the run's time does not include the peer still loading its weights
            const auto before = comm.stats();
            r = pre.run(prompt);
            if (print) {
                print_result(rank == 0U ? "rank 0 prefill" : "rank 1 prefill", r, prompt.size(), a.timing, rank == 0U ? a.loads_out : std::string{});
                const auto& s = comm.stats();
                std::printf("  exchanges %llu, %.1f MB sent, %.2f s in exchange (stage %.2f, wire %.2f, check %.2f)\n",
                            static_cast<unsigned long long>(s.exchanges - before.exchanges),
                            static_cast<double>(s.bytes_sent - before.bytes_sent) / 1e6, s.seconds - before.seconds, s.stage - before.stage,
                            s.wire - before.wire, s.check - before.check);
                std::fflush(stdout);
            }
        }
        if (a.state_timing && print) {  // the handoff to a decode forward: the attention state to the host
            const auto t0 = std::chrono::steady_clock::now();
            const auto exported = pre.export_state();
            std::size_t bytes = 0;
            for (const auto& l : exported.layers) bytes += l.ring.size() + l.kv.size() + l.idx_keys.size() + l.carry.size() * 4U + l.topk.size() * 4U;
            const double ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
            std::uint64_t digest = 14695981039346656037ULL;  // FNV of the state's labelled hashes
            for (const auto& [name, h] : exported.hashes()) digest = (digest ^ h) * 1099511628211ULL;
            std::printf("  export_state: %.2f ms, %.2f MB, state digest %016llx\n", ms, static_cast<double>(bytes) / 1e6,
                        static_cast<unsigned long long>(digest));
        }
        if (decode_reference || out_state != nullptr) state = pre.state_hashes();
    }
    if (out) *out = r;
    if (out_state) *out_state = state;
    if (!decode_reference) return;
    // The two-rank decode forward fed the same tokens over the same communicator (the prefill's weights are freed).
    strix::deepseek41::DeviceForwardOptions fo;
    fo.layers = a.layers;
    fo.max_positions = static_cast<std::uint32_t>(prompt.size() + 8U);
    fo.rank = rank;
    fo.world = 2;
    fo.communicator = &comm;
    fo.op_hashes = true;
    strix::deepseek41::DeviceForward fwd(a.model, fo);
    std::vector<float> logits;
    for (const auto token : prompt) logits = fwd.step(token);
    const auto fh = fwd.state_hashes();
    const auto& sh = fwd.last_layer_stream_hashes();
    int bad = 0;
    if (fnv(logits) != fnv(r.logits) || fwd.next_token() != r.next_token) {
        std::printf("rank %u MISMATCH last-row logits: decode %016llx token %u, prefill %016llx token %u\n", rank,
                    static_cast<unsigned long long>(fnv(logits)), fwd.next_token(), static_cast<unsigned long long>(fnv(r.logits)), r.next_token);
        ++bad;
    }
    // row split: the last row's stream and top-k exist only on the rank that computed the row
    for (std::size_t l = 0; r.last_row_local && l < sh.size() && l < r.stream_hashes.size(); ++l)
        if (sh[l] != r.stream_hashes[l]) {
            if (bad < 20) std::printf("rank %u MISMATCH stream after layer %zu\n", rank, l);
            ++bad;
        }
    for (std::size_t i = 0; i < fh.size() && i < state.size(); ++i)
        if (fh[i] != state[i] && (r.last_row_local || fh[i].first.rfind("topk", 0) != 0)) {
            if (bad < 20) std::printf("rank %u MISMATCH state %s\n", rank, fh[i].first.c_str());
            ++bad;
        }
    std::printf("rank %u reference (two-rank decode forward, %zu steps): %s; %zu state hashes, %zu layer streams; logits %016llx token %u\n",
                rank, prompt.size(), bad == 0 && fh.size() == state.size() ? "bitwise EQUAL" : "DIFFERENT", fh.size(), r.last_row_local ? sh.size() : std::size_t{0},
                static_cast<unsigned long long>(fnv(r.logits)), r.next_token);
    std::fflush(stdout);
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
        if (s == "--prompt") a.prompt_path = next();
        else if (s == "--rows") a.rows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--layers") a.layers = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--world") a.world = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--rank") { a.rank = static_cast<std::uint32_t>(std::stoul(next())); a.world = 2; }
        else if (s == "--transport") { a.transport = next(); a.world = 2; }
        else if (s == "--chunk") a.chunk = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--window") a.window = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--lanes") a.lanes = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--moe-chunk") a.moe_chunk = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--no-tp") a.tp = false;
        else if (s == "--row-split") a.row_split = true;
        else if (s == "--no-row-split") a.row_split = false;  // tensor-parallel attention (every row on both ranks)
        else if (s == "--stream-experts") a.stream_experts = true;
        else if (s == "--repeat") a.repeat = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--timing") a.timing = true;
        else if (s == "--reference") a.reference = true;
        else if (s == "--decode-reference") a.reference = a.decode_reference = true;  // emulate: also the two-rank decode
        else if (s == "--no-cone") a.cone = false;
        else if (s == "--state-timing") a.state_timing = true;
        else if (s == "--warmup") a.warmup = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--split-min-rows") a.split_min_rows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--joint-max-rows") a.joint_max_rows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--loads") a.loads = true;
        else if (s == "--stream-hashes-out") a.hashes_out = next();
        else if (s == "--loads-out") {
            a.loads = true;
            a.loads_out = next();
        }
        else if (s == "--connect-timeout") a.connect_timeout = static_cast<unsigned>(std::stoul(next()));
        else if (s == "--ttft-timing") a.ttft = true;
        else if (s == "--ttft-warmup") a.ttft_warmup = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--ttft-decode") a.ttft_decode = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a.model.empty()) a.model = s;
        else throw std::invalid_argument("unknown argument " + s);
    }
    if (a.model.empty() || a.prompt_path.empty()) throw std::invalid_argument("usage: strix-deepseek41-prefill MODEL.gguf --prompt FILE ...");
    ttft_clock.on = a.ttft;
    ttft_clock.exec_to_main();
    ttft_clock.line("arguments");
    const auto prompt = read_prompt(a.prompt_path, a.rows);
    ttft_clock.line("prompt read");
    if ((a.ttft_warmup != 0U || a.ttft_decode != 0U) && !a.ttft) throw std::invalid_argument("--ttft-warmup and --ttft-decode need --ttft-timing");
    if (a.world == 1U) return run_world1(a, prompt);
    if (a.transport == "emulate") {  // both ranks in this process; rank 0 prints; --reference compares with world 1
        auto [t0, t1] = strix::dist::memory_pair();
        std::exception_ptr failure;
        strix::deepseek41::PrefillResult r0, r1;
        std::vector<std::pair<std::string, std::uint64_t>> s0, s1;
        std::thread peer([&, t = std::move(t1)]() mutable {
            try {
                run_rank(a, prompt, 1U, std::move(t), false, &r1, a.reference ? &s1 : nullptr);
            } catch (...) {
                failure = std::current_exception();
            }
        });
        try {
            run_rank(a, prompt, 0U, std::move(t0), true, &r0, a.reference ? &s0 : nullptr);
        } catch (...) {
            peer.join();
            throw;
        }
        peer.join();
        if (failure) std::rethrow_exception(failure);
        std::printf("ranks: next token %u / %u\n", r0.next_token, r1.next_token);
        if (!a.reference) return 0;
        auto o = options(a, prompt.size());
        o.stream_hashes = true;
        strix::deepseek41::DevicePrefill one(a.model, o);
        const auto r = one.run(prompt);
        // each rank's attention state and (on the last row's rank) the last row's stream after every layer = world 1
        const auto s = one.state_hashes();
        std::size_t state_bad = 0, stream_bad = 0;
        for (const auto* rank : {&r0, &r1}) {
            const auto& st = rank == &r0 ? s0 : s1;
            if (st.size() != s.size()) ++state_bad;
            for (std::size_t i = 0; i < s.size() && i < st.size(); ++i)
                if (st[i] != s[i] && (rank->last_row_local || s[i].first.rfind("topk", 0) != 0)) {
                    if (state_bad < 8) std::printf("rank %d MISMATCH state %s\n", rank == &r0 ? 0 : 1, s[i].first.c_str());
                    ++state_bad;
                }
            if (rank->last_row_local && rank->stream_hashes != r.stream_hashes) ++stream_bad;
        }
        std::printf("reference (world 1 prefill): attention state of both ranks %s (%zu hashes), last row's streams %s (%zu layers)\n",
                    state_bad == 0 ? "EQUAL" : "DIFFERENT", s.size(), stream_bad == 0 ? "EQUAL" : "DIFFERENT", r.stream_hashes.size());
        // each rank holds its vocabulary slice of the logits: compare slice by slice
        std::vector<float> both(r.logits.size(), 0.0F);
        const std::size_t half = r.logits.size() / 2U;
        std::copy(r0.logits.begin(), r0.logits.begin() + static_cast<std::ptrdiff_t>(half), both.begin());
        std::copy(r1.logits.begin() + static_cast<std::ptrdiff_t>(half), r1.logits.end(), both.begin() + static_cast<std::ptrdiff_t>(half));
        const bool equal = fnv(both) == fnv(r.logits) && r0.next_token == r.next_token && r1.next_token == r.next_token;
        std::printf("reference (world 1 prefill): two ranks %s (token %u, logits %016llx vs %016llx)\n", equal ? "bitwise EQUAL" : "DIFFERENT",
                    r.next_token, static_cast<unsigned long long>(fnv(both)), static_cast<unsigned long long>(fnv(r.logits)));
        return equal && state_bad == 0 && stream_bad == 0 ? 0 : 1;
    }
    ::alarm(a.connect_timeout + 30U);  // a stream on a dead USB4 link blocks open(): the alarm ends the process
    std::string receipt;
    auto transport = open_transport(a.transport, a.rank, a.connect_timeout, receipt);
    ::alarm(0);
    ttft_clock.line("transport open (peer handshake)");
    std::printf("connected (%s)%s%s\n", transport->name().c_str(), receipt.empty() ? "" : "\n", receipt.c_str());
    std::fflush(stdout);
    run_rank(a, prompt, a.rank, std::move(transport), true, nullptr);
    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 2;
}
