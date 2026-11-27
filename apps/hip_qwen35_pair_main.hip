// strix-hip-qwen35-pair: Qwen3.8-27B (qwen35 dense) over N ranks as a contiguous layer split.
// Ranks, memory, rates and links come from the topology declaration (strix/device/topology.hpp: --topology FILE --node ID,
// or the legacy two-node --rank R [--transport usb4auto|tcp]); the cuts from the shared planner (strix/pair/chain.hpp via
// strix/qwen35/pair_plan.hpp, override --placement auto|single|cut:A[,B..]); execution from the shared layer pipeline
// (strix/pair/layer_pipeline.hpp: prefill flow shop over one pair::Pipe per link, layer-owned decode with one hidden row
// per cut per token). Each rank loads only its layers (HipEngineOptions::layer_begin/end). One stage = the decode app.
// --self-cut K1[,K2..]: one-process test of the engine range hooks on ONE device: the uncut run, then for each K the same
// prompt as [0,K) -> export_hidden -> import_hidden -> [K,L); logits bytes and ids must be equal (no second node needed).
// Home: specific = app glue (qwen35 engine binding); placement, seam, pipeline, topology are shared.
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <hip/hip_runtime.h>

#include "strix/device/topology.hpp"
#include "strix/env_flags.hpp"
#include "strix/dist/open_transport.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/pair/layer_pipeline.hpp"
#include "strix/qwen35/hip_engine.hpp"
#include "strix/qwen35/pair_plan.hpp"
#include "core/dense_linear.hpp"   // parse_u32, require_capacity
#include "xport/hip_pipe_staging.hpp"

namespace {
using strix::qwen35::HipEngine;
using strix::qwen35::HipEngineOptions;
using strix::qwen35::LogitRows;
namespace layers = strix::pair::layers;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

long sysfs_sclk_mhz() {
    for (int c = 0; c < 4; ++c) {
        std::ifstream f("/sys/class/drm/card" + std::to_string(c) + "/device/pp_dpm_sclk");
        std::string line, best;
        while (std::getline(f, line)) if (line.find('*') != std::string::npos) best = line;
        if (!best.empty()) return std::atol(best.c_str() + best.find(' ') + 1);
    }
    return 0;
}
struct ClockSampler {   // average sclk over a phase (receipt: GPU clock beside every timing)
    std::atomic<bool> run{true};
    std::atomic<long> sum{0}, n{0};
    std::thread t{[this] { while (run) { sum += sysfs_sclk_mhz(); ++n; std::this_thread::sleep_for(std::chrono::milliseconds(100)); } }};
    long stop() { run = false; t.join(); return n ? sum / n : 0L; }
};

std::vector<std::uint32_t> read_ids(std::string arg) {
    if (!arg.empty() && arg[0] == '@') {
        std::ifstream f(arg.substr(1));
        if (!f) throw std::invalid_argument("cannot read " + arg.substr(1));
        arg.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    std::vector<std::uint32_t> r;
    std::stringstream ss(arg);
    std::string t;
    while (std::getline(ss, t, ',')) {
        const auto b = t.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) continue;
        r.push_back(strix::core::parse_u32(t.substr(b, t.find_last_not_of(" \t\r\n") + 1U - b)));
    }
    return r;
}
std::vector<std::uint32_t> parse_list(const std::string& s) {
    std::vector<std::uint32_t> r;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) r.push_back(strix::core::parse_u32(t));
    return r;
}
std::uint64_t fnv_bytes(const void* p, std::size_t n) {
    std::uint64_t h = 14695981039346656037ULL;
    for (std::size_t i = 0; i < n; ++i) { h ^= static_cast<const unsigned char*>(p)[i]; h *= 1099511628211ULL; }
    return h;
}

struct Args {
    std::string model, ids, dump, placement = "auto", transport = "usb4auto", peer, self_cut;
    unsigned gen = 32, wide = 512, maxpos = 0, repeat = 1, port = 29650;
    int legacy_rank = -1;
    double prefill_rate = 0.0;   // one whole node's prefill rows/s (a measured receipt); 0 = nominal (cuts by shape only)
};

// One prompt + decode on a single engine through the range hooks with the given cuts (one process). cuts {0, L} = uncut.
struct Run { std::vector<float> logits; std::vector<std::uint32_t> ids; double prefill_ms = 0, decode_ms = 0; };
Run run_cuts(HipEngine& E, const std::vector<std::uint32_t>& prompt, unsigned width, unsigned gen, const std::vector<std::uint32_t>& cuts) {
    const unsigned V = E.vocab();
    std::vector<float> host(std::size_t(width) * E.hidden_row_bytes() / 4U);
    Run r;
    E.reset();
    strix::qwen35::HipRowsOut o;
    const auto pass = [&](const std::uint32_t* tok, unsigned n, bool head) {
        for (std::size_t s = 0; s + 1U < cuts.size(); ++s) {
            if (s) E.import_hidden(host.data(), n);
            const bool last = s + 2U == cuts.size();
            o = E.forward_range(cuts[s], cuts[s + 1U], tok, n, last && head ? LogitRows::last : LogitRows::none);
            if (!last) E.export_hidden(host.data(), n);
        }
    };
    const auto t0 = Clock::now();
    for (const auto& w : layers::windows(prompt.size(), width)) pass(prompt.data() + w.start, w.rows, w.start + w.rows == prompt.size());
    std::uint32_t top = 0;
    E.read_top(o, 1, &top);
    r.prefill_ms = ms_since(t0);
    r.logits.resize(V);
    E.read_logits(o, 1, r.logits.data());
    r.ids.push_back(top);
    const auto t1 = Clock::now();
    for (unsigned s = 1; s < gen; ++s) {
        const auto tok = r.ids.back();
        pass(&tok, 1U, true);
        E.read_top(o, 1, &top);
        const auto at = r.logits.size();
        r.logits.resize(at + V);
        E.read_logits(o, 1, r.logits.data() + at);
        r.ids.push_back(top);
    }
    r.decode_ms = ms_since(t1);
    return r;
}

int self_cut(const Args& a, const std::vector<std::uint32_t>& prompt) {
    HipEngineOptions opt;
    opt.max_positions = a.maxpos;
    opt.prefill_rows = a.wide;
    opt.wide_exact = false;  // --wide N > 16 stays the FAST wide class here (the engine default is now EXACT wide windows)
    const auto t0 = Clock::now();
    auto E = HipEngine::from_gguf(a.model, opt);
    std::printf("load time %.1f s\n", ms_since(t0) / 1000.0);
    const unsigned L = E->config().layers;
    const auto ref = run_cuts(*E, prompt, a.wide, a.gen, {0U, L});
    std::printf("self-cut uncut: prefill %zu tokens %.1f ms, decode %.2f ms/token, logits fnv %016llx, ids", prompt.size(), ref.prefill_ms,
                a.gen > 1U ? ref.decode_ms / double(a.gen - 1U) : 0.0,
                static_cast<unsigned long long>(fnv_bytes(ref.logits.data(), ref.logits.size() * 4U)));
    for (auto t : ref.ids) std::printf(" %u", t);
    std::printf("\n");
    if (!a.dump.empty()) {   // the one-node reference of the multi-node identity check (same layout as the decode app's dump)
        std::ofstream f(a.dump, std::ios::binary);
        f.write(reinterpret_cast<const char*>(ref.logits.data()), std::streamsize(ref.logits.size() * 4U));
        std::printf("dumped %zu positions x %u to %s\n", ref.logits.size() / E->vocab(), E->vocab(), a.dump.c_str());
    }
    int bad = 0;
    for (const auto k : parse_list(a.self_cut)) {
        if (k == 0U || k >= L) throw std::invalid_argument("--self-cut K must be 1..layers-1");
        const auto r = run_cuts(*E, prompt, a.wide, a.gen, {0U, k, L});
        std::size_t first_diff = r.logits.size();
        for (std::size_t i = 0; i < std::min(r.logits.size(), ref.logits.size()); ++i)
            if (std::memcmp(&r.logits[i], &ref.logits[i], 4U) != 0) { first_diff = i; break; }
        const bool same_logits = r.logits.size() == ref.logits.size() && first_diff == r.logits.size();
        const bool same_ids = r.ids == ref.ids;
        bad += !(same_logits && same_ids);
        std::printf("self-cut k=%u: logits %s (%zu rows) ids %s; prefill %.1f ms\n", k,
                    same_logits ? "IDENTICAL" : ("DIFFER at row " + std::to_string(first_diff / E->vocab())).c_str(),
                    r.logits.size() / E->vocab(), same_ids ? "IDENTICAL" : "DIFFER", r.prefill_ms);
    }
    std::printf("self-cut verdict: %s\n", bad ? "FAIL" : "PASS");
    return bad ? 4 : 0;
}

// Opens the ordered transport of the link between `lower` and lower + 1; the lower rank is role 0 (listener / pipe producer).
std::unique_ptr<strix::dist::Transport> open_link(const strix::device::Topology& t, std::uint32_t lower, bool i_am_lower, const Args& a) {
    strix::dist::OpenTransportOptions o;
    o.usb.rank = i_am_lower ? 0U : 1U;
    std::string spec = a.transport;
    const auto port = std::to_string(strix::pair::chain::line_port(a.port, lower));
    if (spec == "tcp") {
        std::string host = t.nodes.at(lower).host;
        if (host.empty()) host = a.peer;
        if (!i_am_lower && host.empty())
            throw std::invalid_argument("tcp link needs the host of node " + t.nodes.at(lower).id + " (topology host or --peer)");
        spec = i_am_lower ? "tcp-listen:" + port : "tcp-connect:" + host + ":" + port;
        o.usb.timeout = std::chrono::milliseconds(120000);
    } else if (spec != "usb4auto") {
        throw std::invalid_argument("--transport must be usb4auto or tcp");
    }
    return strix::dist::open_transport(spec, o).host;
}
} // namespace

int main(int argc, char** argv) try {
    auto topo_cli = strix::device::take_topology_command_line(argc, argv);
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + s);
            return argv[++i];
        };
        if (s == "--model") a.model = next();
        else if (s == "--prompt-ids") a.ids = next();
        else if (s == "--gen") a.gen = strix::core::parse_u32(next());
        else if (s == "--wide") a.wide = strix::core::parse_u32(next());
        else if (s == "--max-pos") a.maxpos = strix::core::parse_u32(next());
        else if (s == "--repeat") a.repeat = std::max(1U, strix::core::parse_u32(next()));
        else if (s == "--dump-logits") a.dump = next();
        else if (s == "--placement") a.placement = next();
        else if (s == "--transport") a.transport = next();
        else if (s == "--peer") a.peer = next();
        else if (s == "--port") a.port = strix::core::parse_u32(next());
        else if (s == "--rank") a.legacy_rank = int(strix::core::parse_u32(next()));
        else if (s == "--prefill-rate") a.prefill_rate = std::stod(next());
        else if (s == "--self-cut") a.self_cut = next();
        else throw std::invalid_argument("unknown flag " + s);
    }
    if (a.model.empty() || a.ids.empty() || a.wide < 1U)
        throw std::invalid_argument("usage: --model GGUF --prompt-ids @FILE [--wide 512] [--gen 32] [--repeat 2] [--max-pos N] "
                                    "[--dump-logits FILE] [--placement auto|single|cut:A[,B..]] [--topology FILE --node ID | --rank R] "
                                    "[--transport usb4auto|tcp] [--peer HOST] [--prefill-rate ROWS_PER_S] | --self-cut K1[,K2..]");
    const auto prompt = read_ids(a.ids);
    if (prompt.empty()) throw std::invalid_argument("empty prompt");
    const auto required = prompt.size() + a.gen;
    if (!a.maxpos) a.maxpos = unsigned(required);
    strix::core::require_capacity(0, required, a.maxpos);
    if (!a.self_cut.empty()) return self_cut(a, prompt);

    // Topology -> planner -> this rank's stage.
    const auto topo = strix::device::start_topology(topo_cli, a.legacy_rank, a.transport);
    std::printf("%s\n", strix::device::topology_line(topo).c_str());
    const auto info = strix::gguf::read_file(a.model);
    const auto cfg = strix::qwen35::config_from_gguf(info);
    namespace pp = strix::qwen35::pair_plan;
    auto table = pp::layer_table(info.tensors, cfg.layers);
    const auto N = topo.size();
    // Ranks, memory, rates and line links from the declaration (shared binding, pair/chain_topology.hpp).
    namespace chain = strix::pair::chain;
    const auto ranks = chain::ranks(topo, a.prefill_rate > 0.0 ? a.prefill_rate : 1.0);
    const auto hops = chain::line_hops(topo);
    // Layer-owned decode: every stage keeps its layers' state, no rank needs the whole model. With every node's memory
    // declared, the planner admits each stage (its weights + embed/head + its KV rows at max positions + its GDN
    // state) and only fitting plans compete; undeclared memory leaves memory unchecked (weights cleared).
    auto shape = pp::shape(cfg, table, required);
    if (!chain::memory_declared(topo)) shape.weight_bytes.clear();
    const auto devices = pp::devices(cfg, ranks);
    auto request = pp::request(prompt.size(), a.wide, N, a.maxpos, double(a.gen), a.placement);
    // Decode placement from the declaration through the ONE planner: with every node's memory declared, the handoff
    // admission runs first (a decoder that holds the whole model + state receives every stage's layer state once after the
    // prefill and decodes alone: a layer split only serialises the decode). Undeclared memory, no fitting decoder, or a
    // decoder outside the stage chain keep the layer-owned decode. The relay is ON by default (exact: ids equal to the
    // one-node reference and to the layer-owned decode; docs/optimizations.md#qwen-27b-pair-prefill-decode-relay-to-one-node);
    // STRIX_PAIR_DECODE_RELAY=0 keeps the layer-owned decode.
    const bool relay_allowed = strix::envcfg::flag("STRIX_PAIR_DECODE_RELAY", true);
    std::string decode_reason = !relay_allowed ? "STRIX_PAIR_DECODE_RELAY=0 (layer-owned decode)" : !chain::memory_declared(topo) ? "node memory not declared" : "";
    chain::Plan plan;
    if (decode_reason.empty()) {
        request.layer_owned_state = false;
        plan = chain::choose(devices, chain::links(hops), shape, request).best;
        if (!plan.valid) decode_reason = "no decoder holds the whole model: " + plan.reason;
        else if (plan.decoder < plan.first || plan.decoder > plan.last) decode_reason = "decoder outside the stage chain";
    }
    const bool relay = decode_reason.empty() && plan.stages() > 1U;
    if (!decode_reason.empty()) {
        request.layer_owned_state = true;
        request.price_layer_owned_decode = strix::envcfg::flag("STRIX_PAIR_PRICE_LAYER_DECODE", false);
        if (request.price_layer_owned_decode) {
            shape.decode_message_header_bytes = double(layers::decode_header_bytes);
            std::puts("placement pricing: serial layer-owned decode");
        }
        plan = chain::choose(devices, chain::links(hops), shape, request).best;
    }
    std::printf("%s\n", pp::describe(plan, devices, prompt.size()).c_str());
    if (!plan.valid) return 2;
    const std::string placement_text = relay ? "relay to " + devices[plan.decoder].name + " (whole model there, state moved once)"
                                     : plan.stages() == 1U ? std::string("single stage") : "layer-owned (" + decode_reason + ")";
    std::printf("decode placement: %s\n", placement_text.c_str());
    const auto me = topo.local_rank;
    if (me < plan.first || me >= plan.first + plan.stages()) { std::printf("rank %s runs no layers in this plan\n", topo.local().id.c_str()); return 0; }
    layers::Placement place{plan.cuts, me - plan.first};
    place.check();
    const unsigned L = cfg.layers;
    std::printf("stage %u/%u on %s: layers [%u, %u) embed=%d head=%d\n", place.stage, place.stages(), topo.local().id.c_str(),
                place.first(), place.last(), int(place.embeds()), int(place.head()));

    HipEngineOptions opt;
    opt.max_positions = a.maxpos;
    opt.prefill_rows = a.wide;
    opt.wide_exact = false;  // --wide N > 16 stays the FAST wide class here (the engine default is now EXACT wide windows)
    const std::uint32_t decoder_stage = plan.decoder - plan.first;
    const bool decodes_here = !relay || place.stage == decoder_stage;
    opt.layer_begin = relay && decodes_here ? 0U : place.first();   // the relay decoder loads the whole model
    opt.layer_end = relay && decodes_here ? L : place.last();
    const auto t_load = Clock::now();
    auto E = HipEngine::from_gguf(a.model, opt);
    std::printf("load time %.1f s\n", ms_since(t_load) / 1000.0);
    if (E->layer_begin() != opt.layer_begin || E->layer_end() != opt.layer_end) throw std::logic_error("engine range differs from the stage");
    E->set_stage(place.first(), place.last());   // prefill windows run and commit on the stage range

    // Links on the line (stage s <-> s+1 is topology link (first+s, first+s+1)); upstream first: no open deadlock.
    if (a.transport == "usb4auto" && !place.embeds() && !place.head())
        throw std::invalid_argument("usb4auto opens one cable per process: a middle stage needs --transport tcp");
    std::unique_ptr<strix::dist::Transport> up, down;
    if (!place.embeds()) up = open_link(topo, me - 1U, false, a);
    if (!place.head()) down = open_link(topo, me, true, a);
    const std::size_t row_bytes = E->hidden_row_bytes();
    auto stream = static_cast<hipStream_t>(E->stream_handle());
    const std::uint64_t identity = layers::fnv(layers::fnv(14695981039346656037ULL,
        "qwen35|" + std::to_string(info.file_size) + "|" + std::to_string(L) + "|" + std::to_string(a.wide) + "|" + std::to_string(a.gen)),
        std::span<const std::uint32_t>(plan.cuts));
    const auto id_prompt = layers::fnv(identity, std::span<const std::uint32_t>(prompt));
    // Line barrier before each timed phase so every rank starts together (the peer's load skew stays out of the timing).
    const auto barrier = [&] {
        std::uint64_t w = 0x52524142ULL;
        if (down) down->receive(std::as_writable_bytes(std::span(&w, 1)));
        if (up) { up->send(std::as_bytes(std::span(&w, 1))); up->receive(std::as_writable_bytes(std::span(&w, 1))); }
        if (down) down->send(std::as_bytes(std::span(&w, 1)));
    };
    const auto count = layers::windows(prompt.size(), a.wide).size();
    const auto make_pipe = [&](strix::dist::Transport* t, unsigned role, std::uint32_t cut) {
        strix::pair::PipeOptions po;
        po.validate = layers::validator(id_prompt, count, cut, a.wide, row_bytes);
        po.profile_label = "q35-pair-profile";
        return std::make_unique<strix::pair::Pipe>(role, count,
            [t](std::span<const std::byte> d) { t->send(d); }, [t](std::span<std::byte> d) { t->receive(d); },
            [t] { t->abort("qwen35 pair prefill stopped"); }, std::move(po),
            [&] { return strix::pair::make_hip_pipe_staging(std::size_t(a.wide) * row_bytes, [stream] { return stream; }); });
    };
    strix::qwen35::HipRowsOut o;
    layers::StageOps ops;
    ops.row_bytes = row_bytes;
    ops.forward = [&](std::span<const std::uint32_t> tok, const layers::Window& w, bool head) {
        o = E->forward_range(place.first(), place.last(), tok.data(), w.rows, head ? LogitRows::last : LogitRows::none); };
    ops.seam = [&](strix::pair::PipeFrame& f, bool save, std::size_t bytes) { f.staging->copy(save, 0, E->hidden_rows(), bytes); };
    ops.fence = [&] { E->sync(); };
    std::uint32_t first_token = 0;
    std::vector<float> dumped;
    for (unsigned pass = 0; pass < a.repeat; ++pass) {
        if (pass) E->reset();
        E->sync();
        barrier();   // before the pipes: their progress threads read the same transports
        auto pin = up ? make_pipe(up.get(), 1U, place.first()) : nullptr;
        auto pout = down ? make_pipe(down.get(), 0U, place.last()) : nullptr;
        ClockSampler clk;
        const auto t0 = Clock::now();
        const auto st = layers::prefill(place, prompt, a.wide, id_prompt, pin.get(), pout.get(), ops);
        if (place.head()) E->read_top(o, 1, &first_token);
        else E->sync();
        const double wall = ms_since(t0);
        const long mhz = clk.stop();
        pin.reset(); pout.reset();
        std::printf("prefill %u/%u (%s) stage %u: %zu tokens %.1f ms = %.1f tok/s; seam wait %.1f ms, seal %.1f ms, sclk avg %ld MHz\n",
                    pass + 1U, a.repeat, pass == 0U ? "cold" : pass + 1U == a.repeat ? "warm" : "mid", place.stage, prompt.size(), wall,
                    double(prompt.size()) * 1000.0 / wall, st.wait_ms, st.seal_ms, mhz);
        if (place.stages() > 1U) {   // per-window timeline: wait on input / enqueue / sync (GPU drain) in ms
            std::printf("timeline stage %u (wait/enqueue/sync ms):", place.stage);
            for (std::size_t w = 0; w < st.window_ms.size(); ++w)
                std::printf(" %.0f/%.0f/%.0f", st.wait_window_ms[w], st.enqueue_window_ms[w], st.sync_window_ms[w]);
            std::printf("\n");
        }
    }
    if (place.head() && !a.dump.empty()) { dumped.resize(E->vocab()); E->read_logits(o, 1, dumped.data()); }
    {   // Link counter for the decode receipt: round trips of one decode message (header + one row) down and a token back.
        std::vector<std::byte> big(32U + row_bytes), small(32U);
        std::vector<double> rtt;
        for (int i = 0; i < 21; ++i) {
            if (down) { const auto t = Clock::now(); down->send(big); down->receive(small); if (i) rtt.push_back(ms_since(t)); }
            if (up) { up->receive(big); up->send(small); }
        }
        if (!rtt.empty()) {
            std::sort(rtt.begin(), rtt.end());
            std::printf("link ping stage %u->%u: %zu B down + 32 B back, median %.3f ms min %.3f ms max %.3f ms (%zu round trips)\n",
                        place.stage, place.stage + 1U, big.size(), rtt[rtt.size() / 2U], rtt.front(), rtt.back(), rtt.size());
        }
    }
    // Relayed decode: every stage's layer state to the decoder once, then the decoder runs the single-rank decode alone.
    layers::Placement dplace = place;
    if (relay) {
        layers::Link ru, rd;
        if (up) { ru.send = [&](std::span<const std::byte> b) { up->send(b); }; ru.receive = [&](std::span<std::byte> b) { up->receive(b); }; }
        if (down) { rd.send = [&](std::span<const std::byte> b) { down->send(b); }; rd.receive = [&](std::span<std::byte> b) { down->receive(b); }; }
        layers::RelayOps rops;
        rops.save = [&](std::uint32_t layer) {
            std::vector<std::byte> img(E->state_bytes(layer, layer + 1U));
            E->export_state(layer, layer + 1U, img.data());
            return img;
        };
        rops.load = [&](std::uint32_t layer, std::span<const std::byte> img) {
            if (img.size() != E->state_bytes(layer, layer + 1U)) throw std::runtime_error("state relay: byte count differs from this engine's layer");
            E->import_state(layer, layer + 1U, E->committed(), img.data());
        };
        rops.checksum = [](std::span<const std::byte> img) { return strix::dist::crc32c(img); };   // hardware CRC32C
        barrier();
        const auto rs = layers::relay_state(place, decoder_stage, first_token, up ? &ru : nullptr, down ? &rd : nullptr, rops);
        std::printf("state relay stage %u -> decoder stage %u: %.1f MB in %llu layer messages, %.1f ms\n", place.stage, decoder_stage,
                    double(rs.bytes) / 1e6, static_cast<unsigned long long>(rs.messages), rs.ms);
        if (!decodes_here) return 0;
        first_token = rs.token;
        E->set_stage(0U, L);
        dplace = layers::Placement{{0U, L}, 0U};
        up.reset(); down.reset();
    }
    // Layer-owned decode: one hidden row per cut per token, the token relayed back.
    std::vector<float> row(row_bytes / 4U);
    layers::DecodeOps d;
    d.row_bytes = row_bytes;
    d.forward = [&](std::uint32_t tok, bool head) { o = E->forward_range(dplace.first(), dplace.last(), &tok, 1U, head ? LogitRows::last : LogitRows::none); };
    d.top = [&] {
        std::uint32_t t = 0;
        E->read_top(o, 1, &t);
        if (!a.dump.empty()) { const auto at = dumped.size(); dumped.resize(at + E->vocab()); E->read_logits(o, 1, dumped.data() + at); }
        return t;
    };
    d.export_row = [&](std::span<std::byte> b) { E->export_hidden(row.data(), 1U); std::memcpy(b.data(), row.data(), row_bytes); };
    d.import_row = [&](std::span<const std::byte> b) { std::memcpy(row.data(), b.data(), row_bytes); E->import_hidden(row.data(), 1U); };
    layers::Link lu, ld;
    if (up) { lu.send = [&](std::span<const std::byte> b) { up->send(b); }; lu.receive = [&](std::span<std::byte> b) { up->receive(b); }; }
    if (down) { ld.send = [&](std::span<const std::byte> b) { down->send(b); }; ld.receive = [&](std::span<std::byte> b) { down->receive(b); }; }
    if (!relay) barrier();
    ClockSampler clk;
    const auto dec = layers::decode(dplace, first_token, a.gen, up ? &lu : nullptr, down ? &ld : nullptr, d);
    const long mhz = clk.stop();
    std::printf("generated:");
    for (auto t : dec.ids) std::printf(" %u", t);
    std::printf("\n");
    if (a.gen > 1U)
        std::printf("decode: %.2f ms/token, %.2f tok/s (stage %u, %u stages, sclk avg %ld MHz)\n", dec.total_ms / double(a.gen - 1U),
                    1000.0 * double(a.gen - 1U) / dec.total_ms, place.stage, dplace.stages(), mhz);
    if (dplace.head() && !a.dump.empty()) {
        std::ofstream f(a.dump, std::ios::binary);
        f.write(reinterpret_cast<const char*>(dumped.data()), std::streamsize(dumped.size() * 4U));
        std::printf("dumped %zu positions x %u to %s\n", dumped.size() / E->vocab(), E->vocab(), a.dump.c_str());
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
