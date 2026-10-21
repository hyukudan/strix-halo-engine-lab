#include "strix/deepseek41_device_forward.hpp"
#include "strix/dist/communicator.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

// One rank of the two-node deepseek41 decode (expert-parallel halves, split-vocabulary head), or both ranks in one
// process over the in-memory transport (--transport emulate).
//   strix-deepseek41-rank MODEL.gguf --rank R --transport usb4auto | usb4:/dev/tbstreamA,/dev/tbstreamB | tcp-listen:PORT |
//       tcp-connect:HOST:PORT | emulate  --layers N --prompt-ids a,b,c [--generate G] [--op-hashes] [--logits OUT.f32]
//       [--timing] [--engram-direct] [--no-gather] [--op-timing] [--window-rows R[,R...] --windows N] [--op-timing-out FILE]
// --fill N --fill-ids FILE [--fill-window R[,R...]]: N context tokens before the prompt in verify windows of R rows
// (the list cycled; default 8), every row committed; --fill-keep K[,K...]: commit only the first K rows of each window (a
// rejected draft, cycled) and feed the rejected tokens again (the context stays the same).
// --op-timing: per-op device time, host exchange time and weight bytes of the decode steps (R1), then, with
// --window-rows, of N verify windows of R rows after the generation (rows cycled from the sequence, all committed).
// --window-rows R[,R...]: the windows of each size in turn; without --op-timing only their mean step wall is printed
// (with STRIX_DS41_STEP_TRACE=1|2 the device forward writes its host/device timeline, see hip_ds41_step_trace.hpp).
// The per-step lines (and --op-hashes lines) use the format of strix-deepseek41-trimmed, so a rank's output diffs
// against the host forward's directly (ignore the routed_experts ":raw" lines: a rank holds only its expert slots).
namespace {

struct Args {
    std::string model, prompt, transport = "emulate", logits_path, routes_path;
    std::uint32_t rank = 0, layers = 40, generate = 8, max_positions = 4096, fill = 0, windows = 16;
    std::vector<std::uint32_t> window_rows;  // --window-rows R[,R...]
    bool op_timing = false;
    std::string op_timing_out;
    std::string fill_ids;  // --fill: tokens cycled to fill the context before the prompt
    std::vector<std::uint32_t> fill_windows;  // --fill-window R[,R...]: window sizes of the fill, cycled (default 8)
    std::vector<std::uint32_t> fill_keep;     // --fill-keep K[,K...]: rows committed per fill window, cycled (default all)
    unsigned connect_timeout = 60;  // seconds for opening the transport (a dead USB4 link blocks open())
    bool op_hashes = false, timing = false, engram_direct = false, gather = true, tp = false, split_shared = false, stream_experts = false;
};

std::uint64_t fnv(const std::vector<float>& v) {
    std::uint64_t h = 14695981039346656037ULL;
    for (const float f : v) {
        std::uint32_t b;
        std::memcpy(&b, &f, 4);
        for (int k = 0; k < 4; ++k, b >>= 8U) h = (h ^ (b & 0xffU)) * 1099511628211ULL;
    }
    return h;
}

// Sums of DeviceForward::last_op_timing over steps (one entry per plan step).
struct OpSums {
    std::vector<strix::deepseek41::OpTiming> sum;
    std::size_t steps = 0;
    double step_ms = 0.0;
    void add(const std::vector<strix::deepseek41::OpTiming>& t, double ms) {
        if (sum.empty()) sum = t;
        else
            for (std::size_t i = 0; i < t.size(); ++i) {
                sum[i].ms += t[i].ms;
                sum[i].exchange_ms += t[i].exchange_ms;
                sum[i].weight_bytes += t[i].weight_bytes;
            }
        ++steps;
        step_ms += ms;
    }
    // Per op kind (and site), summed over the layers, mean per step: ms, host exchange ms, GB read, GB/s; then per layer.
    void print(const char* label, std::uint32_t rows, const std::string& csv) const {
        if (steps == 0U) return;
        const double n = static_cast<double>(steps);
        struct Row {
            double ms = 0.0, ex = 0.0, bytes = 0.0;
        };
        std::map<std::string, Row> kinds;
        std::map<std::int32_t, Row> layers;
        Row total;
        for (const auto& o : sum) {
            const std::string k = std::string(strix::deepseek41::to_string(o.op)) + (o.site.empty() ? "" : "/" + o.site);
            for (Row* r : {&kinds[k], &layers[o.layer], &total}) {
                r->ms += o.ms / n;
                r->ex += o.exchange_ms / n;
                r->bytes += static_cast<double>(o.weight_bytes) / n;
            }
        }
        std::printf("op timing, %s (%u rows per step, mean over %zu steps; step wall %.2f ms, device events sum %.2f ms):\n", label, rows,
                    steps, step_ms / n, total.ms);
        std::printf("  %-22s %9s %7s %9s %8s %9s\n", "op", "ms", "share", "host-ex", "GB", "GB/s");
        for (const auto& [k, r] : kinds)
            std::printf("  %-22s %9.3f %6.1f%% %9.3f %8.4f %9.1f\n", k.c_str(), r.ms, 100.0 * r.ms / total.ms, r.ex, r.bytes / 1e9,
                        r.ms > 0.0 ? r.bytes / 1e9 / (r.ms / 1e3) : 0.0);
        std::printf("  %-22s %9.3f %6.1f%% %9.3f %8.4f %9.1f\n", "total", total.ms, 100.0, total.ex, total.bytes / 1e9,
                    total.bytes / 1e9 / (total.ms / 1e3));
        std::printf("  per layer ms / GB:");
        for (const auto& [l, r] : layers) std::printf(" %d:%.3f/%.4f", l, r.ms, r.bytes / 1e9);
        std::printf("\n");
        if (!csv.empty()) {
            std::ofstream f(csv, std::ios::app);
            f << "# " << label << " rows " << rows << " steps " << steps << "\nop,layer,site,ms,host_exchange_ms,weight_bytes\n";
            for (const auto& o : sum)
                f << strix::deepseek41::to_string(o.op) << ',' << o.layer << ',' << o.site << ',' << o.ms / n << ',' << o.exchange_ms / n << ','
                  << static_cast<double>(o.weight_bytes) / n << '\n';
        }
    }
};

std::unique_ptr<strix::dist::Transport> open_transport(const std::string& spec, std::uint32_t rank, unsigned timeout) {
    const std::chrono::seconds limit(timeout);
    const auto colon = spec.find(':');
    const std::string kind = spec.substr(0, colon), rest = colon == std::string::npos ? "" : spec.substr(colon + 1);
    if (kind == "tcp-listen") return strix::dist::tcp_listen(static_cast<std::uint16_t>(std::stoul(rest)), limit);
    if (kind == "tcp-connect") {
        const auto c2 = rest.rfind(':');
        return strix::dist::tcp_connect(rest.substr(0, c2), static_cast<std::uint16_t>(std::stoul(rest.substr(c2 + 1))), limit);
    }
    if (kind == "usb4auto") {  // every live stream of every cable, paired by handshake (the communicator's discovery)
        strix::dist::Usb4AutoOptions o;
        o.rank = rank;
        o.timeout = limit;
        strix::dist::Usb4Topology topology;
        auto t = strix::dist::usb4_host_auto(o, &topology);
        std::printf("usb4 topology: %s\n", topology.receipt().c_str());
        return t;
    }
    if (kind == "usb4") {
        std::vector<std::string> devices;
        std::stringstream ss(rest);
        for (std::string d; std::getline(ss, d, ',');) devices.push_back(d);
        return strix::dist::usb4_host(devices);
    }
    (void)rank;
    throw std::invalid_argument("unknown or unavailable transport " + spec);
}

void run_rank(const Args& a, std::uint32_t rank, std::unique_ptr<strix::dist::Transport> transport, bool print) {
    const auto t0 = std::chrono::steady_clock::now();
    // no transport: one node alone (world 1; with --stream-experts the experts are read from the file per layer)
    const bool single = transport == nullptr;
    std::unique_ptr<strix::dist::Communicator> comm_owner;
    if (!single) comm_owner = std::make_unique<strix::dist::Communicator>(strix::dist::Group{rank, 2U, 1U, 0x64733431ULL}, std::move(transport), 512U << 10U);
    const strix::dist::CommunicatorStats no_stats{};
    strix::deepseek41::DeviceForwardOptions o;
    o.layers = a.layers;
    o.max_positions = a.max_positions;
    o.rank = rank;
    o.world = single ? 1U : 2U;
    o.communicator = comm_owner.get();
    o.stream_experts = a.stream_experts;
    o.op_hashes = a.op_hashes;
    o.gather_logits = a.gather;
    o.engram_direct_io = a.engram_direct;
    o.layer_timing = a.timing;
    o.op_timing = a.op_timing;
    o.tp_attention = a.tp;
    o.split_shared = a.split_shared;
    o.record_routes = !a.routes_path.empty();
    strix::deepseek41::DeviceForward fwd(a.model, o);
    if (print)
        std::printf("rank %u of %u (%s): layers [0, %u), device weights %.2f GiB loaded in %.1f s\n", rank, single ? 1U : 2U,
                    single ? (a.stream_experts ? "one node, experts streamed from the file" : "one node") : comm_owner->transport_name().c_str(), a.layers,
                    static_cast<double>(fwd.weight_bytes()) / 1073741824.0, fwd.load_seconds());
    if (print && !fwd.overlay_description().empty()) std::printf("weights overlay: %s\n", fwd.overlay_description().c_str());
    if (print) std::printf("storage: %s\n", fwd.storage_description().c_str());
    std::ofstream logits_out;
    if (print && !a.logits_path.empty()) logits_out.open(a.logits_path, std::ios::binary);
    std::ofstream routes_out;  // one line per step: layer:id,id,... (the router's selection, both ranks the same)
    if (print && !a.routes_path.empty()) routes_out.open(a.routes_path);
    std::vector<std::uint32_t> seq;
    std::stringstream ss(a.prompt);
    for (std::string t; std::getline(ss, t, ',');) seq.push_back(static_cast<std::uint32_t>(std::stoul(t)));
    const std::size_t total = seq.size() + a.generate;
    double decode_seconds = 0.0;
    std::size_t decode_steps = 0;
    std::vector<double> layer_sum(a.layers, 0.0), exch_sum(a.layers, 0.0);
    std::size_t timed = 0;
    OpSums op_r1;
    if (a.fill != 0U) {  // context before the prompt, in verify windows of up to 8 rows (every row committed)
        std::vector<std::uint32_t> filler;
        std::ifstream f(a.fill_ids);
        for (std::string t; std::getline(f, t, ',');) filler.push_back(static_cast<std::uint32_t>(std::stoul(t)));
        if (filler.empty()) throw std::invalid_argument("--fill needs --fill-ids FILE with comma-separated token ids");
        const auto f0 = std::chrono::steady_clock::now();
        std::vector<std::uint32_t> window;
        // the first row alone: a window may not start before the first compressed entry exists (position 0)
        fwd.step(filler[0]);
        for (std::uint32_t i = 1, k = 0; i < a.fill; ++k) {
            window.clear();
            const std::uint32_t cap = a.fill_windows.empty() ? strix::deepseek41::DeviceForward::max_window_rows
                                                             : a.fill_windows[k % a.fill_windows.size()];
            while (window.size() < cap && i < a.fill) window.push_back(filler[i++ % filler.size()]);
            fwd.step_window(window);
            // --fill-keep: commit the first K rows only (a rejected draft); the rejected tokens are fed again next window
            const auto size = static_cast<std::uint32_t>(window.size());
            const std::uint32_t keep = a.fill_keep.empty() ? size : std::min(size, a.fill_keep[k % a.fill_keep.size()]);
            fwd.commit_window(keep);
            i -= size - keep;
        }
        if (print)
            std::printf("filled %u context tokens in %.1f s (%.1f tok/s in windows of 8)\n", a.fill,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - f0).count(),
                        a.fill / std::chrono::duration<double>(std::chrono::steady_clock::now() - f0).count());
    }
    for (std::size_t pos = 0; pos < seq.size(); ++pos) {
        const auto& logits = fwd.step(seq[pos]);
        const auto next = fwd.next_token();
        if (pos >= 1U) {  // the first step pays one-time costs
            decode_seconds += fwd.last_step_seconds();
            ++decode_steps;
        }
        if (a.op_timing && pos >= 1U) op_r1.add(fwd.last_op_timing(), fwd.last_step_seconds() * 1e3);
        if (a.timing && pos >= 1U) {
            for (std::uint32_t l = 0; l < a.layers; ++l) {
                layer_sum[l] += fwd.last_layer_timing()[l].seconds;
                exch_sum[l] += fwd.last_layer_timing()[l].exchange;
            }
            ++timed;
        }
        if (print) {
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const bool full = a.gather || a.op_hashes;
            std::printf("pos %zu fed %u argmax %u top %.4f logits_fnv %016llx (%.1f s)\n", pos, seq[pos], next,
                        full ? static_cast<double>(logits[next]) : 0.0, static_cast<unsigned long long>(full ? fnv(logits) : 0ULL), s);
            if (a.op_hashes)
                for (const auto& [name, hash] : fwd.last_step_hashes()) std::printf("  %s %016llx\n", name.c_str(), static_cast<unsigned long long>(hash));
            if (a.timing)
                std::printf("  step %.2f ms (engram read %.2f ms, head+argmax %.2f ms, expert copies %.2f ms)\n", fwd.last_step_seconds() * 1e3,
                            fwd.last_engram_seconds() * 1e3, fwd.last_head_seconds() * 1e3, fwd.last_stream_seconds() * 1e3);
            if (logits_out) logits_out.write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(logits.size() * 4U));
            if (routes_out) {
                const auto& routes = fwd.last_step_routes();
                for (std::size_t l = 0; l < routes.size(); ++l) {
                    routes_out << (l ? " " : "") << l << ':';
                    for (std::size_t k = 0; k < routes[l].size(); ++k) routes_out << (k ? "," : "") << routes[l][k];
                }
                routes_out << '\n';
            }
            std::fflush(stdout);
        }
        if (pos + 1U == seq.size() && seq.size() < total) seq.push_back(next);
    }
    std::vector<std::pair<std::uint32_t, OpSums>> op_windows;  // per window size
    for (const std::uint32_t rows : a.window_rows) {  // verify windows of R rows (the DSpark verifier's shape), every row committed
        if (rows == 0U || rows > strix::deepseek41::DeviceForward::max_window_rows) throw std::invalid_argument("--window-rows 1..8");
        OpSums& sums = op_windows.emplace_back(rows, OpSums{}).second;
        double wall = 0.0;
        std::vector<std::uint32_t> window;
        for (std::uint32_t w = 0; w < a.windows + 1U; ++w) {
            window.clear();
            for (std::uint32_t r = 0; r < rows; ++r) window.push_back(seq[(w * rows + r) % seq.size()]);
            fwd.step_window(window);
            fwd.commit_window(rows);
            if (w == 0U) continue;  // the first window pays one-time costs
            if (a.op_timing) sums.add(fwd.last_op_timing(), fwd.last_step_seconds() * 1e3);
            wall += fwd.last_step_seconds() * 1e3;
        }
        if (print && !a.op_timing && a.windows != 0U)
            std::printf("verify windows of %u rows: mean step wall %.2f ms over %u windows\n", rows, wall / a.windows, a.windows);
    }
    if (!print) return;
    op_r1.print("decode steps", 1U, a.op_timing_out);
    for (const auto& [rows, sums] : op_windows) sums.print("verify windows", rows, a.op_timing_out);
    const auto& st = single ? no_stats : comm_owner->stats();
    std::printf("decode: %zu steps after the first, %.2f tok/s (%.1f ms/token); exchanges %llu, %.1f MB sent, %.2f s in exchange (stage %.2f, wire %.2f, check %.2f)\n",
                decode_steps, decode_steps ? static_cast<double>(decode_steps) / decode_seconds : 0.0,
                decode_steps ? decode_seconds * 1e3 / static_cast<double>(decode_steps) : 0.0, static_cast<unsigned long long>(st.exchanges),
                static_cast<double>(st.bytes_sent) / 1e6, st.seconds, st.stage, st.wire, st.check);
    if (a.timing && timed != 0U) {
        std::printf("per layer (mean over %zu steps): layer total ms / exchange ms / compute ms\n", timed);
        double tot = 0.0, ex = 0.0;
        for (std::uint32_t l = 0; l < a.layers; ++l) {
            const double t = layer_sum[l] / static_cast<double>(timed) * 1e3, e = exch_sum[l] / static_cast<double>(timed) * 1e3;
            tot += t;
            ex += e;
            std::printf("  layer %2u %7.3f %7.3f %7.3f\n", l, t, e, t - e);
        }
        std::printf("  sum      %7.3f %7.3f %7.3f\n", tot, ex, tot - ex);
    }
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
        if (s == "--rank") a.rank = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--transport") a.transport = next();
        else if (s == "--layers") a.layers = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--prompt-ids") a.prompt = next();
        else if (s == "--generate") a.generate = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--max-positions") a.max_positions = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--logits") a.logits_path = next();
        else if (s == "--op-hashes") a.op_hashes = true;
        else if (s == "--timing") a.timing = true;
        else if (s == "--engram-direct") a.engram_direct = true;
        else if (s == "--no-gather") a.gather = false;
        else if (s == "--tp-attention") a.tp = true;
        else if (s == "--split-shared") a.split_shared = true;
        else if (s == "--stream-experts") a.stream_experts = true;
        else if (s == "--routes") a.routes_path = next();
        else if (s == "--fill") a.fill = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--fill-ids") a.fill_ids = next();
        else if (s == "--fill-window") {
            std::stringstream ws(next());
            for (std::string t; std::getline(ws, t, ',');) {
                const auto r = static_cast<std::uint32_t>(std::stoul(t));
                if (r == 0U || r > strix::deepseek41::DeviceForward::max_window_rows) throw std::invalid_argument("--fill-window 1..8");
                a.fill_windows.push_back(r);
            }
        }
        else if (s == "--fill-keep") {
            std::stringstream ws(next());
            for (std::string t; std::getline(ws, t, ',');) {
                const auto r = static_cast<std::uint32_t>(std::stoul(t));
                if (r == 0U) throw std::invalid_argument("--fill-keep 1..8");
                a.fill_keep.push_back(r);
            }
        }
        else if (s == "--connect-timeout") a.connect_timeout = static_cast<unsigned>(std::stoul(next()));
        else if (s == "--op-timing") a.op_timing = true;
        else if (s == "--window-rows") {
            std::stringstream ws(next());
            for (std::string t; std::getline(ws, t, ',');) a.window_rows.push_back(static_cast<std::uint32_t>(std::stoul(t)));
        }
        else if (s == "--windows") a.windows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--op-timing-out") a.op_timing_out = next();
        else if (a.model.empty()) a.model = s;
        else throw std::invalid_argument("unknown argument " + s);
    }
    if (a.model.empty() || a.prompt.empty()) throw std::invalid_argument("usage: strix-deepseek41-rank MODEL.gguf --rank R --transport T --prompt-ids a,b,c ...");
    if (a.transport == "single") {  // one node alone
        run_rank(a, 0U, nullptr, true);
        return 0;
    }
    if (a.transport == "emulate") {  // both ranks in this process over the in-memory pair; rank 0 prints
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
    // A blocked open() of a stream on a dead link would hold this node (and the peer) indefinitely: the alarm's
    // default action ends the process if the transport is not open within the timeout.
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
