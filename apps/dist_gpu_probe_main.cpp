// Two-rank micro-benchmark of device-buffer exchanges through the communicator: the host path (D2H, wire, H2D) and
// the GPU-direct path (device to device over streams in gpu_direct mode).
//   strix-dist-gpu-probe RANK [SIZES...]          (both ranks with the same sizes and environment)
// Environment: STRIX_PROBE_PATH=host|gpu|both|gpu4 (default both; gpu4: every stream in gpu_direct mode, TCP over Ethernet
// for the rendezvous only, STRIX_PROBE_RENDEZVOUS=HOST:PORT of rank 0), STRIX_PROBE_GPU_LINKS=N (gpu_direct streams used),
// STRIX_PROBE_DEPTH=2|4|8 (slots per direction), STRIX_PROBE_PAUSE_US=N (idle gap before each exchange, not timed), STRIX_PROBE_LOAD=1 (a stream filling 1 GiB in a loop, as load
// on the memory system), STRIX_PROBE_ITERATIONS=N, STRIX_PROBE_LABEL=text, STRIX_DIST_GPU_TIMING=1 (GPU breakdown),
// STRIX_PROBE_MIX="P1 P2 ..." (path both: after the gpu-direct arm, one arm per host share P % of a mixed route
// "inf:mix:RINGS:P", the head of each message over the host streams and the rest over the rings at the same time).
// Both ranks print, per path and size, the maximum and the count above 1 and 10 ms, and one "outlier" line per
// exchange above STRIX_PROBE_OUTLIER_US (default 10000): its index, wall-clock start (ns) and, for GPU-direct, where
// the time went on this rank (producer witness, SUBMIT incl. EAGAIN retries, RX wait, TX tail).
// Each exchange: a GPU producer (memset of the message with a value of rank and iteration), then the timed
// `exchange + stream synchronize`; afterwards (not timed) the received buffer is checked word by word.
// Output (rank 0): one JSON line per path and size, with p50/p99, CPU time per exchange (all threads) and, for
// GPU-direct, the breakdown of the progress thread (witness->submit, submit->RX complete, ->TX complete) and of the
// stream (scatter, wait, gather).

#include "strix/dist/communicator.hpp"
#include "strix/dist/gpu_direct.hpp"
#include "strix/dist/staged_ready_kernels.hpp"
#include "../src/dist_latency_trace.hpp"

#include <hip/hip_runtime.h>
#include <sys/resource.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {
void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
double at(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1U, static_cast<std::size_t>(q * static_cast<double>(v.size())))];
}
double cpu_seconds() {
    rusage u{};
    ::getrusage(RUSAGE_SELF, &u);
    return static_cast<double>(u.ru_utime.tv_sec + u.ru_stime.tv_sec) + static_cast<double>(u.ru_utime.tv_usec + u.ru_stime.tv_usec) * 1e-6;
}
std::string env(const char* name, const char* fallback) {
    const char* v = std::getenv(name);
    return v && *v ? v : fallback;
}
std::uint32_t value_of(unsigned i, unsigned rank) { return 0x9E3779B1U * (i + 1U) ^ (rank ? 0xA5A5A5A5U : 0x5A5A5A5AU); }

// Background load on the memory system: memsets of 1 GiB on its own stream until stopped.
struct Load {
    std::atomic<bool> stop{false};
    std::thread thread;
    void* buffer = nullptr;
    explicit Load(bool on) {
        if (!on) return;
        check(hipMalloc(&buffer, std::size_t{1} << 30U), "load buffer");
        thread = std::thread([this] {
            hipStream_t s{};
            (void)hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
            for (int v = 0; !stop.load(); ++v) {
                (void)hipMemsetAsync(buffer, v & 0xff, std::size_t{1} << 30U, s);
                (void)hipStreamSynchronize(s);
            }
            (void)hipStreamDestroy(s);
        });
    }
    ~Load() {
        stop = true;
        if (thread.joinable()) thread.join();
        if (buffer) (void)hipFree(buffer);
    }
};
} // namespace

int main(int argc, char** argv) try {
    if (argc < 2) { std::cerr << "usage: strix-dist-gpu-probe RANK [SIZES...]\n"; return 2; }
    const auto rank = static_cast<std::uint32_t>(std::stoul(argv[1]));
    std::vector<std::size_t> sizes;
    for (int i = 2; i < argc; ++i) sizes.push_back(std::stoull(argv[i]));
    if (sizes.empty()) sizes = {64, 20480, 40960, 163840};
    const std::string path = env("STRIX_PROBE_PATH", "both"), label = env("STRIX_PROBE_LABEL", "");
    ::setenv("STRIX_DIST_ROUTE", "gpu", 0); // the gpu-direct arm measures the direct path unless a route is given
    const bool load = env("STRIX_PROBE_LOAD", "0") == "1";
    strix::dist::Usb4AutoOptions options;
    options.rank = rank;
    options.timeout = std::chrono::milliseconds(60000);
    strix::dist::Usb4Topology topology;
    std::unique_ptr<strix::dist::Transport> control;
    std::unique_ptr<strix::dist::DeviceExchange> direct;
    strix::dist::GpuDirectSetup setup;
    setup.depth = static_cast<std::uint32_t>(std::stoul(env("STRIX_PROBE_DEPTH", "4")));
    setup.max_streams = static_cast<std::uint32_t>(std::stoul(env("STRIX_PROBE_GPU_LINKS", "8")));
    std::string receipt;
    if (path == "gpu4") {
        const auto rv = env("STRIX_PROBE_RENDEZVOUS", "");
        const auto c = rv.rfind(':');
        if (c == std::string::npos) throw std::invalid_argument("STRIX_PROBE_RENDEZVOUS=HOST:PORT (rank 0's Ethernet address)");
        const auto port = static_cast<std::uint16_t>(std::stoul(rv.substr(c + 1)));
        control = rank == 0U ? strix::dist::tcp_listen(port, options.timeout) : strix::dist::tcp_connect(rv.substr(0, c), port, options.timeout);
        direct = strix::dist::usb4_gpu_direct_all(options, *control, setup, &receipt);
    } else {
        control = strix::dist::usb4_host_auto(options, &topology);
        if (rank == 0U) std::cout << topology.receipt() << '\n';
        if (path != "host") direct = strix::dist::usb4_gpu_direct(options, topology, *control, setup, &receipt);
    }
    if (rank == 0U && direct) std::cout << receipt << '\n';
    const std::size_t largest = *std::max_element(sizes.begin(), sizes.end());
    hipStream_t s{};
    check(hipStreamCreate(&s), "stream");
    void *local = nullptr, *remote = nullptr;
    check(hipMalloc(&local, largest), "local");
    check(hipMalloc(&remote, largest), "remote");
    std::vector<std::uint32_t> got(largest / 4U);
    const unsigned pause_us = static_cast<unsigned>(std::stoul(env("STRIX_PROBE_PAUSE_US", "0")));
    const bool hot_gap = env("STRIX_PROBE_HOT_GAP", "0") == "1";
    const bool staged_ready = env("STRIX_PROBE_STAGED_READY", "0") == "1";
    const double outlier_us = std::stod(env("STRIX_PROBE_OUTLIER_US", "10000"));
    const auto run = [&](strix::dist::Communicator& c, const char* name) {
        Load background(load);
        for (const auto size : sizes) {
            const unsigned iterations = static_cast<unsigned>(std::stoul(env("STRIX_PROBE_ITERATIONS", size > (std::size_t{1} << 20U) ? "20" : "300")));
            std::vector<double> us;
            std::vector<double> available_us;
            bool used_staged_ready = false;
            std::size_t bad = 0;
            const auto* direct_stats = c.device_exchange();
            const auto before = direct_stats ? direct_stats->stats() : strix::dist::DeviceExchangeStats{};
            double cpu = 0.0;
            std::size_t over1 = 0, over10 = 0;
            double worst = 0.0;
            for (unsigned i = 0; i < iterations + 10U; ++i) {
                // Diagnostic rendezvous after HIP/transport warmup. Each rank remains a user process;
                // the external receipt pins every existing thread before releasing this local gate.
                if (i == 10U) {
                    const auto gate = env("STRIX_PROBE_GATE", "");
                    if (!gate.empty()) {
                        { std::ofstream ready(gate + ".ready"); ready << ::getpid() << '\n'; }
                        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(30);
                        while (!std::ifstream(gate).good()) {
                            if (std::chrono::steady_clock::now() >= until) throw std::runtime_error("probe gate timed out");
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        }
                    }
                }
                strix::dist::ProducerStage produced{};
                if (staged_ready) produced = c.prepare_staged_ready(s, size);
                if (produced.data) {
                    strix::dist::enqueue_staged_pattern(produced, size, value_of(i, rank), s);
                    used_staged_ready = true;
                } else {
                    check(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(local), static_cast<int>(value_of(i, rank)), size / 4U, s), "producer");
                }
                check(hipStreamSynchronize(s), "producer");
                if (pause_us && hot_gap) {
                    const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(pause_us);
                    while (std::chrono::steady_clock::now() < until) {
#if defined(__x86_64__)
                        __builtin_ia32_pause();
#endif
                    }
                } else if (pause_us) std::this_thread::sleep_for(std::chrono::microseconds(pause_us));
                const double c0 = cpu_seconds();
                const auto s0 = direct_stats ? direct_stats->stats() : strix::dist::DeviceExchangeStats{};
                const auto wall0 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                const auto t0 = std::chrono::steady_clock::now();
                double available = 0.0;
                if (produced.data) {
                    c.start_staged_ready(s, size, strix::dist::Tag{1U, i, 0U});
                    const auto received = c.finish_staged_ready();
                    available = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                    strix::dist::latency::mark("probe_ready_available", i);
                    // A qualification consumer, not part of the readiness latency. It proves that a GPU kernel
                    // reads the coherent view correctly and publishes its lifetime before the next RX can reuse it.
                    strix::dist::enqueue_staged_consume(received, remote, size, s);
                    c.release_staged_ready(s);
                } else {
                    c.exchange(s, local, remote, size, strix::dist::Tag{1U, i, 0U});
                }
                check(hipStreamSynchronize(s), "exchange");
                strix::dist::latency::mark("probe_h2d_sync_done", i);
                const double t = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                if (!produced.data) available = t;
                const double c1 = cpu_seconds();
                check(hipMemcpy(got.data(), remote, size, hipMemcpyDeviceToHost), "read back");
                const auto expect = value_of(i, 1U - rank);
                bad += std::all_of(got.begin(), got.begin() + static_cast<std::ptrdiff_t>(size / 4U), [&](std::uint32_t w) { return w == expect; }) ? 0U : 1U;
                if (i >= 10U) {
                    us.push_back(t);
                    available_us.push_back(available);
                    cpu += c1 - c0;
                    over1 += t > 1000.0 ? 1U : 0U;
                    over10 += t > 10000.0 ? 1U : 0U;
                    worst = std::max(worst, t);
                }
                if (t > outlier_us) {
                    std::cout << "outlier rank " << rank << " path " << name << " bytes " << size << " i " << i << " wall_ns " << wall0 << " us " << t;
                    if (direct_stats) {
                        const auto s1 = direct_stats->stats();
                        std::cout << " witness_us " << (s1.producer_wait - s0.producer_wait) * 1e6 << " submit_us " << (s1.submit - s0.submit) * 1e6
                                  << " eagain " << (s1.eagain_retries - s0.eagain_retries) << " rx_us " << (s1.rx_wait - s0.rx_wait) * 1e6
                                  << " tx_tail_us " << (s1.tx_tail - s0.tx_tail) * 1e6;
                    }
                    std::cout << '\n' << std::flush;
                }
            }
            if (bad) throw std::runtime_error(std::string(name) + ": received bytes differ (" + std::to_string(bad) + " exchanges)");
            std::cout << "{\"label\":\"" << label << "\",\"rank\":" << rank << ",\"path\":\"" << name << "\",\"load\":" << (load ? 1 : 0) << ",\"pause_us\":" << pause_us << ",\"bytes\":" << size
                      << ",\"p50_us\":" << at(us, 0.5) << ",\"p99_us\":" << at(us, 0.99) << ",\"max_us\":" << worst << ",\"over_1ms\":" << over1
                      << ",\"over_10ms\":" << over10 << ",\"n\":" << us.size() << ",\"cpu_us\":" << cpu * 1e6 / iterations;
            if (staged_ready)
                std::cout << ",\"staged_ready\":" << (used_staged_ready ? 1 : 0)
                          << ",\"available_p50_us\":" << at(available_us, 0.5)
                          << ",\"available_p99_us\":" << at(available_us, 0.99);
            if (direct_stats) {
                const auto after = direct_stats->stats();
                const double n = static_cast<double>(after.exchanges - before.exchanges);
                const double timed = static_cast<double>(after.timed - before.timed);
                if (n > 0)
                    std::cout << ",\"witness_us\":" << (after.producer_wait - before.producer_wait) * 1e6 / n << ",\"submit_us\":"
                              << (after.submit - before.submit) * 1e6 / n << ",\"rx_us\":" << (after.rx_wait - before.rx_wait) * 1e6 / n
                              << ",\"tx_tail_us\":" << (after.tx_tail - before.tx_tail) * 1e6 / n;
                if (timed > 0)
                    std::cout << ",\"scatter_gpu_us\":" << (after.scatter_gpu - before.scatter_gpu) * 1e6 / timed << ",\"wait_gpu_us\":"
                              << (after.wait_gpu - before.wait_gpu) * 1e6 / timed << ",\"gather_gpu_us\":"
                              << (after.gather_gpu - before.gather_gpu) * 1e6 / timed;
            }
            std::cout << "}\n" << std::flush;
        }
    };
    struct Borrowed final : strix::dist::Transport {
        strix::dist::Transport& t;
        explicit Borrowed(strix::dist::Transport& x) : t(x) {}
        void send(std::span<const std::byte> m) override { t.send(m); }
        void receive(std::span<std::byte> m) override { t.receive(m); }
        void exchange(std::span<const std::byte> o, std::span<std::byte> i) override { t.exchange(o, i); }
        [[nodiscard]] std::string name() const override { return t.name(); }
    };
    if (path != "gpu" && path != "gpu4") {
        strix::dist::Communicator host({rank, 2U, 1U, 1U}, std::make_unique<Borrowed>(*control), largest);
        run(host, "host");
    }
    if (direct && env("STRIX_PROBE_DIAG", "0") == "1")
        for (std::size_t b : {std::size_t{20480}, std::size_t{163840}, std::size_t{1} << 20U})
            std::cout << "rank " << rank << " " << direct->diagnose(b) << std::endl;
    if (direct) {
        strix::dist::Communicator gpu({rank, 2U, 1U, 2U}, std::make_unique<Borrowed>(*control), std::move(direct), largest);
        run(gpu, "gpu-direct");
        std::stringstream mix(env("STRIX_PROBE_MIX", ""));
        for (std::string p; mix >> p;) {
            const auto rings = std::to_string(gpu.device_exchange()->rings());
            gpu.set_routes("inf:mix:" + rings + ":" + p);
            const auto name = "mix-" + p;
            run(gpu, name.c_str());
        }
        if (env("STRIX_PROBE_DIAG", "0") == "1") // again, now that the controller has written the RX arena
            for (std::size_t b : {std::size_t{20480}, std::size_t{163840}, std::size_t{1} << 20U})
                std::cout << "rank " << rank << " after DMA: " << const_cast<strix::dist::DeviceExchange*>(gpu.device_exchange())->diagnose(b) << std::endl;
    }
    const auto exit_gate = env("STRIX_PROBE_EXIT_GATE", "");
    if (!exit_gate.empty()) {
        { std::ofstream ready(exit_gate + ".ready"); ready << ::getpid() << '\n'; }
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!std::ifstream(exit_gate).good()) {
            if (std::chrono::steady_clock::now() >= until) throw std::runtime_error("probe exit gate timed out");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    (void)hipFree(local);
    (void)hipFree(remote);
    (void)hipStreamDestroy(s);
    return 0;
} catch (const std::exception& e) {
    std::cerr << "strix-dist-gpu-probe: " << e.what() << '\n';
    return 1;
}
