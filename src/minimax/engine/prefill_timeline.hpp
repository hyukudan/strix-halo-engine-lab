#pragma once

#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace strix::minimax::engine {

// Diagnostic only. All storage (including host records) is allocated at engine
// startup. Never synchronize or allocate while recording the production stream.
struct PrefillTimeline {
    using Clock = std::chrono::steady_clock;
    struct Span {
        const char* category = nullptr;
        std::size_t chunk = 0;
        unsigned layer = 0;
        bool exposed = false;
    };
    struct HostSpan {
        Clock::time_point before{}, after{};
        const char* category = nullptr;
        std::size_t chunk = 0;
        unsigned layer = 0;
    };
    std::vector<hipEvent_t> events;
    std::vector<Span> spans;
    std::vector<HostSpan> host;
    std::size_t used = 0, host_used = 0, chunk = 0;
    unsigned layer = 0;
    bool full = false, received = false;
    Clock::time_point start{};

    static void check(hipError_t e) {
        if (e != hipSuccess) throw std::runtime_error("prefill timeline HIP event failure");
    }
    PrefillTimeline(const PrefillTimeline&) = delete;
    PrefillTimeline& operator=(const PrefillTimeline&) = delete;

    PrefillTimeline(std::size_t capacity, std::size_t max_rows, unsigned layers, unsigned scratch_windows) {
        const std::size_t chunks = (capacity + max_rows - 1) / max_rows;
        // Enough even with one-window scratch batching (two categories/window),
        // plus linears, posts, MoE calls and exchanges.
        const std::size_t per_layer = 64 + 4 * (((max_rows + 15) / 16 + scratch_windows - 1) / scratch_windows);
        spans.resize(chunks * layers * per_layer + 32);
        host.resize(chunks * layers * 8 + 32);
        events.resize(spans.size() * 2, nullptr);
        try {
            for (auto& e : events) check(hipEventCreate(&e));
        } catch (...) {
            for (const auto e : events) if (e != nullptr) (void)hipEventDestroy(e);
            throw;
        }
    }
    ~PrefillTimeline() {
        for (const auto e : events) (void)hipEventDestroy(e);
    }
    void begin() {
        used = host_used = 0;
        full = received = false;
        start = Clock::now();
    }
    void context(std::size_t c, unsigned l) { chunk = c; layer = l; }
    void mark(const char* category, bool before, hipStream_t stream = nullptr) {
        if (full) return;
        if (used == spans.size()) { full = true; return; }
        if (before) {
            spans[used] = {category, chunk, layer, received};
            received = false;
        }
        check(hipEventRecord(events[2 * used + (before ? 0U : 1U)], stream));
        if (!before) ++used;
    }
    std::size_t host_begin(const char* category) {
        return host_begin(category, chunk);
    }
    std::size_t host_begin(const char* category, std::size_t member) {
        if (full || host_used == host.size()) { full = true; return host.size(); }
        const std::size_t i = host_used++;
        host[i] = {Clock::now(), {}, category, member, layer};
        return i;
    }
    void host_end(std::size_t i, bool recv) {
        if (i < host.size()) host[i].after = Clock::now();
        if (recv) received = true;
    }
    static void msa_mark(void* user, const char* label, bool before) {
        static_cast<PrefillTimeline*>(user)->mark(label, before);
    }
    // The existing MoE markers are host callbacks, not timing events. Use
    // just start/combine for one GPU interval per call, after host planning.
    static void moe_mark(const char* label, hipStream_t stream, void* user) {
        auto& t = *static_cast<PrefillTimeline*>(user);
        if (std::strcmp(label, "start") == 0) t.mark("moe", true, stream);
        else if (std::strcmp(label, "combine") == 0) t.mark("moe", false, stream);
    }
    void report(unsigned rank) const {
        const double wall = std::chrono::duration<double>(Clock::now() - start).count();
        struct Total { const char* category = nullptr; double seconds = 0; std::size_t count = 0; };
        std::array<Total, 64> totals{};
        double busy = 0, idle = 0, exposed = 0, recv_host = 0, send_host = 0;
        const auto elapsed = [&](std::size_t a, std::size_t b) {
            float ms = 0;
            check(hipEventElapsedTime(&ms, events[a], events[b]));
            return static_cast<double>(ms) * 0.001;
        };
        for (std::size_t i = 0; i < used; ++i) {
            const double seconds = elapsed(2 * i, 2 * i + 1);
            busy += seconds;
            for (auto& t : totals) {
                if (t.category == nullptr) t.category = spans[i].category;
                if (std::strcmp(t.category, spans[i].category) == 0) {
                    t.seconds += seconds;
                    ++t.count;
                    break;
                }
            }
            if (i != 0) {
                const double gap = std::max(0.0, elapsed(2 * i - 1, 2 * i));
                idle += gap;
                if (spans[i].exposed) exposed += gap;
            }
        }
        for (std::size_t i = 0; i < host_used; ++i) {
            const double s = std::chrono::duration<double>(host[i].after - host[i].before).count();
            if (std::strcmp(host[i].category, "recv") == 0) recv_host += s;
            else send_host += s;
        }
        std::printf("pftimeline rank %u: wall %.6f s | gpu-busy %.6f s | gpu-idle (gaps between consecutive recorded intervals on the stream) %.6f s\n", rank, wall, busy, idle);
        for (const auto& t : totals) if (t.category != nullptr)
            std::printf("pftimeline rank %u: %s %.6f s count %zu\n", rank, t.category, t.seconds, t.count);
        std::printf("pftimeline rank %u: exposed-exchange-wait %.6f s | host recv-wait %.6f s | host send %.6f s%s\n",
                    rank, exposed, recv_host, send_host, full ? " | POOL EXHAUSTED: recording stopped (partial totals)" : "");
        std::fflush(stdout);
    }
};

}  // namespace strix::minimax::engine
