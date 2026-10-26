// Cost of the armed instrumentation on the serve layer (no GPU): per finished request on the engine thread
// (GenerationQueue::finish with the observed path: still_observed, observe_delta, the locked additions), per request on
// the HTTP side (ApiService: the ring record, the response), and the scrape documents, unarmed and armed with an
// aggressive scraper (GET /metrics and GET /api/status?detail=1 every millisecond, far above a 1 s production cadence).
// The checks: armed costs stay flat over thousands of requests served while armed (nothing grows with the number of
// observed requests: no per-request list, ring or vector the paths walk), the scrape documents stop growing once the
// ring is full, the heap does not grow with the observed requests, and the armed engine-thread cost per request stays
// within a few microseconds of the unarmed one. Times are reported; the bounds are loose (the test runs under sanitizers).
#include "strix/serve/api.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <malloc.h>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace strix::serve;
using Clock = std::chrono::steady_clock;
int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::cerr << "FAIL: " << what << '\n'; }
}

std::shared_ptr<const BpeTokenizer> tiny_tokenizer() {
    std::vector<std::string> tokens;
    std::vector<std::int32_t> types;
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        int cp = b;
        const bool direct = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        if (!direct) cp = 256 + extra++;
        std::string s;
        if (cp < 0x80) s.push_back(static_cast<char>(cp));
        else { s.push_back(static_cast<char>(0xC0 | (cp >> 6))); s.push_back(static_cast<char>(0x80 | (cp & 0x3F))); }
        tokens.push_back(s);
        types.push_back(1);
    }
    for (const char* special : {"<|endoftext|>", "<|im_start|>", "<|im_end|>"}) { tokens.push_back(special); types.push_back(3); }
    for (const char* special : {"<tool_call>", "</tool_call>", "<think>", "</think>"}) { tokens.push_back(special); types.push_back(4); }
    return BpeTokenizer::from_parts(std::move(tokens), std::move(types), {}, "qwen35");
}

struct Sink final : HttpResponseWriter {
    std::string body;
    void send(int, std::string_view, std::string_view b, const std::vector<std::pair<std::string, std::string>>&) override { body = b; }
    bool begin_stream(int, std::string_view, const std::vector<std::pair<std::string, std::string>>&) override { return false; }
    bool write(std::string_view) override { return false; }
    void end_stream() override {}
    [[nodiscard]] bool client_gone() const override { return false; }
};
std::string call(ApiService& api, const std::string& method, const std::string& path, const std::string& query = {}, const std::string& body = {},
                 const std::string& id = {}) {
    HttpRequest r;
    r.method = method;
    r.path = path;
    r.query = query;
    r.body = body;
    if (!id.empty()) r.headers = {{"x-request-id", id}};
    Sink s;
    api.handle(r, s);
    return s.body;
}

double median_us(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2), v.end());
    return v[v.size() / 2];
}

struct Phase {
    std::vector<double> finish_us;                                  // engine-thread finish cost, in finishing order
    std::vector<std::pair<double, double>> request_us;              // (start time, HTTP request latency), sorted by start
    std::vector<std::pair<std::size_t, std::size_t>> doc_sizes;     // (requests started, status detail + metrics bytes)
    std::vector<std::vector<std::pair<double, double>>> per_client; // the clients' own samples (reserved up front)
    // bytes the phase's own sample vectors hold (reserved up front: they do not grow during the phase)
    [[nodiscard]] std::size_t own_bytes() const {
        std::size_t b = finish_us.capacity() * sizeof(double) + request_us.capacity() * sizeof(std::pair<double, double>) +
                        doc_sizes.capacity() * sizeof(std::pair<std::size_t, std::size_t>);
        for (const auto& c : per_client) b += c.capacity() * sizeof(std::pair<double, double>);
        return b;
    }
};

// The process's allocated heap: small chunks plus mmapped blocks (large vectors), after returning free memory.
std::size_t heap_bytes() {
    malloc_trim(0);
    const auto m = mallinfo2();
    return m.uordblks + m.hblkhd;
}

// N requests through the API (4 client threads) and a batching engine thread (take / steps / finish) on a queue that
// stays open (the armed queue serves two phases); the engine thread's per-request cost is the time
// GenerationQueue::finish takes (the observed path runs inside it).
Phase run_phase(ApiService& api, GenerationQueue& queue, std::uint32_t im_end, std::size_t n, bool scrape) {
    Phase ph;
    ph.finish_us.reserve(n);
    ph.request_us.reserve(n);
    ph.doc_sizes.reserve(200000);
    ph.per_client.resize(4);
    for (auto& c : ph.per_client) c.reserve(n);
    std::atomic<bool> stop{false};
    std::atomic<std::size_t> next{0}, finished{0};
    const auto t_start = Clock::now();
    std::thread engine([&] {
        while (finished.load() < n) {
            auto t = queue.take(false);
            if (!t) {
                std::this_thread::sleep_for(std::chrono::microseconds(20));
                continue;
            }
            std::uint64_t armed_steps = 0;
            for (int step = 0; step < 8; ++step) armed_steps += observe_armed() ? 1U : 0U;  // the per-step load
            const std::vector<std::uint32_t> out{static_cast<std::uint32_t>('o'), static_cast<std::uint32_t>('k'), im_end};
            (void)t->job.on_tokens(out);
            GenerationStats st;
            st.prompt_tokens = t->job.prompt.size();
            st.cached_tokens = 1;
            st.imported_tokens = 1;
            st.generated_tokens = out.size();
            st.decode_ms = 10.0;
            st.prefill_ms = 5.0;
            st.finish = "stop";
            if (t->job.observation_complete()) {
                st.first_token_ms = 5.0;
                st.proposed = armed_steps;
            }
            const auto t0 = Clock::now();
            queue.finish(*t, st);
            ph.finish_us.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
            finished.fetch_add(1U);
        }
    });
    std::thread scraper([&] {
        while (scrape && !stop.load()) {
            const auto served = next.load();
            const auto bytes = call(api, "GET", "/metrics").size() + call(api, "GET", "/api/status", "detail=1").size();
            if (ph.doc_sizes.size() < ph.doc_sizes.capacity()) ph.doc_sizes.emplace_back(served, bytes);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    std::vector<std::thread> clients;
    for (std::size_t c = 0; c < 4; ++c)
        clients.emplace_back([&, c] {
            for (std::size_t i; (i = next.fetch_add(1)) < n;) {
                const auto t0 = Clock::now();
                (void)call(api, "POST", "/v1/completions", {}, R"({"max_tokens":4,"prompt":"Hello there"})", "cost-" + std::to_string(1000000 + i));
                ph.per_client[c].emplace_back(std::chrono::duration<double>(t0 - t_start).count(),
                                              std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
            }
        });
    for (auto& c : clients) c.join();
    stop.store(true);
    scraper.join();
    engine.join();
    for (const auto& c : ph.per_client) ph.request_us.insert(ph.request_us.end(), c.begin(), c.end());
    std::sort(ph.request_us.begin(), ph.request_us.end());  // chronological: the phase's first and last requests
    return ph;
}

std::vector<double> latencies(const std::vector<std::pair<double, double>>& v, std::size_t from, std::size_t count) {
    std::vector<double> out;
    for (std::size_t i = from; i < from + count && i < v.size(); ++i) out.push_back(v[i].second);
    return out;
}
}  // namespace

int main() try {
    const auto tokenizer = tiny_tokenizer();
    const std::uint32_t im_end = tokenizer->find("<|im_end|>");
    constexpr std::size_t n = 6000;
    ApiConfig config;
    config.model_id = "cost";
    config.stop_tokens = {im_end};
    config.observe_expiry_seconds = 0.0;  // armed until the disarm below

    observe_disarm();
    GenerationQueue q_unarmed;
    ApiService api_unarmed(config, tokenizer, q_unarmed, 4096, false);
    const auto unarmed = run_phase(api_unarmed, q_unarmed, im_end, n, false);
    check(!observe_armed() && q_unarmed.observed().requests == 0U, "unarmed phase: nothing observed");

    // armed: a warm phase (fills the ring, sizes the allocator's pools), then the measured phase on the same service
    GenerationQueue q_armed;
    ApiService api_armed(config, tokenizer, q_armed, 4096, false);
    (void)call(api_armed, "GET", "/metrics");
    const auto warm = run_phase(api_armed, q_armed, im_end, n / 2, true);
    const auto heap_1 = heap_bytes();
    const auto armed = run_phase(api_armed, q_armed, im_end, n, true);
    const auto heap_2 = heap_bytes();
    check(observe_armed(), "armed phases: armed throughout");
    check(q_armed.observed().requests == n + n / 2, "every request of the armed phases observed: " + std::to_string(q_armed.observed().requests));
    // the heap between the two checkpoints, less the measured phase's own reserved sample vectors
    const auto heap_growth = static_cast<long long>(heap_2) - static_cast<long long>(heap_1) - static_cast<long long>(armed.own_bytes());
    const double per_request = static_cast<double>(heap_growth) / static_cast<double>(n);
    (void)warm;

    // the scrape documents: the ring is full (50 records) after 50 requests; they stay the same size from then on
    std::size_t first_doc = 0, largest_doc = 0;
    for (const auto& [served, bytes] : armed.doc_sizes) {
        if (first_doc == 0U) first_doc = bytes;
        largest_doc = std::max(largest_doc, bytes);
    }

    const auto head = [](const std::vector<double>& v) { return std::vector<double>(v.begin(), v.begin() + 1000); };
    const auto tail = [](const std::vector<double>& v) { return std::vector<double>(v.end() - 1000, v.end()); };
    const double uf = median_us(unarmed.finish_us), af = median_us(armed.finish_us);
    const double af_first = median_us(head(armed.finish_us)), af_last = median_us(tail(armed.finish_us));
    const double ur = median_us(latencies(unarmed.request_us, 0, n)), ar = median_us(latencies(armed.request_us, 0, n));
    const double ar_first = median_us(latencies(armed.request_us, 0, 1000)), ar_last = median_us(latencies(armed.request_us, n - 1000, 1000));
    std::printf("engine-thread finish per request: unarmed %.2f us, armed %.2f us (first 1000 %.2f, last 1000 %.2f of the measured phase)\n", uf, af,
                af_first, af_last);
    std::printf("HTTP request latency (chronological): unarmed %.1f us, armed + 1 ms scraper %.1f us (first 1000 %.1f, last 1000 %.1f)\n", ur, ar,
                ar_first, ar_last);
    std::printf("heap over the measured armed phase (after %zu warm armed requests): %+lld bytes beyond its own samples over %zu observed requests "
                "(%.2f B/request); scrape documents %zu..%zu bytes over %zu scrapes\n",
                n / 2, heap_growth, n, per_request, first_doc, largest_doc, armed.doc_sizes.size());
    check(af_last <= 2.0 * af_first + 20.0, "armed engine-thread cost flat over the requests served while armed");
    check(ar_last <= 2.0 * ar_first + 200.0, "armed HTTP request latency flat over the requests served while armed");
    check(af <= 2.0 * uf + 20.0, "armed engine-thread cost per request within a few microseconds of unarmed");
#if defined(__SANITIZE_THREAD__) || defined(__SANITIZE_ADDRESS__)
    std::printf("heap bound not checked: a sanitizer's allocator replaces glibc's (its statistics do not see the allocations)\n");
#else
    check(per_request < 8.0, "no heap growth with the observed requests (< 8 B per request beyond the samples): " + std::to_string(per_request));
#endif
    check(first_doc > 0U && largest_doc <= first_doc + first_doc / 20U, "the scrape documents keep their size once the ring is full");
    q_unarmed.shutdown();
    q_armed.shutdown();
    observe_disarm();
    std::cout << "serve observe cost tests: failures=" << failures << '\n';
    return failures ? 1 : 0;
} catch (const std::exception& e) {
    std::cerr << "serve observe cost tests: " << e.what() << '\n';
    return 1;
}
