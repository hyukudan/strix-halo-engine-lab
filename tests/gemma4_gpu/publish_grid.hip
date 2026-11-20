// Byte gate of the KV commit publication: the (row, producer) grid against the ordered one-block-per-producer
// loop, for counts 1/8/64/512, a commit longer than the sliding ring, starts across a ring wrap, global and
// sliding producers, width 0 (skipped) and a width that is not a multiple of four. K, V and tags must match bit for
// bit, including the slots a commit does not touch. `--bench` also times both orders on the E4B commit geometry
// (24 producers: 20 sliding x 512 floats, 4 global x 1024 floats, 512-row chunk) and prints the median ms.
#include <hip/hip_runtime.h>
#include "gemma4/kv/storage.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace strix::gemma4;
using namespace strix::gemma4::gpu;
namespace g4kv = strix::gemma4::gpu::detail;
namespace {
#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string("check failed: ") + #x + " line " + std::to_string(__LINE__)); } while (false)
void hip_ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
struct Producer {
    unsigned width = 0, slots = 0;
    bool global = false;
};
struct Arena {
    std::vector<Producer> layout;
    std::vector<Buffer> k, v, tags, ok, ov;
    std::vector<std::vector<float>> k0, v0, ok0, ov0;
    std::vector<std::vector<std::uint64_t>> tags0;
    Buffer descriptors;
    Arena(std::vector<Producer> producers, unsigned rows, std::uint32_t seed) : layout(std::move(producers)) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> value(-4.0F, 4.0F);
        std::vector<g4kv::Publication> d(layout.size());
        for (std::size_t l = 0; l < layout.size(); ++l) {
            const auto& p = layout[l];
            const std::size_t cache = std::size_t(p.slots) * p.width, overlay = std::size_t(rows) * p.width;
            auto fill = [&](std::size_t n) { std::vector<float> x(n); for (auto& f : x) f = value(rng); return x; };
            k0.push_back(fill(cache)); v0.push_back(fill(cache)); ok0.push_back(fill(overlay)); ov0.push_back(fill(overlay));
            std::vector<std::uint64_t> t(p.slots);
            for (auto& x : t) x = rng();
            tags0.push_back(t);
            k.emplace_back(std::max<std::size_t>(cache * 4, 4)); v.emplace_back(std::max<std::size_t>(cache * 4, 4));
            tags.emplace_back(std::max<std::size_t>(std::size_t(p.slots) * 8, 8));
            ok.emplace_back(std::max<std::size_t>(overlay * 4, 4)); ov.emplace_back(std::max<std::size_t>(overlay * 4, 4));
            if (overlay) {
                ok.back().upload(ok0.back().data(), overlay * 4);
                ov.back().upload(ov0.back().data(), overlay * 4);
            }
            d[l] = {k.back().as<float>(), v.back().as<float>(), tags.back().as<std::uint64_t>(),
                    ok.back().as<float>(), ov.back().as<float>(), p.slots, p.width, p.global};
        }
        descriptors = Buffer(d.size() * sizeof(g4kv::Publication));
        descriptors.upload(d.data(), descriptors.size());
    }
    void reset() {
        for (std::size_t l = 0; l < layout.size(); ++l) {
            if (!k0[l].empty()) {
                k[l].upload(k0[l].data(), k0[l].size() * 4);
                v[l].upload(v0[l].data(), v0[l].size() * 4);
            }
            if (!tags0[l].empty()) tags[l].upload(tags0[l].data(), tags0[l].size() * 8);
        }
        hip_ok(hipDeviceSynchronize(), "reset");
    }
    void publish(std::uint64_t start, unsigned rows, g4kv::PublishOrder order) {
        g4kv::launch_publication(descriptors.as<g4kv::Publication>(), unsigned(layout.size()), start, rows, order, nullptr);
        hip_ok(hipDeviceSynchronize(), "publish");
    }
    template <class T> static std::vector<T> read(const Buffer& b, std::size_t n) {
        std::vector<T> x(n);
        if (n) hip_ok(hipMemcpy(x.data(), b.as<void>(), n * sizeof(T), hipMemcpyDeviceToHost), "read");
        return x;
    }
    // Snapshot of every cache byte and tag.
    std::vector<std::vector<unsigned char>> state() const {
        std::vector<std::vector<unsigned char>> out;
        for (std::size_t l = 0; l < layout.size(); ++l) {
            const std::size_t cache = std::size_t(layout[l].slots) * layout[l].width;
            for (const auto* b : {&k[l], &v[l]}) {
                const auto x = read<float>(*b, cache);
                out.emplace_back(reinterpret_cast<const unsigned char*>(x.data()),
                                 reinterpret_cast<const unsigned char*>(x.data()) + cache * 4);
            }
            const auto t = read<std::uint64_t>(tags[l], layout[l].slots);
            out.emplace_back(reinterpret_cast<const unsigned char*>(t.data()),
                             reinterpret_cast<const unsigned char*>(t.data()) + t.size() * 8);
        }
        return out;
    }
};
// The host model of the ordered loop: rows in position order, later rows overwrite earlier ones on the ring.
void expect_host(const Arena& a, const std::vector<std::vector<unsigned char>>& got, std::uint64_t start, unsigned rows) {
    for (std::size_t l = 0; l < a.layout.size(); ++l) {
        const auto& p = a.layout[l];
        auto k = a.k0[l], v = a.v0[l];
        auto t = a.tags0[l];
        if (p.width)
            for (unsigned r = 0; r < rows; ++r) {
                const auto pos = start + r;
                const auto slot = p.global ? pos : pos % p.slots;
                std::copy_n(a.ok0[l].begin() + std::ptrdiff_t(std::size_t(r) * p.width), p.width,
                            k.begin() + std::ptrdiff_t(slot * p.width));
                std::copy_n(a.ov0[l].begin() + std::ptrdiff_t(std::size_t(r) * p.width), p.width,
                            v.begin() + std::ptrdiff_t(slot * p.width));
                t[slot] = pos;
            }
        CHECK(std::memcmp(got[3 * l].data(), k.data(), k.size() * 4) == 0);
        CHECK(std::memcmp(got[3 * l + 1].data(), v.data(), v.size() * 4) == 0);
        CHECK(std::memcmp(got[3 * l + 2].data(), t.data(), t.size() * 8) == 0);
    }
}
void gate_case(std::uint64_t start, unsigned rows, std::uint32_t seed) {
    // Sliding ring of 512 (the E4B window), a small odd ring, global caches, a skipped producer and width 6.
    const unsigned capacity = unsigned(start) + rows;
    Arena a({{512, 512, false}, {1024, capacity, true}, {0, 512, false}, {6, 37, false}, {8, capacity, true}}, rows, seed);
    a.reset();
    a.publish(start, rows, g4kv::PublishOrder::ordered);
    const auto ordered = a.state();
    expect_host(a, ordered, start, rows);
    a.reset();
    a.publish(start, rows, g4kv::PublishOrder::grid);
    const auto grid = a.state();
    CHECK(grid == ordered);
}
double median_ms(Arena& a, std::uint64_t start, unsigned rows, g4kv::PublishOrder order, unsigned reps) {
    hipEvent_t b, e;
    hip_ok(hipEventCreate(&b), "event");
    hip_ok(hipEventCreate(&e), "event");
    std::vector<float> ms;
    for (unsigned i = 0; i < reps + 2; ++i) {
        hip_ok(hipEventRecord(b, nullptr), "record");
        g4kv::launch_publication(a.descriptors.as<g4kv::Publication>(), unsigned(a.layout.size()), start, rows, order, nullptr);
        hip_ok(hipEventRecord(e, nullptr), "record");
        hip_ok(hipEventSynchronize(e), "sync");
        float t = 0;
        hip_ok(hipEventElapsedTime(&t, b, e), "elapsed");
        if (i >= 2) ms.push_back(t); // two warm-up launches
    }
    (void)hipEventDestroy(b);
    (void)hipEventDestroy(e);
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}
void bench() {
    const unsigned rows = 512;
    const std::uint64_t start = 4096; // a mid-prompt chunk: sliding slots fully reused, global slots [4096, 4608)
    std::vector<Producer> e4b;
    for (unsigned l = 0; l < 24; ++l) e4b.push_back(l % 6 == 5 ? Producer{1024, 4608, true} : Producer{512, 512, false});
    Arena a(e4b, rows, 7);
    a.reset();
    double bytes = 0;
    for (const auto& p : e4b) bytes += double(rows) * p.width * 4 * 2 * 2; // K+V, read + write
    for (int round = 0; round < 2; ++round)
        for (const auto order : {g4kv::PublishOrder::ordered, g4kv::PublishOrder::grid}) {
            const double t = median_ms(a, start, rows, order, 15);
            std::cout << "{\"publish_bench\":\"" << (order == g4kv::PublishOrder::grid ? "grid" : "ordered")
                      << "\",\"round\":" << round << ",\"rows\":" << rows << ",\"producers\":24,\"median_ms\":" << t
                      << ",\"GBps\":" << bytes / (t * 1e6) << "}\n";
        }
}
} // namespace
int main(int argc, char** argv) try {
    require_device();
    unsigned cases = 0;
    for (const auto& [start, rows] : std::vector<std::pair<std::uint64_t, unsigned>>{
             {0, 1}, {3, 8}, {100, 64}, {0, 512}, {300, 512}, {508, 8}, {511, 1}, {1000, 600}, {77, 1100}}) {
        gate_case(start, rows, 1000U + cases);
        ++cases;
    }
    std::cout << "publish grid == ordered == host model: " << cases << " cases passed\n";
    if (argc > 1 && std::string_view(argv[1]) == "--bench") bench();
    return 0;
} catch (const std::exception& e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
}
