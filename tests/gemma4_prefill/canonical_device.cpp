#include "gemma4/prefill/device.hpp"
#include "../gemma4_gpu/test.hpp"
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <string_view>
#include <utility>

// Whole-chunk canonical attention versus gpu::attention launched eight rows at
// a time on the same descriptors: output bytes and failure words must match.
using namespace strix::gemma4;
namespace pf = strix::gemma4::prefill;
namespace gpu = strix::gemma4::gpu;
namespace {
struct Random {
    std::uint64_t s;
    std::uint32_t next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return std::uint32_t(s >> 33); }
    // Mixed magnitudes, exact zeros and negative zeros; mild keeps scores O(1).
    float value(bool wild) {
        const auto u = next();
        if (u % 61 == 0) return -0.0F;
        if (u % 67 == 0) return 0.0F;
        const float x = float(int(next() % 2001U) - 1000) / 1000.0F;
        if (!wild) return x * 0.25F;
        static constexpr float scales[] = {0.0009765625F, 0.125F, 1.0F, 3.0F};
        return x * scales[u % 4];
    }
};
struct Case {
    unsigned dim, start, rows;
    bool global, wild;
    int poison = -1; // absolute position whose ring tag is invalidated
    bool bench = false;
};
unsigned run(const Case& t) {
    constexpr unsigned head_count = 8, window_size = 512;
    const unsigned width = head_count / group * t.dim, n = t.rows, start = t.start;
    const unsigned slots = t.global ? std::max(1U, start + n) : window_size;
    Random random{0x9e3779b97f4a7c15ULL ^ (std::uint64_t(t.dim) << 40) ^ (std::uint64_t(start) << 20) ^ n ^ (t.wild ? 7 : 0)};
    std::vector<float> q(std::size_t(n) * head_count * t.dim), k(std::size_t(slots) * width), v(k.size());
    std::vector<float> ok(std::size_t(n) * width), ov(ok.size());
    for (auto* a : {&q, &k, &v, &ok, &ov}) for (auto& x : *a) x = random.value(t.wild);
    // One all-zero key row and stale ring rows whose tags never match.
    if (!ok.empty()) std::fill(ok.begin(), ok.begin() + std::min<std::size_t>(width, ok.size()), 0.0F);
    std::vector<std::uint64_t> tags(slots, UINT64_MAX);
    for (unsigned p = t.global || start < window_size ? 0 : start - window_size; p < start; ++p) tags[t.global ? p : p % slots] = p;
    if (t.poison >= 0) tags[t.global ? unsigned(t.poison) : unsigned(t.poison) % slots] = UINT64_MAX;
    gpu::Buffer dq(q.size() * 4), dk(k.size() * 4), dv(v.size() * 4), dok(std::max<std::size_t>(4, ok.size() * 4)),
        dov(std::max<std::size_t>(4, ov.size() * 4)), dt(tags.size() * 8), dr(n * sizeof(gpu::KvRead));
    dq.upload(q.data(), dq.size()); dk.upload(k.data(), dk.size()); dv.upload(v.data(), dv.size());
    dok.upload(ok.data(), ok.size() * 4); dov.upload(ov.data(), ov.size() * 4); dt.upload(tags.data(), dt.size());
    std::vector<gpu::KvRead> reads(n);
    for (unsigned r = 0; r < n; ++r)
        reads[r] = {dk.as<float>(), dv.as<float>(), dt.as<std::uint64_t>(), dok.as<float>(), dov.as<float>(),
                    start, std::uint64_t(start) + r, width, slots, n, window_size, t.global};
    dr.upload(reads.data(), dr.size());
    const unsigned stride = (start + n + 63U) / 64U * 64U, chunks = stride / 64U;
    gpu::Buffer scores(8ULL * head_count * stride * 4), maxima(8ULL * head_count * (chunks + 1ULL) * 4),
        partials(8ULL * head_count * chunks * (t.dim + 1ULL) * 4), old_out(q.size() * 4), new_out(q.size() * 4), f_old(4), f_new(4);
    const auto span = pf::canonical_span(reads[0], n);
    gpu::Buffer canonical(std::size_t(n) * head_count * span.width * 4), denominators(std::size_t(n) * head_count * 4),
        packed(pf::canonical_packed_floats(n, head_count, t.dim, span.width) * 4), new_out2(q.size() * 4), f_new2(4);
    const pf::CanonicalScratch scratch{canonical.as<float>(), canonical.size() / 4, denominators.as<float>(),
                                       packed.as<float>(), packed.size() / 4};
    f_old.zero(); f_new.zero(); f_new2.zero();
    // Distinct poison in all outputs so unwritten elements cannot compare equal.
    std::vector<float> poison(q.size(), bit_cast<float>(0x7fc0dead)), poison2(q.size(), bit_cast<float>(0x7fc0beef));
    old_out.upload(poison.data(), old_out.size()); new_out.upload(poison2.data(), new_out.size());
    new_out2.upload(poison2.data(), new_out2.size());
    for (unsigned r = 0; r < n; r += 8)
        gpu::attention(dq.as<float>() + std::size_t(r) * head_count * t.dim, dr.as<gpu::KvRead>() + r, std::min(8U, n - r), head_count,
                       t.dim, stride, scores.as<float>(), maxima.as<float>(), partials.as<float>(),
                       old_out.as<float>() + std::size_t(r) * head_count * t.dim, f_old.as<unsigned>());
    CHECK(pf::canonical_attention(dq.as<float>(), reads, head_count, t.dim, new_out.as<float>(), f_new.as<unsigned>(), nullptr,
                                  scratch, 0, nullptr, 0));
    CHECK(pf::canonical_attention(dq.as<float>(), reads, head_count, t.dim, new_out2.as<float>(), f_new2.as<unsigned>(), nullptr,
                                  scratch, 1, nullptr, 1));
    if (!t.bench) {
        // Packed but unfused and plain but fused must agree too (third and fourth variants).
        gpu::Buffer extra(q.size() * 4), f_extra(4);
        for (const auto [pack, fuse] : {std::pair{1, 0}, std::pair{0, 1}}) {
            f_extra.zero(); extra.upload(poison2.data(), extra.size());
            CHECK(pf::canonical_attention(dq.as<float>(), reads, head_count, t.dim, extra.as<float>(), f_extra.as<unsigned>(),
                                          nullptr, scratch, pack, nullptr, fuse));
            gpu::synchronize();
            unsigned w = 0, u = 0; f_extra.download(&w, 4); f_old.download(&u, 4);
            CHECK(w == u);
            if (!u || t.poison >= 0) {
                std::vector<float> e1(q.size()), e2(q.size());
                old_out.download(e1.data(), old_out.size()); extra.download(e2.data(), extra.size());
                bytes(e1, e2);
            }
        }
    }
    gpu::synchronize();
    unsigned a = 0, b = 0;
    f_old.download(&a, 4); f_new.download(&b, 4);
    std::vector<float> x(q.size()), y(q.size());
    old_out.download(x.data(), old_out.size()); new_out.download(y.data(), new_out.size());
    if (a != b) {
        std::fprintf(stderr, "failure words %u vs %u\n", a, b);
        throw std::runtime_error("failure word mismatch");
    }
    if (!a || t.poison >= 0) bytes(x, y);
    unsigned c = 0;
    f_new2.download(&c, 4); new_out2.download(y.data(), new_out2.size());
    if (a != c) {
        std::fprintf(stderr, "packed failure words %u vs %u\n", a, c);
        throw std::runtime_error("packed failure word mismatch");
    }
    if (!a || t.poison >= 0) bytes(x, y);
    if (t.bench) {
        using Clock = std::chrono::steady_clock;
        for (unsigned rep = 0; rep < 3; ++rep) {
            gpu::synchronize();
            auto t0 = Clock::now();
            for (unsigned r = 0; r < n; r += 8)
                gpu::attention(dq.as<float>() + std::size_t(r) * head_count * t.dim, dr.as<gpu::KvRead>() + r, std::min(8U, n - r),
                               head_count, t.dim, stride, scores.as<float>(), maxima.as<float>(), partials.as<float>(),
                               old_out.as<float>() + std::size_t(r) * head_count * t.dim, f_old.as<unsigned>());
            gpu::synchronize();
            const double old_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            float phases[3][4] = {};
            double total[3] = {};
            for (int variant = 0; variant < 3; ++variant) {
                t0 = Clock::now();
                CHECK(pf::canonical_attention(dq.as<float>(), reads, head_count, t.dim, new_out.as<float>(), f_new.as<unsigned>(),
                                              nullptr, scratch, variant > 0, phases[variant], variant > 1));
                gpu::synchronize();
                total[variant] = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            }
            std::printf("{\"bench\":\"%s\",\"dim\":%u,\"start\":%u,\"rows\":%u,\"rep\":%u,\"rows8_ms\":%.3f,"
                        "\"plain_ms\":%.3f,\"plain_phases\":[%.3f,%.3f,%.3f,%.3f],\"packed_ms\":%.3f,\"packed_phases\":[%.3f,%.3f,%.3f,%.3f],\"fused_ms\":%.3f,\"fused_phases\":[%.3f,%.3f,%.3f,%.3f]}\n",
                        t.global ? "global" : "sliding", t.dim, start, n, rep, old_ms, total[0], phases[0][0], phases[0][1],
                        phases[0][2], phases[0][3], total[1], phases[1][0], phases[1][1], phases[1][2], phases[1][3], total[2], phases[2][0],
                        phases[2][1], phases[2][2], phases[2][3]);
        }
    }
    return a;
}
} // namespace
int main(int argc, char** argv) try {
    gpu::require_device();
    if (argc > 1 && std::string_view(argv[1]) == "bench") {
        for (const Case& c : {Case{256, 7680, 512, false, false, -1, true}, Case{512, 7680, 512, true, false, -1, true},
                              Case{512, 1536, 512, true, false, -1, true}, Case{256, 1536, 512, false, false, -1, true}})
            run(c);
        return 0;
    }
    unsigned cases = 0;
    for (const unsigned dim : {256U, 512U}) for (const bool global : {false, true}) {
        const bool model = global == (dim == 512U);
        for (const unsigned start : {0U, 13U, 511U, 512U, 513U, 1536U, 4095U})
            for (const unsigned rows : {1U, 7U, 16U, 17U, 512U})
                for (const bool wild : {false, true}) {
                    if (!model && (wild || rows == 7U || rows == 16U)) continue;
                    try {
                        CHECK(run({dim, start, rows, global, wild}) == 0U);
                    } catch (...) {
                        std::fprintf(stderr, "case dim=%u global=%d start=%u rows=%u wild=%d\n", dim, int(global), start, rows, int(wild));
                        throw;
                    }
                    ++cases;
                }
    }
    // Missing ring entries: visible to every row, or only to the first rows.
    for (const Case& c : {Case{256, 1536, 512, false, false, 1535}, Case{256, 1536, 512, false, true, 1536 - 509},
                          Case{512, 1536, 512, true, false, 7}, Case{256, 513, 17, false, false, 2}}) {
        try {
            CHECK((run(c) & 2U) != 0U);
        } catch (...) {
            std::fprintf(stderr, "poison case dim=%u start=%u rows=%u position=%d\n", c.dim, c.start, c.rows, c.poison);
            throw;
        }
        ++cases;
    }
    std::printf("canonical whole-chunk attention == gpu::attention bytes and failure words: %u cases PASS\n", cases);
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
