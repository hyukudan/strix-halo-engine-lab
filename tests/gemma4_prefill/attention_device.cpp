#include "gemma4/prefill/device.hpp"
#include "gemma4/prefill/attention.hpp"
#include "../gemma4_gpu/test.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string_view>
#include <cstring>
#include <numeric>

using namespace strix::gemma4;
namespace pf = strix::gemma4::prefill;
namespace gpu = strix::gemma4::gpu;
namespace {
// Diagnostic timing (not a gate): the last 512-row chunk of an 8K prompt.
int bench() {
    struct Shape { unsigned dim, context; bool global; };
    for (const auto c : {Shape{512, 7680, true}, Shape{256, 7680, false}, Shape{512, 1536, true}}) {
        const unsigned rows = 512, head_count = 8, width = 2 * c.dim, slots = c.global ? c.context : 512U;
        std::vector<float> q(std::size_t(rows) * head_count * c.dim), k(std::size_t(slots) * width), ok(std::size_t(rows) * width);
        for (std::size_t i = 0; i < q.size(); ++i) q[i] = float(int(i % 37) - 18) * 0.015625F;
        for (std::size_t i = 0; i < k.size(); ++i) k[i] = float(int(i % 47) - 23) * 0.0625F;
        for (std::size_t i = 0; i < ok.size(); ++i) ok[i] = float(int(i % 31) - 15) * 0.03125F;
        std::vector<std::uint64_t> tags(slots, UINT64_MAX);
        for (unsigned p = c.context > slots ? c.context - slots : 0; p < c.context; ++p) tags[c.global ? p : p % slots] = p;
        gpu::Buffer dq(q.size() * 4), dk(k.size() * 4), dok(ok.size() * 4), dt(tags.size() * 8), dr(rows * sizeof(gpu::KvRead));
        gpu::Buffer out(dq.size()), failure(4);
        dq.upload(q.data(), dq.size()); dk.upload(k.data(), dk.size()); dok.upload(ok.data(), dok.size()); dt.upload(tags.data(), dt.size()); failure.zero();
        std::vector<gpu::KvRead> reads(rows);
        for (unsigned r = 0; r < rows; ++r)
            reads[r] = {dk.as<float>(), dk.as<float>(), dt.as<std::uint64_t>(), dok.as<float>(), dok.as<float>(),
                        c.context, c.context + r, width, slots, rows, c.global ? 0U : 512U, c.global};
        dr.upload(reads.data(), dr.size());
        const unsigned stride = (c.context + rows + 63U) / 64U * 64U, capacity = c.global ? stride : std::min(stride, 512U + rows - 1);
        gpu::Buffer packed(pf::attention_packed_bytes(stride, width));
        auto run = [&](bool repack) {
            pf::attention(dq.as<float>(), dr.as<gpu::KvRead>(), rows, head_count, c.dim, out.as<float>(), failure.as<unsigned>(),
                          nullptr, packed.data(), stride, capacity, repack);
        };
        for (unsigned i = 0; i < 3; ++i) run(true);
        gpu::synchronize();
        for (const bool repack : {true, false}) {
            constexpr unsigned repetitions = 10;
            const auto begin = std::chrono::steady_clock::now();
            for (unsigned i = 0; i < repetitions; ++i) run(repack);
            gpu::synchronize();
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count() / repetitions;
            std::printf("{\"attention_bench\":true,\"dim\":%u,\"context\":%u,\"global\":%d,\"repack\":%d,\"split\":%d,\"ms\":%.4f}\n",
                        c.dim, c.context, int(c.global), int(repack), int(pf::attention_split_enabled()), ms);
        }
        unsigned flag = 0; failure.download(&flag, 4); CHECK(flag == 0);
    }
    return 0;
}
} // namespace
int main(int argc, char** argv) try {
    gpu::require_device();
    if (argc == 2 && std::string_view(argv[1]) == "--bench") return bench();
    for (unsigned dim : {8U, 32U, 256U, 512U}) for (bool global : {false, true}) {
        const unsigned rows = dim == 32 ? 17U : 3U, head_count = 8, width = 2 * dim;
        const unsigned context = global ? 19U : dim == 32 ? 65519U : 513U, slots = global ? context : 512U;
        const unsigned window_size = global ? 0U : 512U;
        std::vector<float> q(std::size_t(rows) * head_count * dim), k(std::size_t(slots) * width), v(k.size());
        std::vector<float> ok(std::size_t(rows) * width), ov(ok.size());
        std::vector<std::uint64_t> tags(slots, UINT64_MAX);
        for (unsigned i = 0; i < q.size(); ++i) q[i] = float(int(i % 37) - 18) * 0.015625F;
        for (unsigned i = 0; i < k.size(); ++i) { k[i] = float(int(i % 47) - 23) * 0.0625F; v[i] = float(int(i % 53) - 26) * 0.03125F; }
        for (unsigned i = 0; i < ok.size(); ++i) { ok[i] = float(int(i % 31) - 15) * 0.03125F; ov[i] = float(int(i % 43) - 21) * 0.0625F; }
        for (unsigned p = context > slots ? context - slots : 0; p < context; ++p) tags[global ? p : p % slots] = p;
        gpu::Buffer dq(q.size() * 4), dk(k.size() * 4), dv(v.size() * 4), dok(ok.size() * 4), dov(ov.size() * 4);
        gpu::Buffer dt(tags.size() * 8), dr(rows * sizeof(gpu::KvRead)), output(dq.size()), solo(dq.size()), failure(4);
        dq.upload(q.data(), dq.size()); dk.upload(k.data(), dk.size()); dv.upload(v.data(), dv.size());
        dok.upload(ok.data(), dok.size()); dov.upload(ov.data(), dov.size()); dt.upload(tags.data(), dt.size()); failure.zero();
        std::vector<gpu::KvRead> reads(rows);
        for (unsigned r = 0; r < rows; ++r)
            reads[r] = {dk.as<float>(), dv.as<float>(), dt.as<std::uint64_t>(), dok.as<float>(), dov.as<float>(),
                        context, context + r, width, slots, rows, window_size, global};
        dr.upload(reads.data(), dr.size());
        pf::attention(dq.as<float>(), dr.as<gpu::KvRead>(), rows, head_count, dim, output.as<float>(), failure.as<unsigned>());
        for (unsigned r = 0; r < rows; ++r)
            pf::attention(dq.as<float>() + std::size_t(r) * head_count * dim, dr.as<gpu::KvRead>() + r, 1, head_count, dim,
                          solo.as<float>() + std::size_t(r) * head_count * dim, failure.as<unsigned>());
        gpu::synchronize();
        unsigned flag = 0; failure.download(&flag, 4); CHECK(flag == 0);
        std::vector<float> actual(q.size()), expected(q.size()), serial(q.size());
        output.download(actual.data(), output.size()); solo.download(serial.data(), solo.size()); bytes(actual, serial);
        for (unsigned r = 0; r < rows; ++r) for (unsigned h = 0; h < head_count; ++h) {
            const auto offset = (std::size_t(r) * head_count + h) * dim;
            const auto oracle = pf::emulate_attention(std::span(q).subspan(offset, dim), context + r, window_size,
                [&](std::uint64_t p, bool value) {
                    if (p >= context) return (value ? ov : ok).data() + (p - context) * width + (h / group) * dim;
                    return (value ? v : k).data() + (global ? p : p % slots) * width + (h / group) * dim;
                });
            std::copy(oracle.begin(), oracle.end(), expected.begin() + std::ptrdiff_t(offset));
        }
        bytes(actual, expected);
        gpu::Buffer packed(pf::attention_packed_bytes(context + rows, width));
        pf::attention(dq.as<float>(), dr.as<gpu::KvRead>(), rows, head_count, dim, output.as<float>(), failure.as<unsigned>(),
                      nullptr, packed.data(), context + rows, global ? context + rows : window_size + rows - 1);
        gpu::synchronize(); failure.download(&flag, 4); CHECK(flag == 0);
        output.download(serial.data(), output.size()); bytes(actual, serial);
        // Poison a required tag: fail rather than read an aliased ring entry.
        tags[(context - 1) % slots] = UINT64_MAX; dt.upload(tags.data(), dt.size()); failure.zero();
        pf::attention(dq.as<float>(), dr.as<gpu::KvRead>(), rows, head_count, dim, output.as<float>(), failure.as<unsigned>());
        gpu::synchronize(); failure.download(&flag, 4); CHECK((flag & 2U) != 0);
        failure.zero();
        pf::attention(dq.as<float>(), dr.as<gpu::KvRead>(), rows, head_count, dim, output.as<float>(), failure.as<unsigned>(),
                      nullptr, packed.data(), context + rows, global ? context + rows : window_size + rows - 1);
        gpu::synchronize(); failure.download(&flag, 4); CHECK((flag & 2U) != 0);
    }
    // Full prompt chunks: the packed (split by default) route against the
    // unpacked original kernel, byte for byte, plus emulator rows.
    struct Large { unsigned dim, context, rows; bool global; };
    for (const auto c : {Large{256, 0, 512, false}, Large{256, 1531, 512, false}, Large{256, 509, 77, false},
                         Large{512, 1029, 512, true}, Large{512, 0, 512, true}, Large{512, 2048, 33, true}, Large{64, 600, 300, false}}) {
        const unsigned head_count = 8, width = 2 * c.dim, slots = c.global ? std::max(c.context, 1U) : 512U;
        std::uint64_t seed = 0x9e3779b97f4a7c15ULL ^ (c.dim * 131U + c.context);
        const auto next = [&] {
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            const auto u = unsigned(seed >> 40) & 0xffffffU;
            float x = (float(u) / 16777216.0F - 0.5F) * 4.0F;
            if ((u & 255U) == 7U) x = -0.0F;
            if ((u & 1023U) == 9U) x *= 24.0F;
            return x;
        };
        std::vector<float> q(std::size_t(c.rows) * head_count * c.dim), k(std::size_t(slots) * width), v(k.size());
        std::vector<float> ok(std::size_t(c.rows) * width), ov(ok.size());
        for (auto& x : q) x = next() * 0.25F;
        for (auto& x : k) x = next();
        for (auto& x : v) x = next();
        for (auto& x : ok) x = next();
        for (auto& x : ov) x = next();
        std::vector<std::uint64_t> tags(slots, UINT64_MAX);
        for (unsigned p = c.context > slots ? c.context - slots : 0; p < c.context; ++p) tags[c.global ? p : p % slots] = p;
        gpu::Buffer dq(q.size() * 4), dk(k.size() * 4), dv(v.size() * 4), dok(ok.size() * 4), dov(ov.size() * 4);
        gpu::Buffer dt(tags.size() * 8), dr(c.rows * sizeof(gpu::KvRead)), reference(dq.size()), output(dq.size()), failure(4);
        dq.upload(q.data(), dq.size()); dk.upload(k.data(), dk.size()); dv.upload(v.data(), dv.size());
        dok.upload(ok.data(), dok.size()); dov.upload(ov.data(), dov.size()); dt.upload(tags.data(), dt.size()); failure.zero();
        std::vector<gpu::KvRead> reads(c.rows);
        for (unsigned r = 0; r < c.rows; ++r)
            reads[r] = {dk.as<float>(), dv.as<float>(), dt.as<std::uint64_t>(), dok.as<float>(), dov.as<float>(),
                        c.context, c.context + r, width, slots, c.rows, c.global ? 0U : 512U, c.global};
        dr.upload(reads.data(), dr.size());
        const unsigned stride = (c.context + c.rows + 63U) / 64U * 64U;
        gpu::Buffer packed(pf::attention_packed_bytes(stride, width));
        pf::attention(dq.as<float>(), dr.as<gpu::KvRead>(), c.rows, head_count, c.dim, reference.as<float>(), failure.as<unsigned>());
        pf::attention(dq.as<float>(), dr.as<gpu::KvRead>(), c.rows, head_count, c.dim, output.as<float>(), failure.as<unsigned>(),
                      nullptr, packed.data(), stride, c.global ? stride : std::min(stride, 512U + c.rows - 1));
        gpu::synchronize();
        unsigned flag = 0; failure.download(&flag, 4); CHECK(flag == 0);
        std::vector<float> expected(q.size()), actual(q.size());
        reference.download(expected.data(), reference.size()); output.download(actual.data(), output.size());
        bytes(actual, expected);
        for (unsigned r : {0U, c.rows / 2, c.rows - 1}) for (unsigned h : {0U, 5U}) {
            const auto offset = (std::size_t(r) * head_count + h) * c.dim;
            const auto oracle = pf::emulate_attention(std::span(q).subspan(offset, c.dim), c.context + r, c.global ? 0U : 512U,
                [&](std::uint64_t p, bool value) {
                    if (p >= c.context) return (value ? ov : ok).data() + (p - c.context) * width + (h / group) * c.dim;
                    return (value ? v : k).data() + (c.global ? p : p % slots) * width + (h / group) * c.dim;
                });
            CHECK(std::memcmp(oracle.data(), actual.data() + offset, c.dim * sizeof(float)) == 0);
        }
        std::printf("large dim %u context %u rows %u %s: split=%d bytes PASS\n", c.dim, c.context, c.rows,
                    c.global ? "global" : "sliding", int(pf::attention_split_enabled()));
    }
    std::puts("WMMA attention device/emulator bytes, query splits, GQA, ring and masks PASS");
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
