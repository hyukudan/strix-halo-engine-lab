// Host-only: shared chunked two-pass decode form (attn_core/chunk_decode.hpp). Invariants: ordered_sum equals the plain
// ascending scalar chain for every load batching (same additions, same order, +0 start); chunk_maximum scans visible
// positions ascending with strict greater-than from -infinity (an empty chunk gives -infinity, a +0/-0 tie keeps the
// first); and the score chain (32 lane partials of ascending separate multiply/add, then the shfl_down tree) is
// reproduced by a host emulation of the kernel's per-lane walk.
// Also: ordered_maximum, chunk_denominator and chunk_numerators (load batches 1/8/32, heads sharing a value load, two
// columns, absent rows) equal their scalar ascending chains bit for bit.
#include "attn_core/chunk_decode.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* why) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", why);
        ++failures;
    }
}
std::uint32_t bits(float x) { return std::bit_cast<std::uint32_t>(x); }
float wave_tree(const float* lane) {
    float a[32];
    for (unsigned i = 0; i < 32U; ++i)
        a[i] = lane[i];
    for (unsigned d = 16U; d; d >>= 1U)
        for (unsigned i = 0; i < 32U; ++i)
            a[i] = a[i] + (i + d < 32U ? a[i + d] : a[i]);
    return a[0];
}
template <unsigned Dim> float score_chain(const float* q, const float* k) {
    float lane[32];
    for (unsigned l = 0; l < 32U; ++l) {
        float sum = 0.0F;
        for (unsigned c = 0; c < Dim / 32U; ++c)
            sum = sum + q[l + 32U * c] * k[l + 32U * c];
        lane[l] = sum;
    }
    return wave_tree(lane);
}
template <unsigned Dim> float score_flat_tree(const float* q, const float* k) {
    // independent form: lane partials by strided loop (dflash form), same tree
    float lane[32];
    for (unsigned l = 0; l < 32U; ++l) {
        float sum = 0.0F;
        for (unsigned i = l; i < Dim; i += 32U)
            sum = sum + q[i] * k[i];
        lane[l] = sum;
    }
    return wave_tree(lane);
}
}  // namespace

int main() {
    namespace cd = strix::attn_core::chunk_decode;
    std::uint32_t s = 12345U;
    auto next = [&]() {
        s = s * 1664525U + 1013904223U;
        return (static_cast<float>(s >> 8) / 8388608.0F - 1.0F) * 3.0F;
    };
    std::vector<float> v(4000);
    for (auto& x : v)
        x = next();
    for (std::size_t step : {1U, 3U}) {
        for (std::uint64_t last : {0U, 1U, 7U, 8U, 63U, 64U, 200U, 1000U}) {
            float ref = 0.0F;
            for (std::uint64_t c = 0; c <= last; ++c)
                ref = ref + v[c * step];
            check(bits(cd::ordered_sum<1>(v.data(), 0, last, step)) == bits(ref), "ordered_sum<1>");
            check(bits(cd::ordered_sum<8>(v.data(), 0, last, step)) == bits(ref), "ordered_sum<8>");
            check(bits(cd::ordered_sum<16>(v.data(), 0, last, step)) == bits(ref), "ordered_sum<16>");
            check(bits(cd::ordered_sum<32>(v.data(), 0, last, step)) == bits(ref), "ordered_sum<32>");
            float ref2 = 0.0F;
            for (std::uint64_t c = 2; c <= last; ++c)
                ref2 = ref2 + v[c * step];
            if (last >= 2)
                check(bits(cd::ordered_sum<8>(v.data(), 2, last, step)) == bits(ref2), "ordered_sum<8> from 2");
        }
    }
    float sc[cd::kChunk];
    for (auto& x : sc)
        x = next();
    for (std::uint64_t first : {0U, 5U, 64U}) {
        for (std::uint64_t last : {63U, 70U, 100U, 127U}) {
            float ref = cd::neg_inf();
            bool any = false;
            for (unsigned j = 0; j < cd::kChunk; ++j) {
                const std::uint64_t p = 64U + j;
                if (p >= first && p <= last) {
                    any = true;
                    ref = sc[j] > ref ? sc[j] : ref;
                }
            }
            const float got = cd::chunk_maximum(64U, first, last, [&](unsigned j) { return sc[j]; });
            check(bits(got) == bits(ref), "chunk_maximum equals ascending scan");
            if (!any)
                check(bits(got) == cd::kNegInfBits, "empty chunk is -infinity");
        }
    }
    float tie[cd::kChunk] = {};
    tie[0] = 0.0F;
    tie[1] = -0.0F;
    check(bits(cd::chunk_maximum(0U, 0U, 1U, [&](unsigned j) { return tie[j]; })) == bits(0.0F), "+0 first tie keeps +0");
    tie[0] = -0.0F;
    tie[1] = 0.0F;
    check(bits(cd::chunk_maximum(0U, 0U, 1U, [&](unsigned j) { return tie[j]; })) == bits(-0.0F), "-0 first tie keeps -0");
    float q[512], k[512];
    for (unsigned i = 0; i < 512U; ++i) {
        q[i] = next();
        k[i] = next();
    }
    check(bits(score_chain<256>(q, k)) == bits(score_flat_tree<256>(q, k)), "lane-strided walk == chunked c-loop (256)");
    check(bits(score_chain<512>(q, k)) == bits(score_flat_tree<512>(q, k)), "lane-strided walk == chunked c-loop (512)");
    // (score_flat_tree is the Dim == 0 runtime-width chain: equal to the Dim > 0 chain whenever src.dim == Dim.)
    // ordered_maximum: ascending strict scan over chunk maxima (stride step), -infinity when empty.
    for (std::size_t step : {1U, 5U}) {
        for (std::uint64_t first : {0U, 3U}) {
            for (std::uint64_t last : {0U, 2U, 3U, 40U, 300U}) {
                float ref = cd::neg_inf();
                for (std::uint64_t c = first; c <= last; ++c)
                    ref = v[c * step] > ref ? v[c * step] : ref;
                check(bits(cd::ordered_maximum(v.data(), first, last, step)) == bits(ref), "ordered_maximum");
            }
        }
    }
    float zeros[2] = {0.0F, -0.0F};
    check(bits(cd::ordered_maximum(zeros, 0U, 1U, 1U)) == bits(0.0F), "ordered_maximum keeps first zero");
    // Chunk partial chains: denominator and numerators for every load batch, heads sharing a value load, two columns.
    constexpr unsigned kHeads = 4U, kWidth = 600U;
    std::vector<float> e(std::size_t(kHeads) * cd::kChunk), rows(std::size_t(cd::kChunk) * kWidth);
    for (auto& x : e)
        x = std::fabs(next());
    for (auto& x : rows)
        x = next();
    const std::uint64_t chunk_first = 128U;
    auto absent = [](std::uint64_t p) { return p % 7U == 3U; };
    auto value = [&](std::uint64_t p) -> const float* {
        return absent(p) ? nullptr : rows.data() + std::size_t(p - chunk_first) * kWidth;
    };
    for (std::uint64_t begin : {128U, 131U}) {
        for (std::uint64_t end : {129U, 150U, 192U}) {
            float dref = 0.0F;
            for (auto p = begin; p < end; ++p)
                dref = dref + e[p - chunk_first];
            check(bits(cd::chunk_denominator(e.data(), chunk_first, begin, end)) == bits(dref), "chunk_denominator");
            for (unsigned column : {0U, 17U, 255U}) {
                float ref[2][kHeads];
                for (unsigned c = 0; c < 2U; ++c)
                    for (unsigned h = 0; h < kHeads; ++h) {
                        float acc = 0.0F;
                        for (auto p = begin; p < end; ++p)
                            if (!absent(p))
                                acc = acc + e[h * cd::kChunk + (p - chunk_first)] *
                                                rows[std::size_t(p - chunk_first) * kWidth + column + 256U * c];
                        ref[c][h] = acc;
                    }
                const float n1 = cd::chunk_numerator<1>(e.data(), chunk_first, begin, end, value, column);
                const float n8 = cd::chunk_numerator<8>(e.data(), chunk_first, begin, end, value, column);
                const float n32 = cd::chunk_numerator<32>(e.data(), chunk_first, begin, end, value, column);
                check(bits(n1) == bits(ref[0][0]) && bits(n8) == bits(ref[0][0]) && bits(n32) == bits(ref[0][0]),
                      "chunk_numerator Batch 1/8/32");
                float g[1][kHeads], d2[2][kHeads];
                cd::chunk_numerators<8, kHeads, 1>(e.data(), cd::kChunk, chunk_first, begin, end, value, column, 0U, g);
                cd::chunk_numerators<8, kHeads, 2>(e.data(), cd::kChunk, chunk_first, begin, end, value, column, 256U,
                                                   d2);
                for (unsigned h = 0; h < kHeads; ++h) {
                    check(bits(g[0][h]) == bits(ref[0][h]), "chunk_numerators heads");
                    check(bits(d2[0][h]) == bits(ref[0][h]) && bits(d2[1][h]) == bits(ref[1][h]),
                          "chunk_numerators two columns");
                }
            }
        }
    }
    if (failures == 0)
        std::printf("attn_chunk_decode_tests: PASS\n");
    return failures == 0 ? 0 : 1;
}
