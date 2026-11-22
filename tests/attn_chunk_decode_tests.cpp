// Host-only: shared chunked two-pass decode form (attn_core/chunk_decode.hpp). Invariants: ordered_sum equals the plain
// ascending scalar chain for every load batching (same additions, same order, +0 start); chunk_maximum scans visible
// positions ascending with strict greater-than from -infinity (an empty chunk gives -infinity, a +0/-0 tie keeps the
// first); and the score chain (32 lane partials of ascending separate multiply/add, then the shfl_down tree) is
// reproduced by a host emulation of the kernel's per-lane walk.
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
    if (failures == 0)
        std::printf("attn_chunk_decode_tests: PASS\n");
    return failures == 0 ? 0 : 1;
}
