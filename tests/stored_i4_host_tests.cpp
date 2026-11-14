#include "core/iu4.hpp"
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>
using namespace strix::core;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "check failed at %d: %s\n", __LINE__, #x); std::abort(); } } while (0)
int main() {
    for (int outputs : {16, 64, 128}) for (int inputs : {64, 128, 256}) {
        std::vector<unsigned> touched(std::size_t(inputs) * outputs, 0);
        for (int k = 0; k < inputs; ++k) for (int n = 0; n < outputs; ++n) {
            const auto at = iu4_weight_byte(k, n, outputs);
            CHECK(at < touched.size() / 2);
            CHECK(++touched[2 * at + (k & 1)] == 1);
        }
        for (auto count : touched) CHECK(count == 1);
    }
    // A wave reads one consecutive 512-byte stage (32 lanes x 16 bytes).
    for (int k = 0; k < 256; k += 16) for (int tile = 0; tile < 8; tile += 4)
        for (int lane = 0; lane < 32; ++lane)
            CHECK(iu4_weight_byte(k, (tile + lane / 8) * 16 + 2 * (lane % 8), 128) ==
                  iu4_weight_byte(k, tile * 16, 128) + std::size_t(lane) * 16);
    for (int low = -8; low <= 7; ++low) for (int high = -8; high <= 7; ++high) {
        std::uint8_t byte = 0;
        iu4_put(byte, 0, low); iu4_put(byte, 1, high);
        CHECK(iu4_unpack(byte, 0) == low && iu4_unpack(byte, 1) == high);
    }
    std::fesetround(FE_DOWNWARD);
    CHECK(iu4_rne(0.5f) == 0 && iu4_rne(1.5f) == 2 && iu4_rne(2.5f) == 2);
    CHECK(iu4_rne(-0.5f) == 0 && iu4_rne(-1.5f) == -2 && iu4_rne(-2.5f) == -2);
    std::fesetround(FE_TONEAREST);
    constexpr int K = 128, N = 32;
    std::vector<float> w(K * N, 0);
    auto zero = quantise_iu4_weights(w.data(), K, N);
    for (auto scale : zero.scales) CHECK(scale == 1);
    for (auto byte : zero.packed) CHECK(byte == 0);
    // Exactly representable scales isolate packing and K64 group indexing from FP16 rounding.
    for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n)
        w[k * N + n] = float(k % 15 - 7) * (k < 64 ? 0.5f : 2.0f);
    auto q = quantise_iu4_weights(w.data(), K, N);
    for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) {
        CHECK(q.scales[n * 2 + k / 64] == (k < 64 ? 0.5f : 2.0f));
        CHECK(iu4_unpack(q.packed[iu4_weight_byte(k, n, N)], k & 1) == k % 15 - 7);
    }
    w[0] = std::numeric_limits<float>::infinity();
    bool rejected = false;
    try { (void)quantise_iu4_weights(w.data(), K, N); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    std::puts("stored-i4 host: PASS (layout bijection, streaming runs, signed nibbles, RNE, K64 packs)");
}
