// Host check of the gemv_rows lookahead schedules (src/core/gemv_rows_lookahead.inc): the textual body is compiled
// on the host with plain float rows, one call per wave lane, and every (stream, row, lane) partial must equal, bit for
// bit, a scalar reference of the family contract (+0 start, chunks lane, lane + 32, ... ascending, 16 elements
// ascending, acc = acc + w * x; on a partial last chunk an explicit +0 for every padding element). The three
// schedules must also agree with each other whenever inputs % 16 == 0.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct Chunk {
    float v[16] = {};
    float at(unsigned e) const { return v[e]; }
};
Chunk load_chunk(const float* row, unsigned start) {
    Chunk c;
    for (unsigned e = 0; e < 16; ++e)
        c.v[e] = row[start + e];
    return c;
}

template <unsigned Rows>
struct Partials {
    float a[Rows] = {}, b[Rows] = {};
    bool stored = false;
};

template <unsigned Rows, unsigned Streams, unsigned Lookahead>
Partials<Rows> sweep(const float* weights, const float* up, const float* x, unsigned inputs, unsigned outputs,
                     unsigned first, unsigned lane) {
    Partials<Rows> result;
#define GEMV_ROWS_WEIGHT_ROW(stream, o) ((stream) ? up + std::size_t(o) * inputs : weights + std::size_t(o) * inputs)
#define GEMV_ROWS_CHUNK Chunk
#define GEMV_ROWS_LOAD_CHUNK(row, start) load_chunk(row, start)
#define GEMV_ROWS_LANE_SUM(partial) (partial)
#define GEMV_ROWS_TAIL_WEIGHT(stream, o, i) ((stream) ? up[std::size_t(o) * inputs + (i)] : weights[std::size_t(o) * inputs + (i)])
#define GEMV_ROWS_STORE(row, o, sum0, sum1)                                                                            \
    do {                                                                                                               \
        result.a[row] = (sum0);                                                                                        \
        result.b[row] = (sum1);                                                                                        \
        result.stored = (o) < outputs;                                                                                 \
    } while (false)
#include "core/gemv_rows_lookahead.inc"
#undef GEMV_ROWS_WEIGHT_ROW
#undef GEMV_ROWS_CHUNK
#undef GEMV_ROWS_LOAD_CHUNK
#undef GEMV_ROWS_LANE_SUM
#undef GEMV_ROWS_TAIL_WEIGHT
#undef GEMV_ROWS_STORE
    return result;
}

// The scalar contract, written independently of the schedules.
template <unsigned Rows>
Partials<Rows> reference(const float* weights, const float* up, bool gate, const float* x, unsigned inputs,
                         unsigned output, unsigned lane) {
    Partials<Rows> result;
    result.stored = true;
    const float* streams[2] = {weights + std::size_t(output) * inputs, up + std::size_t(output) * inputs};
    for (unsigned s = 0; s < (gate ? 2U : 1U); ++s) {
        float* acc = s == 0 ? result.a : result.b;
        for (unsigned r = 0; r < Rows; ++r) {
            float sum = 0.0F;
            for (unsigned c = lane; c * 16U < inputs; c += 32U) {
                const bool whole = c * 16U + 16U <= inputs;
                for (unsigned e = 0; e < 16; ++e) {
                    const unsigned i = c * 16U + e;
                    if (whole)
                        sum = sum + streams[s][i] * x[std::size_t(r) * inputs + i];
                    else
                        sum = sum + (i < inputs ? streams[s][i] * x[std::size_t(r) * inputs + i] : 0.0F);
                }
            }
            acc[r] = sum;
        }
    }
    return result;
}

int failures = 0;

template <unsigned Rows>
bool same(const Partials<Rows>& p, const Partials<Rows>& q) {
    return p.stored == q.stored && std::memcmp(p.a, q.a, sizeof p.a) == 0 && std::memcmp(p.b, q.b, sizeof p.b) == 0;
}

template <unsigned Rows, unsigned Streams>
void check(unsigned inputs, std::mt19937& rng) {
    constexpr unsigned outputs = 3U;
    std::uniform_real_distribution<float> mantissa(-1.0F, 1.0F);
    std::uniform_int_distribution<int> exponent(-12, 12);
    const auto draw = [&] { return std::ldexp(mantissa(rng), exponent(rng)); };
    std::vector<float> weights(std::size_t(outputs) * inputs), up(weights.size()), x(std::size_t(Rows) * inputs);
    for (auto& v : weights) v = draw();
    for (auto& v : up) v = draw();
    for (auto& v : x) v = draw();
    // Exercise signed zeros: a lane whose products are all -0 keeps +0 only through the explicit padding adds.
    for (unsigned i = 0; i < 16U && i < inputs; ++i) x[i] = -0.0F;
    for (unsigned first = 0; first <= outputs; ++first)
        for (unsigned lane = 0; lane < 32U; ++lane) {
            const auto zero = sweep<Rows, Streams, 0>(weights.data(), up.data(), x.data(), inputs, outputs, first, lane);
            if (first == outputs) {
                if (zero.stored) {
                    std::printf("FAIL store past outputs rows=%u streams=%u inputs=%u\n", Rows, Streams, inputs);
                    ++failures;
                }
                continue;
            }
            const auto ref = reference<Rows>(weights.data(), up.data(), Streams == 2U, x.data(), inputs, first, lane);
            if (!same(zero, ref)) {
                std::printf("FAIL lookahead 0 rows=%u streams=%u inputs=%u first=%u lane=%u\n", Rows, Streams, inputs, first, lane);
                ++failures;
            }
            if (inputs % 16U != 0U)
                continue;
            const auto rolling = sweep<Rows, Streams, 1>(weights.data(), up.data(), x.data(), inputs, outputs, first, lane);
            const auto paired = sweep<Rows, Streams, 2>(weights.data(), up.data(), x.data(), inputs, outputs, first, lane);
            if (!same(rolling, ref) || !same(paired, ref)) {
                std::printf("FAIL lookahead 1/2 rows=%u streams=%u inputs=%u first=%u lane=%u\n", Rows, Streams, inputs, first, lane);
                ++failures;
            }
        }
}

template <unsigned Rows>
void check_rows(std::mt19937& rng) {
    for (const unsigned inputs : {16U, 48U, 512U, 1040U, 2560U, 37U, 1000U, 1553U}) {
        check<Rows, 1>(inputs, rng);
        check<Rows, 2>(inputs, rng);
    }
}

}  // namespace

int main() {
    std::mt19937 rng(20261122U);
    check_rows<1>(rng);
    check_rows<2>(rng);
    check_rows<3>(rng);
    check_rows<8>(rng);
    if (failures != 0) {
        std::printf("gemv_rows_lookahead_tests: %d failures\n", failures);
        return 1;
    }
    std::printf("gemv_rows_lookahead_tests: PASS\n");
    return 0;
}
