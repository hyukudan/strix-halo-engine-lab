#include "attn_core/dense_flash_prefill.hpp"
#include "attn_core/relative_bias.hpp"
#include "strix/ds41_scalar_codecs.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>

namespace {
namespace dense = strix::attn_core::dense;
constexpr unsigned dim = 128, heads = 3, rows = 2, extent = 2;
constexpr unsigned pos0 = 17, window = 3, capacity = 32;
constexpr float ln2 = 0.6931471805599453F;

struct Fixture {
    alignas(16) float query[rows][heads][dim]{};
    alignas(16) float out[rows][heads][dim]{};
    alignas(16) std::uint16_t key[capacity][dim]{};
    alignas(16) std::uint16_t value[capacity][dim]{};
    float bias[rows * heads * extent]{};
    float sink[rows * heads]{};
    alignas(16) float workspace[dense::workspace_floats(rows, heads, dim, 2U)]{};
    unsigned failure = 0;
};

struct NoBiasPolicy {
    using Args = Fixture*;
    using Shape = dense::Shape<dim, heads, 1>;
    static constexpr auto kOp = strix::attn_core::tile::Operand::bf16;
    static constexpr bool kValueRows = true;
    static constexpr float kScaleLog2 = 1.0F;
    __device__ static bool valid(Args a) { return a != nullptr; }
    __host__ __device__ static unsigned rows(Args) { return ::rows; }
    __device__ static unsigned position(Args) { return pos0; }
    __device__ static unsigned window(Args) { return ::window; }
    __device__ static const float* query(Args a, unsigned row, unsigned head) { return a->query[row][head]; }
    __device__ static float* out(Args a, unsigned row, unsigned head) { return a->out[row][head]; }
    __device__ static const std::uint16_t* key(Args a, unsigned p, unsigned) { return a->key[p]; }
    __device__ static const std::uint16_t* value(Args a, unsigned p, unsigned) { return a->value[p]; }
    __device__ static unsigned* failure(Args a) { return &a->failure; }
    __device__ static float sink(Args a, unsigned row, unsigned head) {
        return strix::attn_core::AttentionSinkRows{a->sink, heads}.sink(row, head);
    }
};

struct BiasPolicy : NoBiasPolicy {
    __device__ static float score_bias(Args a, unsigned row, unsigned head, unsigned p, unsigned k) {
        if (row >= ::rows || head >= heads || k > p || p - k >= ::window) {
            atomicOr(&a->failure, 8U);
            return 0.0F;
        }
        return strix::attn_core::RelativeBiasRows{a->bias, heads, extent}.score_bias(row, head, p, k);
    }
};

static_assert(dense::DenseScoreBias<BiasPolicy>);
static_assert(!dense::DenseScoreBias<NoBiasPolicy>);

int key_dot(unsigned p) { return static_cast<int>(p % 3U) - 1; }
int distance_bias(unsigned row, unsigned head, unsigned lag) {
    return lag < extent ? static_cast<int>(row + head + lag) - 1 : 0;
}
int sink_log2(unsigned row, unsigned head) { return static_cast<int>(row + head) - 1; }

std::uint16_t bf16(float x) {
    return static_cast<std::uint16_t>(strix::ds41::codec::f2u(x) >> 16U);
}

void fill(Fixture& fixture) {
    for (unsigned p = 0; p < capacity; ++p) {
        fixture.key[p][0] = bf16(static_cast<float>(key_dot(p)));
        for (auto& v : fixture.value[p]) v = bf16(static_cast<float>(p + 1U));
    }
    for (unsigned row = 0; row < rows; ++row) {
        for (unsigned head = 0; head < heads; ++head) {
            fixture.query[row][head][0] = 1.0F;
            fixture.sink[row * heads + head] = ln2 * static_cast<float>(sink_log2(row, head));
            for (unsigned lag = 0; lag < extent; ++lag) {
                fixture.bias[(row * heads + head) * extent + lag] =
                    ln2 * static_cast<float>(distance_bias(row, head, lag));
            }
        }
    }
}

void require_hip(hipError_t result) {
    if (result == hipSuccess) return;
    std::cerr << hipGetErrorString(result) << '\n';
    std::exit(1);
}

template<class Policy>
bool run(Fixture* fixture, unsigned splits) {
    for (auto& row : fixture->out)
        for (auto& head : row)
            for (float& value : head) value = 0.0F;
    dense::launch<Policy>(fixture, heads, nullptr, fixture->workspace, splits);
    require_hip(hipGetLastError());
    require_hip(hipDeviceSynchronize());
    if (fixture->failure != 0) return false;

    for (unsigned row = 0; row < rows; ++row) {
        const unsigned p = pos0 + row;
        for (unsigned head = 0; head < heads; ++head) {
            float numerator = 0.0F;
            float denominator = std::ldexp(1.0F, sink_log2(row, head));
            for (unsigned lag = 0; lag < window; ++lag) {
                const unsigned key = p - lag;
                int logit = key_dot(key);
                if constexpr (dense::DenseScoreBias<Policy>) logit += distance_bias(row, head, lag);
                const float mass = std::ldexp(1.0F, logit);
                denominator += mass;
                numerator += mass * static_cast<float>(key + 1U);
            }
            const float expected = numerator / denominator;
            for (float actual : fixture->out[row][head]) {
                if (!(std::abs(actual - expected) <= 2e-4F)) return false;
            }
        }
    }
    return true;
}
} // namespace

int main() {
    void* storage = nullptr;
    require_hip(hipMallocManaged(&storage, sizeof(Fixture)));
    auto* fixture = static_cast<Fixture*>(storage);
    std::construct_at(fixture);
    fill(*fixture);
    const bool ok = run<NoBiasPolicy>(fixture, 1U) && run<BiasPolicy>(fixture, 1U) && run<BiasPolicy>(fixture, 2U);
    std::destroy_at(fixture);
    require_hip(hipFree(fixture));
    if (!ok) std::cerr << "dense flash relative bias mismatch\n";
    return ok ? 0 : 1;
}
