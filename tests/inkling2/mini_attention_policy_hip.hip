#include "attn_core/prepared_bf16_attention.hpp"
#include "strix/ds41_scalar_codecs.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>

namespace {
namespace dense = strix::attn_core::dense;
constexpr unsigned dim = 128, heads = 3, rows = 2, extent = 2;
constexpr unsigned pos0 = 17, window = 3, capacity = pos0 + rows;
constexpr unsigned ring_slots = window + rows - 1U;
constexpr float ln2 = 0.6931471805599453F;

struct Fixture {
    alignas(16) float query[rows * heads * dim]{};
    alignas(16) float out[rows * heads * dim]{};
    alignas(16) std::uint16_t key[capacity * dim]{};
    alignas(16) std::uint16_t value[capacity * dim]{};
    alignas(16) std::uint16_t zeros[dim]{};
    float bias[rows * heads * extent]{};
    float sink[rows * heads]{};
    alignas(16) float workspace[dense::workspace_floats(rows, heads, dim, 2U)]{};
    unsigned failure = 0;
};

struct NoBiasPolicy : dense::PreparedBf16Policy<dim, heads, false, true> {
    static constexpr float kScaleLog2 = 1.0F;
};

using BiasBase = dense::PreparedBf16Policy<dim, heads, true, true>;
struct BiasPolicy : BiasBase {
    static constexpr float kScaleLog2 = NoBiasPolicy::kScaleLog2;
    __device__ static float score_bias(const Args& a, unsigned row, unsigned head, unsigned p, unsigned k) {
        if (row >= a.rows || head >= heads || k > p || p - k >= a.window) {
            atomicOr(a.failure, 8U);
            return 0.0F;
        }
        return BiasBase::score_bias(a, row, head, p, k);
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

void fill(Fixture& fixture, bool wrapped) {
    for (unsigned p = pos0 - window + 1U; p < capacity; ++p) {
        const unsigned slot = wrapped ? p % ring_slots : p;
        fixture.key[slot * dim] = bf16(static_cast<float>(key_dot(p)));
        std::fill_n(fixture.value + slot * dim, dim, bf16(static_cast<float>(p + 1U)));
    }
    for (unsigned row = 0; row < rows; ++row) {
        for (unsigned head = 0; head < heads; ++head) {
            fixture.query[(row * heads + head) * dim] = 1.0F;
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
bool run(Fixture* fixture, unsigned splits, bool wrapped) {
    std::ranges::fill(fixture->out, 0.0F);
    const dense::PreparedBf16Args args{
        fixture->query, fixture->out, fixture->key, fixture->value, fixture->zeros, &fixture->failure,
        rows, 1U, pos0, window, wrapped ? ring_slots : capacity, wrapped,
        {fixture->bias, heads, extent}, {fixture->sink, heads},
    };
    require_hip(dense::launch_prepared_bf16<Policy>(args, nullptr, fixture->workspace, splits));
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
            const auto actual_head = std::span(fixture->out).subspan((row * heads + head) * dim, dim);
            for (float actual : actual_head) {
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
    fill(*fixture, false);
    const bool contiguous_ok = run<NoBiasPolicy>(fixture, 1U, false) &&
                               run<BiasPolicy>(fixture, 1U, false) && run<BiasPolicy>(fixture, 2U, false);
    fill(*fixture, true);
    const bool wrapped_ok = run<BiasPolicy>(fixture, 1U, true) && run<BiasPolicy>(fixture, 2U, true);
    std::destroy_at(fixture);
    require_hip(hipFree(fixture));
    if (!contiguous_ok || !wrapped_ok) std::cerr << "dense flash relative bias mismatch\n";
    return contiguous_ok && wrapped_ok ? 0 : 1;
}
