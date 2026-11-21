#include "attn_core/prepared_bf16_attention.hpp"
#include "inkling2/inkling2_mini_attention_hip.hpp"
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

dense::PreparedBf16Args prepared_args(Fixture* fixture, bool wrapped) {
    return {
        fixture->query, fixture->out, fixture->key, fixture->value, fixture->zeros, &fixture->failure,
        rows, 1U, pos0, window, wrapped ? ring_slots : capacity, wrapped,
        {fixture->bias, heads, extent}, {fixture->sink, heads},
    };
}

template<class Policy>
bool run(Fixture* fixture, unsigned splits, bool wrapped) {
    std::ranges::fill(fixture->out, 0.0F);
    const auto args = prepared_args(fixture, wrapped);
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

bool run_extreme_sink(Fixture* fixture) {
    static_assert(dense::DenseRebasedSink<NoBiasPolicy>);
    std::ranges::fill(fixture->query, 0.0F);
    std::ranges::fill(fixture->value, bf16(32.0F));
    std::ranges::fill(fixture->sink, 130.0F * ln2);
    const auto args = prepared_args(fixture, false);
    // All visible keys have score zero. The sink overflows a single FP32
    // exponent and its reciprocal is subnormal, but the result is normal.
    const float expected = std::ldexp(32.0F * static_cast<float>(window), -130);
    for (unsigned splits : {1U, 2U}) {
        std::ranges::fill(fixture->out, 0.0F);
        require_hip(dense::launch_prepared_bf16<NoBiasPolicy>(args, nullptr, fixture->workspace, splits));
        require_hip(hipDeviceSynchronize());
        if (fixture->failure != 0) return false;
        for (float actual : fixture->out) {
            if (!(std::abs(actual / expected - 1.0F) <= 2e-4F)) return false;
        }
    }
    return true;
}

bool run_mini_append() {
    using namespace strix::inkling2;
    constexpr unsigned slots = rows + 1U;
    struct MiniFixture {
        alignas(16) float q[rows * kQueryWidth]{};
        float k[rows * kKvWidth]{};
        float v[rows * kKvWidth]{};
        float relative[rows * kQueryHeads * kRelativeExtent]{};
        float sink[rows * kQueryHeads]{};
        alignas(16) float out[rows * kQueryWidth]{};
        alignas(16) std::uint16_t key_cache[slots * kKvWidth]{};
        alignas(16) std::uint16_t value_cache[slots * kKvWidth]{};
        alignas(16) std::uint16_t zeros[kHeadWidth]{};
        alignas(16) float workspace[dense::workspace_floats(rows, kQueryHeads, kHeadWidth, 2U)]{};
        unsigned failure = 0;
    };

    void* storage = nullptr;
    require_hip(hipMallocManaged(&storage, sizeof(MiniFixture)));
    auto* fixture = std::construct_at(static_cast<MiniFixture*>(storage));
    hipStream_t stream = nullptr;
    require_hip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    for (unsigned kv = 0; kv < kKvHeads; ++kv) {
        std::fill_n(fixture->value_cache + kv * kHeadWidth, kHeadWidth, bf16(static_cast<float>(kv + 1U)));
        for (unsigned row = 0; row < rows; ++row) {
            const unsigned offset = row * kKvWidth + kv * kHeadWidth;
            fixture->k[offset] = row == 0 ? 1.00390625F : 1.01171875F;  // BF16 halfway cases
            std::fill_n(fixture->v + offset, kHeadWidth, static_cast<float>(kv + 2U + 2U * row));
        }
    }
    for (unsigned row = 0; row < rows; ++row) {
        for (unsigned head = 0; head < kQueryHeads; ++head) {
            const unsigned at = row * kQueryHeads + head;
            fixture->relative[at * kRelativeExtent] = ln2;
            fixture->sink[at] = head % (kQueryHeads / kKvHeads) == 0 ? ln2 : 0.0F;
        }
    }

    constexpr auto visit = mini_schedule()[5];
    static_assert(is_full_attention(visit.physical_layer));
    const auto chunk = [&](unsigned offset, unsigned count) {
        return MiniHipAttentionRows{
            fixture->q + offset * kQueryWidth, fixture->k + offset * kKvWidth, fixture->v + offset * kKvWidth,
            fixture->relative + offset * kQueryHeads * kRelativeExtent, fixture->sink + offset * kQueryHeads,
            fixture->out + offset * kQueryWidth, 1U + offset, count,
        };
    };
    MiniHipAttentionCache cache{fixture->key_cache, fixture->value_cache, fixture->zeros, slots - 1U, false};
    bool ok = enqueue_mini_hip_attention(visit, chunk(0U, rows), cache, &fixture->failure, stream) == hipErrorInvalidValue;
    cache.slots = slots;
    ok &= enqueue_mini_hip_attention(visit, chunk(0U, rows), cache, &fixture->failure, stream, {}, 2U) == hipErrorInvalidValue;
    const MiniHipAttentionCache aliased{fixture->key_cache, fixture->key_cache, fixture->zeros, slots, false};
    ok &= enqueue_mini_hip_attention(visit, chunk(0U, rows), aliased, &fixture->failure, stream) == hipErrorInvalidValue;
    using strix::attn_core::cache_rows::StoreArgs;
    const StoreArgs bad_wrap{fixture->k, kKvWidth, nullptr, 0U, rows, {kKvWidth, 1U}, fixture->key_cache, &fixture->failure};
    const StoreArgs valid_wrap{fixture->v, kKvWidth, nullptr, 0U, rows, {kKvWidth, 2U}, fixture->value_cache, &fixture->failure};
    ok &= strix::attn_core::cache_rows::store_pair(bad_wrap, valid_wrap, stream) == hipErrorInvalidValue;
    require_hip(hipStreamSynchronize(stream));
    ok &= fixture->key_cache[0] == 0U && fixture->key_cache[kKvWidth] == 0U && fixture->value_cache[kKvWidth] == 0U;

    const auto outputs_match = [&] {
        if (fixture->failure != 0) return false;
        for (unsigned row = 0; row < rows; ++row) {
            const auto slot = (row + 1U) * kKvWidth;
            if (fixture->key_cache[slot] != (row == 0 ? 0x3f80U : 0x3f82U) ||
                fixture->value_cache[slot] != bf16(static_cast<float>(2U + 2U * row))) return false;
            for (unsigned head = 0; head < kQueryHeads; ++head) {
                const unsigned group = head / (kQueryHeads / kKvHeads);
                const float prev = static_cast<float>(group + 1U + (row == 0 ? 0U : group + 2U));
                const float current = static_cast<float>(group + 2U + 2U * row);
                const float sink_mass = head % (kQueryHeads / kKvHeads) == 0 ? 2.0F : 1.0F;
                const float expected = (prev + 2.0F * current) / (3.0F + static_cast<float>(row) + sink_mass);
                const auto actual = std::span(fixture->out).subspan(row * kQueryWidth + head * kHeadWidth, kHeadWidth);
                for (float value : actual) {
                    if (!(std::abs(value - expected) <= 2e-4F)) return false;
                }
            }
        }
        return true;
    };
    require_hip(enqueue_mini_hip_attention(visit, chunk(0U, rows), cache, &fixture->failure, stream, fixture->workspace, 2U));
    require_hip(hipStreamSynchronize(stream));
    ok &= outputs_match();

    std::ranges::fill(fixture->out, 0.0F);
    for (unsigned row = 0; row < rows; ++row) {
        require_hip(enqueue_mini_hip_attention(visit, chunk(row, 1U), cache, &fixture->failure, stream));
    }
    require_hip(hipStreamSynchronize(stream));
    ok &= outputs_match();
    require_hip(hipStreamDestroy(stream));
    std::destroy_at(fixture);
    require_hip(hipFree(fixture));
    return ok;
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
    const bool sink_ok = run_extreme_sink(fixture);
    std::destroy_at(fixture);
    require_hip(hipFree(fixture));
    const bool append_ok = run_mini_append();
    if (!contiguous_ok || !wrapped_ok) std::cerr << "dense flash relative bias mismatch\n";
    if (!sink_ok) std::cerr << "dense flash extreme sink mismatch\n";
    if (!append_ok) std::cerr << "Mini cache append and attention mismatch\n";
    return contiguous_ok && wrapped_ok && sink_ok && append_ok ? 0 : 1;
}
