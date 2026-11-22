#include "attn_core/prepared_gqa.hpp"
#include "core/bf16_projection_batch.hpp"

#include <stdexcept>

namespace {
namespace ac = strix::attn_core;
using strix::core::Bf16RowProjection;
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct Shape {
    static constexpr unsigned kMaxContext = 64, kHiddenWidth = 32, kQueryWidth = 32, kHeadWidth = 16, kKvWidth = 16,
                              kQueryHeads = 2, kKvHeads = 1, kConvTaps = 3, kRelativeWidth = 16;
    static constexpr float kQkClip = 1, kRelativeClip = 1, kFullConvDivisor = 1, kLocalConvDivisor = 1;
    static unsigned projection_width(ac::PreparedGqaProjection role) {
        return role == ac::PreparedGqaProjection::sink ? 2U : 16U;
    }
};

template <bool Deferred> struct Probe {
    using Status = bool;
    static constexpr bool success = true, invalid = false;
    float* raw_key;
    bool reject;
    mutable strix::core::Bf16ProjectionBatch batch{};
    mutable unsigned finishes{}, rounds{};
    bool linear(const float* input, unsigned width, unsigned, std::span<const Bf16RowProjection> views) const {
        return finishes == 0 && rounds == 0 && batch.append(input, width, views);
    }
    bool finish_projections() const requires Deferred {
        ++finishes;
        return !reject && batch.valid_r1_operands();
    }
    bool round(float* p, std::size_t, unsigned* = nullptr) const {
        if (rounds == 0 && (p != raw_key || finishes != static_cast<unsigned>(Deferred))) return false;
        ++rounds;
        return true;
    }
    template <class... T> bool convolve(T&&...) const { return rounds != 0; }
    template <class... T> bool clip(T&&...) const { return rounds != 0; }
    template <class... T> bool rms_values(T&&...) const { return rounds != 0; }
    template <class... T> bool relative_logits(T&&...) const { return rounds != 0; }
};

void run_boundary(bool full, unsigned alias) {
    alignas(16) float a[16][64]{};
    alignas(16) std::uint16_t w[32 * 32]{};
    const ac::PreparedGqaInputs in{a[0],alias == 0 ? a[0] : a[1],alias == 0 ? a[0] : alias == 1 ? a[1] : a[2]};
    ac::PreparedGqaWeights weights{};
    weights.matrix.fill(w);
    weights.query_gain = weights.relative_gain = w;
    weights.key_convolution = weights.value_convolution = a[0];
    const ac::PreparedGqaWorkspace work{a[3], a[4], a[5], a[6], a[7], a[8], a[9]};
    const ac::PreparedGqaRows out{a[10], a[11], a[12], a[13], a[14], a[15]};
    const auto run = [&](const auto& probe) {
        return ac::enqueue_prepared_gqa<Shape>(probe, full, 0, 1, in, weights, work, out);
    };
    Probe<true> accepted{work.raw_keys, false};
    check(run(accepted) && accepted.finishes == 1 && accepted.rounds > 0, "producer boundary must precede raw K");
    std::size_t jobs = 0;
    for (const auto& group : accepted.batch.groups()) jobs += group.count;
    check(jobs == (full ? 8U : 6U), "lost projection");
    check(accepted.batch.groups().size() == (full ? 3U : alias + 1U), "changed input alias groups");
    unsigned failure{};
    check(!accepted.batch.try_round_store(work.raw_keys, 16, &failure), "must retain failure-checking round");
    check(!accepted.batch.try_round_store(out.sinks, 2), "untiled output consumed a round");
    check(accepted.batch.try_round_store(work.raw_keys, 16) && !accepted.batch.try_round_store(work.raw_keys, 16),
          "round must bind once to the matching output");
    Probe<true> declined{work.raw_keys, true};
    check(!run(declined) && declined.finishes == 1 && declined.rounds == 0, "consumer ran after failed collection");
    Probe<false> immediate{work.raw_keys, false};
    check(run(immediate) && immediate.finishes == 0 && immediate.rounds > 0, "ordinary backend fallback");
    const std::array request_groups{ac::PreparedGqaGroup{0, 1, a[8], a[9]},
                                    ac::PreparedGqaGroup{5, 1, a[8] + 16, a[9] + 16}};
    Probe<false> grouped{work.raw_keys, false};
    check(ac::enqueue_prepared_gqa<Shape>(grouped, full, 0, 2, in, weights, work, out, nullptr, nullptr,
                                          request_groups) && grouped.rounds > immediate.rounds,
          "ordinary grouped request consumers dropped");
}
}  // namespace

void check_projection_boundary() {
    for (unsigned alias = 0; alias < 3; ++alias) run_boundary(false, alias);
    run_boundary(true, 2);
    alignas(16) float x[64]{};
    alignas(16) std::uint16_t w[32 * 16]{};
    strix::core::Bf16ProjectionBatch batch;
    const Bf16RowProjection p{w, 16, x + 4};
    check(batch.append(x, 32, std::span(&p, 1)) && !batch.valid_r1_operands(), "partial input/output overlap accepted");
    std::array<Bf16RowProjection, 8> excess{};
    check(!batch.append(x, 32, excess) && batch.groups().size() == 1, "capacity rejection mutated batch");
}
