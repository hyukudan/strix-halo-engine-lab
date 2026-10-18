#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace strix {
// Cost estimate of one window owner (or of a serial row per emitted token), from cycle wall-time samples.
// legacy: EMA (alpha 0.25), each sample capped at twice the estimate; the first sample sets it.
// robust: the last 8 samples; with fewer than 3 the minimum (a provisional estimate a single slow sample cannot
// inflate), then their median (one outlier never moves it).
class MtpWindowCostEstimate {
public:
    enum class Model {legacy,robust};
    explicit MtpWindowCostEstimate(Model model=Model::robust):model_(model) {}
    void add(double ms);
    [[nodiscard]] double value() const noexcept {return value_;}
    [[nodiscard]] std::uint64_t samples() const noexcept {return samples_;}
    // Robust model: fewer than 3 samples, the estimate is provisional.
    [[nodiscard]] bool provisional() const noexcept {return samples_<3U;}
private:
    Model model_;
    double value_=0.0;
    std::uint64_t samples_=0;
    std::array<double,8> recent_{};
};

// The D-022 window controller (min-p cut, per-bin acceptance calibration, owner chosen by expected tokens per
// estimated ms). Host-only and deterministic given its inputs, so the verifier and the tests share it.
// With the robust cost model it also re-probes: an owner whose estimate is provisional, or implausible (above
// a larger owner's), and that has gone `reprobe_cycles` choices unused is chosen once more when it fits; each
// probe doubles that owner's interval, so probes stay O(log cycles).
class MtpMinPWindowController {
public:
    static constexpr std::size_t serial=static_cast<std::size_t>(-1);
    struct OwnerStats {std::uint32_t rows=0;std::uint64_t cycles=0;double cycle_ms=0.0;std::uint64_t probes=0;};
    MtpMinPWindowController(std::vector<std::uint32_t> owner_rows,float min_probability,
        MtpWindowCostEstimate::Model model=MtpWindowCostEstimate::Model::robust,std::uint64_t reprobe_cycles=64);
    // Whether one more draft can still pay off (see HipMtpWindowVerifier::worth_drafting).
    [[nodiscard]] bool worth_drafting(std::span<const float> p) const;
    // Owner index (ascending rows) or `serial`; fits[o] != 0 when owner o is admissible this cycle.
    [[nodiscard]] std::size_t choose(std::span<const float> p,std::size_t candidates,std::span<const std::uint8_t> fits);
    void calibrate(std::span<const float> p,std::size_t accepted_drafts,std::size_t verified_rows);
    // Wall time from the previous verified cycle to the end of this one (owner or `serial`, emitted > 0).
    // An owner's first window pays one-time admission and first touch: it is not a sample.
    void observe(std::size_t owner,std::size_t emitted,double ms);
    [[nodiscard]] std::vector<OwnerStats> owner_stats() const; // serial first (rows 1), then the owners
    [[nodiscard]] MtpWindowCostEstimate::Model model() const noexcept {return model_;}
private:
    struct Owner {
        std::uint32_t rows=0;MtpWindowCostEstimate cost;bool warmed=false;
        std::uint64_t last_used=0,interval=0,probes=0;
    };
    [[nodiscard]] double acceptance(float p) const;
    [[nodiscard]] double expected_tokens(std::uint32_t rows,std::span<const float> p) const;
    [[nodiscard]] bool implausible(const Owner&) const;
    std::vector<Owner> owners_;
    MtpWindowCostEstimate serial_;
    MtpWindowCostEstimate::Model model_;
    float min_probability_;
    std::uint64_t reprobe_cycles_,choices_=0;
    std::array<double,10> tested_{},accepted_{};
};
} // namespace strix
