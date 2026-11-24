// Multi-request decode step lanes on HIP (strix/exec/step_lanes.hpp, strix/state/lane_view.hpp) identity, EXACT.
// A lane-batched stage that resolves each lane's state and position on the device (slot directory + step block)
// must equal the incumbent one-row stage of each valid lane bit for bit, and padded lanes must leave every
// session state untouched:
//   conv  causal_conv::launch_lanes vs launch_with_store (rows = 1) per lane: output + history bytes of every
//         slot (plain, dynamic taps, rounded output, rounded input + output; 2..16 lanes; 1,024 and 2,304 channels)
//   clip  l2_clip_bf16_rows vs l2_clip_bf16 at first = the lane position, with and without position scale
// The engine floating-point options (no FMA contraction, IEEE division / sqrt), gfx1151.
#include "core/row_transforms.hpp"
#include "core/short_convolution.hpp"
#include "strix/exec/step_lanes.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
namespace ex = strix::exec;
namespace conv = strix::core::causal_conv;
namespace rt = strix::core::row_transforms;

void check(hipError_t status, const char* operation) {
    if (status == hipSuccess) return;
    std::fprintf(stderr, "step-lanes-hip %s: %s\n", operation, hipGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

template <typename T>
class Device {
public:
    explicit Device(std::size_t count) : count_(count) { check(hipMalloc(&ptr_, count_ * sizeof(T)), "hipMalloc"); }
    explicit Device(const std::vector<T>& data) : Device(data.size()) { upload(data); }
    ~Device() {
        if (ptr_) (void)hipFree(ptr_);
    }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void upload(const std::vector<T>& data) {
        check(hipMemcpy(ptr_, data.data(), count_ * sizeof(T), hipMemcpyHostToDevice), "upload");
    }
    [[nodiscard]] T* data() const { return ptr_; }
    [[nodiscard]] std::vector<T> download() const {
        std::vector<T> out(count_);
        check(hipMemcpy(out.data(), ptr_, count_ * sizeof(T), hipMemcpyDeviceToHost), "download");
        return out;
    }

private:
    std::size_t count_;
    T* ptr_ = nullptr;
};

class Values {
public:
    explicit Values(std::uint32_t seed) : state_(seed) {}
    float next(float scale) {
        state_ = state_ * 1664525U + 1013904223U;
        return static_cast<float>(static_cast<int>(state_ >> 16) - 32768) * (scale / 32768.0F);
    }
    std::vector<float> make(std::size_t count, float scale) {
        std::vector<float> out(count);
        for (auto& v : out) v = next(scale);
        return out;
    }

private:
    std::uint32_t state_;
};

template <typename T>
bool same_bytes(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

template <typename T>
bool same_rows(const std::vector<T>& a, const std::vector<T>& b, std::size_t row, std::size_t width) {
    return std::memcmp(a.data() + row * width, b.data() + row * width, width * sizeof(T)) == 0;
}

std::vector<ex::CohortLane> lanes_of(const std::vector<std::uint32_t>& slots,
                                     const std::vector<std::uint32_t>& positions) {
    std::vector<ex::CohortLane> out;
    for (std::size_t i = 0; i < slots.size(); ++i)
        out.push_back({slots[i], positions[i], 7U + static_cast<std::uint32_t>(i)});
    return out;
}

// conv: R valid lanes in a launch of `width` lanes; rows [R, width) are padding (repeat lane 0, commit nothing).
template <bool RoundOutput, bool RoundInput>
bool conv_case(const char* name, std::size_t channels, bool dynamic, std::vector<std::uint32_t> slots,
               std::uint32_t width, std::uint32_t seed) {
    constexpr std::uint32_t kSlots = 17;
    const strix::core::ShortConvolution plan{channels, 4, dynamic ? 8U : 0U, 1.0F, dynamic ? 3.0F : 2.0F};
    const std::size_t history_floats = (plan.taps - 1) * plan.channels;
    const std::size_t visit_offset = 3 * history_floats;  // the plane holds other visits first
    const std::size_t slot_floats = visit_offset + history_floats + 256;
    Values values(seed);
    const auto lanes = static_cast<std::uint32_t>(slots.size());
    const auto input = values.make(std::size_t{width} * plan.channels, 2.0F);
    const auto weights = values.make(plan.taps * plan.channels, 1.0F);
    const auto taps = values.make(std::size_t{width} * (dynamic ? plan.dynamic_groups * plan.taps : 1U), 0.5F);
    const auto pool = values.make(kSlots * slot_floats, 1.5F);

    const Device<float> d_input(input), d_weights(weights), d_taps(taps);
    const Device<float> ref_pool(pool), lane_pool(pool);
    const Device<float> ref_out(std::vector<float>(std::size_t{width} * plan.channels, NAN));
    const Device<float> lane_out(std::vector<float>(std::size_t{width} * plan.channels, -NAN));
    for (std::uint32_t i = 0; i < lanes; ++i) {
        const conv::Args args{1U, d_input.data() + std::size_t{i} * plan.channels, d_weights.data(),
                              dynamic ? d_taps.data() + std::size_t{i} * plan.dynamic_groups * plan.taps : nullptr,
                              ref_pool.data() + slots[i] * slot_floats + visit_offset,
                              ref_out.data() + std::size_t{i} * plan.channels};
        check(conv::launch_with_store<RoundOutput, RoundInput>(plan, args, nullptr), "per-lane conv");
    }
    std::vector<strix::state::SlotEntry> directory(kSlots);
    for (std::uint32_t s = 0; s < kSlots; ++s)
        directory[s] = {reinterpret_cast<std::uint64_t>(lane_pool.data() + s * slot_floats), 0, 0, 0, 0};
    const Device<strix::state::SlotEntry> d_directory(directory);
    const std::vector<std::uint32_t> positions(slots.size(), 5U);
    const Device<ex::StepBlock> d_block(
        std::vector<ex::StepBlock>{ex::make_cohort_step(lanes_of(slots, positions), width, kSlots, 4096U)});
    const strix::state::LaneState<float> history{ex::step_lane_view(d_block.data(), d_directory.data(), kSlots, width),
                                                 {0U, visit_offset * sizeof(float)}};
    const conv::LaneArgs<strix::state::LaneState<float>> args{width, d_input.data(), d_weights.data(),
                                                             dynamic ? d_taps.data() : nullptr, history, lane_out.data()};
    check(conv::launch_lanes<RoundOutput, RoundInput>(plan, args, nullptr), "lanes conv");
    check(hipDeviceSynchronize(), "sync");
    const auto ro = ref_out.download(), lo = lane_out.download();
    bool ok = same_bytes(ref_pool.download(), lane_pool.download());  // valid slots committed, others untouched
    for (std::uint32_t i = 0; i < lanes; ++i) ok = ok && same_rows(ro, lo, i, plan.channels);
    for (std::uint32_t i = lanes; i < width; ++i) {
        for (std::size_t c = 0; c < plan.channels; ++c) ok = ok && std::isfinite(lo[i * plan.channels + c]);
    }
    std::printf("step-lanes-hip conv-%s %s\n", name, ok ? "PASS" : "FAIL");
    return ok;
}

// clip: one row per lane at its own device position == the one-row clip at that host position.
bool clip_case(const char* name, bool scale, unsigned groups, unsigned width, std::vector<std::uint32_t> positions,
               std::uint32_t seed) {
    const auto rows = static_cast<unsigned>(positions.size());
    Values values(seed);
    const auto input = values.make(std::size_t{rows} * groups * width, 4.0F);
    std::vector<std::uint16_t> gain(std::size_t{groups} * width);
    for (auto& g : gain) g = static_cast<std::uint16_t>(0x3F80U + (values.next(1.0F) > 0.0F ? 1U : 0U));
    const Device<float> ref(input), lanes(input);
    const Device<std::uint16_t> d_gain(gain);
    const rt::LogPositionScale position_scale = scale ? rt::LogPositionScale{0U, 8U, 0.1F} : rt::LogPositionScale{};
    for (unsigned r = 0; r < rows; ++r) {
        auto* row = ref.data() + std::size_t{r} * groups * width;
        auto per_row = position_scale;
        per_row.first = positions[r];
        check(rt::l2_clip_bf16({row, d_gain.data(), row, 1U, groups, width, 1.0e-6F, 22.6F, per_row}, nullptr),
              "per-row clip");
    }
    std::vector<ex::StepBlock> block(1);
    for (unsigned r = 0; r < rows; ++r) block[0].row[r].position = positions[r];
    const Device<ex::StepBlock> d_block(block);
    check(rt::l2_clip_bf16_rows({lanes.data(), d_gain.data(), lanes.data(), rows, groups, width, 1.0e-6F, 22.6F,
                                 position_scale}, nullptr,
                                ex::step_lane_view(d_block.data(), nullptr, 0U, rows).position, ex::kStepRowWords),
          "rows clip");
    check(hipDeviceSynchronize(), "sync");
    const bool ok = same_bytes(ref.download(), lanes.download());
    std::printf("step-lanes-hip clip-%s %s\n", name, ok ? "PASS" : "FAIL");
    return ok;
}

std::vector<std::uint32_t> first_slots(std::uint32_t count, std::uint32_t stride) {
    std::vector<std::uint32_t> out;
    for (std::uint32_t i = 0; i < count; ++i) out.push_back((i * stride) % 17U);
    return out;
}

}  // namespace

int main() {
    int cases = 0, failed = 0;
    const auto run = [&](bool ok) {
        ++cases;
        failed += ok ? 0 : 1;
    };
    // K / V convolutions (1,024 channels; dynamic taps on full visits), unfolded and folded rounds.
    run(conv_case<false, false>("kv-static-2of2", 1024, false, {4, 1}, 2, 21));
    run(conv_case<false, false>("kv-dynamic-8of8", 1024, true, first_slots(8, 3), 8, 22));
    run(conv_case<false, true>("kv-folded-16of16", 1024, true, first_slots(16, 5), 16, 23));
    run(conv_case<true, true>("kv-folded-round-5of8", 1024, false, {0, 5, 3, 2, 1}, 8, 24));
    // Post convolutions (2,304 channels, rounded output).
    run(conv_case<true, false>("post-1of1", 2304, false, {6}, 1, 25));
    run(conv_case<true, false>("post-2of2", 2304, false, {3, 0}, 2, 26));
    run(conv_case<true, false>("post-8of8", 2304, false, first_slots(8, 7), 8, 27));
    run(conv_case<true, false>("post-16of16", 2304, false, first_slots(16, 1), 16, 28));
    run(conv_case<true, false>("post-3of4-pad", 2304, false, {9, 2, 16}, 4, 29));
    // Position-scaled clips: positions below / at / far past the scale floor (query 24 x 128, relative 24 x 16).
    run(clip_case("query-scaled", true, 24, 128, {0, 7, 8, 9, 300, 4095, 70000, 12}, 31));
    run(clip_case("query-plain", false, 24, 128, {0, 7, 8, 9}, 32));
    run(clip_case("relative-scaled-16", true, 24, 16,
                  {1, 2, 3, 1023, 1024, 1025, 2047, 2048, 5, 6, 7, 8, 9, 10, 131071, 11}, 33));
    if (failed != 0) {
        std::printf("step-lanes-hip EXACT FAIL failed=%d cases=%d\n", failed, cases);
        return 1;
    }
    std::printf("step-lanes-hip EXACT PASS cases=%d\n", cases);
    return 0;
}
