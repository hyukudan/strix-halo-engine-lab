// Multi-request decode step lanes (strix/exec/step_lanes.hpp, strix/state/lane_view.hpp) identity, EXACT. Every lane-batched stage
// that resolves its operands on the device (slot directory + X3 step block) must equal the incumbent one-row
// stage of each valid lane bit for bit, padded lanes must leave every session state untouched, and a program
// captured once for a width class must replay a DIFFERENT request set (other slots, other positions) with the
// bits of the one-row stages on that set:
//   conv      causal_conv::launch_lanes vs launch_with_store (rows = 1) per lane: output + history bytes
//   clip      l2_clip_bf16_rows vs l2_clip_bf16 at first = the lane position, with and without position scale
//   attn      append_attention_prepared_bf16_step_lanes vs append_attention_prepared_bf16 per lane: output,
//             K/V cache bytes of every slot, failure flag; full and sliding caches, both pipes
//   graph     WidthStepGraphs: capture with request set A, replay with set B, then set C; outputs and the
//             state of every slot equal the per-lane eager chain A, B, C
// Production FP flags (no FMA contraction, IEEE division/sqrt), sm_120a.
#include "core/row_transforms.hpp"
#include "core/short_convolution.hpp"
#include "strix/cuda_core/attention_step_lanes.hpp"
#include "strix/exec/width_graphs.hpp"
#include "strix/exec/step_lanes.hpp"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace {
namespace cc = strix::cuda_core;
namespace ex = strix::exec;
namespace conv = strix::core::causal_conv;
namespace rt = strix::core::row_transforms;

void check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "step-lanes %s: %s\n", operation, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

template <typename T>
class Device {
public:
    explicit Device(std::size_t count) : count_(count) { check(cudaMalloc(&ptr_, count_ * sizeof(T)), "cudaMalloc"); }
    explicit Device(const std::vector<T>& data) : Device(data.size()) { upload(data); }
    ~Device() { if (ptr_) (void)cudaFree(ptr_); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void upload(const std::vector<T>& data) {
        check(cudaMemcpy(ptr_, data.data(), count_ * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    [[nodiscard]] T* data() const { return ptr_; }
    [[nodiscard]] std::vector<T> download() const {
        std::vector<T> out(count_);
        check(cudaMemcpy(out.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost), "download");
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
    template <typename T>
    std::vector<T> make(std::size_t count, float scale) {
        std::vector<T> out(count);
        for (auto& v : out) v = T(next(scale));
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

// A device step block + slot directory for one test.
struct Directory {
    explicit Directory(std::uint32_t states, std::uint32_t slots)
        : states_(states), slots_(slots), host(std::size_t{states} * slots), device(host.size()), block(1) {}
    void set(std::uint32_t state, std::uint32_t slot, const void* base) {
        host[std::size_t{state} * slots_ + slot] = {reinterpret_cast<std::uint64_t>(base), 0, 0, 0, 0};
    }
    void publish() { device.upload(host); }
    void upload_step(const ex::StepBlock& step) { block.upload(std::vector<ex::StepBlock>{step}); }
    [[nodiscard]] strix::state::LaneView lanes(std::uint32_t width) const {
        return ex::step_lane_view(block.data(), device.data(), slots_, width);
    }

    std::uint32_t states_, slots_;
    std::vector<strix::state::SlotEntry> host;
    Device<strix::state::SlotEntry> device;
    Device<ex::StepBlock> block;
};

std::vector<ex::CohortLane> lanes_of(const std::vector<std::uint32_t>& slots, const std::vector<std::uint32_t>& positions) {
    std::vector<ex::CohortLane> out;
    for (std::size_t i = 0; i < slots.size(); ++i) out.push_back({slots[i], positions[i], 7U + static_cast<std::uint32_t>(i)});
    return out;
}

// ---- conv ------------------------------------------------------------------------------------------------

template <bool RoundOutput, bool RoundInput>
bool conv_case(const char* name, bool dynamic, std::vector<std::uint32_t> slots, std::uint32_t width, std::uint32_t seed) {
    constexpr std::uint32_t kSlots = 6;
    const strix::core::ShortConvolution plan{1024, 4, dynamic ? 8U : 0U, 1.0F, dynamic ? 3.0F : 2.0F};
    const std::size_t history_floats = (plan.taps - 1) * plan.channels;
    const std::size_t visit_offset = 3 * history_floats;  // the plane holds other visits first
    const std::size_t slot_floats = visit_offset + history_floats + 256;
    Values values(seed);
    const auto lanes = static_cast<std::uint32_t>(slots.size());
    const auto input = values.make<float>(std::size_t{width} * plan.channels, 2.0F);
    const auto weights = values.make<float>(plan.taps * plan.channels, 1.0F);
    const auto taps = values.make<float>(std::size_t{width} * (dynamic ? plan.dynamic_groups * plan.taps : 1U), 0.5F);
    const auto pool = values.make<float>(kSlots * slot_floats, 1.5F);

    const Device<float> d_input(input), d_weights(weights), d_taps(taps);
    Device<float> ref_pool(pool), lane_pool(pool);
    const Device<float> ref_out(std::vector<float>(std::size_t{width} * plan.channels, NAN));
    const Device<float> lane_out(std::vector<float>(std::size_t{width} * plan.channels, -NAN));
    for (std::uint32_t i = 0; i < lanes; ++i) {
        const conv::Args args{1U, d_input.data() + std::size_t{i} * plan.channels, d_weights.data(),
                              dynamic ? d_taps.data() + std::size_t{i} * plan.dynamic_groups * plan.taps : nullptr,
                              ref_pool.data() + slots[i] * slot_floats + visit_offset,
                              ref_out.data() + std::size_t{i} * plan.channels};
        check(conv::launch_with_store<RoundOutput, RoundInput>(plan, args, nullptr), "per-lane conv");
    }
    Directory directory(1, kSlots);
    for (std::uint32_t s = 0; s < kSlots; ++s) directory.set(0, s, lane_pool.data() + s * slot_floats);
    directory.publish();
    std::vector<std::uint32_t> positions(slots.size(), 5U);
    directory.upload_step(ex::make_cohort_step(lanes_of(slots, positions), width, kSlots, 4096U));
    const strix::state::LaneState<float> history{directory.lanes(width), {0U, visit_offset * sizeof(float)}};
    const conv::LaneArgs<strix::state::LaneState<float>> args{width, d_input.data(), d_weights.data(),
                                                        dynamic ? d_taps.data() : nullptr, history, lane_out.data()};
    check(conv::launch_lanes<RoundOutput, RoundInput>(plan, args, nullptr), "lanes conv");
    check(cudaDeviceSynchronize(), "sync");
    const auto ro = ref_out.download(), lo = lane_out.download();
    bool ok = same_bytes(ref_pool.download(), lane_pool.download());  // valid slots committed, others untouched
    for (std::uint32_t i = 0; i < lanes; ++i) ok = ok && same_rows(ro, lo, i, plan.channels);
    for (std::uint32_t i = lanes; i < width; ++i) {
        for (std::size_t c = 0; c < plan.channels; ++c) ok = ok && std::isfinite(lo[i * plan.channels + c]);
    }
    std::printf("step-lanes conv-%s %s\n", name, ok ? "PASS" : "FAIL");
    return ok;
}

// ---- clip ------------------------------------------------------------------------------------------------

bool clip_case(const char* name, bool scale, std::vector<std::uint32_t> positions, std::uint32_t seed) {
    constexpr unsigned kGroups = 24, kWidth = 128;
    const auto rows = static_cast<unsigned>(positions.size());
    Values values(seed);
    const auto input = values.make<float>(std::size_t{rows} * kGroups * kWidth, 4.0F);
    std::vector<std::uint16_t> gain(kGroups * kWidth);
    for (auto& g : gain) g = static_cast<std::uint16_t>(0x3F80U + (values.next(1.0F) > 0.0F ? 1U : 0U));
    const Device<float> ref(input), lanes(input);
    const Device<std::uint16_t> d_gain(gain);
    const rt::LogPositionScale position_scale = scale ? rt::LogPositionScale{0U, 8U, 0.1F} : rt::LogPositionScale{};
    for (unsigned r = 0; r < rows; ++r) {
        auto* row = ref.data() + std::size_t{r} * kGroups * kWidth;
        auto per_row = position_scale;
        per_row.first = positions[r];
        check(rt::l2_clip_bf16({row, d_gain.data(), row, 1U, kGroups, kWidth, 1.0e-6F, 22.6F, per_row}, nullptr),
              "per-row clip");
    }
    std::vector<ex::StepBlock> block(1);
    for (unsigned r = 0; r < rows; ++r) block[0].row[r].position = positions[r];
    const Device<ex::StepBlock> d_block(block);
    check(rt::l2_clip_bf16_rows({lanes.data(), d_gain.data(), lanes.data(), rows, kGroups, kWidth, 1.0e-6F, 22.6F,
                                 position_scale}, nullptr,
                                ex::step_lane_view(d_block.data(), nullptr, 0U, rows).position, ex::kStepRowWords),
          "rows clip");
    check(cudaDeviceSynchronize(), "sync");
    const bool ok = same_bytes(ref.download(), lanes.download());
    std::printf("step-lanes clip-%s %s\n", name, ok ? "PASS" : "FAIL");
    return ok;
}

// ---- attention -------------------------------------------------------------------------------------------

constexpr unsigned kHeads = 24, kKvHeads = 8, kDim = 128, kExtent = 1024, kWindow = 1024;

struct AttnPool {
    AttnPool(std::uint32_t slots, std::size_t cache, std::uint32_t seed)
        : slots_(slots), cache_(cache), plane_(2 * cache + 512) {
        Values values(seed);
        image = values.make<__nv_bfloat16>(std::size_t{slots} * 2 * plane_, 1.0F);
    }
    // Slot s: keys plane then values plane; this visit sits at element offset `cache` of each plane.
    [[nodiscard]] std::size_t keys(std::uint32_t slot) const { return std::size_t{slot} * 2 * plane_; }
    [[nodiscard]] std::size_t values(std::uint32_t slot) const { return keys(slot) + plane_; }
    [[nodiscard]] std::size_t offset() const { return cache_; }

    std::uint32_t slots_;
    std::size_t cache_, plane_;
    std::vector<__nv_bfloat16> image;
};

struct AttnRows {
    AttnRows(std::uint32_t width, std::uint32_t seed) {
        Values values(seed);
        q = values.make<float>(std::size_t{width} * kHeads * kDim, 2.0F);
        k = values.make<float>(std::size_t{width} * kKvHeads * kDim, 2.0F);
        v = values.make<float>(std::size_t{width} * kKvHeads * kDim, 1.0F);
        bias = values.make<float>(std::size_t{width} * kHeads * kExtent, 0.5F);
        sinks = values.make<float>(std::size_t{width} * kHeads, 1.0F);
    }
    std::vector<float> q, k, v, bias, sinks;
};

// Reference: the incumbent one-row append of each valid lane on its own caches, in lane order.
void reference_attention(const AttnPool& pool, Device<__nv_bfloat16>& image, const AttnRows& rows,
                         const Device<float>& dq, const Device<float>& dk, const Device<float>& dv,
                         const Device<float>& db, const Device<float>& ds, float* out, unsigned* failure,
                         bool sliding, cc::FlashPipe pipe, const std::vector<std::uint32_t>& slots,
                         const std::vector<std::uint32_t>& positions, std::uint32_t cache_slots, cudaStream_t stream) {
    static_cast<void>(rows);
    const cc::AttentionDims dims{kHeads, kKvHeads, kDim, 1.0F / std::sqrt(static_cast<float>(kDim)), cache_slots};
    for (std::size_t i = 0; i < slots.size(); ++i) {
        cc::PreparedBf16Flash prepared{db.data() + i * kHeads * kExtent, ds.data() + i * kHeads, kExtent,
                                       sliding ? kWindow : 0U, sliding};
        check(cc::append_attention_prepared_bf16(dims, prepared, dq.data() + i * kHeads * kDim,
                  dk.data() + i * kKvHeads * kDim, dv.data() + i * kKvHeads * kDim,
                  image.data() + pool.keys(slots[i]) + pool.offset(), image.data() + pool.values(slots[i]) + pool.offset(),
                  positions[i], 1U, out + i * kHeads * kDim, stream, pipe, failure), "per-lane append");
    }
}

cc::PreparedBf16StepLanes step_lanes(const Directory& directory, const AttnPool& pool, std::uint32_t width,
                                     std::uint32_t cache_slots, bool sliding) {
    const auto bytes = pool.offset() * sizeof(__nv_bfloat16);
    return {directory.lanes(width), {0U, bytes}, {1U, bytes}, cache_slots, sliding ? 1U : 0U};
}

void bind_directory(Directory& directory, const AttnPool& pool, const Device<__nv_bfloat16>& image) {
    for (std::uint32_t s = 0; s < pool.slots_; ++s) {
        directory.set(0, s, image.data() + pool.keys(s));
        directory.set(1, s, image.data() + pool.values(s));
    }
    directory.publish();
}

bool attn_case(const char* name, bool sliding, cc::FlashPipe pipe, std::vector<std::uint32_t> slots,
               std::vector<std::uint32_t> positions, std::uint32_t width, std::uint32_t seed) {
    constexpr std::uint32_t kSlots = 7;
    const std::uint32_t cache_slots = sliding ? kWindow + 64U : 4096U;
    const AttnPool pool(kSlots, std::size_t{cache_slots} * kKvHeads * kDim, seed);
    const AttnRows rows(width, seed + 1U);
    const Device<float> dq(rows.q), dk(rows.k), dv(rows.v), db(rows.bias), ds(rows.sinks);
    Device<__nv_bfloat16> ref_image(pool.image), lane_image(pool.image);
    const Device<float> ref_out(std::vector<float>(std::size_t{width} * kHeads * kDim, NAN));
    const Device<float> lane_out(std::vector<float>(std::size_t{width} * kHeads * kDim, -NAN));
    const Device<unsigned> ref_fail(std::vector<unsigned>{0}), lane_fail(std::vector<unsigned>{0});
    reference_attention(pool, ref_image, rows, dq, dk, dv, db, ds, ref_out.data(), ref_fail.data(), sliding, pipe,
                        slots, positions, cache_slots, nullptr);

    Directory directory(2, kSlots);
    bind_directory(directory, pool, lane_image);
    directory.upload_step(ex::make_cohort_step(lanes_of(slots, positions), width, kSlots, cache_slots));
    const cc::AttentionDims dims{kHeads, kKvHeads, kDim, 1.0F / std::sqrt(static_cast<float>(kDim)), 0U};
    const cc::PreparedBf16Flash shape{nullptr, nullptr, kExtent, sliding ? kWindow : 0U, false};
    const cc::PreparedBf16LaneImages images{dq.data(), dk.data(), dv.data(), db.data(), ds.data(), lane_out.data()};
    check(cc::append_attention_prepared_bf16_step_lanes(dims, shape, images,
              step_lanes(directory, pool, width, cache_slots, sliding), nullptr, pipe, lane_fail.data()),
          "step-lanes append");
    check(cudaDeviceSynchronize(), "sync");
    const auto ro = ref_out.download(), lo = lane_out.download();
    bool ok = same_bytes(ref_image.download(), lane_image.download()) && same_bytes(ref_fail.download(), lane_fail.download()) &&
              lane_fail.download()[0] == 0U;
    for (std::size_t i = 0; i < slots.size(); ++i) ok = ok && same_rows(ro, lo, i, std::size_t{kHeads} * kDim);
    std::printf("step-lanes attn-%s %s\n", name, ok ? "PASS" : "FAIL");
    return ok;
}

// ---- graph: capture once, replay other request sets ------------------------------------------------------

struct Request {
    std::vector<std::uint32_t> slots, positions;
};

bool graph_case(const char* name, cc::FlashPipe pipe, std::uint32_t width, const std::vector<Request>& steps,
                std::uint32_t seed) {
    constexpr std::uint32_t kSlots = 8;
    constexpr std::uint32_t cache_slots = 4096U;
    const strix::core::ShortConvolution plan{kKvHeads * kDim, 4, 0U, 1.0F, 2.0F};
    const std::size_t history_floats = (plan.taps - 1) * plan.channels;
    const AttnPool pool(kSlots, std::size_t{cache_slots} * kKvHeads * kDim, seed);
    Values values(seed + 3U);
    const auto histories = values.make<float>(kSlots * history_floats, 1.0F);
    const auto weights = values.make<float>(plan.taps * plan.channels, 1.0F);
    const Device<float> d_weights(weights);
    Device<__nv_bfloat16> ref_image(pool.image), lane_image(pool.image);
    Device<float> ref_hist(histories), lane_hist(histories);

    Directory directory(3, kSlots);
    bind_directory(directory, pool, lane_image);
    for (std::uint32_t s = 0; s < kSlots; ++s) directory.set(2, s, lane_hist.data() + s * history_floats);
    directory.publish();

    strix::device::native::StreamOwner owner;
    const auto stream = owner.get();
    ex::WidthStepGraphs graphs;
    // The program reads the graph object's block: point the directory view at it.
    const auto view = ex::step_lane_view(graphs.block(), directory.device.data(), kSlots, width);
    const cc::AttentionDims dims{kHeads, kKvHeads, kDim, 1.0F / std::sqrt(static_cast<float>(kDim)), 0U};
    const cc::PreparedBf16Flash shape{nullptr, nullptr, kExtent, 0U, false};
    const auto bytes = pool.offset() * sizeof(__nv_bfloat16);
    const cc::PreparedBf16StepLanes attention{view, {0U, bytes}, {1U, bytes}, cache_slots, 0U};

    // Fixed-address operands: a replayed program reads them where the capture saw them.
    const std::size_t q_rows = std::size_t{width} * kHeads * kDim, kv_rows = std::size_t{width} * kKvHeads * kDim;
    Device<float> lane_q(q_rows), lane_k(kv_rows), lane_v(kv_rows), lane_conv(kv_rows), lane_out(q_rows),
        lane_bias(std::size_t{width} * kHeads * kExtent), lane_sinks(std::size_t{width} * kHeads);
    const Device<unsigned> lane_fail(std::vector<unsigned>{0});
    const cc::PreparedBf16LaneImages images{lane_q.data(), lane_conv.data(), lane_v.data(), lane_bias.data(),
                                            lane_sinks.data(), lane_out.data()};
    bool ok = true;
    for (std::size_t step = 0; step < steps.size(); ++step) {
        const auto& request = steps[step];
        const AttnRows rows(width, seed + 11U * static_cast<std::uint32_t>(step));
        const Device<float> dq(rows.q), dk(rows.k), dv(rows.v), db(rows.bias), ds(rows.sinks);
        lane_q.upload(rows.q);
        lane_k.upload(rows.k);
        lane_v.upload(rows.v);
        lane_bias.upload(rows.bias);
        lane_sinks.upload(rows.sinks);
        const Device<float> ref_out(q_rows), ref_conv(kv_rows);
        const Device<unsigned> ref_fail(std::vector<unsigned>{0});
        // Reference chain: the per-lane one-row conv of K, then the per-lane append of the convolved K.
        for (std::size_t i = 0; i < request.slots.size(); ++i) {
            const conv::Args args{1U, dk.data() + i * plan.channels, d_weights.data(), nullptr,
                                  ref_hist.data() + request.slots[i] * history_floats, ref_conv.data() + i * plan.channels};
            check(conv::launch_with_store<false>(plan, args, stream), "reference conv");
        }
        reference_attention(pool, ref_image, rows, dq, ref_conv, dv, db, ds, ref_out.data(), ref_fail.data(), false, pipe,
                            request.slots, request.positions, cache_slots, stream);
        // Program: conv lanes then attention lanes, every per-request operand resolved from the replayed block.
        const auto block = ex::make_cohort_step(lanes_of(request.slots, request.positions), width, kSlots, cache_slots);
        const bool launched = graphs.try_launch(stream, true, true, block, [&] {
            const conv::LaneArgs<strix::state::LaneState<float>> args{width, lane_k.data(), d_weights.data(), nullptr,
                                                                {view, {2U, 0U}}, lane_conv.data()};
            check(conv::launch_lanes<false>(plan, args, stream), "lanes conv");
            check(cc::append_attention_prepared_bf16_step_lanes(dims, shape, images, attention, stream, pipe,
                                                                lane_fail.data()), "step-lanes append");
        });
        check(cudaStreamSynchronize(stream), "sync");
        if (!launched) {
            std::printf("step-lanes graph-%s FAIL (capture declined)\n", name);
            return false;
        }
        const auto ro = ref_out.download(), lo = lane_out.download();
        for (std::size_t i = 0; i < request.slots.size(); ++i) ok = ok && same_rows(ro, lo, i, std::size_t{kHeads} * kDim);
        ok = ok && same_bytes(ref_image.download(), lane_image.download()) &&
             same_bytes(ref_hist.download(), lane_hist.download()) && lane_fail.download()[0] == 0U &&
             ref_fail.download()[0] == 0U;
    }
    ok = ok && graphs.instantiations() == 1U && graphs.nodes(width) != 0U;
    std::printf("step-lanes graph-%s %s (instantiations %u nodes %zu)\n", name, ok ? "PASS" : "FAIL",
                graphs.instantiations(), graphs.nodes(width));
    return ok;
}

}  // namespace

int main() {
    int cases = 0, failed = 0;
    const auto run = [&](bool ok) { ++cases; failed += ok ? 0 : 1; };
    // conv: R valid lanes in a width class, padded rows repeat lane 0 and commit nothing.
    run(conv_case<false, false>("static-3of4", false, {4, 1, 2}, 4, 21));
    run(conv_case<false, false>("dynamic-5of8", true, {0, 5, 3, 2, 1}, 8, 22));
    run(conv_case<true, false>("post-2of2", false, {3, 0}, 2, 23));
    run(conv_case<true, true>("folded-6of8", true, {5, 4, 3, 2, 1, 0}, 8, 24));
    // clip: positions below / at / far past the scale floor, across a grid row each.
    run(clip_case("scaled", true, {0, 7, 8, 9, 300, 4095, 70000, 12}, 31));
    run(clip_case("plain", false, {0, 7, 8, 9}, 32));
    // attention: full / sliding (wrapped), both pipes, 1..32 lanes, pads.
    for (const auto pipe : {cc::FlashPipe::pad2, cc::FlashPipe::swz3}) {
        const bool pad = pipe == cc::FlashPipe::pad2;
        run(attn_case(pad ? "full-3of4-pad2" : "full-3of4-swz3", false, pipe, {6, 0, 3}, {311, 0, 1500}, 4, 41));
        run(attn_case(pad ? "sliding-5of8-pad2" : "sliding-5of8-swz3", true, pipe, {1, 2, 3, 4, 5},
                      {5, 1030, 2048, 4000, 77}, 8, 42));
    }
    {
        std::vector<std::uint32_t> slots, positions;
        for (std::uint32_t i = 0; i < 7; ++i) {
            slots.push_back(i);
            positions.push_back(3U + 397U * i);
        }
        run(attn_case("full-7of32-pad2", false, cc::FlashPipe::pad2, slots, positions, 32, 43));
    }
    // graph: one capture per width, replayed for other slots and positions.
    run(graph_case("w4", cc::FlashPipe::pad2, 4,
                   {{{0, 2}, {10, 600}}, {{5, 1, 3}, {44, 3000, 1}}, {{2, 7, 6, 4}, {11, 9, 4000, 128}}}, 51));
    if (failed != 0) {
        std::printf("step-lanes EXACT FAIL failed=%d cases=%d\n", failed, cases);
        return 1;
    }
    std::printf("step-lanes EXACT PASS cases=%d\n", cases);
    return 0;
}
