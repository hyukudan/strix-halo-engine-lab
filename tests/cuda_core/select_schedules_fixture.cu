#include "cuda/core/select_rows.cuh"
#include "select_fixture_common.cuh"
#include "select_qwen_baseline.cuh"

#include <algorithm>
#include <array>

namespace {
namespace sel = strix::cuda::core::select;
namespace exec = strix::exec;
namespace ref = select_qwen_baseline;
using k5c_fixture::check;
using k5c_fixture::compare;
using k5c_fixture::Device;
using k5c_fixture::poison;
using k5c_fixture::score_poison;
using k5c_fixture::values;
using sel::ScoreSchedule;

constexpr sel::ScoreSpec spec{ref::kTopK, ref::kIndexHeads, ref::kIndexDim, ref::kRatio};
constexpr sel::PickSpec pick_spec{ref::kTopK, ref::kRatio, sel::KeyOrder::eligible_float};
constexpr unsigned max_groups = 1100, query_offset = 3, score_offset = 2;
constexpr sel::RowStrides stride{519, 1109, ref::kMaxSelected};
constexpr auto score_pitch = static_cast<unsigned>(stride.scores);

unsigned tiles(unsigned n, unsigned width) { return (n + width - 1U) / width; }

std::vector<float> pad_query(const std::vector<float>& input, unsigned rows) {
    std::vector<float> padded((rows + 1U) * stride.query, score_poison);
    for (unsigned row = 0; row < rows; ++row)
        std::copy_n(input.data() + row * ref::kIndexQWidth, ref::kIndexQWidth,
                    padded.data() + query_offset + row * stride.query);
    return padded;
}

struct Buffers {
    unsigned rows;
    std::vector<float> host_query;
    Device<float> ref_query, query, scores, expected;

    Buffers(unsigned count, unsigned seed)
        : rows(count), host_query(values(count * ref::kIndexQWidth, seed)), ref_query(host_query),
          query(pad_query(host_query, rows)), scores((rows + 1U) * stride.scores, score_poison),
          expected((rows + 1U) * stride.scores, score_poison) {}

    sel::RowBases bases(const float* keys, unsigned* selected = nullptr) const {
        return {query.data + query_offset, keys, scores.data + score_offset, selected};
    }

    void verify(const char* name) const { compare(name, expected, scores); }
};

template <ScoreSchedule Schedule>
void frozen_single(const Buffers& b, unsigned first, const float* keys) {
    const auto groups = (first + b.rows) / ref::kRatio;
    if constexpr (Schedule == ScoreSchedule::warp_shuffle) {
        const auto waves = tiles(b.rows, 4) * tiles(groups, 32);
        ref::block_score_kernel<<<tiles(waves, 8), 256>>>(b.ref_query.data, b.rows, first, keys, groups,
                                                        b.expected.data + score_offset, score_pitch);
    } else {
        ref::block_score_tree_kernel<<<dim3(tiles(b.rows, 64), tiles(groups, 256)), 256>>>(
            b.ref_query.data, b.rows, first, keys, groups, b.expected.data + score_offset, score_pitch);
    }
    check(cudaGetLastError(), "frozen single score enqueue");
}

template <ScoreSchedule Schedule>
void affine_case() {
    constexpr unsigned first = 2046, count = 71;
    const auto key_values = values((first + count) / ref::kRatio * ref::kIndexDim, 3);
    const Device<float> keys(key_values);
    auto unaligned_values = key_values;
    unaligned_values.insert(unaligned_values.begin(), score_poison);
    const Device<float> unaligned_keys(unaligned_values);
    const Buffers b(count, 17);
    const sel::AffineRows rows{b.bases(unaligned_keys.data + 1), stride, first};
    check(sel::enqueue_scores<sel::ScoreForm::relu_tree, Schedule>(rows, count, max_groups, spec, nullptr),
          "affine score enqueue");
    frozen_single<Schedule>(b, first, keys.data);
    b.verify("affine score bytes, below-cutoff rows and sentinels");
}

template <ScoreSchedule Schedule>
void x3_case() {
    constexpr unsigned first_a = 2046, first_b = 4094, count_a = 11, count_b = 9;
    const Device<float> keys_a(values((first_a + count_a) / ref::kRatio * ref::kIndexDim, 37));
    const Device<float> keys_b(values((first_b + count_b) / ref::kRatio * ref::kIndexDim, 57));
    const Buffers a(count_a, 61), b(count_b, 71);
    const Device table(std::vector{sel::SlotRowBase{a.bases(keys_a.data), first_a},
                                   sel::SlotRowBase{b.bases(keys_b.data), first_b}});
    constexpr unsigned active = count_a + count_b;
    exec::StepBlock step{};
    step.h.rows = active + 2U;
    step.h.valid_mask = (1U << active) - 1U;
    for (unsigned i = 0; i < count_b; ++i) {
        step.row[2U * i].slot = 0;
        step.row[2U * i].position = first_a + count_a - i - 1U;
        step.row[2U * i + 1U].slot = 1;
        step.row[2U * i + 1U].position = first_b + i;
    }
    step.row[active - 2U].position = first_a + 1U;
    step.row[active - 1U].position = first_a;
    for (unsigned i = 0; i < active; ++i) step.row[i].flags = exec::step_valid;
    // Missing mask, missing flag and outside width all point into the guard row.
    for (unsigned i = active; i < active + 3U; ++i) step.row[i].position = first_a + count_a;
    step.row[active].flags = step.row[active + 2U].flags = exec::step_valid;
    step.h.valid_mask |= (1U << (active + 1U)) | (1U << (active + 2U));
    const Device descriptor(std::vector{step});
    const sel::SlotRows rows{descriptor.data, table.data, 2, stride};
    check(sel::enqueue_scores<sel::ScoreForm::relu_tree, Schedule>(rows, active + 3U, max_groups, spec, nullptr),
          "X3 score enqueue");
    frozen_single<Schedule>(a, first_a, keys_a.data);
    frozen_single<Schedule>(b, first_b, keys_b.data);
    a.verify("X3 first slot score bytes and sentinels");
    b.verify("X3 second slot score bytes and sentinels");
}

using Map = sel::SegmentScoreMap<8>;
constexpr sel::SegmentScoreShape shape{120, 72, max_groups};

ref::SegmentMap frozen_map(std::span<const sel::ScoreSegmentInput> input) {
    ref::SegmentMap map{};
    for (const auto& item : input) {
        const auto s = map.n++;
        map.first_row[s + 1U] = map.first_row[s] + item.rows;
        map.position[s] = item.first_position;
        map.pooled[s] = item.keys;
        map.scored[s] = static_cast<unsigned>((item.first_position + item.rows) / ref::kRatio);
        map.score_begin[s + 1U] = map.score_begin[s];
        if (map.scored[s] > ref::kTopK) map.score_begin[s + 1U] += tiles(item.rows, 4) * tiles(map.scored[s], 32);
    }
    return map;
}

template <ScoreSchedule Schedule, class MapArg>
void frozen_cohort(const Buffers& b, MapArg map, unsigned count) {
    if (count == 0) return;
    if constexpr (Schedule == ScoreSchedule::warp_shuffle) {
        ref::block_score_multi_kernel<<<2, 256>>>(b.ref_query.data, map, b.expected.data + score_offset, score_pitch);
    } else {
        ref::block_score_tree_multi_kernel<<<dim3(tiles(shape.rows_per_segment, 64), tiles(max_groups, 256), count), 256>>>(
            b.ref_query.data, map, b.expected.data + score_offset, score_pitch);
    }
    check(cudaGetLastError(), "frozen cohort score enqueue");
}

template <class Producer>
void verify_picks(const Buffers& b, Producer rows, const ref::SegmentMap& map, const Device<unsigned>& expected_ids,
                  const Device<unsigned>& actual_ids) {
    for (unsigned s = 0; s < map.n; ++s) {
        const auto row = map.first_row[s], count = map.first_row[s + 1U] - row;
        ref::select_kernel<<<count, ref::kSelectThreads>>>(b.expected.data + score_offset + row * score_pitch,
            score_pitch, map.position[s], expected_ids.data + row * stride.selected);
        check(cudaGetLastError(), "frozen segment pick enqueue");
    }
    check(sel::enqueue_pick(rows, shape.rows, pick_spec, nullptr), "segment pick enqueue");
    b.verify("segment score mask and sentinels");
    compare("segment selected IDs, tails and sentinels", expected_ids, actual_ids);
}

template <ScoreSchedule Schedule>
void segment_case() {
    const Device<float> keys_a(values(max_groups * ref::kIndexDim, 101));
    const Device<float> keys_b(values(max_groups * ref::kIndexDim, 103));
    const Buffers b(shape.rows, 107);
    const std::array first{sel::ScoreSegmentInput{3, 2045, nullptr},
                           sel::ScoreSegmentInput{69, 2046, keys_a.data},
                           sel::ScoreSegmentInput{5, 4094, keys_b.data}};
    const std::array next{sel::ScoreSegmentInput{35, 4094, keys_b.data},
                          sel::ScoreSegmentInput{66, 2048, keys_a.data},
                          sel::ScoreSegmentInput{1, 28, nullptr},
                          sel::ScoreSegmentInput{6, 2049, keys_b.data}};
    Map map;
    if (!sel::pack_score_segments(first, shape, spec, map)) std::exit(EXIT_FAILURE);
    const auto first_ref = frozen_map(first);
    {
        const Device<unsigned> expected_ids((shape.rows + 1U) * stride.selected, poison);
        const Device<unsigned> actual_ids((shape.rows + 1U) * stride.selected, poison);
        const sel::SegmentRows eager{b.bases(nullptr, actual_ids.data), stride, map};
        check(sel::enqueue_scores<sel::ScoreForm::relu_tree, Schedule>(eager, shape, spec, nullptr),
              "eager segment score enqueue");
        frozen_cohort<Schedule>(b, first_ref, first_ref.n);
        b.verify("eager cohort scores and sentinels");
        verify_picks(b, eager, first_ref, expected_ids, actual_ids);
    }

    const Device device_map(std::vector{Map{}});
    const Device device_reference(std::vector{ref::SegmentMap{}});
    const Device<unsigned> expected_ids((shape.rows + 1U) * stride.selected, poison);
    const Device<unsigned> actual_ids((shape.rows + 1U) * stride.selected, poison);
    const auto* map_ptr = device_map.data;
    const sel::SegmentRows replay{b.bases(nullptr, actual_ids.data), stride, map_ptr};
    cudaStream_t stream{};
    cudaGraph_t graph{};
    cudaGraphExec_t instance{};
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "graph stream");
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture begin");
    check(sel::enqueue_scores<sel::ScoreForm::relu_tree, Schedule>(replay, shape, spec, stream, 2), "capture score");
    check(cudaStreamEndCapture(stream, &graph), "capture end");
    check(cudaGraphInstantiate(&instance, graph, 0), "instantiate graph");
    const std::vector<float> initial((shape.rows + 1U) * stride.scores, score_poison);
    const auto replay_case = [&](std::span<const sel::ScoreSegmentInput> input) {
        check(cudaMemcpyAsync(b.scores.data, initial.data(), initial.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
              "reset replay scores");
        if (!sel::pack_score_segments(input, shape, spec, map)) std::exit(EXIT_FAILURE);
        check(cudaMemcpyAsync(device_map.data, &map, sizeof(map), cudaMemcpyHostToDevice, stream),
              "publish replay descriptor");
        check(cudaGraphLaunch(instance, stream), "graph replay");
        const auto packed_ref = frozen_map(input);
        check(cudaMemcpy(b.expected.data, initial.data(), initial.size() * sizeof(float), cudaMemcpyHostToDevice),
              "reset reference scores");
        check(cudaMemcpy(device_reference.data, &packed_ref, sizeof(packed_ref), cudaMemcpyHostToDevice),
              "publish frozen descriptor");
        const auto* ref_ptr = device_reference.data;
        frozen_cohort<Schedule>(b, ref_ptr, packed_ref.n);
        check(cudaStreamSynchronize(stream), "score replay completion");
        b.verify("graph replay scores and sentinels");
    };
    replay_case(first);
    replay_case({});
    replay_case(next);
    verify_picks(b, replay, frozen_map(next), expected_ids, actual_ids);
    check(cudaGraphExecDestroy(instance), "destroy graph instance");
    check(cudaGraphDestroy(graph), "destroy graph");
    check(cudaStreamDestroy(stream), "destroy stream");
}
} // namespace

int main() {
    affine_case<ScoreSchedule::warp_shuffle>();
    affine_case<ScoreSchedule::row_tree>();
    std::puts("K5B affine warp-shuffle and 64-row tree scores and sentinels EXACT");
    x3_case<ScoreSchedule::warp_shuffle>();
    x3_case<ScoreSchedule::row_tree>();
    std::puts("K5B X3 warp-shuffle and 64-row tree scores and sentinels EXACT");
    segment_case<ScoreSchedule::warp_shuffle>();
    segment_case<ScoreSchedule::row_tree>();
    std::puts("K5B segment eager and graph replay scores, picks, masks and sentinels EXACT");
}
