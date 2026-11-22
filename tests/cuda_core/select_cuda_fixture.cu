#include "cuda/core/select_score.cuh"
#include "cuda/models/glm/dsa_selection.hpp"
#include "cuda/models/minimax/mm_rtx_kernels.cuh"
#include "select_oracle.hpp"
#include "strix/cuda/ds41_attention.hpp"
#include "strix/cuda/ds41_fast_attention.hpp"
#include "strix/cuda/qwen_qsa.hpp"

#include <cuda_fp8.h>

#include "select_incumbents.inc"

#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace k5c_fixture {
namespace sel = strix::cuda::core::select;
namespace qwen = strix::cuda::qwen_qsa;
namespace ds = strix::ds41::cuda::attn;
namespace glm = strix::glm::rtxpf;
namespace mm = strix::minimax::rtxpf::cuda;

void check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "k5c %s: %s\n", operation, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

template <class T>
class Device {
public:
    explicit Device(const std::vector<T>& values) : size_(values.size()) {
        check(cudaMalloc(&data, size_ * sizeof(T)), "allocation");
        check(cudaMemcpy(data, values.data(), size_ * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    Device(std::size_t size, T initial) : Device(std::vector<T>(size, initial)) {}
    ~Device() { (void)cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    std::vector<T> read() const {
        std::vector<T> values(size_);
        check(cudaMemcpy(values.data(), data, size_ * sizeof(T), cudaMemcpyDeviceToHost), "download");
        return values;
    }

    T* data = nullptr;
private:
    std::size_t size_;
};

template <class T>
void compare(const char* name, const std::vector<T>& expected, const std::vector<T>& actual) {
    if (expected.size() == actual.size() &&
        std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(T)) == 0) return;
    std::size_t i = 0;
    while (i < expected.size() && i < actual.size() && std::memcmp(&expected[i], &actual[i], sizeof(T)) == 0) ++i;
    std::fprintf(stderr, "k5c %s FAIL at element %zu\n", name, i);
    std::exit(EXIT_FAILURE);
}

template <class T>
void compare(const char* name, const Device<T>& expected, const Device<T>& actual) {
    compare(name, expected.read(), actual.read());
}

std::vector<float> values(std::size_t n, unsigned seed) {
    std::vector<float> result(n);
    for (auto& value : result) {
        seed = seed * 1664525U + 1013904223U;
        value = static_cast<float>(static_cast<int>(seed >> 16U) - 32768) / 65536.0F;
    }
    return result;
}

constexpr unsigned poison = 0xdeadbeefU;
constexpr float score_poison = -991.0F;

void qwen_case() {
    constexpr unsigned rows = 13, first = 2046, stride = 523;
    constexpr unsigned groups = (first + rows) / 4;
    Device<float> query(values(rows * 512, 77)), keys(values(stride * 128, 15));
    Device<float> ref(rows * stride, score_poison), tree(rows * stride, score_poison), got(rows * stride, score_poison);
    std::vector<sel::ScoreRow> score_rows(rows);
    for (unsigned r = 0; r < rows; ++r) {
        score_rows[r].query = query.data + r * 512;
        score_rows[r].keys = keys.data;
        score_rows[r].visible = (first + r + 1U) / 4U;
        score_rows[r].scores = got.data + r * stride;
    }
    Device score_args(score_rows);
    const unsigned waves = ((rows + 3U) / 4U) * ((groups + 31U) / 32U);
    qwen::block_score_kernel<<<(waves + 7U) / 8U, 256>>>(query.data, rows, first, keys.data, groups, ref.data, stride);
    qwen::block_score_tree_kernel<<<dim3(1, (groups + 255U) / 256U), 256>>>(query.data, rows, first, keys.data, groups, tree.data, stride);
    check(cudaGetLastError(), "Qwen incumbent scores");
    check(sel::enqueue_scores<sel::ScoreForm::relu_tree>(score_args.data, rows, stride, {512, 4, 128}, nullptr), "Qwen scores");
    compare("Qwen shuffle score bytes", ref, got);
    compare("Qwen tree score bytes", tree, got);

    auto scores = ref.read();
    const unsigned special = rows - 1U;
    for (unsigned i = 0; i < groups; ++i) scores[special * stride + i] = select_oracle::ninf;
    scores[special * stride + 1U] = -0.0F;
    scores[special * stride + 2U] = 0.0F;
    scores[special * stride + 3U] = std::bit_cast<float>(0x7fc12345U);
    Device<float> ref_mask(scores), got_mask(scores);
    Device<unsigned> ref_ids(rows * qwen::kMaxSelected, poison), got_ids(rows * qwen::kMaxSelected, poison), counts(rows, poison);
    std::vector<sel::PickRow> pick_rows(rows);
    for (unsigned r = 0; r < rows; ++r) {
        auto& row = pick_rows[r];
        row.groups = score_rows[r].visible;
        row.tail = (first + r + 1U) % 4U;
        row.scores = row.mask_scores = got_mask.data + r * stride;
        row.selected = got_ids.data + r * qwen::kMaxSelected;
        row.count = counts.data + r;
    }
    Device pick_args(pick_rows);
    qwen::select_kernel<<<rows, 1024>>>(ref_mask.data, stride, first, ref_ids.data);
    check(cudaGetLastError(), "Qwen incumbent pick");
    const sel::PickSpec spec{512, 4, sel::KeyOrder::eligible_float};
    check(sel::enqueue_pick(pick_args.data, rows, spec, nullptr), "Qwen pick");
    compare("Qwen ids", ref_ids, got_ids);
    compare("Qwen masked scores", ref_mask, got_mask);
    const auto ids = got_ids.read(), lengths = counts.read();
    for (unsigned r = 0; r < rows; ++r) {
        const auto expected = select_oracle::pick(spec, pick_rows[r].groups, pick_rows[r].tail,
            std::span(scores).subspan(r * stride, stride));
        compare("Qwen oracle ids", expected.ids,
                std::vector<unsigned>(ids.begin() + r * qwen::kMaxSelected, ids.begin() + r * qwen::kMaxSelected + lengths[r]));
    }
    std::puts("k5c Qwen QSA ids and scores PASS");
}

void ds_case() {
    constexpr unsigned rows = 4, stride = 8201, top = 512, pool_width = 64, pool_stride = 130;
    constexpr std::array groups{0U, 513U, 520U, 8195U};
    Device<std::uint64_t> pos(std::vector<std::uint64_t>{0, 1025, 1039, 16389});
    std::vector<sel::PackedQuery> prep(rows);
    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned h = 0; h < 32; ++h) {
            for (unsigned i = 0; i < 32; ++i) prep[r].codes[h][i] = (i + h + r) % 2U ? 0x0103ff02 : 0x03fe0101;
            for (unsigned i = 0; i < 4; ++i) prep[r].scales[h][i] = (h + i) % 2U ? 0.5F : 0.25F;
            prep[r].weights[h] = static_cast<float>(static_cast<int>(h) - 11) * 0.03125F;
        }
    }
    static_assert(sizeof(sel::PackedQuery) == sizeof(ds::QPrep));
    Device query(prep);
    std::vector<std::uint8_t> packed(stride * 68);
    for (unsigned i = 0; i < stride; ++i) {
        for (unsigned j = 0; j < 64; ++j) packed[i * 68U + j] = static_cast<std::uint8_t>(i * 11U + j * 17U);
        for (unsigned j = 64; j < 68; ++j) packed[i * 68U + j] = static_cast<std::uint8_t>(124U + (i + j) % 9U);
    }
    Device keys(packed);
    Device<float> ref(rows * stride, score_poison), got(rows * stride, score_poison);
    std::vector<sel::ScoreRow> score_rows(rows);
    for (unsigned r = 0; r < rows; ++r) {
        auto& row = score_rows[r];
        row.packed_query = query.data + r;
        row.packed_keys = keys.data;
        row.visible = groups[r];
        row.scores = got.data + r * stride;
    }
    Device score_args(score_rows);
    ds::IndexView original{rows, 2, stride, nullptr, nullptr, nullptr, pos.data, keys.data, ref.data, query.data};
    ds::index_scores_kernel<1, 4><<<dim3((stride + 31U) / 32U, 1), 128>>>(original);
    check(cudaGetLastError(), "DeepSeek incumbent scores");
    check(sel::enqueue_scores<sel::ScoreForm::weighted_fp4>(score_args.data, rows, stride, {0, 32, 128}, nullptr), "DeepSeek scores");
    compare("DeepSeek score bytes", ref, got);
    // Explicit signed-zero order in the raw float trait, and repeated boundary scores.
    const auto scored = ref.read();
    auto selection_scores = scored;
    selection_scores[stride] = -0.0F;
    selection_scores[stride + 1U] = 0.0F;
    Device<float> ref_scores(selection_scores), got_scores(selection_scores);
    Device<unsigned> ref_ids(rows * top, poison), got_ids(rows * top, poison);
    Device<unsigned> ref_count(rows, poison), got_count(rows, poison);
    Device<unsigned> ref_pool_keys(rows * pool_stride, poison), got_pool_keys(rows * pool_stride, poison);
    Device<std::uint8_t> ref_pool(rows * pool_stride, 0x7f), got_pool(rows * pool_stride, 0x7f);
    constexpr unsigned slices = (stride + ds::kSelSlice - 1U) / ds::kSelSlice;
    Device<unsigned> scratch(rows * slices * (2U * top + 1U), poison);
    ds::SelectView original_pick{rows, 2, stride, top, 2, pool_width, ds::Pool::candidate_source, pos.data,
        ref_scores.data, ref_pool.data, pool_stride, ref_pool_keys.data, ref_ids.data, ref_count.data, scratch.data};
    std::vector<sel::PickRow> pick_rows(rows);
    for (unsigned r = 0; r < rows; ++r) {
        auto& row = pick_rows[r];
        row.scores = got_scores.data + r * stride;
        row.groups = groups[r];
        row.selected = got_ids.data + r * top;
        row.count = got_count.data + r;
        row.pool = sel::Pool::source;
        row.pool_width = pool_width;
        row.pool_top = 2;
        row.candidates = got_pool.data + r * pool_stride;
        row.pool_keys = got_pool_keys.data + r * pool_stride;
    }
    const auto run = [&] {
        Device args(pick_rows);
        ds::select_kernel<<<rows, 1024>>>(original_pick);
        ds::select_slice_kernel<<<dim3(slices, rows), 1024>>>(original_pick, slices);
        ds::select_merge_kernel<128><<<rows, 1024>>>(original_pick, slices);
        check(cudaGetLastError(), "DeepSeek incumbent pick");
        check(sel::enqueue_pick(args.data, rows, {top, 1, sel::KeyOrder::raw_float}, nullptr), "DeepSeek pick");
        compare("DeepSeek ids", ref_ids, got_ids);
        compare("DeepSeek counts", ref_count, got_count);
        compare("DeepSeek candidate membership", ref_pool, got_pool);
        compare("DeepSeek candidate keys", ref_pool_keys, got_pool_keys);
        compare("DeepSeek unmodified scores", ref_scores, got_scores);
    };
    run();
    original_pick.pool = ds::Pool::candidate_user;
    for (auto& row : pick_rows) row.pool = sel::Pool::user;
    run();
}

void ds_fp8_case() {
    constexpr unsigned rows = 3, stride = 39, budget = 8;
    constexpr std::array groups{0U, 21U, 35U};
    constexpr std::array<std::uint8_t, 8> e4m3{0x00, 0x21, 0x34, 0x38, 0x47, 0x9c, 0xac, 0xb5};
    static_assert(sizeof(sel::PackedFp8Query) == sizeof(ds::FastQPrep));
    std::vector<sel::PackedFp8Query> prep(rows);
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned h = 0; h < 32; ++h) {
            for (unsigned i = 0; i < 128; ++i) prep[r].codes[h][i] = e4m3[(r + h / 3U + i * 3U) % e4m3.size()];
            prep[r].weights[h] = static_cast<float>(static_cast<int>(h) - 11) * 0.03125F;
        }
    std::vector<std::uint8_t> codes(stride * 128);
    std::vector<float> scales(stride);
    for (unsigned p = 0; p < stride; ++p) {
        scales[p] = 0.5F * static_cast<float>(1U + p % 3U);
        for (unsigned j = 0; j < 128; ++j) codes[p * 128U + j] = e4m3[(p / 4U + j) % e4m3.size()];
    }
    Device query(prep);
    Device keys(codes);
    Device key_scales(scales);
    Device<std::uint64_t> positions(std::vector<std::uint64_t>{0, 41, 69});
    Device<float> ref(rows * stride, score_poison), got(rows * stride, score_poison);
    std::vector<sel::ScoreRow> score_rows(rows);
    for (unsigned r = 0; r < rows; ++r) {
        auto& row = score_rows[r];
        row.fp8_query = query.data + r;
        row.packed_keys = keys.data;
        row.key_scales = key_scales.data;
        row.visible = groups[r];
        row.scores = got.data + r * stride;
    }
    Device score_args(score_rows);
    ds::FastIndexView original{};
    original.rows = rows;
    original.ratio = 2;
    original.max_count = stride;
    original.pos = positions.data;
    original.k8 = keys.data;
    original.kscale = key_scales.data;
    original.scores = ref.data;
    original.qprep = query.data;
    ds::fast_index_scores_kernel<<<dim3(1, 1), ds::kXThreads>>>(original);
    check(cudaGetLastError(), "DeepSeek incumbent E4M3 scores");
    check(sel::enqueue_scores<sel::ScoreForm::weighted_fp8>(score_args.data, rows, stride, {0, 32, 128}, nullptr), "DeepSeek E4M3 scores");
    compare("DeepSeek E4M3 score bytes", ref, got);
    Device<unsigned> ref_ids(rows * budget, poison), got_ids(rows * budget, poison);
    Device<unsigned> ref_count(rows, poison), got_count(rows, poison);
    Device<std::uint8_t> unused_candidates(rows, 0U);
    ds::SelectView original_pick{rows, 2, stride, budget, 0, 1, ds::Pool::plain, positions.data, ref.data,
        unused_candidates.data, 1, nullptr, ref_ids.data, ref_count.data};
    std::vector<sel::PickRow> pick_rows(rows);
    for (unsigned r = 0; r < rows; ++r) {
        pick_rows[r].scores = got.data + r * stride;
        pick_rows[r].groups = groups[r];
        pick_rows[r].selected = got_ids.data + r * budget;
        pick_rows[r].count = got_count.data + r;
    }
    Device pick_args(pick_rows);
    ds::select_kernel<<<rows, 1024>>>(original_pick);
    check(cudaGetLastError(), "DeepSeek incumbent E4M3 pick");
    check(sel::enqueue_pick(pick_args.data, rows, {budget, 1, sel::KeyOrder::raw_float}, nullptr), "DeepSeek E4M3 pick");
    compare("DeepSeek E4M3 ids", ref_ids, got_ids);
    compare("DeepSeek E4M3 counts", ref_count, got_count);
}

template <sel::ScoreForm Form>
void glm_case(unsigned first) {
    constexpr unsigned rows = 11, stride = 1033, list_stride = glm::dsa_max_tokens;
    const unsigned groups = (first + rows) / 4U;
    Device<float> q(values(rows * 4096, 93)), weights(values(rows * 32, 21)), keys(values(stride * 128, 1008));
    Device<float> ref(rows * stride, score_poison), got(rows * stride, score_poison);
    Device<unsigned> ref_keys(rows * stride, poison), got_keys(rows * stride, poison), flags(1, 0U), got_flags(1, 0U);
    std::vector<sel::ScoreRow> score_rows(rows);
    for (unsigned r = 0; r < rows; ++r) {
        auto& row = score_rows[r];
        row.query = q.data + r * 4096;
        row.keys = keys.data;
        row.weights = weights.data + r * 32;
        row.visible = (first + r + 1U) / 4U;
        row.scores = got.data + r * stride;
        row.ordered = got_keys.data + r * stride;
        row.nonfinite = got_flags.data;
    }
    Device score_args(score_rows);
    if constexpr (Form == sel::ScoreForm::weighted_fma) {
        Device<float> scalar(rows * stride, score_poison);
        Device<unsigned> scalar_keys(rows * stride, poison);
        glm::index_scores<1><<<dim3((groups + 15U) / 16U, rows), 32>>>(
            keys.data, q.data, weights.data, scalar.data, scalar_keys.data, stride, first, flags.data);
        glm::index_fp32_tiled<<<dim3((groups + 15U) / 16U, (rows + 1U) / 2U), 128>>>(
            keys.data, q.data, weights.data, ref.data, ref_keys.data, stride, first, rows, flags.data);
        check(cudaGetLastError(), "GLM incumbent FP32 scores");
        compare("GLM scalar and tiled FP32 scores", scalar, ref);
        compare("GLM scalar and tiled FP32 ordered keys", scalar_keys, ref_keys);
    } else {
        constexpr int kind = Form == sel::ScoreForm::weighted_bf16x3 ? 2 : 0;
        glm::index_scores<kind><<<dim3((groups + 15U) / 16U, rows), 32>>>(
            keys.data, q.data, weights.data, ref.data, ref_keys.data, stride, first, flags.data);
    }
    check(cudaGetLastError(), "GLM incumbent scores");
    check(sel::enqueue_scores<Form>(score_args.data, rows, stride, {512, 32, 128}, nullptr), "GLM scores");
    compare("GLM float score bytes", ref, got);
    compare("GLM ordered score keys", ref_keys, got_keys);
    compare("GLM nonfinite flags", flags, got_flags);
    if constexpr (Form == sel::ScoreForm::weighted_bf16x3) {
        Device<float> tiled(rows * stride, score_poison);
        Device<unsigned> tiled_keys(rows * stride, poison);
        constexpr int shared_bytes = sizeof(glm::TiledIndexShared);
        check(cudaFuncSetAttribute(glm::tiled_index, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes), "GLM tiled smem");
        glm::tiled_index<<<dim3((groups + 31U) / 32U, (rows + 1U) / 2U), 128, shared_bytes>>>(
            keys.data, q.data, weights.data, tiled.data, tiled_keys.data, stride, first, rows, flags.data);
        check(cudaGetLastError(), "GLM incumbent tiled BF16x3");
        compare("GLM tiled BF16x3 scores", tiled, got);
        compare("GLM tiled BF16x3 ordered keys", tiled_keys, got_keys);
    }
    Device<unsigned> ref_ids(rows * list_stride, poison), streamed(rows * list_stride, poison), got_ids(rows * list_stride, poison);
    Device<unsigned> ref_count(rows, poison), stream_count(rows, poison), got_count(rows, poison);
    std::vector<sel::PickRow> pick_rows(rows);
    for (unsigned r = 0; r < rows; ++r) {
        auto& row = pick_rows[r];
        row.keys = got_keys.data + r * stride;
        row.groups = score_rows[r].visible;
        row.tail = (first + r + 1U) % 4U;
        row.selected = got_ids.data + r * list_stride;
        row.count = got_count.data + r;
    }
    Device pick_args(pick_rows);
    glm::select_rows<<<rows, 1024>>>(ref_keys.data, stride, first, ref_ids.data, ref_count.data);
    glm::streaming_index<<<rows, 256>>>(ref_keys.data, stride, first, streamed.data, stream_count.data);
    check(cudaGetLastError(), "GLM incumbent selection");
    check(sel::enqueue_pick(pick_args.data, rows, {512, 4, sel::KeyOrder::ordered}, nullptr), "GLM pick");
    compare("GLM whole-row ids", ref_ids, got_ids);
    compare("GLM streaming ids", streamed, got_ids);
    compare("GLM whole-row counts", ref_count, got_count);
    compare("GLM streaming counts", stream_count, got_count);
}

template <sel::ScoreForm Form>
void mm_case() {
    constexpr unsigned rows = 3, pos0 = 2047, end = pos0 + rows, blocks = 19, heads = 4;
    const auto input = values(end * 128, 109);
    std::vector<std::uint16_t> bf16(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) bf16[i] = strix::weight::scalar_float::bf16_encode(input[i]);
    Device keys(bf16);
    Device<float> q(values(rows * heads * 128, 7));
    Device seq(std::vector{mm::SeqView{0, rows, end, nullptr, nullptr, keys.data, pos0}});
    Device<unsigned> seq_ids(rows, 0U);
    Device<float> ref(rows * heads * blocks, score_poison), got(rows * heads * blocks, score_poison);
    std::vector<sel::ScoreRow> score_rows(rows);
    for (unsigned r = 0; r < rows; ++r) {
        auto& row = score_rows[r];
        row.query = q.data + r * heads * 128;
        row.bf16_keys = keys.data;
        row.visible = pos0 + r + 1U;
        row.scores = got.data + r * heads * blocks;
    }
    Device score_args(score_rows);
    if constexpr (Form == sel::ScoreForm::block_max)
        mm::index_scores_kernel<<<rows * blocks, 128>>>(seq.data, seq_ids.data, rows, blocks, 0, q.data, ref.data);
    else {
        constexpr int smem = (128 * mm::kKs4 + 32 * 32) * 16;
        check(cudaFuncSetAttribute(mm::index_scores_tiled_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "MiniMax tiled smem");
        Device tiles(std::vector{make_uint2(0, rows)});
        mm::index_scores_tiled_kernel<<<dim3(1, heads, blocks), 256, smem>>>(
            seq.data, seq_ids.data, tiles.data, blocks, 0, q.data, ref.data);
        check(cudaGetLastError(), "MiniMax incumbent tiled scores");
        check(cudaDeviceSynchronize(), "MiniMax tiled score completion");
    }
    check(cudaGetLastError(), "MiniMax incumbent scores");
    check(sel::enqueue_scores<Form>(score_args.data, rows, blocks, {16, heads, 128, 128}, nullptr), "MiniMax scores");
    compare("MiniMax score bytes", ref, got);
    Device<unsigned> ref_ids(rows * heads * 16, poison), got_ids(rows * heads * 16, poison);
    Device<unsigned> ref_count(rows * heads, poison), got_count(rows * heads, poison);
    std::vector<sel::PickRow> pick_rows(rows * heads);
    for (unsigned i = 0; i < rows * heads; ++i) {
        auto& row = pick_rows[i];
        row.scores = got.data + i * blocks;
        row.groups = (pos0 + i / heads + 128U) / 128U;
        row.selected = got_ids.data + i * 16;
        row.count = got_count.data + i;
    }
    Device pick_args(pick_rows);
    mm::select_kernel<<<1, 128>>>(seq.data, seq_ids.data, rows, blocks, ref.data, ref_ids.data, ref_count.data);
    check(cudaGetLastError(), "MiniMax incumbent selection");
    check(sel::enqueue_pick(pick_args.data, rows * heads, {16, 1, sel::KeyOrder::canonical_float, true}, nullptr), "MiniMax pick");
    compare("MiniMax ids", ref_ids, got_ids);
    compare("MiniMax counts", ref_count, got_count);
}

void shard_case() {
    const std::vector keys{7U, 9U, 9U, 7U, 10U, 5U, 9U, 3U};
    const std::vector ids{2U, 3U, 8U, 10U, 14U, 18U, 27U, 31U};
    Device staged_keys(keys), staged_ids(ids);
    Device<unsigned> survivors(6, poison), survivor_keys(6, poison);
    Device<unsigned> result(3, poison), result_keys(3, poison), count(1, poison);
    std::vector<sel::PickRow> shards(2);
    for (unsigned i = 0; i < 2; ++i) {
        shards[i].keys = staged_keys.data + 4U * i;
        shards[i].ids = staged_ids.data + 4U * i;
        shards[i].groups = 4;
        shards[i].selected = survivors.data + 3U * i;
        shards[i].selected_keys = survivor_keys.data + 3U * i;
    }
    Device local_args(shards);
    constexpr sel::PickSpec spec{3, 1, sel::KeyOrder::ordered};
    check(sel::enqueue_pick(local_args.data, 2, spec, nullptr), "shard-local candidates");
    sel::PickRow row;
    row.keys = survivor_keys.data;
    row.ids = survivors.data;
    row.groups = 6;
    row.selected = result.data;
    row.selected_keys = result_keys.data;
    row.count = count.data;
    Device args(std::vector{row});
    check(sel::enqueue_pick(args.data, 1, spec, nullptr), "shard join");
    compare("shard global IDs", select_oracle::pick(spec, static_cast<unsigned>(keys.size()), 0, {}, keys, ids).ids, result.read());
    compare("shard selected keys", std::vector{9U, 9U, 10U}, result_keys.read());
    compare("shard count", std::vector{3U}, count.read());
    std::puts("k5c sharded stable join PASS");
}
} // namespace k5c_fixture

int main() {
    using namespace k5c_fixture;
    qwen_case();
    ds_case();
    ds_fp8_case();
    std::puts("k5c DeepSeek CSA2 ids and scores PASS");
    glm_case<sel::ScoreForm::weighted_fma>(2046);
    glm_case<sel::ScoreForm::weighted_bf16>(4094);
    glm_case<sel::ScoreForm::weighted_bf16x3>(4094);
    std::puts("k5c GLM DSA ids and scores PASS");
    mm_case<sel::ScoreForm::block_max>();
    mm_case<sel::ScoreForm::block_max_tiled>();
    std::puts("k5c MiniMax MSA ids and scores PASS");
    shard_case();
}
