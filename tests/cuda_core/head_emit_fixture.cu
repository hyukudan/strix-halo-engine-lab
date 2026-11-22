// Compile locally. Run only on the native test node.
#include "cuda/core/head_emit.cuh"
#include "strix/cuda_logprob_partials.hpp"
#include "strix/ds41_dspark.hpp"

// Both private draft kernels are the incumbent oracles. Include the Qwen TU and the
// unmodified, fixture-only DS41 kernel slice generated from its engine TU by K11c.cmake.
#include "cuda/models/qwen/qwen_mtp_draft.cu"
namespace markov {
using namespace strix::ds41::dspark;
#include "k11c_markov_incumbent.cuh"
}  // namespace markov

#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
namespace head = strix::cuda::core::head;
namespace ha = strix::head_argmax;
namespace greedy = strix::cuda::core::head_argmax;
namespace qwen = strix::cuda::qwen_mtp_draft;

void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

void require(bool ok, const char* compare) {
    if (!ok) throw std::runtime_error(compare);
}

template <class T>
struct Device {
    explicit Device(std::size_t size) { check(cudaMalloc(&data, size * sizeof(T))); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    ~Device() { cudaFree(data); }
    T* data = nullptr;

    void upload(std::span<const T> src) {
        check(cudaMemcpy(data, src.data(), src.size_bytes(), cudaMemcpyHostToDevice));
    }
    std::vector<T> download(std::size_t count) const {
        std::vector<T> out(count);
        check(cudaMemcpy(out.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
        return out;
    }
};

bool same(float a, float b) {
    return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
}

__global__ void incumbent_lse(const strix::LogprobPartial* partials, head::RowLse* out) {
    const auto& p = partials[threadIdx.x];
    out[threadIdx.x] = {p.max, p.sum, p.max + logf(p.sum), p.nonfinite};
}

struct IncumbentPick {
    ha::Pick* out;
    __device__ void operator()(std::uint32_t row, const ha::Pick& p) const { out[row] = p; }
};

void check_greedy(head::Rows v, std::span<const float> logits) {
    Device<std::uint32_t> ids(v.rows), second(v.rows), old_ids(v.rows), old_second(v.rows);
    Device<float> gap(v.rows), old_gap(v.rows);
    const auto chunks = ha::default_chunks(v.count);
    Device<ha::Pick> partial(std::size_t{v.rows} * chunks), pick(v.rows), old_pick(v.rows);
    Device<ha::Pick2> partial2(std::size_t{v.rows} * chunks);
    const greedy::View incumbent{v.logits, v.stride, v.count, v.rows, v.first, chunks, partial.data};

    check(head::enqueue(v, head::EmitIds{ids.data, partial.data, true}, nullptr));
    check(greedy::pick_rows(incumbent, greedy::EmitIndexOrNone{old_ids.data}, nullptr));
    const auto strict = ids.download(v.rows);
    require(strict == old_ids.download(v.rows), "ids status (Inkling and Qwen flash-next)");

    check(head::enqueue<head::Launch::pdl>(v, head::EmitIds{ids.data, partial.data}, nullptr));
    check(greedy::pick_rows<greedy::Launch::pdl>(incumbent, greedy::EmitIndex{old_ids.data}, nullptr));
    require(ids.download(v.rows) == old_ids.download(v.rows), "ids PDL (Qwen 27B)");

    check(head::enqueue(v, head::EmitPick{pick.data, partial.data}, nullptr));
    check(greedy::pick_rows(incumbent, IncumbentPick{old_pick.data}, nullptr));
    const auto choices = pick.download(v.rows), old_choices = old_pick.download(v.rows);
    const auto id = ids.download(v.rows);
    for (std::uint32_t r = 0; r < v.rows; ++r) {
        const auto expected = ha::reference(logits.subspan(std::size_t{r} * v.stride, v.count), v.first);
        const auto& a = choices[r];
        const auto& b = old_choices[r];
        require(strict[r] == (expected.nonfinite == 0U ? expected.index : ha::kNone), "nonfinite refusal");
        require(a.index == id[r] && a.index == b.index && a.index == expected.index &&
                a.nonfinite == b.nonfinite && a.nonfinite == expected.nonfinite &&
                same(a.value, b.value) && same(a.value, expected.value), "pick payload (GLM and DS41)");
    }

    check(head::enqueue<head::Launch::pdl>(v, head::EmitTop2{ids.data, second.data, gap.data, partial2.data}, nullptr));
    check(greedy::pick2_rows<greedy::Launch::pdl>(incumbent, partial2.data,
        greedy::EmitIndex2{old_ids.data, old_second.data, old_gap.data}, nullptr));
    require(ids.download(v.rows) == old_ids.download(v.rows) && id == old_ids.download(v.rows) &&
            second.download(v.rows) == old_second.download(v.rows), "top2 ids PDL (Qwen 27B)");
    const auto actual_gap = gap.download(v.rows), expected_gap = old_gap.download(v.rows);
    for (std::uint32_t r = 0; r < v.rows; ++r) require(same(actual_gap[r], expected_gap[r]), "top2 gap bits");
}

void check_scoring(head::Rows v, std::span<const float> logits) {
    Device<head::Candidate> candidates(std::size_t{v.rows} * head::kMaxCandidates);
    Device<head::RowLse> lse(v.rows), old_lse(v.rows), combined_lse(v.rows);
    Device<strix::LogprobPartial> old(v.rows);
    strix::enqueue_cuda_logprob_partials({v.logits, v.rows, v.count, v.stride, v.first,
                                         strix::kLogprobPartialTop, nullptr, old.data}, nullptr);
    incumbent_lse<<<1, v.rows>>>(old.data, old_lse.data);
    check(cudaGetLastError());
    check(head::enqueue(v, head::EmitLse{lse.data}, nullptr));
    const auto scoring = old.download(v.rows);
    const auto actual_lse = lse.download(v.rows), expected_lse = old_lse.download(v.rows);

    for (const auto top : {1U, 2U, strix::kLogprobPartialTop, head::kMaxCandidates}) {
        check(head::enqueue(v, head::EmitCandidates{candidates.data, top, combined_lse.data}, nullptr));
        const auto combined = combined_lse.download(v.rows);
        for (std::uint32_t r = 0; r < v.rows; ++r) {
            const auto& b = expected_lse[r];
            for (const auto& a : {actual_lse[r], combined[r]}) {
                require(same(a.maximum, b.maximum) && same(a.sum, b.sum) && same(a.lse, b.lse) &&
                        a.nonfinite == b.nonfinite, "LSE incumbent bits");
            }
        }
        const auto both = candidates.download(std::size_t{v.rows} * top);
        check(head::enqueue(v, head::EmitCandidates{candidates.data, top}, nullptr));
        const auto actual = candidates.download(both.size());
        for (std::uint32_t r = 0; r < v.rows; ++r) {
            std::array<head::Candidate, head::kMaxCandidates> buffer;
            auto expected = std::span(buffer).first(top);
            head::reference_candidates(logits.subspan(std::size_t{r} * v.stride, v.count), v.first, expected);
            for (std::uint32_t k = 0; k < top; ++k) {
                const auto index = std::size_t{r} * top + k;
                const auto& a = actual[index];
                const auto& b = expected[k];
                require(a.id == b.id && same(a.value, b.value) && a.id == both[index].id &&
                        same(a.value, both[index].value), "candidate host oracle bits/order");
                if (k < scoring[r].top_count) {
                    require(a.id == scoring[r].top_id[k] && same(a.value, scoring[r].top_value[k]),
                            "candidate incumbent bits/order");
                }
            }
        }
    }
}

void check_draft_confidence(head::Rows v) {
    Device<std::uint32_t> map(v.count), old(std::size_t{v.rows} * 2U), old_v2(std::size_t{v.rows} * 2U), feed(v.rows);
    Device<head::DraftChoice> choice(v.rows);
    std::vector<std::uint32_t> ids(v.count);
    std::iota(ids.rbegin(), ids.rend(), 10U);  // intentionally non-monotonic in the shortlist order
    map.upload(ids);
    check(head::enqueue(v, head::EmitDraftConfidence{choice.data}, nullptr));
    for (std::uint32_t r = 0; r < v.rows; ++r) {
        const auto* row = v.logits + std::size_t{r} * v.stride;
        qwen::select_kernel<<<1, 1024>>>(row, v.count, map.data, old.data + r * 2U, nullptr);
        qwen::select_v2_kernel<<<1, 1024>>>(row, v.count, map.data, old_v2.data + r * 2U, feed.data + r);
    }
    check(cudaGetLastError());
    const auto a = choice.download(v.rows);
    const auto b = old.download(std::size_t{v.rows} * 2U), b2 = old_v2.download(b.size()), f = feed.download(v.rows);
    require(b == b2, "draft incumbent v1/v2 bits");
    for (std::uint32_t r = 0; r < v.rows; ++r) {
        const auto token = a[r].id == ha::kNone ? 0U : ids[a[r].id - v.first];
        require(token == b[r * 2U] && token == f[r] &&
                std::bit_cast<std::uint32_t>(a[r].confidence) == b[r * 2U + 1U], "draft confidence bits");
    }
}

void check_markov(head::Rows v, const Device<std::uint16_t>& zero_head, const float* packed_logits) {
    // Markov's head receives packed rows. Its zero weights leave each fixture logit unchanged.
    Device<std::uint32_t> tokens(v.rows + 1U), out(v.rows), best_ids(markov::markov_blocks);
    Device<markov::DraftParams> params(1);
    Device<float> markov_state(std::size_t{v.rows} * markov::markov_rank), best_values(markov::markov_blocks);
    Device<ha::Pick> partial(std::size_t{v.rows} * ha::default_chunks(v.count));
    check(cudaMemset(tokens.data, 0, (v.rows + 1U) * sizeof(std::uint32_t)));
    check(cudaMemset(params.data, 0, sizeof(markov::DraftParams)));
    markov::HeadView view{};
    view.logits_core = packed_logits;
    view.core = v.count;
    view.prm = params.data;
    view.embed = zero_head.data;
    view.head = zero_head.data;
    view.tokens = tokens.data;
    view.markov = markov_state.data;
    view.best_values = best_values.data;
    view.best_ids = best_ids.data;
    for (std::uint32_t r = 0; r < v.rows; ++r) {
        markov::markov_pass_kernel<<<markov::markov_blocks, 256>>>(view, r);
        markov::markov_pick_kernel<<<1, markov::markov_blocks>>>(view, r);
    }
    check(cudaGetLastError());
    v.first = 0;
    check(head::enqueue(v, head::EmitIds{out.data, partial.data}, nullptr));
    const auto a = out.download(v.rows), b = tokens.download(v.rows + 1U);
    require(std::equal(a.begin(), a.end(), b.begin() + 1), "DS41 DFlash Markov ids");
}

std::vector<float> shape_logits(std::uint32_t vocab, std::uint32_t stride, std::uint32_t rows) {
    std::vector<float> logits(std::size_t{stride} * rows, 1000.0F);
    for (std::uint32_t r = 0; r < rows; ++r) {
        auto* row = logits.data() + std::size_t{r} * stride;
        for (std::uint32_t i = 0; i < vocab; ++i) {
            auto bits = (i + 1U) * 0x9e3779b9U ^ ((r + 1U) * 0x85ebca6bU);
            bits ^= bits >> 16U;
            row[i] = static_cast<float>(bits % 1024U) / 64.0F - 12.0F;
            // Enough entries ahead of the Kth lane winner to force the pool's overflow path for K >= 20.
            if (r == 1U && i % 256U < 8U) row[i] += 16.0F;
        }
        row[17U + r] = row[vocab / 2U + r] = row[vocab - 1U] = 26.0F;
    }
    return logits;
}

void recorded_shapes(std::uint32_t vocab) {
    constexpr std::uint32_t max_rows = 8;
    const std::uint32_t stride = vocab + 17U;
    const auto logits = shape_logits(vocab, stride, max_rows);
    Device<float> device_logits(logits.size()), packed_logits(std::size_t{vocab} * max_rows);
    device_logits.upload(logits);
    check(cudaMemcpy2D(packed_logits.data, std::size_t{vocab} * sizeof(float), device_logits.data,
                      std::size_t{stride} * sizeof(float), std::size_t{vocab} * sizeof(float), max_rows,
                      cudaMemcpyDeviceToDevice));
    const auto weights = std::size_t{vocab} * markov::markov_rank;
    Device<std::uint16_t> zero_head(weights);
    check(cudaMemset(zero_head.data, 0, weights * sizeof(std::uint16_t)));
    for (std::uint32_t rows = 1; rows <= max_rows; ++rows) {
        const head::Rows v{device_logits.data, rows, vocab, stride, 31U};
        check_greedy(v, logits);
        check_scoring(v, logits);
        check_draft_confidence(v);
        check_markov(v, zero_head, packed_logits.data);
        std::printf("k11c vocab%u-rows%u PASS\n", vocab, rows);
    }
}

void edges() {
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    const std::array logits{-0.0F, 0.0F, 0.0F, inf, nan, -inf, nan, -inf, inf, nan, -inf, nan};
    Device<float> device_logits(logits.size());
    device_logits.upload(logits);
    const head::Rows v{device_logits.data, 2, 6, 6, ha::kNone - 6U};
    check_greedy(v, logits);
    check_scoring(v, logits);
    // Host-side guards enqueue no kernels, including when a zero-row view has null pointers.
    Device<head::Candidate> candidate(1);
    require(head::enqueue({}, head::EmitLse{nullptr}, nullptr) == cudaSuccess, "zero rows");
    require(head::enqueue(v, head::EmitCandidates{candidate.data, 65U}, nullptr) == cudaErrorInvalidValue,
            "candidate width guard");
    const head::Rows bad_ids{device_logits.data, 1, 6, 0, ha::kNone - 5U};
    require(head::enqueue(bad_ids, head::EmitCandidates{candidate.data, 1U}, nullptr) == cudaErrorInvalidValue,
            "reserved ID guard");
    std::puts("k11c edges PASS");
}
}  // namespace

int main() {
    try {
        recorded_shapes(151936);
        recorded_shapes(201088);
        edges();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "k11c FAIL %s\n", e.what());
        return 1;
    }
}
