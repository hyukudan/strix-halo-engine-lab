// QSA index block scores: each bound schedule against its frozen kernel, byte for byte, on random and adversarial inputs.
// Unwritten entries keep a sentinel. --bench: median time of each form at long positions.
//   qwen_qsa_score_tree_test [--bench]
#include "strix/cuda/qwen_qsa.hpp"
#include "../cuda_core/select_qwen_baseline.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace q = strix::cuda::qwen_qsa;
namespace frozen = select_qwen_baseline;

#define CK(x)                                                                                 \
    do {                                                                                      \
        cudaError_t e_ = (x);                                                                 \
        if (e_ != cudaSuccess) {                                                              \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            std::exit(2);                                                                     \
        }                                                                                     \
    } while (0)

static std::uint32_t mix(std::uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return static_cast<std::uint32_t>(x);
}
static float value(std::uint64_t i, std::uint32_t seed, bool adversarial) {
    const std::uint32_t h = mix(i * 0x9E3779B97F4A7C15ULL + seed);
    float v = (static_cast<float>(h & 0xFFFFFF) / 8388608.0F - 1.0F) * 1.5F;
    if (adversarial) {
        switch ((h >> 24) % 64U) {
            case 0: v = 0.0F; break;
            case 1: v = -0.0F; break;
            case 2: v = std::numeric_limits<float>::infinity(); break;
            case 3: v = -std::numeric_limits<float>::infinity(); break;
            case 4: v = std::numeric_limits<float>::quiet_NaN(); break;
            case 5: v = 3.0e38F; break;
            case 6: v = 1.0e-39F; break;
            case 7: v = -1.0e-41F; break;
            default: break;
        }
    }
    return v;
}

struct Case {
    std::uint32_t rows;
    std::uint64_t p0;
    bool adversarial;
};

static double run(const Case& c, bool timing, std::size_t& bad) {
    const auto groups = static_cast<std::uint32_t>((c.p0 + c.rows) / q::kRatio);
    const std::uint32_t max_blocks = groups + 7U;
    const std::size_t qn = static_cast<std::size_t>(c.rows) * q::kIndexQWidth, kn = static_cast<std::size_t>(max_blocks) * q::kIndexDim;
    const std::size_t sn = static_cast<std::size_t>(c.rows) * max_blocks;
    std::vector<float> hq(qn), hk(kn);
    for (std::size_t i = 0; i < qn; ++i) hq[i] = value(i, 101U, c.adversarial && i % 7 == 0);
    for (std::size_t i = 0; i < kn; ++i) hk[i] = value(i, 202U, c.adversarial && i % 5 == 0);
    float *dq = nullptr, *dk = nullptr, *ds = nullptr;
    CK(cudaMalloc(&dq, qn * 4));
    CK(cudaMalloc(&dk, kn * 4));
    CK(cudaMalloc(&ds, sn * 4));
    CK(cudaMemcpy(dq, hq.data(), qn * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dk, hk.data(), kn * 4, cudaMemcpyHostToDevice));
    std::vector<float> out[2];
    double ms_form[2] = {0, 0};
    const q::ScoreForm forms[2] = {q::ScoreForm::shuffle, q::ScoreForm::tree};
    for (int f = 0; f < 2; ++f) {
        CK(cudaMemset(ds, 0x7F, sn * 4));
        q::block_scores_form(forms[f], dq, c.rows, c.p0, dk, ds, max_blocks, nullptr);
        CK(cudaDeviceSynchronize());
        out[f].resize(sn);
        CK(cudaMemcpy(out[f].data(), ds, sn * 4, cudaMemcpyDeviceToHost));
    }
    if (!timing) {
        std::vector<float> old(sn);
        for (int f = 0; f < 2; ++f) {
            CK(cudaMemset(ds, 0x7F, sn * sizeof(float)));
            if (groups > q::kTopK) {
                if (forms[f] == q::ScoreForm::tree) {
                    frozen::block_score_tree_kernel<<<dim3((c.rows + 63U) / 64U, (groups + 255U) / 256U), 256>>>(
                        dq, c.rows, c.p0, dk, groups, ds, max_blocks);
                } else {
                    const auto waves = ((c.rows + 3U) / 4U) * ((groups + 31U) / 32U);
                    frozen::block_score_kernel<<<(waves + 7U) / 8U, 256>>>(dq, c.rows, c.p0, dk, groups, ds, max_blocks);
                }
                CK(cudaGetLastError());
            }
            CK(cudaMemcpy(old.data(), ds, sn * sizeof(float), cudaMemcpyDeviceToHost));
            for (std::size_t i = 0; i < sn; ++i)
                bad += std::memcmp(&old[i], &out[f][i], sizeof(float)) != 0;
        }
    }
    // 16-B alignment admission: the tree form with query rows / pooled keys one float off a 16-B boundary must give the
    // shuffle scorer's bytes (scalar fallback), not a misaligned float4 load
    std::size_t misaligned = 0;
    {
        float *mq = nullptr, *mk = nullptr;
        CK(cudaMalloc(&mq, (qn + 1U) * 4));
        CK(cudaMalloc(&mk, (kn + 1U) * 4));
        CK(cudaMemcpy(mq + 1, hq.data(), qn * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(mk + 1, hk.data(), kn * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(ds, 0x7F, sn * 4));
        q::block_scores_form(q::ScoreForm::tree, mq + 1, c.rows, c.p0, mk + 1, ds, max_blocks, nullptr);
        CK(cudaDeviceSynchronize());
        std::vector<float> mo(sn);
        CK(cudaMemcpy(mo.data(), ds, sn * 4, cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < sn; ++i) misaligned += std::memcmp(&out[0][i], &mo[i], 4) != 0;
        CK(cudaFree(mq));
        CK(cudaFree(mk));
    }
    bad += misaligned;
    if (timing) {
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        for (int rep = 0; rep < 2; ++rep)  // ABAB: shuffle, tree, shuffle, tree; keep each form's last median
            for (int f = 0; f < 2; ++f) {
                std::vector<float> ms(5);
                for (float& m : ms) {
                    CK(cudaEventRecord(e0));
                    q::block_scores_form(forms[f], dq, c.rows, c.p0, dk, ds, max_blocks, nullptr);
                    CK(cudaEventRecord(e1));
                    CK(cudaEventSynchronize(e1));
                    CK(cudaEventElapsedTime(&m, e0, e1));
                }
                std::sort(ms.begin(), ms.end());
                ms_form[f] = ms[2];
            }
    }
    std::size_t diff = 0, written = 0;
    for (std::size_t i = 0; i < sn; ++i) {
        diff += std::memcmp(&out[0][i], &out[1][i], 4) != 0;
        std::uint32_t b;
        std::memcpy(&b, &out[0][i], 4);
        written += b != 0x7F7F7F7FU;
    }
    bad += diff;
    double pairs = 0;
    for (std::uint32_t r = 0; r < c.rows; ++r) {
        const std::uint64_t complete = (c.p0 + r + 1U) / q::kRatio;
        if (complete > q::kTopK) pairs += static_cast<double>(complete);
    }
    bad += written != static_cast<std::size_t>(pairs);
    std::printf("rows=%u p0=%llu adv=%d written=%zu mismatch=%zu misaligned_mismatch=%zu", c.rows, static_cast<unsigned long long>(c.p0),
                c.adversarial ? 1 : 0, written, diff, misaligned);
    if (timing)
        std::printf(" shuffle_ms=%.3f tree_ms=%.3f ratio=%.3f pairs/ns shuffle %.2f tree %.2f", ms_form[0], ms_form[1],
                    ms_form[1] / ms_form[0], pairs / (ms_form[0] * 1e6), pairs / (ms_form[1] * 1e6));
    std::printf("\n");
    std::fflush(stdout);
    CK(cudaFree(dq));
    CK(cudaFree(dk));
    CK(cudaFree(ds));
    return ms_form[1];
}

static bool check_pick() {
    constexpr std::uint64_t first = 2047;
    constexpr unsigned rows = 10, stride = 520;
    const auto inf = std::numeric_limits<float>::infinity();
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> input(rows * stride, 0.0F);
    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned i = 0; i < stride; ++i) {
            // The late rows exercise signed-zero cutoff ties, nonfinite exclusions, and underfill.
            input[r * stride + i] = i % 2U ? -0.0F : 0.0F;
            if (r >= 6 && i % 5U == 0U) input[r * stride + i] = i % 2U ? -inf : nan;
            if (r >= 6 && i % 53U == 1U) input[r * stride + i] = inf;
            if (r == rows - 1U) input[r * stride + i] = i % 2U ? -inf : nan;
        }
    }
    constexpr std::size_t score_bytes = rows * stride * sizeof(float);
    constexpr std::size_t id_count = rows * q::kMaxSelected;
    constexpr std::size_t id_bytes = id_count * sizeof(std::uint32_t);
    float *shared = nullptr, *old = nullptr;
    std::uint32_t *shared_ids = nullptr, *old_ids = nullptr;
    CK(cudaMalloc(&shared, score_bytes));
    CK(cudaMalloc(&old, score_bytes));
    CK(cudaMalloc(&shared_ids, id_bytes));
    CK(cudaMalloc(&old_ids, id_bytes));
    CK(cudaMemcpy(shared, input.data(), score_bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(old, input.data(), score_bytes, cudaMemcpyHostToDevice));
    CK(cudaMemset(shared_ids, 0xA5, id_bytes));
    CK(cudaMemset(old_ids, 0xA5, id_bytes));
    q::select_rows(shared, stride, rows, first, shared_ids, nullptr);
    frozen::select_kernel<<<rows, frozen::kSelectThreads>>>(old, stride, first, old_ids);
    CK(cudaGetLastError());
    std::vector<float> shared_scores(rows * stride), old_scores(rows * stride);
    std::vector<std::uint32_t> shared_tokens(id_count), old_tokens(id_count);
    CK(cudaMemcpy(shared_scores.data(), shared, score_bytes, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(old_scores.data(), old, score_bytes, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(shared_tokens.data(), shared_ids, id_bytes, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(old_tokens.data(), old_ids, id_bytes, cudaMemcpyDeviceToHost));
    CK(cudaFree(shared));
    CK(cudaFree(old));
    CK(cudaFree(shared_ids));
    CK(cudaFree(old_ids));
    return shared_tokens == old_tokens && std::memcmp(shared_scores.data(), old_scores.data(), score_bytes) == 0;
}

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    std::size_t bad = 0;
    const Case checks[] = {{5, 2046, false}, {5, 2047, false}, {65, 2048, true},
                           {1024, 1024, false}, {1024, 2000, true}, {77, 5000, true}, {1, 9000, false}, {1000, 30001, true},
                           {130, 2047, false}, {64, 2049, true}, {1024, 65536 - 1024, true}};
    for (const Case& c : checks) run(c, false, bad);
    const bool picks = check_pick();
    std::printf("QSA affine picker %s\n", picks ? "EXACT" : "MISMATCH");
    if (bench) {
        const Case b[] = {{1024, 32768 - 1024, false}, {1024, 131072 - 1024, false}, {1024, 262144 - 1024, false}};
        for (const Case& c : b) run(c, true, bad);
    }
    std::printf("qwen_qsa_score_tree_test: %s (%zu mismatching floats)\n", bad == 0 ? "IDENTICAL" : "MISMATCH", bad);
    return bad == 0 && picks ? 0 : 1;
}
