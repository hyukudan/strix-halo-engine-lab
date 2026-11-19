// Model-free device fixture of the exact QSA fused attention: qsa::attention_fused against the three-kernel exact form
// (attention_scores + attention_softmax + attention_values) on the same random inputs. attn_out must be equal byte for
// byte (the fused output buffer is poisoned first). Selection is built like select_kernel's output: all tokens while
// <= 512 complete blocks, else 512 random 4-token blocks in random order + the tail. Then ABAB timing of an 8,192-row
// prefill (history 0) in 1,024-row chunks, the production chunk.
//   qwen_qsa_fused_test <tables dir> [--no-timing]
#include "strix/cuda/qwen_qsa.hpp"

#include "strix/cuda/amd_math_tables.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace qsa = strix::cuda::qwen_qsa;
namespace am = strix::cuda::amdmath;

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
std::mt19937 rng(20261117);
float uniform(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }
std::uint16_t to_bf16(float x) { std::uint32_t u; std::memcpy(&u, &x, 4); return static_cast<std::uint16_t>((u + 0x7fffU + ((u >> 16) & 1U)) >> 16); }
float bf16_value(float x) { std::uint32_t u = static_cast<std::uint32_t>(to_bf16(x)) << 16; float y; std::memcpy(&y, &u, 4); return y; }
template <class T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    ck(cudaMalloc(&d, std::max<std::size_t>(v.size(), 1) * sizeof(T)), "malloc");
    ck(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}

struct Inputs { std::uint16_t* kv = nullptr; float* q = nullptr; std::uint32_t* sel = nullptr; };
Inputs make_inputs(std::uint32_t rows, std::uint64_t first, float qscale) {
    const std::uint64_t positions = first + rows;
    std::vector<std::uint16_t> kv(positions * qsa::kKvWidth * 2ULL);
    for (auto& x : kv) x = to_bf16(uniform(-1.5F, 1.5F));
    std::vector<float> q(std::uint64_t{rows} * qsa::kQGateWidth);
    for (auto& x : q) x = bf16_value(uniform(-qscale, qscale));
    std::vector<std::uint32_t> sel(std::uint64_t{rows} * qsa::kMaxSelected, 0U), blocks;
    for (std::uint32_t r = 0; r < rows; ++r) {
        const auto visible = static_cast<std::uint32_t>(first + r + 1), nb = visible / qsa::kRatio;
        auto* s = &sel[std::uint64_t{r} * qsa::kMaxSelected];
        if (nb <= qsa::kTopK) { for (std::uint32_t i = 0; i < visible; ++i) s[i] = i; continue; }
        blocks.resize(nb);
        for (std::uint32_t i = 0; i < nb; ++i) blocks[i] = i;
        for (std::uint32_t i = 0; i < qsa::kTopK; ++i) std::swap(blocks[i], blocks[i + rng() % (nb - i)]);
        for (std::uint32_t i = 0; i < qsa::kTopK; ++i)
            for (std::uint32_t o = 0; o < qsa::kRatio; ++o) s[i * qsa::kRatio + o] = blocks[i] * qsa::kRatio + o;
        for (std::uint32_t t = nb * qsa::kRatio; t < visible; ++t) s[qsa::kTopK * qsa::kRatio + t - nb * qsa::kRatio] = t;
    }
    return {upload(kv), upload(q), upload(sel)};
}
void free_inputs(Inputs& in) { cudaFree(in.kv); cudaFree(in.q); cudaFree(in.sel); }

void three(const Inputs& in, std::uint32_t n, std::uint64_t first, std::uint32_t c, float* scores, float* out, const am::Tables& t) {
    const auto* q = in.q + std::uint64_t{c} * qsa::kQGateWidth;
    const auto* s = in.sel + std::uint64_t{c} * qsa::kMaxSelected;
    auto* o = out + std::uint64_t{c} * qsa::kAttnWidth;
    qsa::attention_scores(q, n, first + c, s, in.kv, scores, nullptr);
    qsa::attention_softmax(n, first + c, scores, t, nullptr);
    qsa::attention_values(q, n, first + c, s, in.kv, scores, o, t, nullptr);
}
void fused(const Inputs& in, std::uint32_t n, std::uint64_t first, std::uint32_t c, float* out, const am::Tables& t) {
    qsa::attention_fused(in.q + std::uint64_t{c} * qsa::kQGateWidth, n, first + c, in.sel + std::uint64_t{c} * qsa::kMaxSelected, in.kv,
        out + std::uint64_t{c} * qsa::kAttnWidth, t, nullptr);
}

int failures = 0;
void compare(std::uint32_t rows, std::uint64_t first, float qscale, const am::Tables& tables) {
    auto in = make_inputs(rows, first, qscale);
    const std::uint64_t n_out = std::uint64_t{rows} * qsa::kAttnWidth;
    float *scores, *oa, *ob;
    ck(cudaMalloc(&scores, std::uint64_t{rows} * qsa::kHeads * qsa::kMaxSelected * 4), "scores");
    ck(cudaMalloc(&oa, n_out * 4), "oa");
    ck(cudaMalloc(&ob, n_out * 4), "ob");
    ck(cudaMemset(ob, 0xff, n_out * 4), "poison");
    three(in, rows, first, 0, scores, oa, tables);
    fused(in, rows, first, 0, ob, tables);
    ck(cudaDeviceSynchronize(), "run");
    std::vector<std::uint32_t> a(n_out), b(n_out);
    ck(cudaMemcpy(a.data(), oa, n_out * 4, cudaMemcpyDeviceToHost), "dl a");
    ck(cudaMemcpy(b.data(), ob, n_out * 4, cudaMemcpyDeviceToHost), "dl b");
    std::uint64_t diff = 0, first_bad = n_out;
    for (std::uint64_t i = 0; i < n_out; ++i)
        if (a[i] != b[i]) { if (!diff) first_bad = i; ++diff; }
    std::printf("QSA_FUSED compare rows %u first %llu qscale %.1f: %s (%llu of %llu words differ%s)\n", rows,
        static_cast<unsigned long long>(first), qscale, diff ? "DIFF" : "BYTE-EQUAL", static_cast<unsigned long long>(diff),
        static_cast<unsigned long long>(n_out), diff ? ", see first_bad" : "");
    if (diff) {
        std::printf("  first_bad word %llu (row %llu col %llu): three %08x fused %08x\n", static_cast<unsigned long long>(first_bad),
            static_cast<unsigned long long>(first_bad / qsa::kAttnWidth), static_cast<unsigned long long>(first_bad % qsa::kAttnWidth),
            a[first_bad], b[first_bad]);
        ++failures;
    }
    cudaFree(scores); cudaFree(oa); cudaFree(ob); free_inputs(in);
}

void timing(std::uint32_t rows, const am::Tables& tables) {
    const std::uint32_t chunk = 1024;
    auto in = make_inputs(rows, 0, 2.0F);
    float *scores, *out;
    ck(cudaMalloc(&scores, std::uint64_t{chunk} * qsa::kHeads * qsa::kMaxSelected * 4), "scores");
    ck(cudaMalloc(&out, std::uint64_t{rows} * qsa::kAttnWidth * 4), "out");
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const auto run = [&](int mode) {
        for (std::uint32_t c = 0; c < rows; c += chunk) {
            const auto n = std::min(chunk, rows - c);
            if (mode == 0) three(in, n, 0, c, scores, out, tables);
            else fused(in, n, 0, c, out, tables);
        }
    };
    run(0); run(1); ck(cudaDeviceSynchronize(), "warm");
    float sum[2] = {0.0F, 0.0F};
    for (int pair = 0; pair < 3; ++pair)
        for (int mode = 0; mode < 2; ++mode) {
            cudaEventRecord(e0);
            run(mode);
            cudaEventRecord(e1); ck(cudaEventSynchronize(e1), "time");
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            sum[mode] += ms;
            std::printf("QSA_FUSED timing pair %d %s: %.3f ms per layer (%u rows, 1024-row chunks)\n", pair, mode ? "fused" : "three", ms, rows);
        }
    std::printf("QSA_FUSED timing mean three %.3f ms fused %.3f ms per layer; x12 layers: three %.1f ms fused %.1f ms\n", sum[0] / 3.0F,
        sum[1] / 3.0F, 4.0F * sum[0], 4.0F * sum[1]);
    cudaFree(scores); cudaFree(out); free_inputs(in);
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::strcmp(argv[2], "--no-timing") != 0)) {
        std::fprintf(stderr, "usage: qwen_qsa_fused_test <tables dir> [--no-timing]\n"); return 1;
    }
    const auto tables = am::load_tables(argv[1]);
    for (const std::uint64_t first : {0ULL, 9000ULL})
        for (const std::uint32_t rows : {1U, 17U, 1000U, 1024U}) compare(rows, first, 2.0F, tables);
    compare(1024, 9000, 6.0F, tables);   // peaked softmax
    compare(17, 2040, 2.0F, tables);     // crosses the 512-block boundary (count 2048 -> selection starts; count 2051 max)
    compare(1024, 7052, 2.0F, tables);   // the last chunk of the 8,076-token cell
    if (argc == 2) timing(8192, tables);
    std::printf(failures ? "QSA_FUSED FAIL (%d)\n" : "QSA_FUSED PASS\n", failures);
    return failures ? 1 : 0;
}
