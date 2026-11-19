// Model-free device fixture of the FAST QSA form v2 (qsa::attention(..., mma = true) -> shared row-group core):
//   1. union byte-compare: the device union lists (tokens, row masks, counts) left in the workspace equal the host oracle
//      built with the same rowgroup_union.hpp functions (an exact invariant of the FAST form);
//   2. accuracy against the exact three-kernel attention() on the same inputs (FAST: reported, loose sanity bound);
//   3. ABAB timing of an 8,192-row prefill (history 0) in 1,024-row chunks, exact three kernels vs v2.
// Selections are shaped like select_kernel_body's output (ascending): all tokens while <= 512 complete blocks, else 512
// ascending 4-token blocks per row, drawn per 4-row tile from a shared base set with churn 0.12 (4-row union / per-row
// count ~1.4, the in-model 8K figure is 1.37), then the tail.
//   qwen_qsa_rowgroup_test <tables dir> [--no-timing]
#include "strix/cuda/qwen_qsa.hpp"

#include "strix/cuda/amd_math_tables.hpp"
#include "../../src/cuda/core/attention_rowgroup.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <vector>

namespace qsa = strix::cuda::qwen_qsa;
namespace am = strix::cuda::amdmath;
namespace rg = strix::cuda::core::rowgroup;

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
// Layout twin of the Qwen binding (qwen_qsa.cu QsaRowgroup): only Shape / carve are used here.
struct Layout { static constexpr unsigned D = qsa::kHeadDim, Heads = 12, Rows = 4, Cap = qsa::kMaxSelected; };
constexpr unsigned kR = Layout::Rows, kUnionCap = rg::Shape<Layout>::UnionCap;

std::mt19937 rng(20261118);
float uniform(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }
std::uint16_t to_bf16(float x) { std::uint32_t u; std::memcpy(&u, &x, 4); return static_cast<std::uint16_t>((u + 0x7fffU + ((u >> 16) & 1U)) >> 16); }
float bf16_value(float x) { std::uint32_t u = static_cast<std::uint32_t>(to_bf16(x)) << 16; float y; std::memcpy(&y, &u, 4); return y; }
template <class T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    ck(cudaMalloc(&d, std::max<std::size_t>(v.size(), 1) * sizeof(T)), "malloc");
    ck(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}

struct Inputs { std::uint16_t* kv = nullptr; float* q = nullptr; std::uint32_t* sel = nullptr; std::vector<std::uint32_t> host_sel; };
Inputs make_inputs(std::uint32_t rows, std::uint64_t first, float qscale, double churn) {
    const std::uint64_t positions = first + rows;
    std::vector<std::uint16_t> kv(positions * qsa::kKvWidth * 2ULL);
    for (auto& x : kv) x = to_bf16(uniform(-1.5F, 1.5F));
    std::vector<float> q(std::uint64_t{rows} * qsa::kQGateWidth);
    for (auto& x : q) x = bf16_value(uniform(-qscale, qscale));
    Inputs in;
    in.host_sel.assign(std::uint64_t{rows} * qsa::kMaxSelected, 0U);
    std::vector<std::uint32_t> base;
    for (std::uint32_t r = 0; r < rows; ++r) {
        const auto visible = static_cast<std::uint32_t>(first + r + 1), nb = visible / qsa::kRatio;
        auto* s = &in.host_sel[std::uint64_t{r} * qsa::kMaxSelected];
        if (nb <= qsa::kTopK) { for (std::uint32_t i = 0; i < visible; ++i) s[i] = i; continue; }
        if (r % kR == 0U || base.empty()) {  // the tile's shared base set
            base.resize(nb);
            for (std::uint32_t i = 0; i < nb; ++i) base[i] = i;
            for (std::uint32_t i = 0; i < qsa::kTopK; ++i) std::swap(base[i], base[i + rng() % (nb - i)]);
            base.resize(qsa::kTopK);
        }
        std::set<std::uint32_t> blocks;
        for (const auto b : base) if (b < nb && std::uniform_real_distribution<double>(0, 1)(rng) >= churn) blocks.insert(b);
        while (blocks.size() < qsa::kTopK) blocks.insert(static_cast<std::uint32_t>(rng() % nb));
        std::uint32_t i = 0;
        for (const auto b : blocks) for (std::uint32_t o = 0; o < qsa::kRatio; ++o) s[i++] = b * qsa::kRatio + o;
        for (std::uint32_t t = nb * qsa::kRatio; t < visible; ++t) s[i++] = t;
    }
    in.kv = upload(kv); in.q = upload(q); in.sel = upload(in.host_sel);
    return in;
}
void free_inputs(Inputs& in) { cudaFree(in.kv); cudaFree(in.q); cudaFree(in.sel); }

void run(const Inputs& in, std::uint32_t n, std::uint64_t first, std::uint32_t c, float* scores, float* out, const am::Tables& t, bool mma) {
    qsa::attention(in.q + std::uint64_t{c} * qsa::kQGateWidth, n, first + c, in.sel + std::uint64_t{c} * qsa::kMaxSelected, in.kv, scores,
        out + std::uint64_t{c} * qsa::kAttnWidth, t, nullptr, mma);
}

int failures = 0;
// Host union of tile `tile` (same functions as the device union_body; ascending input -> rank form).
void host_union(const Inputs& in, std::uint32_t rows, std::uint64_t first, unsigned tile, std::vector<std::uint32_t>& tok, std::vector<std::uint8_t>& mask) {
    std::vector<std::uint32_t> lists(kR * Layout::Cap, 0U);
    std::vector<std::uint16_t> pre(kR * (Layout::Cap + 1U), 0U);
    unsigned n[kR];
    for (unsigned r = 0; r < kR; ++r) {
        const unsigned row = tile * kR + r;
        n[r] = row < rows ? qsa::selected_count(first + row) : 0U;
        std::copy_n(&in.host_sel[std::uint64_t{row} * qsa::kMaxSelected], n[r], lists.begin() + r * Layout::Cap);
    }
    unsigned total = 0;
    for (unsigned r = 0; r < kR; ++r) {
        unsigned run = 0;
        for (unsigned j = 0; j < n[r]; ++j) {
            pre[r * (Layout::Cap + 1U) + j] = static_cast<std::uint16_t>(run);
            run += rg::owned(lists.data(), Layout::Cap, n, r, lists[r * Layout::Cap + j]) ? 1U : 0U;
        }
        pre[r * (Layout::Cap + 1U) + n[r]] = static_cast<std::uint16_t>(run);
        total += run;
    }
    tok.assign(total, 0U); mask.assign(total, 0U);
    for (unsigned r = 0; r < kR; ++r)
        for (unsigned j = 0; j < n[r]; ++j) {
            if (!rg::owned(lists.data(), Layout::Cap, n, r, lists[r * Layout::Cap + j])) continue;
            unsigned m = 0;
            const unsigned at = rg::rank_and_mask<kR>(lists.data(), Layout::Cap, n, pre.data(), Layout::Cap + 1U, r, j, m);
            tok[at] = lists[r * Layout::Cap + j]; mask[at] = static_cast<std::uint8_t>(m);
        }
}

void compare(std::uint32_t rows, std::uint64_t first, float qscale, const am::Tables& tables) {
    auto in = make_inputs(rows, first, qscale, 0.12);
    const std::uint64_t n_out = std::uint64_t{rows} * qsa::kAttnWidth;
    const std::size_t ws_bytes = std::uint64_t{rows} * qsa::kHeads * qsa::kMaxSelected * 4;
    float *scores, *oa, *ob;
    ck(cudaMalloc(&scores, ws_bytes), "scores");
    ck(cudaMalloc(&oa, n_out * 4), "oa");
    ck(cudaMalloc(&ob, n_out * 4), "ob");
    ck(cudaMemset(ob, 0xff, n_out * 4), "poison");
    run(in, rows, first, 0, scores, oa, tables, false);
    run(in, rows, first, 0, scores, ob, tables, true);  // leaves the union lists in `scores`
    ck(cudaDeviceSynchronize(), "run");
    // 1. union byte-compare
    const unsigned tiles = (rows + kR - 1U) / kR;
    std::vector<std::uint8_t> ws(rg::workspace_bytes<Layout>(tiles));
    ck(cudaMemcpy(ws.data(), scores, ws.size(), cudaMemcpyDeviceToHost), "dl union");
    const auto u = rg::carve<Layout>(ws.data(), tiles);
    std::uint64_t union_bad = 0, union_entries = 0, per_row = 0;
    for (unsigned t = 0; t < tiles; ++t) {
        std::vector<std::uint32_t> tok; std::vector<std::uint8_t> mask;
        host_union(in, rows, first, t, tok, mask);
        union_entries += tok.size();
        bool ok = u.count[t] == tok.size();
        for (std::size_t i = 0; ok && i < tok.size(); ++i)
            ok = u.tok[std::size_t{t} * kUnionCap + i] == tok[i] && u.mask[std::size_t{t} * kUnionCap + i] == mask[i];
        union_bad += ok ? 0U : 1U;
    }
    for (std::uint32_t r = 0; r < rows; ++r) per_row += qsa::selected_count(first + r);
    // 2. accuracy vs the exact path
    std::vector<float> a(n_out), b(n_out);
    ck(cudaMemcpy(a.data(), oa, n_out * 4, cudaMemcpyDeviceToHost), "dl a");
    ck(cudaMemcpy(b.data(), ob, n_out * 4, cudaMemcpyDeviceToHost), "dl b");
    double max_ref = 0.0, max_diff = 0.0, sum_diff = 0.0;
    std::uint64_t nonfinite = 0;
    for (std::uint64_t i = 0; i < n_out; ++i) {
        if (!std::isfinite(b[i])) { ++nonfinite; continue; }
        const double d = std::fabs(static_cast<double>(a[i]) - b[i]);
        max_ref = std::max(max_ref, std::fabs(static_cast<double>(a[i])));
        max_diff = std::max(max_diff, d);
        sum_diff += d;
    }
    const bool ok = union_bad == 0U && nonfinite == 0U && max_diff <= 0.02 * max_ref;
    failures += ok ? 0 : 1;
    std::printf("QSA_RG compare rows %u first %llu qscale %.1f: union %s (%llu of %u tiles differ, inflation %.3f) | vs exact max %.3e "
                "(ref max %.3e) mean %.3e nonfinite %llu: %s\n", rows, static_cast<unsigned long long>(first), qscale,
        union_bad ? "DIFF" : "BYTE-EQUAL", static_cast<unsigned long long>(union_bad), tiles,
        per_row ? static_cast<double>(union_entries) * kR / static_cast<double>(per_row) : 0.0, max_diff, max_ref,
        sum_diff / static_cast<double>(n_out), static_cast<unsigned long long>(nonfinite), ok ? "PASS" : "FAIL");
    cudaFree(scores); cudaFree(oa); cudaFree(ob); free_inputs(in);
}

void timing(std::uint32_t rows, const am::Tables& tables) {
    const std::uint32_t chunk = 1024;
    auto in = make_inputs(rows, 0, 2.0F, 0.12);
    float *scores, *out;
    ck(cudaMalloc(&scores, std::uint64_t{chunk} * qsa::kHeads * qsa::kMaxSelected * 4), "scores");
    ck(cudaMalloc(&out, std::uint64_t{rows} * qsa::kAttnWidth * 4), "out");
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const auto all = [&](bool mma) {
        for (std::uint32_t c = 0; c < rows; c += chunk) run(in, std::min(chunk, rows - c), 0, c, scores, out, tables, mma);
    };
    all(false); all(true); ck(cudaDeviceSynchronize(), "warm");
    float sum[2] = {0.0F, 0.0F};
    for (int pair = 0; pair < 3; ++pair)
        for (int mode = 0; mode < 2; ++mode) {
            cudaEventRecord(e0);
            all(mode == 1);
            cudaEventRecord(e1); ck(cudaEventSynchronize(e1), "time");
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            sum[mode] += ms;
            std::printf("QSA_RG timing pair %d %s: %.3f ms per layer (%u rows, 1024-row chunks)\n", pair, mode ? "rowgroup_v2" : "exact_three", ms, rows);
        }
    std::printf("QSA_RG timing mean exact_three %.3f ms rowgroup_v2 %.3f ms per layer; x12 layers: %.1f ms vs %.1f ms\n", sum[0] / 3.0F,
        sum[1] / 3.0F, 4.0F * sum[0], 4.0F * sum[1]);
    cudaFree(scores); cudaFree(out); free_inputs(in);
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::strcmp(argv[2], "--no-timing") != 0)) {
        std::fprintf(stderr, "usage: qwen_qsa_rowgroup_test <tables dir> [--no-timing]\n"); return 1;
    }
    const auto tables = am::load_tables(argv[1]);
    for (const std::uint64_t first : {0ULL, 9000ULL})
        for (const std::uint32_t rows : {1U, 17U, 1024U}) compare(rows, first, 2.0F, tables);
    compare(1024, 9000, 6.0F, tables);   // peaked softmax
    compare(17, 2040, 2.0F, tables);     // crosses the dense -> selected boundary inside one tile
    compare(1024, 7052, 2.0F, tables);   // the last chunk of the 8,076-token cell
    if (argc == 2) timing(8192, tables);
    std::printf(failures ? "QSA_RG FAIL (%d)\n" : "QSA_RG PASS\n", failures);
    return failures ? 1 : 0;
}
