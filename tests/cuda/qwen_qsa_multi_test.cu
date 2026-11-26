// Model-free device test of the multi-segment QSA core: qsa::layer_core_multi over several sessions' segments of one
// window must give exactly what per-segment qsa::layer_core gives (attention output, normed/rotated q/k/index rows, the
// stored KV / raw index rows, the pooled keys and the host pooled-valid counts), on random BF16 data, at mixed positions
// below and above the pooled-selection regime (> 2048 positions), several long sessions per window, two consecutive
// windows on one captured graph (the second refreshes the descriptor and reuses the pooled caches).
//   qwen_qsa_multi_test <tables dir>
#include "strix/cuda/qwen_qsa.hpp"
#include "../../src/cuda/models/qwen/qwen_qsa_segments.hpp"
#include "strix/env_flags.hpp"
#include "../cuda_core/select_qwen_baseline.cuh"

#include "strix/cuda/amd_math_tables.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace qsa = strix::cuda::qwen_qsa;
namespace am = strix::cuda::amdmath;
namespace frozen = select_qwen_baseline;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
std::mt19937 rng(20261102);
float uniform(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }
std::uint16_t to_bf16(float x) { std::uint32_t u; std::memcpy(&u, &x, 4); return static_cast<std::uint16_t>(u >> 16); }
float bf16_value(float x) { std::uint32_t u; std::memcpy(&u, &x, 4); u &= 0xffff0000U; float y; std::memcpy(&y, &u, 4); return y; }

template <class T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    ck(cudaMalloc(&d, std::max<std::size_t>(v.size(), 1) * sizeof(T)), "malloc");
    ck(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
template <class T> std::vector<T> download(const T* d, std::size_t n) {
    std::vector<T> v(n);
    ck(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}
std::vector<std::uint16_t> bf16_random(std::size_t n, float scale) {
    std::vector<std::uint16_t> v(n);
    for (auto& x : v) x = to_bf16(uniform(-scale, scale));
    return v;
}
std::vector<float> bf16_valued(std::size_t n, float scale) {
    std::vector<float> v(n);
    for (auto& x : v) x = bf16_value(uniform(-scale, scale));
    return v;
}

constexpr std::uint64_t P = 8192;  // positions per session

// One session's layer state (two identical copies are made per case: per-segment and multi).
struct State {
    std::uint16_t *kv = nullptr, *raw = nullptr;
    float* pooled = nullptr;
    std::uint32_t valid = 0;
};
State make_state(const std::vector<std::uint16_t>& kv, const std::vector<std::uint16_t>& raw) {
    State s;
    s.kv = upload(kv);
    s.raw = upload(raw);
    ck(cudaMalloc(&s.pooled, P / 4 * qsa::kIndexDim * sizeof(float)), "pooled");
    ck(cudaMemset(s.pooled, 0, P / 4 * qsa::kIndexDim * sizeof(float)), "pooled clear");
    return s;
}

struct Rows {
    std::uint16_t* hidden; float *q_gate, *k, *v, *iq, *ik, *out;
    qsa::RowBuffers view(std::uint32_t r) const {
        return {hidden + std::uint64_t{r} * qsa::kHidden, q_gate + std::uint64_t{r} * qsa::kQGateWidth, k + std::uint64_t{r} * qsa::kKvWidth,
                v + std::uint64_t{r} * qsa::kKvWidth, iq + std::uint64_t{r} * qsa::kIndexQWidth, ik + std::uint64_t{r} * qsa::kIndexDim,
                out + std::uint64_t{r} * qsa::kAttnWidth};
    }
};
Rows make_rows(const std::vector<std::uint16_t>& hidden, const std::vector<float>& q, const std::vector<float>& k,
               const std::vector<float>& v, std::uint32_t rows) {
    Rows r{};
    r.hidden = upload(hidden);
    r.q_gate = upload(q);
    r.k = upload(k);
    r.v = upload(v);
    ck(cudaMalloc(&r.iq, std::uint64_t{rows} * qsa::kIndexQWidth * 4), "iq");
    ck(cudaMalloc(&r.ik, std::uint64_t{rows} * qsa::kIndexDim * 4), "ik");
    ck(cudaMalloc(&r.out, std::uint64_t{rows} * qsa::kAttnWidth * 4), "out");
    return r;
}

int failures = 0;
template <class T> void expect_same(const T* a, const T* b, std::size_t n, const std::string& what) {
    const auto x = download(a, n), y = download(b, n);
    if (std::memcmp(x.data(), y.data(), n * sizeof(T)) != 0) {
        std::size_t first = 0;
        while (first < n && std::memcmp(&x[first], &y[first], sizeof(T)) == 0) ++first;
        std::printf("    DIFF %s (first at element %zu of %zu)\n", what.c_str(), first, n);
        ++failures;
    }
}

void sentinel(const qsa::Workspace& ws, std::uint32_t rows, cudaStream_t stream) {
    ck(cudaMemsetAsync(ws.block_scores, 0x7F, std::size_t{rows} * ws.max_blocks * sizeof(float), stream), "score sentinel");
    ck(cudaMemsetAsync(ws.selected, 0xA5, std::size_t{rows} * qsa::kMaxSelected * sizeof(std::uint32_t), stream), "selection sentinel");
}

void frozen_selection(const float* query, const std::vector<State>& states,
                      const std::vector<std::pair<std::uint64_t, std::uint32_t>>& segments,
                      const qsa::Workspace& ws, cudaStream_t stream) {
    frozen::SegmentMap map;
    map.n = static_cast<unsigned>(segments.size());
    unsigned max_rows = 0, max_groups = 0;
    for (unsigned i = 0; i < map.n; ++i) {
        const auto [first, rows] = segments[i];
        const auto groups = static_cast<unsigned>((first + rows) / qsa::kRatio);
        map.first_row[i + 1U] = map.first_row[i] + rows;
        map.position[i] = first;
        map.pooled[i] = states[i].pooled;
        map.scored[i] = groups;
        map.score_begin[i + 1U] = map.score_begin[i];
        if (groups > qsa::kTopK) {
            max_rows = std::max(max_rows, rows);
            max_groups = std::max(max_groups, groups);
            map.score_begin[i + 1U] += ((rows + 3U) / 4U) * ((groups + 31U) / 32U);
        }
    }
    const auto waves = map.score_begin[map.n];
    if (waves) {
        if (strix::envcfg::flag("STRIX_QRTX_QSA_SCORE_TREE", true)) {
            frozen::block_score_tree_multi_kernel<<<dim3((max_rows + 63U) / 64U, (max_groups + 255U) / 256U, map.n),
                256, 0, stream>>>(query, map, ws.block_scores, ws.max_blocks);
        } else {
            frozen::block_score_multi_kernel<<<(waves + 7U) / 8U, 256, 0, stream>>>(query, map, ws.block_scores, ws.max_blocks);
        }
        ck(cudaGetLastError(), "frozen cohort score");
    }
    for (unsigned i = 0; i < map.n; ++i) {
        const auto row = map.first_row[i], count = map.first_row[i + 1U] - row;
        frozen::select_kernel<<<count, frozen::kSelectThreads, 0, stream>>>(
            ws.block_scores + std::size_t{row} * ws.max_blocks, ws.max_blocks, map.position[i],
            ws.selected + std::size_t{row} * qsa::kMaxSelected);
        ck(cudaGetLastError(), "frozen picker");
    }
}

template <class T>
void refresh(T* dest, const std::vector<T>& data, cudaStream_t stream) {
    ck(cudaMemcpyAsync(dest, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice, stream), "refresh graph rows");
}

void free_rows(const Rows& rows) {
    for (auto* p : {static_cast<void*>(rows.hidden), static_cast<void*>(rows.q_gate), static_cast<void*>(rows.k),
                    static_cast<void*>(rows.v), static_cast<void*>(rows.iq), static_cast<void*>(rows.ik),
                    static_cast<void*>(rows.out)})
        ck(cudaFree(p), "free rows");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: qwen_qsa_multi_test <tables dir>\n"); return 1; }
    const auto tables = am::load_tables(argv[1]);
    cudaStream_t stream;
    ck(cudaStreamCreate(&stream), "stream");
    // weights (BF16 indexer projections, F32 norms) and a random rope table (any values: the test compares two paths)
    const auto* wq = upload(bf16_random(std::uint64_t{qsa::kIndexQWidth} * qsa::kHidden, 0.02F));
    const auto* wk = upload(bf16_random(std::uint64_t{qsa::kIndexDim} * qsa::kHidden, 0.02F));
    std::vector<float> qn(qsa::kHeadDim), kn(qsa::kHeadDim), iqn(qsa::kIndexDim), ikn(qsa::kIndexDim);
    for (auto* v : {&qn, &kn, &iqn, &ikn}) for (auto& x : *v) x = uniform(0.5F, 1.5F);
    const qsa::LayerWeights w{wq, wk, upload(qn), upload(kn), upload(iqn), upload(ikn)};
    std::vector<float2> cs(P * qsa::kRotaryPairs);
    for (auto& c : cs) { const float a = uniform(0.0F, 6.2831853F); c = make_float2(std::cos(a), std::sin(a)); }
    const auto* rope = upload(cs);
    constexpr unsigned workspace_rows = 9;  // include unused score and ID rows in the byte comparison
    auto ws_a = qsa::allocate_workspace(workspace_rows, P), ws_b = qsa::allocate_workspace(workspace_rows, P),
        ws_g = qsa::allocate_workspace(workspace_rows, P);
    qsa::ReplaySegmentMap* dmap = nullptr;
    qsa::ReplaySegmentMap* hmap = nullptr;
    qsa::SegmentMap* legacy_dmap = nullptr;
    qsa::SegmentMap* legacy_hmap = nullptr;
    ck(cudaMalloc(&dmap, sizeof(*dmap)), "device map");
    ck(cudaMallocHost(&hmap, sizeof(*hmap)), "pinned map");
    ck(cudaMalloc(&legacy_dmap, sizeof(*legacy_dmap)), "legacy device map");
    ck(cudaMallocHost(&legacy_hmap, sizeof(*legacy_hmap)), "legacy pinned map");
    const bool expect_graph = !strix::envcfg::flag("STRIX_QSA_SCORE_WAVE", false);
    unsigned packed_replays = 0, legacy_replays = 0;

    // Cases include a dense-only cohort, dense holes between scored segments and all eight entries.
    const std::vector<std::vector<std::pair<std::uint64_t, std::uint32_t>>> cases = {
        {{64, 1}, {2051, 2}, {4097, 3}, {2600, 1}},
        {{4096, 2}, {63, 1}, {2999, 3}},
        {{5000, 1}, {10, 1}, {2100, 1}, {3000, 1}, {64, 1}, {4500, 1}, {2048, 1}, {7000, 1}},
        {{2047, 4}, {2049, 4}},
        {{3, 2}, {5, 3}, {700, 3}},
    };
    for (std::size_t c = 0; c < cases.size(); ++c) {
        auto segs = cases[c];
        const auto n = static_cast<std::uint32_t>(segs.size());
        std::vector<State> a, b, g;
        for (const auto& sg : segs) {
            auto kv = bf16_random(P * 1024, 1.0F), raw = bf16_random(P * 128, 1.0F);
            a.push_back(make_state(kv, raw));
            b.push_back(make_state(kv, raw));
            g.push_back(make_state(kv, raw));
        }
        std::uint32_t rows = 0;
        std::vector<std::uint32_t> seg_rows;
        for (const auto& sg : segs) {
            rows += sg.second;
            seg_rows.push_back(sg.second);
        }
        qsa::MultiShape gshape{};
        const bool graph_ok = qsa::graph_shape(seg_rows.data(), n, ws_g, gshape);
        if (graph_ok != expect_graph) {
            std::fprintf(stderr, "unexpected graph admission for case %zu\n", c);
            return 3;
        }
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graph_exec = nullptr;
        Rows rg{};
        for (int window = 0; window < 2; ++window) {
            const auto hidden = bf16_random(std::uint64_t{rows} * qsa::kHidden, 1.0F);
            const auto q = bf16_valued(std::uint64_t{rows} * qsa::kQGateWidth, 2.0F);
            const auto k = bf16_valued(std::uint64_t{rows} * qsa::kKvWidth, 2.0F);
            const auto v = bf16_valued(std::uint64_t{rows} * qsa::kKvWidth, 2.0F);
            const Rows ra = make_rows(hidden, q, k, v, rows), rb = make_rows(hidden, q, k, v, rows);
            if (window == 0) {
                rg = make_rows(hidden, q, k, v, rows);
            } else {
                refresh(rg.hidden, hidden, stream);
                refresh(rg.q_gate, q, stream);
                refresh(rg.k, k, stream);
                refresh(rg.v, v, stream);
            }
            // per segment
            std::uint32_t off = 0;
            for (std::uint32_t i = 0; i < n; ++i) {
                const qsa::LayerState st{a[i].kv, a[i].raw, a[i].pooled, &a[i].valid};
                qsa::layer_core(ra.view(off), segs[i].second, segs[i].first, w, st, rope, tables, ws_a, stream);
                off += segs[i].second;
            }
            // multi
            std::vector<qsa::SegmentState> ms;
            for (std::uint32_t i = 0; i < n; ++i)
                ms.push_back(qsa::SegmentState{segs[i].second, segs[i].first, qsa::LayerState{b[i].kv, b[i].raw, b[i].pooled, &b[i].valid}});
            sentinel(ws_b, workspace_rows, stream);
            sentinel(ws_g, workspace_rows, stream);
            qsa::layer_core_multi(rb.view(0), ms.data(), n, w, rope, tables, ws_b, stream);
            // graph step: fixed graph shape + device map (else the eager multi core keeps the sessions in step)
            std::vector<qsa::SegmentState> gs;
            for (std::uint32_t i = 0; i < n; ++i)
                gs.push_back(qsa::SegmentState{segs[i].second, segs[i].first, qsa::LayerState{g[i].kv, g[i].raw, g[i].pooled, &g[i].valid}});
            const auto replay = [&](auto* host_map, auto* device_map) {
                qsa::prepare_replay(gs.data(), n, ws_g, *host_map);
                if (window == 0) {
                    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
                    ck(cudaMemcpyAsync(device_map, host_map, sizeof(*host_map), cudaMemcpyHostToDevice, stream), "map copy");
                    qsa::layer_core_device(rg.view(0), device_map, gshape, w, rope, tables, ws_g, stream);
                    ck(cudaStreamEndCapture(stream, &graph), "end capture");
                    ck(cudaGraphInstantiate(&graph_exec, graph, 0), "instantiate");
                }
                ck(cudaGraphLaunch(graph_exec, stream), "replay");
            };
            if (!graph_ok) {
                qsa::layer_core_multi(rg.view(0), gs.data(), n, w, rope, tables, ws_g, stream);
            } else if (c == 1) {
                replay(legacy_hmap, legacy_dmap);
                ++legacy_replays;
            } else {
                replay(hmap, dmap);
                ++packed_replays;
            }
            sentinel(ws_a, workspace_rows, stream);
            frozen_selection(ra.iq, a, segs, ws_a, stream);
            ck(cudaStreamSynchronize(stream), "case");
            const int before = failures;
            const std::string tag = "case " + std::to_string(c) + " window " + std::to_string(window);
            expect_same(ra.out, rb.out, std::uint64_t{rows} * qsa::kAttnWidth, tag + " attention output");
            expect_same(ra.q_gate, rb.q_gate, std::uint64_t{rows} * qsa::kQGateWidth, tag + " q");
            expect_same(ra.k, rb.k, std::uint64_t{rows} * qsa::kKvWidth, tag + " k");
            expect_same(ra.iq, rb.iq, std::uint64_t{rows} * qsa::kIndexQWidth, tag + " index q");
            expect_same(ra.ik, rb.ik, std::uint64_t{rows} * qsa::kIndexDim, tag + " index k");
            expect_same(ra.out, rg.out, std::uint64_t{rows} * qsa::kAttnWidth, tag + " graph attention output");
            // Multi can fall back to per-segment calls, which reuse scratch starting at row zero.
            if (graph_ok) {
                const auto score_count = std::size_t{workspace_rows} * ws_a.max_blocks;
                const auto id_count = std::size_t{workspace_rows} * qsa::kMaxSelected;
                expect_same(ws_a.block_scores, ws_b.block_scores, score_count, tag + " masked scores");
                expect_same(ws_a.selected, ws_b.selected, id_count, tag + " selected IDs");
                expect_same(ws_a.block_scores, ws_g.block_scores, score_count, tag + " graph masked scores");
                expect_same(ws_a.selected, ws_g.selected, id_count, tag + " graph selected IDs");
            }
            for (std::uint32_t i = 0; i < n; ++i) {
                const std::string s = tag + " segment " + std::to_string(i);
                expect_same(a[i].kv, b[i].kv, P * 1024, s + " kv");
                expect_same(a[i].raw, b[i].raw, P * 128, s + " raw index");
                expect_same(a[i].pooled, b[i].pooled, P / 4 * qsa::kIndexDim, s + " pooled keys");
                expect_same(a[i].kv, g[i].kv, P * 1024, s + " graph kv");
                expect_same(a[i].raw, g[i].raw, P * 128, s + " graph raw index");
                expect_same(a[i].pooled, g[i].pooled, P / 4 * qsa::kIndexDim, s + " graph pooled keys");
                if (a[i].valid != g[i].valid) { std::printf("    DIFF %s graph pooled valid %u vs %u\n", s.c_str(), a[i].valid, g[i].valid); ++failures; }
                if (a[i].valid != b[i].valid) { std::printf("    DIFF %s pooled valid %u vs %u\n", s.c_str(), a[i].valid, b[i].valid); ++failures; }
            }
            std::printf("%s (%u segments, %u rows, pooled valid", tag.c_str(), n, rows);
            for (std::uint32_t i = 0; i < n; ++i) std::printf(" %u", b[i].valid);
            std::printf(") graph=%s %s\n", graph_ok ? "device-map" : "eager", failures == before ? "BIT-IDENTICAL" : "DIFFER");
            free_rows(ra);
            free_rows(rb);
            for (auto& sg : segs) sg.first += sg.second;  // the next window continues every session
        }
        if (graph_ok) {
            ck(cudaGraphExecDestroy(graph_exec), "destroy exec");
            ck(cudaGraphDestroy(graph), "destroy graph");
        }
        free_rows(rg);
        for (auto& s : a) { cudaFree(s.kv); cudaFree(s.raw); cudaFree(s.pooled); }
        for (auto& s : b) { cudaFree(s.kv); cudaFree(s.raw); cudaFree(s.pooled); }
        for (auto& s : g) { cudaFree(s.kv); cudaFree(s.raw); cudaFree(s.pooled); }
    }
    ck(cudaFree(dmap), "free device map");
    ck(cudaFreeHost(hmap), "free pinned map");
    ck(cudaFree(legacy_dmap), "free legacy device map");
    ck(cudaFreeHost(legacy_hmap), "free legacy pinned map");
    ck(cudaStreamDestroy(stream), "destroy stream");
    qsa::free_workspace(ws_a);
    qsa::free_workspace(ws_b);
    qsa::free_workspace(ws_g);
    if (failures == 0 && expect_graph)
        std::printf("QSA cohort frozen picks, masks and scratch EXACT (%u packed, %u legacy replays)\n", packed_replays, legacy_replays);
    std::printf("%s (%d differences)\n", failures == 0 ? "QSA MULTI PASS" : "QSA MULTI FAIL", failures);
    return failures == 0 ? 0 : 3;
}
