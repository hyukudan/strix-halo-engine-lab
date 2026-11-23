// No-model identity fixture and bench of the R1 MoE decode route + plan forms (moe_core/r1_route.hpp):
//   ROUTED  gate_up with an inline checked route (no route launch) vs the route launch + gate_up + down_combine:
//           ids, weights, status, activations and outputs bitwise, rows 1 and 3, two route policies (selection order /
//           float keys, ascending ids / ordered keys), a failed row (non-finite logit).
//   UNION   expert-union gate_up / down vs the rows launches: activations and outputs bitwise for 2..8 rows, routes with
//           distinct, shared and repeated experts, a failed row, an invalid id and a row the step block marks invalid;
//           the inline-routed union vs route launch + union.
//   TIME    (unless --check-only) graph-captured decode visits on synthetic top-8 experts (hidden 2304, intermediate
//           1152, 128 experts; 16 rotating route sets so expert bytes come from DRAM):
//           r1: route launch + gate_up + down (no PDL) | the same with PDL R1 entries | routed gate_up + down (PDL);
//           union: rows launches vs union launches at R = 4 and 8, uniform and correlated routes (distinct experts).
// Synthetic shapes and policies only: no model data. RESULT PASS requires every identity line EQUAL.

#include "cuda/core/common.cuh"
#include "moe_core/microscaled_r1.hpp"
#include "moe_core/route_wave_kernel.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

namespace {

namespace packed = strix::weight::microscaled;
namespace r1 = strix::moe_core::microscaled_r1;
namespace route = strix::moe_core::r1_route;
namespace grouped = strix::moe_core::device_grouped;
namespace moe = strix::moe;
using packed::MatrixFormat;

constexpr unsigned kHidden = 2304, kInter = 1152, kExperts = 128, kTop = 8, kSets = 16, kMax = route::kMaxUnionRows;

// Two checked route policies of the R1 decode shape (data of this bench, not a model).
struct SelectionPolicy {
    static constexpr moe::RoutePolicy value{moe::RouteScore::sigmoid, true, moe::RouteKey::float_serial,
                                            moe::RouteOrder::selection, kExperts, kTop, 1U, 1U, true, true, 1e-20F, 2.5F};
};
struct AscendingPolicy {
    static constexpr moe::RoutePolicy value{moe::RouteScore::sigmoid, true, moe::RouteKey::ordered_bits,
                                            moe::RouteOrder::ascending_id, kExperts, kTop, 1U, 1U, true, true, 1e-20F, 1.0F};
};
struct Math {
    __host__ __device__ static float exp(float v) { return ::expf(v); }
};
using Traits = moe::RouteWaveTraits<true>;

bool g_fail = false;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::printf("CUDA error %s: %s\nRESULT FAIL\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct Rng {
    std::uint64_t s;
    std::uint32_t operator()() {
        s ^= s << 13U;
        s ^= s >> 7U;
        s ^= s << 17U;
        return static_cast<std::uint32_t>(s >> 11U);
    }
    float uniform(float lo, float hi) { return lo + (hi - lo) * static_cast<float>((*this)() & 0xffffffU) / 16777216.0F; }
};

template <class T>
T* device(std::size_t n) {
    T* p = nullptr;
    check(cudaMalloc(&p, std::max<std::size_t>(n, 1U) * sizeof(T)), "malloc");
    return p;
}
template <class T>
T* upload(const std::vector<T>& host) {
    T* p = device<T>(host.size());
    check(cudaMemcpy(p, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return p;
}
template <class T>
std::vector<T> download(const T* p, std::size_t n) {
    std::vector<T> host(n);
    check(cudaMemcpy(host.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return host;
}
template <class T>
void fill(T* p, std::size_t n, T value) {
    check(cudaMemcpy(p, std::vector<T>(n, value).data(), n * sizeof(T), cudaMemcpyHostToDevice), "fill");
}
template <class T>
bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

template <MatrixFormat F>
struct Pack {
    static constexpr unsigned ratio = F == MatrixFormat::nvfp4 ? 2U : 1U, block = F == MatrixFormat::nvfp4 ? 16U : 32U;
    static packed::PackedMatrix matrix(Rng& rng, unsigned rows, unsigned cols, packed::RowOrder order) {
        std::vector<std::uint8_t> codes(std::size_t{rows} * cols / ratio), scales(std::size_t{rows} * cols / block);
        for (auto& c : codes) {
            c = static_cast<std::uint8_t>(rng());
            if constexpr (F == MatrixFormat::mxfp8) {
                if ((c & 0x7fU) == 0x7fU) c = static_cast<std::uint8_t>(c & 0xf0U);  // no E4M3 NaN
            }
        }
        for (auto& s : scales) {
            s = F == MatrixFormat::nvfp4 ? static_cast<std::uint8_t>(0x30U + rng() % 16U)
                                         : static_cast<std::uint8_t>(120U + rng() % 12U);
        }
        return {upload(codes), upload(scales), rows, cols, F == MatrixFormat::nvfp4 ? 1.0F / 64.0F : 1.0F, order};
    }
};

// Device state of one format: experts, inputs and two independent sets of route / output buffers (arms A and B).
struct Arm {
    unsigned *ids, *status;
    float *weights, *output;
    std::uint16_t* activation;
};

struct Bench {
    grouped::ExpertWeights* matrices;
    std::uint64_t expert_bytes;
    float *input, *logits, *bias;  // input [kMax][hidden]; logits [kSets][kMax][experts]
    Arm a, b;
    strix::exec::StepBlock* step;
};

template <MatrixFormat F>
Bench make() {
    Rng rng{F == MatrixFormat::nvfp4 ? 0x5eed41ULL : 0x5eed43ULL};
    std::vector<grouped::ExpertWeights> experts(kExperts);
    for (auto& e : experts) {
        e.gate_up = Pack<F>::matrix(rng, 2U * kInter, kHidden, packed::RowOrder::interleaved_halves);
        e.down = Pack<F>::matrix(rng, kHidden, kInter, packed::RowOrder::contiguous);
    }
    Bench b{};
    b.matrices = upload(experts);
    b.expert_bytes = (3ULL * kInter * kHidden) / Pack<F>::ratio + (3ULL * kInter * kHidden) / Pack<F>::block;
    std::vector<float> input(std::size_t{kMax} * kHidden), logits(std::size_t{kSets} * kMax * kExperts), bias(kExperts);
    for (auto& x : input) x = rng.uniform(-1.0F, 1.0F);
    for (auto& x : bias) x = rng.uniform(-0.25F, 0.25F);
    // Even sets: independent rows (uniform routes). Odd sets: one base row plus small per-row noise (rows of a verify
    // window share most experts).
    for (unsigned set = 0; set < kSets; ++set) {
        std::vector<float> base(kExperts);
        for (auto& x : base) x = rng.uniform(-4.0F, 4.0F);
        for (unsigned row = 0; row < kMax; ++row) {
            for (unsigned e = 0; e < kExperts; ++e) {
                logits[(std::size_t{set} * kMax + row) * kExperts + e] =
                    set % 2U == 0U ? rng.uniform(-4.0F, 4.0F) : base[e] + rng.uniform(-0.35F, 0.35F);
            }
        }
    }
    b.input = upload(input);
    b.logits = upload(logits);
    b.bias = upload(bias);
    for (Arm* arm : {&b.a, &b.b}) {
        arm->ids = device<unsigned>(kMax * kTop);
        arm->status = device<unsigned>(kMax);
        arm->weights = device<float>(kMax * kTop);
        arm->output = device<float>(std::size_t{kMax} * kHidden);
        arm->activation = device<std::uint16_t>(std::size_t{kMax} * kTop * kInter);
    }
    b.step = device<strix::exec::StepBlock>(1U);
    return b;
}

r1::Args args_of(const Bench& b, const Arm& arm) {
    return {b.matrices, arm.ids, arm.weights, arm.status, b.input, arm.activation, arm.output, kExperts, kHidden, kInter,
            30.0F, 30.0F};
}

// The rows launches of `rows` routed rows, in launches of at most r1::kMaxRows rows (row_offsets layout).
template <MatrixFormat F, class Entry = r1::EntryComplete>
void rows_launches(r1::Args args, unsigned rows, cudaStream_t stream) {
    for (unsigned first = 0; first < rows; first += r1::kMaxRows) {
        const unsigned n = std::min(rows - first, r1::kMaxRows);
        const auto at = r1::row_offsets<kTop>(first, kHidden, kInter);
        r1::Args part = args;
        part.ids += at.route;
        part.weights += at.route;
        part.failure += first;
        part.input += at.input;
        part.activation += at.activation;
        part.output += at.output;
        check(r1::launch_gate_up<F, kTop, Entry>(part, stream, n), "rows gate_up");
        check(r1::launch_down_combine<F, kTop, Entry>(part, stream, n), "rows down");
    }
}

template <class Policy>
route::Inline<Policy, Math, Traits> source(const Bench& b, const Arm& arm, unsigned set) {
    return {b.logits + std::size_t{set} * kMax * kExperts, b.bias, arm.ids, arm.weights, arm.status};
}

void clear(const Arm& arm) {
    fill(arm.ids, kMax * kTop, 0x7777U);
    fill(arm.status, kMax, 0x5555U);
    fill(arm.weights, kMax * kTop, -1.0F);
    fill(arm.output, std::size_t{kMax} * kHidden, -0x1p+120F);
    fill(arm.activation, std::size_t{kMax} * kTop * kInter, std::uint16_t{0x7654U});
}

struct Snapshot {
    std::vector<unsigned> ids, status;
    std::vector<float> weights, output;
    std::vector<std::uint16_t> activation;
};
Snapshot snap(const Arm& arm) {
    check(cudaDeviceSynchronize(), "sync");
    return {download(arm.ids, kMax * kTop), download(arm.status, kMax), download(arm.weights, kMax * kTop),
            download(arm.output, std::size_t{kMax} * kHidden),
            download(arm.activation, std::size_t{kMax} * kTop * kInter)};
}

void verdict(const char* what, bool equal, const char* detail) {
    std::printf("ID %s %s %s\n", what, detail, equal ? "EQUAL" : "DIFF");
    if (!equal) g_fail = true;
}

template <MatrixFormat F>
const char* name() { return F == MatrixFormat::nvfp4 ? "nvfp4" : "mxfp8"; }

// ROUTED: inline-routed gate_up + down_combine vs route launch + gate_up + down_combine.
template <MatrixFormat F, class Policy>
void check_routed(Bench& b, unsigned rows, bool poison, const char* policy) {
    const unsigned set = 3U;
    float* poisoned = b.logits + (std::size_t{set} * kMax + 1U) * kExperts + 17U;
    float original = 0.0F;
    check(cudaMemcpy(&original, poisoned, sizeof(original), cudaMemcpyDeviceToHost), "read logit");
    if (poison) {  // row 1 routes on a NaN logit: checked status 1, no expert, the row's output untouched
        const float nan = std::numeric_limits<float>::quiet_NaN();
        check(cudaMemcpy(poisoned, &nan, sizeof(nan), cudaMemcpyHostToDevice), "poison");
    }
    clear(b.a);
    clear(b.b);
    const auto la = args_of(b, b.a), lb = args_of(b, b.b);
    check(strix::moe_core::enqueue_route_wave<Policy, Math, Traits>(b.logits + std::size_t{set} * kMax * kExperts,
                                                                    b.bias, b.a.ids, b.a.weights, b.a.status, rows, nullptr),
          "route launch");
    check(r1::launch_gate_up<F, kTop>(la, nullptr, rows), "gate_up");
    check(r1::launch_down_combine<F, kTop>(la, nullptr, rows), "down");
    check(r1::launch_gate_up_routed<F, kTop, r1::EntryPdl>(lb, source<Policy>(b, b.b, set), nullptr, rows), "routed");
    check(r1::launch_down_combine<F, kTop, r1::EntryPdl>(lb, nullptr, rows), "down pdl");
    const auto x = snap(b.a), y = snap(b.b);
    char detail[96];
    std::snprintf(detail, sizeof(detail), "%s policy=%s rows=%u%s status0=%u status1=%u", name<F>(), policy, rows,
                  poison ? " poisoned" : "", x.status[0], rows > 1U ? x.status[1] : 0U);
    const bool equal = same(x.ids, y.ids) && same(x.status, y.status) && same(x.weights, y.weights) &&
                       same(x.activation, y.activation) && same(x.output, y.output) && (!poison || x.status[1] == 1U);
    verdict("ROUTED", equal, detail);
    check(cudaMemcpy(poisoned, &original, sizeof(original), cudaMemcpyHostToDevice), "restore logit");
}

// Explicit route sets for the union identity (ids written directly, not routed).
enum class Shape { distinct, shared, repeated };
std::vector<unsigned> ids_of(Shape shape, unsigned rows, Rng& rng) {
    std::vector<unsigned> ids(kMax * kTop, 0U), pool(kExperts);
    std::iota(pool.begin(), pool.end(), 0U);
    for (unsigned row = 0; row < rows; ++row) {
        const unsigned span = shape == Shape::shared ? 12U : kExperts;
        for (unsigned i = 0; i < kTop; ++i) std::swap(pool[i], pool[i + rng() % (span - i)]);
        std::copy_n(pool.begin(), kTop, ids.begin() + row * kTop);
        if (shape == Shape::repeated) {
            ids[row * kTop + 7U] = ids[row * kTop];  // the same expert twice in one row (two assignments)
            if (row == 1U) ids[row * kTop + 3U] = kExperts;  // an invalid id: the row is not stored
        }
    }
    return ids;
}

template <MatrixFormat F, unsigned MaxRows>
void check_union(Bench& b, unsigned rows, Shape shape, Rng& rng, bool failed_row, bool step_row) {
    const auto ids = ids_of(shape, rows, rng);
    std::vector<float> weights(kMax * kTop);
    for (auto& w : weights) w = rng.uniform(0.02F, 0.4F);
    std::vector<unsigned> status(kMax, 0U), status_rows = status;
    if (failed_row) status[rows - 1U] = status_rows[rows - 1U] = 1U;
    strix::exec::StepBlock block{};
    const strix::exec::StepBlock* step = nullptr;
    if (step_row) {  // the step block marks row 0 invalid: the rows arm models it as a failed row
        block.h.rows = rows;
        block.h.valid_mask = ((1U << rows) - 1U) & ~1U;
        for (unsigned r = 0; r < rows; ++r) block.row[r].flags = strix::exec::step_valid;
        check(cudaMemcpy(b.step, &block, sizeof(block), cudaMemcpyHostToDevice), "step");
        step = b.step;
        status_rows[0] = 1U;
    }
    clear(b.a);
    clear(b.b);
    for (const Arm* arm : {&b.a, &b.b}) {
        check(cudaMemcpy(arm->ids, ids.data(), ids.size() * sizeof(unsigned), cudaMemcpyHostToDevice), "ids");
        check(cudaMemcpy(arm->weights, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice), "w");
    }
    check(cudaMemcpy(b.a.status, status_rows.data(), kMax * sizeof(unsigned), cudaMemcpyHostToDevice), "status");
    check(cudaMemcpy(b.b.status, status.data(), kMax * sizeof(unsigned), cudaMemcpyHostToDevice), "status");
    rows_launches<F>(args_of(b, b.a), rows, nullptr);
    check(r1::launch_gate_up_union<F, kTop, MaxRows, r1::EntryPdl>(args_of(b, b.b), rows, nullptr, step), "union gate_up");
    check(r1::launch_down_union<F, kTop, MaxRows, r1::EntryPdl>(args_of(b, b.b), rows, nullptr, step), "union down");
    const auto x = snap(b.a), y = snap(b.b);
    std::vector<std::uint8_t> valid(rows, 1U);
    if (step_row) valid[0] = 0U;
    const auto groups = route::expert_union(ids, status_rows, {}, rows, kTop, kExperts);
    char detail[128];
    static const char* shapes[] = {"distinct", "shared", "repeated"};
    std::snprintf(detail, sizeof(detail), "%s max=%u rows=%u %s%s%s distinct=%zu", name<F>(), MaxRows, rows,
                  shapes[static_cast<unsigned>(shape)], failed_row ? " failed-row" : "", step_row ? " step-row" : "",
                  groups.size());
    verdict("UNION", same(x.activation, y.activation) && same(x.output, y.output), detail);
}

// The inline-routed union vs route launch + union (routes from correlated logits).
template <MatrixFormat F, unsigned MaxRows, class Policy>
void check_routed_union(Bench& b, unsigned rows, unsigned set) {
    clear(b.a);
    clear(b.b);
    check(strix::moe_core::enqueue_route_wave<Policy, Math, Traits>(b.logits + std::size_t{set} * kMax * kExperts,
                                                                    b.bias, b.a.ids, b.a.weights, b.a.status, rows, nullptr),
          "route launch");
    check(r1::launch_gate_up_union<F, kTop, MaxRows>(args_of(b, b.a), rows, nullptr), "union gate_up");
    check(r1::launch_down_union<F, kTop, MaxRows>(args_of(b, b.a), rows, nullptr), "union down");
    check(r1::launch_gate_up_union_routed<F, kTop, MaxRows, r1::EntryPdl>(args_of(b, b.b), source<Policy>(b, b.b, set),
                                                                         rows, nullptr), "routed union");
    check(r1::launch_down_union<F, kTop, MaxRows, r1::EntryPdl>(args_of(b, b.b), rows, nullptr), "union down");
    const auto x = snap(b.a), y = snap(b.b);
    char detail[96];
    std::snprintf(detail, sizeof(detail), "%s max=%u rows=%u set=%u", name<F>(), MaxRows, rows, set);
    verdict("ROUTED-UNION", same(x.ids, y.ids) && same(x.status, y.status) && same(x.weights, y.weights) &&
                                same(x.activation, y.activation) && same(x.output, y.output), detail);
}

// ---- timing ----
float graph_us(cudaStream_t stream, cudaGraphExec_t exec, int iters) {
    cudaEvent_t e0, e1;
    check(cudaEventCreate(&e0), "event");
    check(cudaEventCreate(&e1), "event");
    for (int i = 0; i < 3; ++i) check(cudaGraphLaunch(exec, stream), "warm");
    check(cudaEventRecord(e0, stream), "record");
    for (int i = 0; i < iters; ++i) check(cudaGraphLaunch(exec, stream), "launch");
    check(cudaEventRecord(e1, stream), "record");
    check(cudaEventSynchronize(e1), "sync");
    float ms = 0.0F;
    check(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return 1000.0F * ms / static_cast<float>(iters);
}

template <class Body>
cudaGraphExec_t capture(cudaStream_t stream, const Body& body) {
    cudaGraph_t graph;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture");
    body();
    check(cudaStreamEndCapture(stream, &graph), "end capture");
    cudaGraphExec_t exec;
    check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    check(cudaGraphDestroy(graph), "graph");
    return exec;
}

// R1 decode visits per graph: one per route set (kSets distinct expert sets, DRAM-resident experts).
template <MatrixFormat F>
void time_r1(Bench& b, int iters) {
    using Policy = SelectionPolicy;
    using Route = route::Inline<Policy, Math, Traits>;
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    const auto threads = dim3(r1::detail::kThreads);
    const dim3 gate_grid(r1::gate_blocks(kInter, r1::detail::kGateColumns), kTop);
    const dim3 down_grid(r1::down_blocks(kHidden, r1::detail::kDownColumns), 1U);
    const auto args = args_of(b, b.a);
    const auto routed = [&](unsigned set) { return source<Policy>(b, b.a, set); };
    const auto logits = [&](unsigned set) { return b.logits + std::size_t{set} * kMax * kExperts; };
    namespace cc = strix::cuda_core;
    const auto incumbent = capture(stream, [&] {
        for (unsigned set = 0; set < kSets; ++set) {
            check(strix::moe_core::enqueue_route_wave<Policy, Math, Traits>(logits(set), b.bias, b.a.ids, b.a.weights,
                                                                            b.a.status, 1U, stream), "route");
            r1::detail::gate_up<F, kTop><<<gate_grid, threads, 0, stream>>>(args);
            r1::detail::down_combine<F, kTop><<<down_grid, threads, 0, stream>>>(args);
        }
    });
    const auto pdl = capture(stream, [&] {
        for (unsigned set = 0; set < kSets; ++set) {
            check(strix::moe_core::enqueue_route_wave<Policy, Math, Traits>(logits(set), b.bias, b.a.ids, b.a.weights,
                                                                            b.a.status, 1U, stream), "route");
            check(cc::launch_pdl(r1::detail::gate_up<F, kTop, r1::detail::kGateColumns, r1::EntryPdl>, gate_grid,
                                 threads, stream, args), "gate pdl");
            check(cc::launch_pdl(r1::detail::down_combine<F, kTop, r1::detail::kDownColumns, r1::EntryPdl>, down_grid,
                                 threads, stream, args), "down pdl");
        }
    });
    const auto inline_route = capture(stream, [&] {
        for (unsigned set = 0; set < kSets; ++set) {
            check(cc::launch_pdl(r1::detail::gate_up<F, kTop, r1::detail::kGateColumns, r1::EntryPdl, Route>, gate_grid,
                                 threads, stream, args, routed(set)), "routed pdl");
            check(cc::launch_pdl(r1::detail::down_combine<F, kTop, r1::detail::kDownColumns, r1::EntryPdl>, down_grid,
                                 threads, stream, args), "down pdl");
        }
    });
    // interleaved A/B/C rounds, median of the rounds
    std::vector<float> t0, t1, t2;
    for (int round = 0; round < 5; ++round) {
        t0.push_back(graph_us(stream, incumbent, iters) / kSets);
        t1.push_back(graph_us(stream, pdl, iters) / kSets);
        t2.push_back(graph_us(stream, inline_route, iters) / kSets);
    }
    const auto median = [](std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    const double bytes = 8.0 * static_cast<double>(b.expert_bytes);
    const float m0 = median(t0), m1 = median(t1), m2 = median(t2);
    std::printf("TIME r1 %s us_per_visit route+gate+down=%.2f pdl=%.2f routed_pdl=%.2f (saved %.2f vs pdl, %.2f vs "
                "incumbent) routed_GBs=%.0f\n", name<F>(), m0, m1, m2, m1 - m2, m0 - m2, bytes / (m2 * 1e3));
    for (auto exec : {incumbent, pdl, inline_route}) cudaGraphExecDestroy(exec);
    cudaStreamDestroy(stream);
}

// Union vs rows launches at R rows: kSets visits per graph, routes from the route launch on the logits sets
// (even = uniform rows, odd = correlated rows), timed per parity.
template <MatrixFormat F, unsigned MaxRows>
void time_union(Bench& b, unsigned rows, int iters) {
    using Policy = SelectionPolicy;
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    // one route buffer per set (graphs replay the routes, so the timed visits touch kSets distinct expert sets)
    std::vector<unsigned*> ids(kSets), status(kSets);
    std::vector<float*> weights(kSets);
    std::vector<double> distinct(2, 0.0);
    for (unsigned set = 0; set < kSets; ++set) {
        ids[set] = device<unsigned>(kMax * kTop);
        status[set] = device<unsigned>(kMax);
        weights[set] = device<float>(kMax * kTop);
        check(strix::moe_core::enqueue_route_wave<Policy, Math, Traits>(b.logits + std::size_t{set} * kMax * kExperts,
                                                                        b.bias, ids[set], weights[set], status[set], rows,
                                                                        stream), "route");
        check(cudaStreamSynchronize(stream), "route sync");
        const auto host_ids = download(ids[set], std::size_t{rows} * kTop);
        distinct[set % 2U] += static_cast<double>(route::expert_union(host_ids, {}, {}, rows, kTop, kExperts).size());
    }
    const auto args = [&](unsigned set) {
        return r1::Args{b.matrices, ids[set], weights[set], status[set], b.input, b.a.activation, b.a.output,
                        kExperts, kHidden, kInter, 30.0F, 30.0F};
    };
    for (unsigned parity = 0; parity < 2U; ++parity) {
        const auto rows_form = capture(stream, [&] {
            for (unsigned set = parity; set < kSets; set += 2U) rows_launches<F, r1::EntryPdl>(args(set), rows, stream);
        });
        const auto union_form = capture(stream, [&] {
            for (unsigned set = parity; set < kSets; set += 2U) {
                check(r1::launch_gate_up_union<F, kTop, MaxRows, r1::EntryPdl>(args(set), rows, stream), "union gate");
                check(r1::launch_down_union<F, kTop, MaxRows, r1::EntryPdl>(args(set), rows, stream), "union down");
            }
        });
        std::vector<float> ta, tb;
        for (int round = 0; round < 5; ++round) {
            ta.push_back(graph_us(stream, rows_form, iters) / (kSets / 2U));
            tb.push_back(graph_us(stream, union_form, iters) / (kSets / 2U));
        }
        std::sort(ta.begin(), ta.end());
        std::sort(tb.begin(), tb.end());
        const double groups = distinct[parity] / (kSets / 2U);
        std::printf("TIME union %s R=%u routes=%s distinct=%.1f/%u rows_us=%.2f union_us=%.2f ratio=%.3f "
                    "union_GBs=%.0f\n", name<F>(), rows, parity == 0U ? "uniform" : "correlated", groups, rows * kTop,
                    ta[2], tb[2], tb[2] / ta[2], groups * static_cast<double>(b.expert_bytes) / (tb[2] * 1e3));
        cudaGraphExecDestroy(rows_form);
        cudaGraphExecDestroy(union_form);
    }
    for (unsigned set = 0; set < kSets; ++set) {
        cudaFree(ids[set]);
        cudaFree(status[set]);
        cudaFree(weights[set]);
    }
    cudaStreamDestroy(stream);
}

template <MatrixFormat F>
void run(int iters, bool check_only) {
    Bench b = make<F>();
    for (const unsigned rows : {1U, 3U}) {
        check_routed<F, SelectionPolicy>(b, rows, false, "selection");
        check_routed<F, AscendingPolicy>(b, rows, false, "ascending");
    }
    check_routed<F, SelectionPolicy>(b, 3U, true, "selection");
    Rng rng{0x1234ULL};
    for (const auto shape : {Shape::distinct, Shape::shared, Shape::repeated}) {
        check_union<F, 4U>(b, 2U, shape, rng, false, false);
        check_union<F, 4U>(b, 4U, shape, rng, shape == Shape::shared, shape == Shape::repeated);
        check_union<F, 8U>(b, 5U, shape, rng, false, shape == Shape::distinct);
        check_union<F, 8U>(b, 8U, shape, rng, shape != Shape::distinct, false);
        check_union<F, 16U>(b, 16U, shape, rng, shape == Shape::shared, shape == Shape::repeated);
    }
    check_routed_union<F, 4U, SelectionPolicy>(b, 4U, 1U);
    check_routed_union<F, 8U, SelectionPolicy>(b, 8U, 1U);
    check_routed_union<F, 8U, AscendingPolicy>(b, 8U, 2U);
    if (check_only || g_fail) return;  // timings only after every identity line is EQUAL
    time_r1<F>(b, iters);
    time_union<F, 4U>(b, 4U, iters);
    time_union<F, 8U>(b, 8U, iters);
    time_union<F, 16U>(b, 16U, iters);
}

}  // namespace

int main(int argc, char** argv) {
    int iters = 50;
    bool check_only = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::max(1, std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--check-only")) check_only = true;
    }
    run<MatrixFormat::nvfp4>(iters, check_only);
    run<MatrixFormat::mxfp8>(iters, check_only);
    std::printf("RESULT %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
