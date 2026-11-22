#include "recur_cases.hpp"

#include "cuda/core/recur_cuda.cuh"
#include "cuda/core/recur_math.cuh"
#include "strix/cuda_core/gdn.hpp"
#include "strix/gdn_chunk.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

namespace rec = strix::cuda::recur;
namespace cc = strix::cuda_core;
using strix::state::Rollback;

void ck(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

class Buffer {
public:
    explicit Buffer(std::size_t n) { ck(cudaMalloc(&p, n * sizeof(float))); }
    explicit Buffer(const std::vector<float>& v) : Buffer(v.size()) { upload(v); }
    ~Buffer() { (void)cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    void upload(const std::vector<float>& v, std::size_t offset = 0) {
        ck(cudaMemcpy(p + offset, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    float* p{};
};

std::vector<float> read(const float* p, std::size_t n) {
    std::vector<float> v(n);
    ck(cudaMemcpy(v.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost));
    return v;
}

std::vector<float> data(std::size_t n, std::uint32_t seed, float scale, float bias = 0.0f) {
    std::vector<float> v(n);
    for (float& f : v) {
        seed = seed * 1664525u + 1013904223u;
        f = bias + scale * static_cast<float>(static_cast<int>(seed >> 23) - 256);
    }
    return v;
}

void same(const std::vector<float>& got, const std::vector<float>& want, const char* label) {
    if (got.size() != want.size()) throw std::runtime_error("fixture length mismatch");
    if (std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) == 0) return;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const auto a = std::bit_cast<std::uint32_t>(got[i]);
        const auto b = std::bit_cast<std::uint32_t>(want[i]);
        if (a != b) {
            std::fprintf(stderr, "%s at %zu: %08x vs %08x\n", label, i, a, b);
            break;
        }
    }
    throw std::runtime_error(label);
}

struct Result {
    std::vector<float> prepared, output, matrix, taps;
};

class FamilyWindow {
public:
    FamilyWindow(rec::Dims d, Rollback matrix_policy, Rollback tap_policy, rec::Execution execution, std::uint32_t steps,
                 const std::vector<float>& initial_matrix, const std::vector<float>& initial_taps)
        : decls_(rec::test::declarations(d, matrix_policy, tap_policy)),
          layout_(rec::test::layout(decls_, execution == rec::Execution::committed ? 1U : steps)),
          pool_(layout_.pool_bytes / sizeof(float)),
          plan(rec::make_plan(d, decls_[0], decls_[1], execution).value()) {
        matrix = bind(plan.matrix, 0);
        history = bind(plan.history, 1);
        const auto& mi = layout_.instances[0];
        const auto& hi = layout_.instances[1];
        pool_.upload(initial_matrix, mi.pool_offset / sizeof(float) + matrix.stride * matrix.published);
        pool_.upload(initial_taps, hi.pool_offset / sizeof(float) + history.stride * history.published);
    }

    Result run(const rec::Weights& weights, rec::Rows rows, std::uint32_t steps) const {
        ck(rec::enqueue(plan, weights, rows, matrix, history, steps, nullptr));
        auto result = state(steps);
        result.output = read(rows.out, std::size_t{steps} * plan.value_width);
        if (plan.matrix.write == rec::Write::replay_input) {
            const auto log = read(matrix.input, std::size_t{steps} * plan.matrix.replay_floats);
            for (std::uint32_t r = 0; r < steps; ++r) {
                const auto* begin = log.data() + std::size_t{r} * plan.matrix.replay_floats;
                result.prepared.insert(result.prepared.end(), begin, begin + plan.channels);
            }
        } else {
            result.prepared = read(rows.prepared, std::size_t{steps} * plan.channels);
        }
        return result;
    }

    Result state(std::uint32_t accepted) const {
        const auto index = [accepted](rec::StatePlan state_plan, rec::StateView view) {
            return accepted ? rec::image_slot(state_plan.write, view, accepted - 1) : view.published;
        };
        return {{}, {}, read(matrix.images + matrix.stride * index(plan.matrix, matrix), plan.matrix.image_floats),
                read(history.images + history.stride * index(plan.history, history), plan.history.image_floats)};
    }

private:
    rec::StateView bind(rec::StatePlan state_plan, std::size_t instance) {
        const auto& part = layout_.instances[instance];
        // Start near a wrap point in both forms, with neither candidate at the published index.
        const auto published = static_cast<std::uint32_t>(part.rows - 1);
        return rec::bind_state(state_plan, part, pool_.p + part.pool_offset / sizeof(float), published,
                               layout_.max_pending).value();
    }

    std::array<strix::state::Decl, 2> decls_;
    strix::state::Layout layout_;
    Buffer pool_;

public:
    const rec::Plan plan;
    rec::StateView matrix{}, history{};
};

Result incumbent(rec::Dims dims, const rec::Weights& weights, rec::Rows rows, std::uint32_t steps,
                 const std::vector<float>& initial_matrix, const std::vector<float>& initial_taps) {
    const cc::GdnDims d{dims.key_heads, dims.key_dim, dims.value_heads, dims.value_dim, dims.taps};
    const cc::GdnLayer w{weights.conv, weights.ssm_a, weights.dt_bias, weights.gain};
    Buffer matrix(2 * initial_matrix.size());
    Buffer taps(initial_taps);
    Buffer parity(1);
    matrix.upload(initial_matrix);
    auto* side = reinterpret_cast<std::uint32_t*>(parity.p);
    ck(cudaMemset(side, 0, sizeof(*side)));
    // The actual donor admits up to 32 rows per call. A 64-step reference
    // continues from its first committed window through the same launchers.
    for (std::uint32_t r = 0; r < steps;) {
        const auto n = std::min(steps - r, 32U);
        const auto qkv_offset = std::size_t{r} * d.conv_channels();
        const auto gates_offset = std::size_t{r} * d.value_heads;
        const auto v_offset = std::size_t{r} * d.v_width();
        const cc::GdnCommitLayer layer{w, {taps.p, matrix.p},
            {rows.raw + qkv_offset, rows.prepared + qkv_offset, rows.a + gates_offset, rows.b + gates_offset}};
        ck(cc::gdn_window(d, layer.layer, layer.state, layer.win, rows.z + v_offset, rows.out + v_offset,
                          n, weights.rms_eps, side, nullptr));
        ck(cc::gdn_commit(d, &layer, 1, n, n, side, nullptr));
        r += n;
    }
    std::uint32_t current{};
    ck(cudaMemcpy(&current, side, sizeof(current), cudaMemcpyDeviceToHost));
    return {read(rows.prepared, std::size_t{steps} * d.conv_channels()), read(rows.out, std::size_t{steps} * d.v_width()),
            read(matrix.p + current * initial_matrix.size(), initial_matrix.size()), read(taps.p, initial_taps.size())};
}

void compare_state(const Result& got, const Result& want) {
    same(got.matrix, want.matrix, "recurrent matrix");
    same(got.taps, want.taps, "convolution history");
}

void compare_forward(const Result& got, const Result& want) {
    same(got.prepared, want.prepared, "prepared rows");
    same(got.output, want.output, "gated output");
    compare_state(got, want);
}

void run(rec::Dims d, const char* label, std::uint32_t steps, bool donor) {
    const auto v = std::size_t{d.value_heads} * d.value_dim;
    const auto c = 2 * std::size_t{d.key_heads} * d.key_dim + v;
    const auto initial_matrix = data(v * d.key_dim, 31, 0x1p-14f);
    const auto initial_taps = data(c * (d.taps - 1), 37, 0x1p-10f);
    const Buffer conv(data(c * d.taps, 41, 0x1p-10f));
    const Buffer ssm(data(d.value_heads, 43, 0x1p-13f, -0.0625f));
    const Buffer dt(data(d.value_heads, 47, 0x1p-10f));
    const Buffer gain(data(d.value_dim, 53, 0x1p-10f, 0.75f));
    const rec::Weights weights{conv.p, ssm.p, dt.p, gain.p, 1e-6f};
    auto raw_host = data(c * steps, 59, 0x1p-10f);
    raw_host[0] = -0.0f;
    const Buffer raw(raw_host);
    auto a_host = data(std::size_t{d.value_heads} * steps, 61, 0x1p-8f);
    a_host[0] = 21.0f;  // softplus shortcut
    const Buffer a(a_host);
    const Buffer b(data(a_host.size(), 67, 0x1p-8f));
    const Buffer z(data(v * steps, 71, 0x1p-9f));
    Buffer prepared(c * steps), out(v * steps);
    const rec::Rows rows{raw.p, a.p, b.p, z.p, prepared.p, out.p};

    const auto accepted = steps - 1;
    const Result initial{{}, {}, initial_matrix, initial_taps};
    const auto committed = [&](std::uint32_t n) {
        const FamilyWindow window(d, Rollback::snapshot, Rollback::snapshot, rec::Execution::committed, n,
                                  initial_matrix, initial_taps);
        return window.run(weights, rows, n);
    };
    const auto full = committed(steps);
    const auto prefix = accepted ? committed(accepted) : initial;
    if (donor) {
        compare_forward(full, incumbent(d, weights, rows, steps, initial_matrix, initial_taps));
        if (accepted) compare_forward(prefix, incumbent(d, weights, rows, accepted, initial_matrix, initial_taps));
    }

    const auto speculative = rec::Execution::speculative;
    const FamilyWindow ring(d, Rollback::snapshot, Rollback::snapshot, speculative, steps, initial_matrix, initial_taps);
    const FamilyWindow replay(d, Rollback::replay, Rollback::replay, speculative, steps, initial_matrix, initial_taps);
    compare_forward(ring.run(weights, rows, steps), full);
    auto retained = rows;
    retained.prepared = nullptr;
    compare_forward(replay.run(weights, retained, steps), full);
    std::optional<FamilyWindow> only_matrix, only_taps;
    if (!donor) {
        only_matrix.emplace(d, Rollback::replay, Rollback::snapshot, speculative, steps, initial_matrix, initial_taps);
        only_taps.emplace(d, Rollback::snapshot, Rollback::replay, speculative, steps, initial_matrix, initial_taps);
        compare_forward(only_matrix->run(weights, retained, steps), full);
        compare_forward(only_taps->run(weights, rows, steps), full);
    }

    // Destroy transient inputs and scratch. The accepted replay must use
    // the retained records and the original published images exclusively.
    ck(cudaMemset(raw.p, 0, c * steps * sizeof(float)));
    ck(cudaMemset(prepared.p, 0, c * steps * sizeof(float)));
    ck(cudaMemset(a.p, 0, a_host.size() * sizeof(float)));
    ck(cudaMemset(b.p, 0, a_host.size() * sizeof(float)));
    compare_state(ring.state(accepted), prefix);
    compare_state(ring.state(0), initial);
    const auto check_replay = [&](const FamilyWindow& window) {
        ck(rec::enqueue_replay(window.plan, weights, window.matrix, window.history, accepted, nullptr));
        compare_state(window.state(accepted), prefix);
        ck(rec::enqueue_replay(window.plan, weights, window.matrix, window.history, 0, nullptr));
        compare_state(window.state(0), initial);
    };
    check_replay(replay);
    if (!donor) {
        check_replay(*only_matrix);
        check_replay(*only_taps);
    }
    std::printf("k9c %s-%u PASS\n", label, steps);
}

constexpr rec::Dims abi_dims{rec::Kind::gdn, 1, 128, 2, 128, 4};
constexpr std::uint32_t channels = 512, values = 256, heads = 2;
constexpr std::size_t matrix_size = values * 128, taps_size = channels * 3;
constexpr float eps = 1e-6f;

struct AbiLayer {
    explicit AbiLayer(std::uint32_t rows, std::uint32_t seed)
        : conv(data(channels * 4, seed + 1, 0x1p-10f)),
          ssm(data(heads, seed + 2, 0x1p-13f, -0.0625f)),
          dt(data(heads, seed + 3, 0x1p-10f)),
          gain(data(128, seed + 4, 0x1p-10f, 0.75f)),
          raw(data(channels * rows, seed + 5, 0x1p-10f)),
          a(data(heads * rows, seed + 6, 0x1p-8f)),
          b(data(heads * rows, seed + 7, 0x1p-8f)),
          z(data(values * rows, seed + 8, 0x1p-9f)),
          prepared(channels * rows), out(values * rows) {}

    cc::GdnCommitLayer bind(float* history, float* matrix) const {
        return {{conv.p, ssm.p, dt.p, gain.p}, {history, matrix}, {raw.p, prepared.p, a.p, b.p}};
    }

    Result oracle(const Result& initial, std::span<const std::uint32_t> path) const {
        if (path.empty()) return initial;
        const auto rows = static_cast<std::uint32_t>(path.size());
        Buffer rq(channels * rows), ra(heads * rows), rb(heads * rows), rz(values * rows);
        Buffer pq(channels * rows), po(values * rows);
        const auto gather = [&](float* to, const float* from, std::size_t width) {
            for (std::size_t i = 0; i < path.size(); ++i)
                ck(cudaMemcpy(to + i * width, from + path[i] * width, width * sizeof(float), cudaMemcpyDeviceToDevice));
        };
        gather(rq.p, raw.p, channels);
        gather(ra.p, a.p, heads);
        gather(rb.p, b.p, heads);
        gather(rz.p, z.p, values);
        const FamilyWindow oracle(abi_dims, Rollback::snapshot, Rollback::snapshot, rec::Execution::committed,
                                  rows, initial.matrix, initial.taps);
        return oracle.run({conv.p, ssm.p, dt.p, gain.p, eps}, {rq.p, ra.p, rb.p, rz.p, pq.p, po.p}, rows);
    }

    void compare_rows(std::uint32_t first, std::uint32_t rows, const Result& expected, std::uint32_t ref = 0) const {
        const auto check = [first, rows, ref](const Buffer& buffer, const std::vector<float>& want,
                                             std::size_t width, const char* label) {
            auto begin = want.begin() + static_cast<std::ptrdiff_t>(ref * width);
            same(read(buffer.p + first * width, rows * width),
                 {begin, begin + static_cast<std::ptrdiff_t>(rows * width)}, label);
        };
        check(prepared, expected.prepared, channels, "production convolution");
        check(out, expected.output, values, "production gated output");
    }

    Buffer conv, ssm, dt, gain, raw, a, b, z, prepared, out;
};

Result initial_abi(std::uint32_t seed) {
    return {{}, {}, data(matrix_size, seed, 0x1p-14f), data(taps_size, seed + 1, 0x1p-10f)};
}

void compare_abi_state(cc::GdnState state, const Result& want, std::uint32_t side) {
    same(read(state.delta_base + side * matrix_size, matrix_size), want.matrix, "production matrix");
    same(read(state.conv, taps_size), want.taps, "production taps");
}

template <std::size_t N>
void upload_u32(const Buffer& buffer, const std::array<std::uint32_t, N>& words) {
    ck(cudaMemcpy(buffer.p, words.data(), sizeof(words), cudaMemcpyHostToDevice));
}

std::uint32_t word(const std::uint32_t* ptr) {
    std::uint32_t value{};
    ck(cudaMemcpy(&value, ptr, sizeof(value), cudaMemcpyDeviceToHost));
    return value;
}

void require(bool ok, const char* label) {
    if (!ok) throw std::runtime_error(label);
}

void production_linear_tree() {
    constexpr cc::GdnDims d{1, 128, 2, 128, 4};
    const AbiLayer first(5, 501), second(5, 525);
    const std::array layers{&first, &second};
    const std::array initial{initial_abi(503), initial_abi(529)};
    Buffer m0(2 * matrix_size), m1(2 * matrix_size), t0(taps_size), t1(taps_size), parity(1), leaf(1);
    auto* side = reinterpret_cast<std::uint32_t*>(parity.p);
    const auto* leaves = reinterpret_cast<const std::uint32_t*>(leaf.p);
    const std::array bound{first.bind(t0.p, m0.p), second.bind(t1.p, m1.p)};
    const auto reset = [&] {
        upload_u32(parity, std::array{1U});
        m0.upload(initial[0].matrix, matrix_size);
        m1.upload(initial[1].matrix, matrix_size);
        t0.upload(initial[0].taps);
        t1.upload(initial[1].taps);
    };
    const auto launch = [&](const std::uint32_t* mask) {
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const auto& l = bound[i];
            ck(cc::gdn_window(d, l.layer, l.state, l.win, layers[i]->z.p, layers[i]->out.p, 5, eps, side, nullptr, mask));
            compare_abi_state(l.state, initial[i], 1);
        }
        require(word(side) == 1, "window published state");
    };
    const std::array linear{0U, 1U, 2U, 3U, 4U};
    for (const bool pdl : {false, true}) {
        cc::pdl_enabled() = pdl;
        reset();
        launch(nullptr);
        for (std::size_t i = 0; i < layers.size(); ++i)
            layers[i]->compare_rows(0, 5, layers[i]->oracle(initial[i], linear));
        ck(cc::gdn_commit(d, bound.data(), 2, 5, 0, side, nullptr));
        require(word(side) == 1, "zero accept flipped parity");
        ck(cc::gdn_commit(d, bound.data(), 2, 5, 2, side, nullptr));
        require(word(side) == 0, "linear commit parity");
        for (std::size_t i = 0; i < layers.size(); ++i)
            compare_abi_state(bound[i].state, layers[i]->oracle(initial[i], std::span(linear).first(2)), 0);
    }
    std::puts("k9c production linear-pdl PASS");
    std::puts("k9c production batched-prefix PASS");

    reset();
    upload_u32(leaf, std::array{0b01010U});
    launch(leaves);
    constexpr std::array<std::array<std::uint32_t, 3>, 5> paths{{
        {0, 0, 0}, {0, 1, 0}, {0, 2, 0}, {0, 2, 3}, {0, 2, 4}}};
    constexpr std::array counts{1U, 2U, 2U, 3U, 3U};
    for (std::size_t i = 0; i < layers.size(); ++i)
        for (std::uint32_t r = 0; r < paths.size(); ++r)
            layers[i]->compare_rows(r, 1, layers[i]->oracle(initial[i], std::span(paths[r]).first(counts[r])), counts[r] - 1);
    std::puts("k9c production tree-window PASS");
    ck(cc::gdn_commit_path(d, bound.data(), 2, 5, 4, 0b00010U, true, side, nullptr));
    require(word(side) == 0, "tree commit parity");
    for (std::size_t i = 0; i < layers.size(); ++i)
        compare_abi_state(bound[i].state, layers[i]->oracle(initial[i], paths[3]), 0);
    std::puts("k9c production tree-replay PASS");
}

void production_slots() {
    constexpr cc::GdnDims d{1, 128, 2, 128, 4};
    constexpr auto conv_stride = taps_size + 4;
    constexpr auto delta_stride = 2 * matrix_size + 8;
    constexpr std::array starts{0U, 3U, 5U}, lengths{3U, 2U, 2U};
    constexpr std::array all_rows{0U, 1U, 2U, 3U, 4U, 5U, 6U};
    const AbiLayer layer(7, 401);
    Buffer matrix(3 * delta_stride), taps(3 * conv_stride), parity(3), row0(3), nrows(3), slot(3), accept(3);
    auto* side = reinterpret_cast<std::uint32_t*>(parity.p);
    const cc::GdnSlotTable table{reinterpret_cast<const std::uint32_t*>(row0.p),
        reinterpret_cast<const std::uint32_t*>(nrows.p), reinterpret_cast<const std::uint32_t*>(slot.p),
        side, conv_stride, delta_stride};
    auto expected = std::array{initial_abi(17), initial_abi(31), initial_abi(71)};
    auto sides = std::array{1U, 0U, 1U};
    for (std::size_t s = 0; s < expected.size(); ++s) {
        matrix.upload(expected[s].matrix, s * delta_stride + sides[s] * matrix_size);
        taps.upload(expected[s].taps, s * conv_stride);
    }
    upload_u32(row0, starts);
    upload_u32(nrows, lengths);
    upload_u32(parity, sides);

    cudaStream_t stream{};
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const auto l = layer.bind(taps.p, matrix.p);
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    ck(cc::gdn_window_slots(d, l.layer, l.state, l.win, layer.z.p, layer.out.p, 3, table, eps, stream));
    ck(cc::gdn_commit_slots(d, &l, 1, 3, table, reinterpret_cast<const std::uint32_t*>(accept.p), true, stream));
    ck(cudaStreamEndCapture(stream, &graph));
    ck(cudaGraphInstantiate(&executable, graph, 0));
    const auto replay = [&](std::array<std::uint32_t, 3> slots, std::array<std::uint32_t, 3> accepted) {
        upload_u32(slot, slots);
        upload_u32(accept, accepted);
        ck(cudaGraphLaunch(executable, stream));
        ck(cudaStreamSynchronize(stream));
        for (std::size_t g = 0; g < slots.size(); ++g) {
            const auto s = slots[g];
            const auto path = std::span(all_rows).subspan(starts[g], lengths[g]);
            const auto full = layer.oracle(expected[s], path);
            layer.compare_rows(starts[g], lengths[g], full);
            const auto k = accepted[g];
            expected[s] = k == lengths[g] ? full : layer.oracle(expected[s], path.first(k));
            sides[s] ^= static_cast<std::uint32_t>(k != 0);
            require(word(side + s) == sides[s], "slot commit parity");
            compare_abi_state({taps.p + s * conv_stride, matrix.p + s * delta_stride}, expected[s], sides[s]);
        }
    };
    replay({2, 0, 1}, {1, 2, 0});
    replay({1, 2, 0}, {0, 1, 2});
    ck(cudaGraphExecDestroy(executable));
    ck(cudaGraphDestroy(graph));
    ck(cudaStreamDestroy(stream));
    std::puts("k9c production slot-graph-window PASS");
    std::puts("k9c production slot-replay PASS");
    std::puts("k9c production slot-taps PASS");
    std::puts("k9c production slot-parity PASS");
}

void chunk_close(const std::vector<float>& got, const std::vector<float>& want) {
    double delta = 0, reference = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double v = want[i];
        const double e = got[i] - v;
        delta += e * e;
        reference += v * v;
    }
    require(std::sqrt(delta / reference) <= 1e-4, "production chunk numerical bound");
}

void production_prefill() {
    constexpr cc::GdnDims d{1, 128, 2, 128, 4};
    constexpr std::uint32_t rows = 35;
    const AbiLayer layer(rows, 307);
    const auto initial = initial_abi(337);
    std::array<std::uint32_t, rows> path{};
    for (std::uint32_t i = 0; i < rows; ++i) path[i] = i;
    const auto expected = layer.oracle(initial, path);
    Buffer matrix(2 * matrix_size), taps(taps_size), parity(1), workspace(cc::gdn_chunk_workspace_floats(d, rows));
    auto* side = reinterpret_cast<std::uint32_t*>(parity.p);
    const auto l = layer.bind(taps.p, matrix.p);
    const auto reset = [&] {
        matrix.upload(initial.matrix, matrix_size);
        taps.upload(initial.taps);
        upload_u32(parity, std::array{1U});
    };
    reset();
    ck(cc::gdn_prefill_layer(d, l.layer, l.state, l.win, layer.z.p, layer.out.p, rows, eps, side, nullptr));
    layer.compare_rows(0, rows, expected);
    compare_abi_state(l.state, expected, 0);
    require(word(side) == 1, "prefill published early");
    ck(cc::gdn_prefill_flip(side, nullptr));
    require(word(side) == 0, "prefill commit parity");
    std::puts("k9c production serial-prefill PASS");
    std::puts("k9c production prefill-parity PASS");

    using strix::gdn_chunk::Scan;
    for (const auto scan : {Scan::f32, Scan::tf32}) {
        const auto run_chunk = [&](bool defer_norm) {
            reset();
            const cc::GdnChunkWork work{workspace.p, cc::gdn_chunk_workspace_floats(d, rows), defer_norm,
                                       static_cast<std::uint8_t>(scan)};
            ck(cc::gdn_prefill_layer(d, l.layer, l.state, l.win, layer.z.p, layer.out.p, rows, eps, side, work, nullptr));
            if (defer_norm)
                ck(cc::gdn_chunk_norm(d, l.layer.norm_gain, layer.z.p, layer.out.p, rows, eps, nullptr));
            require(word(side) == 1, "chunk prefill published early");
            same(read(matrix.p + matrix_size, matrix_size), initial.matrix, "chunk overwrote committed matrix");
            return Result{read(layer.prepared.p, rows * channels), read(layer.out.p, rows * values),
                          read(matrix.p, matrix_size), read(taps.p, taps_size)};
        };
        const auto fused = run_chunk(false);
        const auto deferred = run_chunk(true);
        compare_forward(deferred, fused);
        same(fused.prepared, expected.prepared, "chunk prefill convolution");
        same(fused.taps, expected.taps, "chunk prefill taps");
        chunk_close(fused.matrix, expected.matrix);
        chunk_close(fused.output, expected.output);
        std::puts(scan == Scan::f32 ? "k9c production chunk-f32 PASS" : "k9c production chunk-tf32 PASS");
    }
    std::puts("k9c production chunk-norm PASS");
}

}  // namespace

int main() {
    try {
        for (const auto steps : {1U, 7U, 64U}) run(rec::test::qwen27, "qwen27", steps, true);
        for (const auto steps : {1U, 7U, 64U}) run(rec::test::other_heads, "other-heads", steps, true);
        for (const auto steps : {1U, 7U, 64U}) run(rec::test::other_dims, "other-dims", steps, false);
        production_linear_tree();
        production_slots();
        production_prefill();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "k9c fixture FAIL: %s\n", error.what());
        return 1;
    }
}
