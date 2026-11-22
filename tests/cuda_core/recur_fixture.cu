#include "recur_cases.hpp"

#include "cuda/core/recur_cuda.cuh"
#include "strix/cuda_core/gdn.hpp"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstring>
#include <optional>
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

}  // namespace

int main() {
    try {
        for (const auto steps : {1U, 7U, 64U}) run(rec::test::qwen27, "qwen27", steps, true);
        for (const auto steps : {1U, 7U, 64U}) run(rec::test::other_heads, "other-heads", steps, true);
        for (const auto steps : {1U, 7U, 64U}) run(rec::test::other_dims, "other-dims", steps, false);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "k9c fixture FAIL: %s\n", error.what());
        return 1;
    }
}
