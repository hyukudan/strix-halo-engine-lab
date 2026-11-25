#include "qwen_gdn_replay.hpp"
#include "cuda/core/recur_cuda.cuh"
#include "strix/cuda/amd_math_tables.hpp"
#include "strix/cuda/qwen_gdn.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace g = strix::cuda::qwen_gdn;
namespace r = strix::cuda::recur;
namespace am = strix::cuda::amdmath;
constexpr auto kShape = g::replay_shape_for(
    {g::kHidden, g::kKeyHeads, g::kValueHeads, g::kHeadDim, g::kConvKernel, g::kA8Pitch});
using Layout = strix::ops::recur::GdnDecodeLayout<kShape>;
using strix::RecurrentStateType;

constexpr unsigned kRows = g::kMaxReplayRows, kLayer = 1, kLayers = 2;
constexpr unsigned kStride = g::kConvWidth + 2 * g::kValueWidth + 2 * g::kValueHeads;
constexpr auto kHistoryBytes = g::kConvStateHalfs * sizeof(std::uint16_t);
static_assert(Layout::channels == g::kConvWidth && Layout::value_width == g::kValueWidth &&
              Layout::value_heads == g::kValueHeads && Layout::hidden == g::kHidden &&
              Layout::history_elements == g::kConvStateHalfs && Layout::matrix_elements == g::kMatrixFloats &&
              Layout::a8_pitch == g::kA8Pitch);

static void check(cudaError_t result, const char* action) {
    if (result == cudaSuccess) return;
    std::fprintf(stderr, "%s: %s\n", action, cudaGetErrorString(result));
    std::exit(2);
}

template <class T>
struct Buffer {
    T* ptr{};
    std::size_t count;
    explicit Buffer(std::size_t n) : count(n) { check(cudaMalloc(&ptr, count * sizeof(T)), "allocate"); }
    ~Buffer() { cudaFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    void upload(const std::vector<T>& data) const {
        check(cudaMemcpy(ptr, data.data(), count * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    void fill(int byte) const { check(cudaMemset(ptr, byte, count * sizeof(T)), "fill"); }
    void equal(const Buffer& expected, const char* field, unsigned rows) const {
        std::vector<T> got(count), want(count);
        const auto bytes = count * sizeof(T);
        check(cudaMemcpy(got.data(), ptr, bytes, cudaMemcpyDeviceToHost), "read candidate");
        check(cudaMemcpy(want.data(), expected.ptr, bytes, cudaMemcpyDeviceToHost), "read reference");
        if (std::memcmp(got.data(), want.data(), bytes) == 0) return;
        std::fprintf(stderr, "Qwen GDN replay mismatch: %s (R=%u)\n", field, rows);
        std::exit(1);
    }
};

static std::vector<float> values(std::size_t count, unsigned seed, float scale) {
    std::vector<float> result(count);
    for (auto& x : result) {
        seed = seed * 1664525U + 1013904223U;
        x = static_cast<float>(static_cast<int>(seed >> 24U) - 128) * scale;
    }
    return result;
}

struct Session {
    Buffer<std::uint16_t> history{kLayers * g::kConvStateHalfs}, before{kLayers * g::kConvStateHalfs};
    Buffer<float> bank0{kLayers * g::kMatrixFloats}, bank1{kLayers * g::kMatrixFloats};
    [[nodiscard]] float* matrix(unsigned parity) const { return parity ? bank1.ptr : bank0.ptr; }
    void reset(unsigned seed) const {
        std::vector<std::uint16_t> taps(history.count);
        for (std::size_t i = 0; i < taps.size(); ++i)
            taps[i] = static_cast<std::uint16_t>(0x3d00U + ((i * 17U + seed) % 151U));
        history.upload(taps);
        before.fill(0xcd);
        bank0.upload(values(bank0.count, seed, 0x1p-10F));
        bank1.upload(values(bank1.count, seed + 9, 0x1p-10F));
    }
    void save(cudaStream_t stream) const {
        check(cudaMemcpyAsync(before.ptr, history.ptr, kLayers * kHistoryBytes, cudaMemcpyDeviceToDevice, stream), "save taps");
    }
    void undo(cudaStream_t stream) const {
        check(cudaMemcpyAsync(history.ptr, before.ptr, kLayers * kHistoryBytes, cudaMemcpyDeviceToDevice, stream), "undo taps");
    }
    void equal(const Session& expected, unsigned rows) const {
        history.equal(expected.history, "tap images", rows);
        before.equal(expected.before, "tap snapshots", rows);
        bank0.equal(expected.bank0, "matrix bank 0", rows);
        bank1.equal(expected.bank1, "matrix bank 1", rows);
    }
};

struct Workspace {
    Buffer<float> data{kRows * kStride}, raw{kRows * g::kConvWidth}, deltas{Layout::a8_deltas};
    Buffer<std::int8_t> codes{kRows * g::kValueWidth};
    Buffer<unsigned char> scratch{g::core_scratch_bytes(kRows)};
    [[nodiscard]] g::Rows rows() const {
        auto* q = data.ptr;
        auto* z = q + g::kConvWidth;
        auto* beta = z + g::kValueWidth;
        auto* gate = beta + g::kValueHeads;
        return {q, z, beta, gate, gate + g::kValueHeads, kStride};
    }
    void reset(const std::vector<float>& inputs) const {
        data.upload(inputs);
        raw.fill(0xcd);
        codes.fill(0xcd);
        deltas.fill(0xcd);
    }
    void equal(const Workspace& expected, unsigned rows) const {
        data.equal(expected.data, "QKV, z, gates, FP32 and padding", rows);
        raw.equal(expected.raw, "raw capture and padding", rows);
        codes.equal(expected.codes, "A8 codes and padding", rows);
        deltas.equal(expected.deltas, "A8 deltas and padding", rows);
    }
};

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: qwen_gdn_replay_test TABLES_DIR\n");
        return 2;
    }
    auto tables = am::load_tables(argv[1]);
    cudaStream_t stream{};
    check(cudaStreamCreate(&stream), "create stream");
    g::ReplayWindow* pinned{};
    check(cudaMallocHost(&pinned, sizeof(*pinned)), "pinned replay packet");
    *pinned = {};
    Buffer<g::ReplayWindow> packet(1);
    Buffer<std::uint16_t> hidden(kRows * g::kHidden);
    Buffer<float> beta(g::kValueHeads * g::kHidden), alpha(beta.count), dt(g::kValueHeads), a(dt.count),
        conv(g::kConvStateHalfs), gain(g::kHeadDim);
    beta.upload(values(beta.count, 7, 0x1p-13F));
    alpha.upload(values(alpha.count, 9, 0x1p-12F));
    dt.upload(values(dt.count, 13, 0x1p-7F));
    a.upload(std::vector<float>(a.count, -0.75F));
    conv.upload(values(conv.count, 11, 0x1p-8F));
    gain.upload(std::vector<float>(gain.count, 0.875F));
    const g::Weights incumbent{beta.ptr, alpha.ptr, conv.ptr, dt.ptr, a.ptr, gain.ptr};
    const r::DecodeWeights shared{beta.ptr, alpha.ptr, dt.ptr, a.ptr, conv.ptr, gain.ptr};
    const g::Tuning tuning{4, RecurrentStateType::f32, true};
    const auto layer_matrix = [](const Session& s, unsigned parity) { return s.matrix(parity) + kLayer * g::kMatrixFloats; };
    const auto layer_history = [](const Buffer<std::uint16_t>& b) { return b.ptr + kLayer * g::kConvStateHalfs; };
    constexpr std::array<std::uint32_t, kRows> tokens{7, 5, 9, 3, 11, 2, 24, 44};
    std::array<Session, 2> candidate, reference;
    Workspace got, expected;
    const auto row = got.rows();
    const auto step = g::bind_replay_layer(kShape, *packet.ptr,
        {hidden.ptr, row.qkv, row.z, row.qkv, row.beta, row.gate, row.core, got.raw.ptr},
        {g::kHidden, kStride, kStride, kStride, kStride, kStride, g::kConvWidth, g::kValueWidth}, kLayer,
        RecurrentStateType::f32, {got.codes.ptr, got.deltas.ptr});
    const auto submit = [&](unsigned rows) {
        check(cudaMemcpyAsync(packet.ptr, pinned, sizeof(*pinned), cudaMemcpyHostToDevice, stream), "copy replay packet");
        check(r::enqueue_decode_replay_step(kShape, std::bit_ceil(rows), shared, step, tables, stream), "enqueue replay");
    };

    for (unsigned rows = 1; rows <= kRows; ++rows) {
        cudaGraph_t graph{};
        cudaGraphExec_t executable{};
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
        submit(rows);
        check(cudaStreamEndCapture(stream, &graph), "end capture");
        check(cudaGraphInstantiate(&executable, graph, 0), "instantiate graph");
        for (unsigned trial = 0; trial < 4; ++trial) {
            const unsigned session = trial % 2U;
            const unsigned parity = trial / 2U;
            for (unsigned i = 0; i < candidate.size(); ++i) {
                candidate[i].reset(7U + i + trial);
                reference[i].reset(7U + i + trial);
            }
            const auto input = values(got.data.count, 19U + trial + rows, 0x1p-7F);
            std::vector<std::uint16_t> hidden_input(hidden.count);
            for (std::size_t i = 0; i < hidden_input.size(); ++i)
                hidden_input[i] = static_cast<std::uint16_t>(0x3e20U + ((i + trial * 31U) % 135U));
            hidden.upload(hidden_input);
            got.reset(input);
            expected.reset(input);
            const auto& s = candidate[session];
            const auto& ref = reference[session];
            *pinned = g::stage_replay_window(std::span(tokens).first(rows), 71U + 3U * trial, 128,
                s.before.ptr, s.matrix(parity), s.history.ptr, s.matrix(1U - parity));
            s.save(stream);
            ref.save(stream);
            if (trial == 0) submit(rows);
            else check(cudaGraphLaunch(executable, stream), "replay graph");
            const auto reference_forward = [&](unsigned count) {
                const g::A8Out a8{expected.codes.ptr, expected.deltas.ptr};
                g::launch_core_split(hidden.ptr, g::kHidden, expected.rows(), count, layer_history(ref.history),
                    layer_matrix(ref, parity), layer_matrix(ref, 1U - parity), expected.raw.ptr, expected.scratch.ptr,
                    incumbent, tables, stream, &a8, tuning);
            };
            reference_forward(rows);
            check(cudaStreamSynchronize(stream), "verify forward");
            got.equal(expected, rows);
            for (unsigned i = 0; i < candidate.size(); ++i) candidate[i].equal(reference[i], rows);

            // Positive partial acceptance consumes the retained prepared rows
            // and must yield the same images as an independent shorter forward.
            for (unsigned keep = 1; keep < rows; ++keep) {
                g::launch_commit_replay(row, keep, layer_matrix(s, parity), layer_matrix(s, 1U - parity), got.raw.ptr,
                    layer_history(s.before), layer_history(s.history), got.scratch.ptr, tables, stream, tuning);
                check(cudaStreamSynchronize(stream), "commit candidate");
                expected.reset(input);
                ref.undo(stream);
                reference_forward(keep);
                check(cudaStreamSynchronize(stream), "verify committed prefix");
                s.equal(ref, rows);
            }
            // Zero acceptance keeps the active matrix and restores all taps.
            s.undo(stream);
            ref.undo(stream);
            check(cudaStreamSynchronize(stream), "verify undo");
            s.equal(ref, rows);
        }
        check(cudaGraphExecDestroy(executable), "destroy executable");
        check(cudaGraphDestroy(graph), "destroy graph");
        if (rows == 1)
            std::puts("Qwen GDN replay f32 R=1 EXACT (eager, 3 graph launches, 2 sessions, parity, layer offset)");
    }
    std::puts("Qwen GDN replay f32 R=2..8 EXACT (padding, all prefix lengths, zero accept)");
    check(cudaFreeHost(pinned), "free replay packet");
    check(cudaStreamDestroy(stream), "destroy stream");
    am::free_tables(tables);
}
