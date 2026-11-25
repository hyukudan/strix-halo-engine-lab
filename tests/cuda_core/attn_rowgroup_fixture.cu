#include "attn_rowgroup_baseline.cuh"
#include "cuda/core/attn_rowgroup_launch.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace rg = strix::cuda::core::rowgroup;
namespace baseline = attn_rowgroup_baseline;

namespace {

using Layout = baseline::FrozenRowgroup;

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <class T> struct Buffer {
    T* ptr = nullptr;
    std::size_t bytes;
    explicit Buffer(std::size_t count) : bytes(count * sizeof(T)) { check(cudaMalloc(&ptr, bytes)); }
    explicit Buffer(const std::vector<T>& src) : Buffer(src.size()) {
        check(cudaMemcpy(ptr, src.data(), bytes, cudaMemcpyHostToDevice));
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { cudaFree(ptr); }
    std::vector<T> read() const {
        std::vector<T> dst(bytes / sizeof(T));
        check(cudaMemcpy(dst.data(), ptr, bytes, cudaMemcpyDeviceToHost));
        return dst;
    }
};

int compare(cudaStream_t stream) {
    constexpr unsigned rows = 7, first = 15;
    constexpr auto plan = *rg::make_launch_plan<Layout>(rows, baseline::kKvHeads, rg::kFlashGridLimits);
    constexpr auto scratch_bytes = rg::workspace_bytes<Layout>(plan.tiles);
    static_assert(scratch_bytes == 82048);
    std::vector<std::uint32_t> selected(std::size_t{rows} * baseline::kMaxSelected);
    for (unsigned row = 0; row < rows; ++row) {
        auto* list = selected.data() + std::size_t{row} * baseline::kMaxSelected;
        for (unsigned j = 0; j < baseline::selected_count(first + row); ++j) list[j] = j;
        if (row == 5) list[3] = 1;  // The second tile uses concatenation and retains a duplicate.
    }
    std::vector<float> q_gate(std::size_t{rows} * baseline::kQGateWidth);
    for (std::size_t i = 0; i < q_gate.size(); ++i)
        q_gate[i] = static_cast<float>(static_cast<int>((i * 7U) % 29U) - 14) / 16.0F;
    constexpr std::uint16_t values[] = {0x3d80, 0x3e00, 0x3e80, 0xbe80, 0x3f00, 0xbf00, 0x3f80};
    std::vector<std::uint16_t> kv(std::size_t{first + rows} * 2U * baseline::kKvWidth);
    for (std::size_t i = 0; i < kv.size(); ++i) kv[i] = values[i % 7U];

    const Buffer<float> queries(q_gate);
    const Buffer<std::uint32_t> lists(selected);
    const Buffer<std::uint16_t> cache(kv);
    const Buffer<std::uint8_t> frozen_scratch(scratch_bytes), shared_scratch(scratch_bytes);
    const Buffer<float> frozen_out(rows * baseline::kAttnWidth), shared_out(rows * baseline::kAttnWidth);
    check(cudaMemsetAsync(frozen_scratch.ptr, 0xa5, frozen_scratch.bytes, stream));
    check(cudaMemsetAsync(shared_scratch.ptr, 0xa5, shared_scratch.bytes, stream));
    check(cudaMemsetAsync(frozen_out.ptr, 0xff, frozen_out.bytes, stream));
    check(cudaMemsetAsync(shared_out.ptr, 0x7f, shared_out.bytes, stream));

    baseline::launch(Layout{queries.ptr, first, lists.ptr, cache.ptr, frozen_out.ptr, rows},
                     frozen_scratch.ptr, stream);
    rg::enqueue_rowgroup(Layout{queries.ptr, first, lists.ptr, cache.ptr, shared_out.ptr, rows},
                         baseline::kKvHeads, shared_scratch.ptr, shared_scratch.bytes, stream);
    check(cudaStreamSynchronize(stream));

    const auto frozen_union = frozen_scratch.read();
    auto shared_union = shared_scratch.read();
    std::uint32_t counts[2]{};
    std::memcpy(counts, rg::carve<Layout>(shared_union.data(), plan.tiles).count, sizeof(counts));
    const bool union_ok = frozen_union == shared_union && counts[0] == 19U && counts[1] == 63U;
    const auto expected = frozen_out.read(), actual = shared_out.read();
    const bool output_ok = std::memcmp(expected.data(), actual.data(), shared_out.bytes) == 0 &&
        std::all_of(actual.begin(), actual.end(), [](float x) { return std::isfinite(x); });
    std::printf("QSA union %s: 2 tiles, counts 19 and 63\n", union_ok ? "PASS" : "FAIL");
    std::printf("QSA flash %s: 43008 outputs bitwise\n", output_ok ? "PASS" : "FAIL");
    return union_ok && output_ok ? 0 : 1;
}
}  // namespace

int main() {
    cudaStream_t stream = nullptr;
    try {
        rg::prepare_rowgroup<Layout>();
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const int result = compare(stream);
        check(cudaStreamDestroy(stream));
        return result;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "QSA fixture FAIL: %s\n", e.what());
        if (stream != nullptr) cudaStreamDestroy(stream);
        return 1;
    }
}
