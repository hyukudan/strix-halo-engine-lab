#include "mini_dense_cuda_support.hpp"
#include "inkling2/inkling2_mini_resident_decoder.hpp"
#include "mini_staged_source.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <memory_resource>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mini = strix::inkling2;
namespace cc = strix::cuda_core;
namespace core = strix::core;
namespace wei = strix::weight;
using Role = mini::MiniAttentionProjection;

namespace {

// The resident attention operand class through the production wiring, on a checkpoint-shaped staged model. The model is
// the real Mini text schema (every tensor name, dtype and extent) generated on the fly as a safetensors-indexed source:
// BF16 tensors hold small finite values, the rest zeros, plus an optional poisoned coefficient. It goes through
// MiniTextWeights::bind, MiniResidentWeightPlan::make and upload_mini_resident_weights (the upload the decoder runs,
// including its attention certification), a MiniResidentBackend is built over the bound weights, and the attention
// linears run on attention_operand_backend() (the backend the resident attention() prepares with), with the planes
// observed at the shared dense dispatch. Three loads:
//   finite      every attention tensor and gain finite          -> certified, ONE plane
//   refused     an Inf coefficient in the layer 0 query weight   -> refused, THREE planes
//   no scratch  finite weights, scan scratch allocation refused  -> load completes, empty certificate, THREE planes
// (the scratch exhaustion is injected through the scan's own allocator seam: the default memory resource the shared
// Bf16StagedScan allocates from). Each linear's raw output is compared bitwise with an independent three-plane arm
// (dense_big_split + dense_big_gemm called directly); the refused load also checks that a forced one-plane dispatch
// of the Inf weight differs from it (the zero planes meet 0 x inf), so the refused case is discriminating.
constexpr unsigned kRows = 512U;

class Refusing final : public std::pmr::memory_resource {
    void* do_allocate(std::size_t, std::size_t) override { throw std::bad_alloc(); }
    void do_deallocate(void*, std::size_t, std::size_t) override {}
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override { return this == &other; }
};

struct Outcome {
    bool certified;
    int planes;
    bool equals_three_planes;
    bool forced_one_plane_differs;
    bool forced_one_plane_ran;
};

Outcome run(const char* label, mini_staged::Poison poison, bool refuse_scan_scratch) {
    using namespace mini;
    using namespace mini_dense_fixture;
    const std::string query_name = "layers.0." + std::string(mini_attention_projection_spec(Role::query).tensor_suffix);
    const auto source = std::make_shared<mini_staged::StagedSource>(std::move(poison));
    wei::TensorIndex index;
    if (const auto status = wei::index_safetensors(source, index); !status) throw std::runtime_error(status.detail);
    MiniTextWeights text;
    if (const auto status = text.bind(std::move(index)); !status) throw std::runtime_error(status.detail);
    const auto plan = MiniResidentWeightPlan::make(text, wei::kBf16WeightStorage, MiniResidentBackend::resident_row_pitch_rule());
    mini_gpu::StreamOwner stream;
    mini_gpu::Buffer<std::uint8_t> storage(plan.bytes());
    Refusing refusing;
    auto* const previous = refuse_scan_scratch ? std::pmr::set_default_resource(&refusing) : nullptr;
    std::optional<MiniResidentWeights> bound;
    try {
        bound.emplace(upload_mini_resident_weights(plan, storage.span(), stream.get()));
    } catch (...) {
        if (refuse_scan_scratch) std::pmr::set_default_resource(previous);
        throw;
    }
    if (refuse_scan_scratch) std::pmr::set_default_resource(previous);
    const MiniResidentWeights& weights = *bound;
    MiniResidentBackend backend(weights, kRows, stream.get());

    std::unique_ptr<cc::BigCtx, decltype(&cc::dense_big_ctx_destroy)> ctx(cc::dense_big_ctx_create(), cc::dense_big_ctx_destroy);
    if (!ctx) throw std::runtime_error("dense context");
    std::uint32_t seed = 0x0DDBA11U;
    Buffer<float> raw(std::size_t{kRows} * kHiddenWidth, 0.0F), normalised(std::size_t{kRows} * kHiddenWidth);
    for (std::size_t i = 0; i < raw.size; ++i) {
        seed = 1664525U * seed + 1013904223U;
        raw.data[i] = static_cast<float>(static_cast<int>(seed >> 8U) - 8388608) * 0x1p-21F;
    }
    Buffer<std::uint16_t> gain(kHiddenWidth, 0x3F80U);
    require(enqueue_mini_cuda_rmsnorm(raw.data, gain.data, normalised.data, kRows, nullptr));
    require(cudaDeviceSynchronize());
    Buffer<float> reference(std::size_t{kRows} * kQueryWidth), result(std::size_t{kRows} * kQueryWidth),
        forced(std::size_t{kRows} * kQueryWidth);

    // The linear the resident attention prepare issues for the query projection, on the production-built backend.
    const auto query = weights.bf16(query_name);
    int planes = 0;
    const auto native = backend.attention_operand_backend(&planes);
    const std::array view{MiniAttentionLinear{query.data(), kQueryWidth, result.data, false}};
    require(native.linear(normalised.data, kHiddenWidth, kRows, view));
    require(cudaDeviceSynchronize());
    three_plane_reference(ctx.get(), query.data(), kQueryWidth, kHiddenWidth, kRows, normalised.data, reference.data);

    // Control: the same weight forced through the one-plane dispatch.
    int forced_planes = 0;
    const std::array forced_view{MiniAttentionLinear{query.data(), kQueryWidth, forced.data, false}};
    require(enqueue_mini_cuda_dense_rows(normalised.data, kHiddenWidth, kRows, forced_view, nullptr, cc::DenseArith::mma3,
                                         ctx.get(), core::Bf16Operand::bf16_exact, &forced_planes));
    require(cudaDeviceSynchronize());
    const Outcome outcome{weights.attention_certificate().route(core::Bf16Operand::bf16_exact) == core::Bf16Operand::bf16_exact,
                          planes, same_floats(result.data, reference.data, result.size),
                          !same_floats(forced.data, reference.data, forced.size), forced_planes == 1};
    std::cout << "CUDA Mini operand class [" << label << "]: certified " << outcome.certified << " planes " << outcome.planes
              << " equals-three-plane-arm " << outcome.equals_three_planes << " forced-one-plane-differs "
              << outcome.forced_one_plane_differs << std::endl;
    return outcome;
}

bool check_resident_operand_class() {
    const std::string full_query =
        std::string(mini::kMiniTextTensorPrefix) + "layers.0." +
        std::string(mini::mini_attention_projection_spec(Role::query).tensor_suffix);
    const auto finite = run("finite", {}, false);
    const auto refused = run("refused certificate", {full_query, 7U, 0xFF80U}, false);  // -inf in the layer 0 query
    const auto no_scratch = run("scan scratch exhausted", {}, true);
    return finite.certified && finite.planes == 1 && finite.equals_three_planes && finite.forced_one_plane_ran &&
           !finite.forced_one_plane_differs && !refused.certified && refused.planes == 3 && refused.equals_three_planes &&
           refused.forced_one_plane_ran && refused.forced_one_plane_differs && !no_scratch.certified &&
           no_scratch.planes == 3 && no_scratch.equals_three_planes;
}

}  // namespace

int main() {
    try {
        if (!check_resident_operand_class()) {
            std::cerr << "CUDA Mini operand class differs from the reference fixture\n";
            return 1;
        }
    } catch (const std::exception& error) {
        std::cerr << "CUDA Mini operand class: exception: " << error.what() << '\n';
        return 1;
    }
    std::cout << "CUDA Mini operand class: PASSED" << std::endl;
    return 0;
}
