#pragma once

#include "core/bf16_reference.hpp"
#include "core/weight_slice_loader.hpp"
#include "inkling2_mini_dspark_checkpoint.hpp"
#include "inkling2_mini_resident_backend.hpp"
#include "ops/rope/split_half_device.hpp"

#if defined(__HIPCC__)
#include "core/hip_weight_uploader.hpp"
#else
#include "cuda/core/weight_uploader.hpp"
#endif

namespace strix::inkling2 {

class MiniDSparkDeviceWeights {
    using D = MiniDSparkCheckpoint;

    static void require(weight::Status status) {
        if (!status) throw std::runtime_error("DSpark: " + status.detail);
    }

    static weight::ResidentSlicePlan read(const weight::TensorLookup& source) {
        std::vector<weight::ResidentSlicePlan::NamedTensor> tensors;
        for (const auto& tensor : D::schema().tensors)
            tensors.push_back({tensor.name, source.find(tensor.name)});
        return weight::ResidentSlicePlan::make(std::move(tensors), MiniResidentWeightPlan::kAlignment);
    }

    static weight::TensorLookup open(const std::string& path) {
        weight::TensorLookup input;
        require(input.open_safetensors(path, D::schema()));
        return input;
    }

    MiniDSparkDeviceWeights(const weight::TensorLookup& source, mini_gpu::Stream stream)
        : MiniDSparkDeviceWeights(source, read(source), stream) {}

    MiniDSparkDeviceWeights(const weight::TensorLookup& source, const weight::ResidentSlicePlan& plan,
                            mini_gpu::Stream stream)
        : storage_(plan.bytes()), tensors_(weight::ResidentSlices::bind(plan, storage_.span())),
          fusion_(MiniDSparkTapPlan::count * MiniDSparkTapPlan::count) {
#if defined(__HIPCC__)
        core::HipWeightUploader uploader;
#else
        cuda::core::WeightUploader uploader;
#endif
        core::WeightSliceLoader loader(uploader, {});
        loader.load_resident(plan, storage_.span());
        std::array<float, MiniDSparkTapPlan::count * MiniDSparkTapPlan::count> coefficients{};
        require(source.read_floats("fuse_weight", coefficients));
        for (unsigned row = 0; row < MiniDSparkTapPlan::count; ++row)
            core::softmax_bf16_reference(std::span(coefficients).subspan(row * MiniDSparkTapPlan::count,
                                                                       MiniDSparkTapPlan::count));
        mini_gpu::copy(fusion_.data(), coefficients.data(), sizeof(coefficients), mini_gpu::host_to_device, stream);
        mini_gpu::synchronize(stream);
    }

public:
    MiniDSparkDeviceWeights(const std::string& path, mini_gpu::Stream stream)
        : MiniDSparkDeviceWeights(open(path), stream) {}

    [[nodiscard]] const std::uint16_t* get(std::string_view name) const { return tensors_.bf16(name).data(); }
    [[nodiscard]] const float* fusion(unsigned row) const { return fusion_.data() + row * MiniDSparkTapPlan::count; }

private:
    mini_gpu::Buffer<std::uint8_t> storage_;
    weight::ResidentSlices tensors_;
    mini_gpu::Buffer<float> fusion_;
};

namespace mini_dspark_device {

inline void rms(float* input, const std::uint16_t* gain, float* output,
                unsigned rows, unsigned width, mini_gpu::Stream stream) {
#if defined(__HIPCC__)
    const auto status = ops::norm::enqueue_rms_bf16_rows({input, gain, output, rows, width, width, width},
        MiniDSparkCheckpoint::epsilon, ops::norm::RmsRowsReduction::parallel, stream);
#else
    const cuda_core::GroupedView layout{rows, 1U, width};
    const auto status = cuda_core::rmsnorm_bf16(input, layout, gain, output, layout, MiniDSparkCheckpoint::epsilon, stream);
#endif
    mini_gpu::check(status, "normalize DSpark rows");
}

inline void norm_rotary(float* input, const std::uint16_t* gain, unsigned rows, unsigned heads,
                        unsigned first, mini_gpu::Stream stream) {
    rms(input, gain, input, rows * heads, kHeadWidth, stream);
    mini_gpu::check(core::pointwise::elements(std::size_t{rows} * heads * (kHeadWidth / 2U),
        ops::rope::SplitHalfBf16{input, kHeadWidth, heads, first, MiniDSparkCheckpoint::rope_theta}, stream),
        "rotate DSpark heads");
}

inline void fuse(const std::uint16_t* taps, const float* coefficients, const std::uint16_t* gain,
                 float* output, unsigned rows, mini_gpu::Stream stream) {
    mini_gpu::check(core::pointwise::elements(std::size_t{rows} * kHiddenWidth,
        core::row_transforms::MixBf16Rows<MiniDSparkTapPlan::count>{taps, coefficients, output, kHiddenWidth}, stream),
        "mix DSpark taps");
    rms(output, gain, output, rows, kHiddenWidth, stream);
}

struct KvCache {
    explicit KvCache(unsigned capacity, bool ring)
        : keys(std::size_t{capacity} * kKvWidth), values(keys.span().size()), slots(capacity), wrapped(ring) {}

    void store(const float* k, const float* v, unsigned first, unsigned rows, unsigned* failure,
               mini_gpu::Stream stream) const {
#if defined(__HIPCC__)
        namespace cache = attn_core::cache_rows;
        const attn::cache_rows::Rows layout{kKvWidth, wrapped ? slots : 0U};
        mini_gpu::check(cache::store_pair({k, kKvWidth, nullptr, first, rows, layout, keys.data(), failure},
                                         {v, kKvWidth, nullptr, first, rows, layout, values.data(), failure}, stream),
                        "store DSpark KV");
#else
        (void)failure;
        const cuda_core::AttentionDims dims{kQueryHeads, kKvHeads, kHeadWidth, 1.0F, slots};
        mini_gpu::check(cuda_core::kv_store_prepared_bf16(dims, k, v,
            reinterpret_cast<__nv_bfloat16*>(keys.data()), reinterpret_cast<__nv_bfloat16*>(values.data()),
            first, rows, wrapped, stream), "store DSpark KV");
#endif
    }

    mini_gpu::Buffer<std::uint16_t> keys, values;
    unsigned slots;
    bool wrapped;
};

}  // namespace mini_dspark_device

class MiniDSparkDeviceContext {
    using D = MiniDSparkCheckpoint;
    static constexpr unsigned kChunk = 256U;

public:
    MiniDSparkDeviceContext(unsigned positions, const MiniDSparkDeviceWeights& weights)
        : weights_(weights), positions_(positions),
          cache_{mini_dspark_device::KvCache{D::context_window + D::block, true},
                 mini_dspark_device::KvCache{std::min(kMaxContext, positions + D::block - 1U), false},
                 mini_dspark_device::KvCache{D::context_window + D::block, true}},
          mixed_(kChunk * kHiddenWidth), keys_(kChunk * kKvWidth), values_(kChunk * kKvWidth),
          markov_(kHiddenWidth) {}

    // Consumes an append-only view bounded by the target's committed frontier.
    // No tap bytes move to the host. A late refresh skips already expired local rows.
    template <class Backend>
    void refresh(std::span<const std::uint16_t> taps, Backend& backend, unsigned max_rows,
                 unsigned* failure, mini_gpu::Stream stream) {
        constexpr auto width = MiniDSparkTapPlan::width;
        const auto count = taps.size() / width;
        if (max_rows == 0U || taps.size() % width != 0U || count < projected_ || count > positions_)
            throw std::invalid_argument("invalid committed DSpark context");
        const auto end = static_cast<unsigned>(count);
        if (end == projected_) return;
        for (unsigned layer = 0; layer < D::layers; ++layer) {
            const auto prefix = "layers." + std::to_string(layer) + ".";
            const auto* gain = weights_.get(prefix + "dflare_norm.weight");
            const auto* key_gain = weights_.get(prefix + "self_attn.k_norm_ctx.weight");
            const std::array projections{
                core::Bf16RowProjection{weights_.get(prefix + "self_attn.k_proj_ctx.weight"), kKvWidth, keys_.data(), true},
                core::Bf16RowProjection{weights_.get(prefix + "self_attn.v_proj_ctx.weight"), kKvWidth, values_.data(), true}};
            const auto window = D::window(layer);
            auto first = std::max(projected_, window == 0U ? 0U : end - std::min(end, window));
            while (first < end) {
                const auto rows = std::min({kChunk, max_rows, end - first});
                mini_dspark_device::fuse(taps.data() + MiniDSparkTapPlan::elements(first), weights_.fusion(layer),
                                         gain, mixed_.data(), rows, stream);
                backend.dense(mixed_.data(), kHiddenWidth, rows, projections);
                mini_dspark_device::norm_rotary(keys_.data(), key_gain, rows, kKvHeads, first, stream);
                cache_[layer].store(keys_.data(), values_.data(), first, rows, failure, stream);
                first += rows;
            }
        }
        mini_dspark_device::fuse(taps.last(width).data(), weights_.fusion(D::layers),
                                 weights_.get("markov_head.dflare_norm.weight"), markov_.data(), 1U, stream);
        projected_ = end;
    }

    [[nodiscard]] const mini_dspark_device::KvCache& cache(unsigned layer) const { return cache_.at(layer); }
    [[nodiscard]] const float* markov() const { return markov_.data(); }

private:
    const MiniDSparkDeviceWeights& weights_;
    unsigned positions_;
    unsigned projected_ = 0U;
    std::array<mini_dspark_device::KvCache, D::layers> cache_;
    mini_gpu::Buffer<float> mixed_, keys_, values_, markov_;
};

}  // namespace strix::inkling2
