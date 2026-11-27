#pragma once

// Block-diffusion drafter (the DFlash family), device-resident forward.
// Home: shared (strix::spec::dflash). Same contract as the host reference (include/strix/spec/dflash_drafter.hpp): the
// geometry, block, window, taps, conv and selector are the drafter's config.json (spec::dflash::Config), not code. The
// target supplies its boundary through a small adapter (the Target concept below): its BF16 dense rows, its embedding
// table + embedding norm, its head and its committed tap plane. The composition reuses shared device pieces only: the
// backend's BF16 dense rows, the RMS rows, split-half RoPE, the prepared BF16 attention (bidirectional block + preceding
// window, attn_core::mask::block_range), SwiGLU, the KV row store. The pieces this drafter adds are pointwise functors
// (grouped two-tap convolution, the canonical top-k of the target logits, the selector edge scores and its path), so
// one body serves CUDA and HIP.
//
// Numerics: BF16-rounded projections and residual rows (the DSpark device convention); the selector scores in FP32.
// Ties of the top-k keep the lower id (the host reference's order). Exactness of the emitted ids never depends on the
// drafter: verification decides every token.

#include "core/row_transforms.hpp"
#include "spec/dflash_rows.hpp"
#include "core/weight_slice_loader.hpp"
#include "ops/glu/swiglu_rows.hpp"
#include "ops/rope/split_half_device.hpp"
#include "strix/device/runtime.hpp"
#include "strix/spec/block_parallel.hpp"
#include "strix/spec/dflash_drafter.hpp"
#include "strix/weight/resident_slices.hpp"
#include "strix/weight/tensor_access.hpp"

#if defined(__HIPCC__)
#include "attn_core/cache_rows.hpp"
#include "attn_core/prepared_bf16_attention.hpp"
#include "core/hip_weight_uploader.hpp"
#include "ops/norm/rms_bf16_rows.hpp"
#else
#include "cuda/core/weight_uploader.hpp"
#include "strix/cuda_core/attention.hpp"
#include "strix/cuda_core/norms.hpp"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::spec::dflash {

namespace device_ops {

namespace gpu = strix::device::native;

inline constexpr unsigned kHeadDim = 128U;     // the prepared attention cores' head width
inline constexpr unsigned kGroup = 4U;         // query heads per KV head of the bound geometry (HIP policy)
using namespace row_ops;

inline void check(gpu::Status status, const char* what) { gpu::check(status, what); }

template <class Transform>
void transform(std::size_t count, Transform operation, gpu::Stream stream, const char* what) {
    if (count) check(core::pointwise::elements(count, operation, stream), what);
}

inline void rms(const float* input, const std::uint16_t* gain, float* output, unsigned rows, unsigned width, float eps,
                gpu::Stream stream) {
#if defined(__HIPCC__)
    const auto status = ops::norm::enqueue_rms_bf16_rows({const_cast<float*>(input), gain, output, rows, width, width, width},
        eps, ops::norm::RmsRowsReduction::parallel, stream);
#else
    const cuda_core::GroupedView layout{rows, 1U, width};
    const auto status = cuda_core::rmsnorm_bf16(const_cast<float*>(input), layout, gain, output, layout, eps, stream);
#endif
    check(status, "normalize DFlash rows");
}

inline void norm_rotary(float* rows_data, const std::uint16_t* gain, unsigned rows, unsigned heads, unsigned first,
                        float eps, float theta, gpu::Stream stream) {
    rms(rows_data, gain, rows_data, rows * heads, kHeadDim, eps, stream);
    transform(std::size_t{rows} * heads * (kHeadDim / 2U), ops::rope::SplitHalfBf16{rows_data, kHeadDim, heads, first, theta},
              stream, "rotate DFlash heads");
}

// One layer's context K/V ring (BF16 prepared rows, slot = position % slots).
struct KvRing {
    KvRing(unsigned capacity, unsigned kv_width)
        : keys(std::size_t{capacity} * kv_width), values(keys.span().size()), slots(capacity), width(kv_width) {}

    void store(const float* k, const float* v, unsigned first, unsigned rows, unsigned heads, unsigned* failure,
               gpu::Stream stream) const {
#if defined(__HIPCC__)
        (void)heads;
        namespace cache = attn_core::cache_rows;
        const attn::cache_rows::Rows layout{width, slots};
        check(cache::store_pair({k, width, nullptr, first, rows, layout, keys.data(), failure},
                                {v, width, nullptr, first, rows, layout, values.data(), failure}, stream),
              "store DFlash KV");
#else
        (void)failure;
        const cuda_core::AttentionDims dims{heads * kGroup, heads, kHeadDim, 1.0F, slots};
        check(cuda_core::kv_store_prepared_bf16(dims, k, v, reinterpret_cast<__nv_bfloat16*>(keys.data()),
                                                reinterpret_cast<__nv_bfloat16*>(values.data()), first, rows, true, stream),
              "store DFlash KV");
#endif
    }

    gpu::Buffer<std::uint16_t> keys, values;
    unsigned slots, width;
};

#if defined(__HIPCC__)
struct HipAttentionPolicy
    : attn_core::dense::PreparedBf16Policy<kHeadDim, kGroup, false, false, 1U, 16U, true> {
    static constexpr float kScaleLog2 = 0.08838834764831845F * 1.4426950408889634F;
};
#endif

// Block rows [anchor, anchor + rows) attend every block row and the `window` context keys preceding the anchor.
inline void attend(unsigned heads, unsigned kv_heads, unsigned window, unsigned anchor, unsigned rows, const KvRing& ring,
                   const float* q, const float* k, const float* v, float* output, const std::uint16_t* zeros,
                   unsigned* failure, gpu::Stream stream) {
#if defined(__HIPCC__)
    (void)heads;
    const attn_core::PreparedGqaAttentionRows block{q, k, v, nullptr, nullptr, output, anchor, rows};
    const attn_core::dense::PreparedBf16Cache view{ring.keys.data(), ring.values.data(), zeros, ring.slots, true};
    check(attn_core::dense::append_prepared_bf16<HipAttentionPolicy, true>(kv_heads, window, 0U, block, view, failure,
                                                                           stream, {}, 1U),
          "attend DFlash block");
#else
    (void)zeros;
    const cuda_core::AttentionDims dims{heads, kv_heads, kHeadDim, 1.0F / std::sqrt(static_cast<float>(kHeadDim)), ring.slots};
    const cuda_core::PreparedBf16Flash policy{nullptr, nullptr, 0U, window, true, true};
    check(cuda_core::append_attention_prepared_bf16(dims, policy, q, k, v,
        reinterpret_cast<__nv_bfloat16*>(ring.keys.data()), reinterpret_cast<__nv_bfloat16*>(ring.values.data()),
        anchor, rows, output, stream, cuda_core::FlashPipe::pad2, failure), "attend DFlash block");
#endif
}

}  // namespace device_ops

// The drafter's config.json + model.safetensors, resident (BF16, the file's tensors by name).
class ResidentWeights {
public:
    ResidentWeights(const std::filesystem::path& dir, device_ops::gpu::Stream stream)
        : config_(load_config(dir)), lookup_(open_lookup(config_, dir)), plan_(resident_plan(config_, lookup_, 256U)),
          storage_(plan_.bytes()),
          tensors_(weight::ResidentSlices::bind(plan_, storage_.span())) {
#if defined(__HIPCC__)
        core::HipWeightUploader uploader;
#else
        cuda::core::WeightUploader uploader;
#endif
        core::WeightSliceLoader loader(uploader, {});
        loader.load_resident(plan_, storage_.span());
        device_ops::gpu::synchronize(stream);
    }

    [[nodiscard]] const Config& config() const noexcept { return config_; }
    [[nodiscard]] const std::uint16_t* get(const std::string& name) const { return tensors_.bf16(name).data(); }
    [[nodiscard]] const std::uint16_t* layer(unsigned l, const char* rest) const {
        return get("layers." + std::to_string(l) + "." + rest);
    }

private:
    Config config_;
    weight::TensorLookup lookup_;
    weight::ResidentSlicePlan plan_;
    device_ops::gpu::Buffer<std::uint8_t> storage_;
    weight::ResidentSlices tensors_;
};

// Target adapter concept (the only model code):
//   gpu::Stream stream();                             the target's stream (the drafter is ordered after its commit)
//   unsigned max_rows();                              rows of one dense launch
//   void dense(const float* in, unsigned width, unsigned rows, std::span<const core::Bf16RowProjection>);
//   const std::uint16_t* embed_table();               BF16 [vocab][hidden]
//   const std::uint16_t* embed_norm();                BF16 gain of the embedding boundary (nullptr: none)
//   float embed_eps();
//   void head(const float* rows, unsigned count, float* logits);  raw head logits [count][vocab_out()]
//   unsigned vocab_out();
//   std::span<const std::uint16_t> committed_taps();  BF16 [position][tap_width], device
template <class Target>
class Resident final : public ChainDrafter {
    using Projection = core::Bf16RowProjection;
    static constexpr unsigned kChunk = 256U;

public:
    Resident(Target target, const std::filesystem::path& dir)
        : target_(std::move(target)), stream_(target_.stream()), weights_(dir, stream_), c_(weights_.config()),
          qw_(c_.heads * c_.head_dim), kvw_(c_.kv_heads * c_.head_dim), dw_(2U * c_.conv_taps * c_.conv_groups()),
          top_k_(c_.selector_rank ? c_.selector_top_k : 1U), vocab_out_(target_.vocab_out()),
          forward_binding_(*this), controller_(Drafter::policy(c_), forward_binding_),
          x_(std::size_t{c_.block} * c_.hidden), n_(x_.span().size()), t_(x_.span().size()), branch_(x_.span().size()),
          q_(std::size_t{c_.block} * qw_), att_(q_.span().size()), k_(std::size_t{c_.block} * kvw_), v_(k_.span().size()),
          gate_(std::size_t{c_.block} * c_.intermediate), up_(gate_.span().size()),
          delta_(std::size_t{c_.block} * std::max(dw_, 1U)), zero_(device_ops::kHeadDim), failure_(1U),
          wide_(std::size_t{kChunk} * c_.tap_width()), fused_(std::size_t{kChunk} * c_.hidden),
          ck_(std::size_t{kChunk} * kvw_), cv_(ck_.span().size()),
          logits_(std::size_t{max_drafts()} * vocab_out_),
          top_values_(std::size_t{max_drafts()} * device_ops::kTopChunks * top_k_), top_ids_(top_values_.span().size()),
          mid_values_(top_values_.span().size()), mid_ids_(top_values_.span().size()),
          unary_(std::size_t{max_drafts()} * top_k_), cand_(unary_.span().size()),
          hsel_(std::size_t{max_drafts()} * std::max(c_.selector_rank, 1U)),
          edges_(std::size_t{max_drafts()} * top_k_ * top_k_), picks_(max_drafts()) {
        // The bound geometry of the shared attention cores (head width, GQA group) and of the registers of the top-k.
        if (c_.head_dim != device_ops::kHeadDim || c_.heads != c_.kv_heads * device_ops::kGroup || c_.window == 0U ||
            top_k_ > device_ops::kMaxTopK || c_.embed_scale != 1.0F || (c_.conv_taps && c_.hidden % c_.conv_group))
            throw std::invalid_argument("DFlash device drafter: unsupported geometry in config.json");
        if (c_.selector_rank && vocab_out_ > c_.vocab) throw std::invalid_argument("DFlash: head wider than the codebooks");
        rings_.reserve(c_.layers);
        for (unsigned l = 0; l < c_.layers; ++l) rings_.push_back(std::make_unique<device_ops::KvRing>(c_.window + c_.block, kvw_));
        device_ops::gpu::clear(zero_.data(), zero_.span().size_bytes(), stream_);
        device_ops::gpu::synchronize(stream_);
    }

    [[nodiscard]] const Config& config() const noexcept { return c_; }
    [[nodiscard]] unsigned max_drafts() const noexcept { return c_.block - 1U; }

    void start(std::uint32_t anchor, std::uint32_t position) {
        if (target_.committed_taps().size() != std::size_t{position} * c_.tap_width())
            throw std::logic_error("invalid resident DFlash start frontier");
        controller_.start(anchor, position);
    }
    // A served slot reused by a new sequence: the next forward re-projects its context from the new taps.
    void restart_sequence() noexcept { projected_ = 0U; }
    [[nodiscard]] bool failed() const noexcept { return failed_; }

    void draft(std::span<const std::uint32_t> prefix, std::uint32_t count, std::vector<std::uint32_t>& out) override {
        if (failed_) throw std::logic_error("invalid resident DFlash session");
        try {
            controller_.draft(prefix, count, out);
        } catch (...) {
            failed_ = true;
            try { device_ops::gpu::synchronize(stream_); } catch (...) {}
            throw;
        }
    }
    void accept(std::span<const std::uint32_t> emitted) override { controller_.accept(emitted); }

private:
    class ForwardBinding final : public BlockForward {
    public:
        explicit ForwardBinding(Resident& owner) : owner_(owner) {}
        void forward(const BlockRequest& request, std::vector<std::uint32_t>& proposals, std::vector<float>& confidence) override {
            owner_.forward(request, proposals, confidence);
        }
        void accept(std::span<const std::uint32_t>) override {}
    private:
        Resident& owner_;
    };

    void dense(const float* input, unsigned width, unsigned rows, std::span<const Projection> projections) {
        target_.dense(input, width, rows, projections);
    }

    // Context rows [projected_, end): c = hidden_norm(fc taps); per layer K = rope(k_norm(W_k c)), V = W_v c. Only the
    // last `window` positions are read by the next block.
    void refresh(std::span<const std::uint16_t> taps) {
        const auto tw = c_.tap_width();
        const auto count = taps.size() / tw;
        if (taps.size() % tw != 0U || count < projected_) throw std::invalid_argument("invalid committed DFlash context");
        const auto end = static_cast<unsigned>(count);
        auto first = std::max(projected_, end - std::min(end, c_.window));
        const auto max_rows = std::max(1U, std::min(kChunk, target_.max_rows()));
        while (first < end) {
            const auto rows = std::min(max_rows, end - first);
            device_ops::transform(std::size_t{rows} * tw, core::row_transforms::WidenRepeatedBf16Row{
                taps.data() + std::size_t{first} * tw, wide_.data(), rows * tw}, stream_, "widen DFlash taps");
            const std::array fc{Projection{weights_.get("fc.weight"), c_.hidden, fused_.data(), true}};
            dense(wide_.data(), tw, rows, fc);
            device_ops::rms(fused_.data(), weights_.get("hidden_norm.weight"), fused_.data(), rows, c_.hidden, c_.eps, stream_);
            for (unsigned l = 0; l < c_.layers; ++l) {
                const std::array kv{Projection{weights_.layer(l, "self_attn.k_proj.weight"), kvw_, ck_.data(), true},
                                    Projection{weights_.layer(l, "self_attn.v_proj.weight"), kvw_, cv_.data(), true}};
                dense(fused_.data(), c_.hidden, rows, kv);
                device_ops::norm_rotary(ck_.data(), weights_.layer(l, "self_attn.k_norm.weight"), rows, c_.kv_heads, first,
                                        c_.eps, c_.rope_theta, stream_);
                rings_[l]->store(ck_.data(), cv_.data(), first, rows, c_.kv_heads, failure_.data(), stream_);
            }
            first += rows;
        }
        projected_ = end;
    }

    // Side 0 filters `rows` in place after computing the coefficients from it; side 1 filters the sub-block output.
    void conv(unsigned l, const char* which, unsigned side, float* rows) {
        const std::string prefix = std::string(which);
        if (side == 0U) {
            const std::array kp{Projection{weights_.layer(l, (prefix + "kernel_projection.weight").c_str()), dw_, delta_.data(), false}};
            dense(rows, c_.hidden, c_.block, kp);
        }
        device_ops::transform(std::size_t{c_.block} * c_.hidden, device_ops::GroupedConv{rows,
            weights_.layer(l, (prefix + "base_kernel").c_str()), delta_.data(), t_.data(), c_.hidden, c_.conv_taps,
            c_.conv_group, side}, stream_, "DFlash grouped convolution");
        device_ops::gpu::copy(rows, t_.data(), t_.span().size_bytes(), device_ops::gpu::device_to_device, stream_);
    }

    void run_layer(unsigned l, unsigned anchor) {
        const auto R = c_.block;
        device_ops::rms(x_.data(), weights_.layer(l, "input_layernorm.weight"), n_.data(), R, c_.hidden, c_.eps, stream_);
        if (c_.conv_taps) conv(l, "attention_conv.", 0U, n_.data());
        const std::array qkv{Projection{weights_.layer(l, "self_attn.q_proj.weight"), qw_, q_.data(), true},
                             Projection{weights_.layer(l, "self_attn.k_proj.weight"), kvw_, k_.data(), true},
                             Projection{weights_.layer(l, "self_attn.v_proj.weight"), kvw_, v_.data(), true}};
        dense(n_.data(), c_.hidden, R, qkv);
        device_ops::norm_rotary(q_.data(), weights_.layer(l, "self_attn.q_norm.weight"), R, c_.heads, anchor, c_.eps,
                                c_.rope_theta, stream_);
        device_ops::norm_rotary(k_.data(), weights_.layer(l, "self_attn.k_norm.weight"), R, c_.kv_heads, anchor, c_.eps,
                                c_.rope_theta, stream_);
        // Window: the reference row r sees context keys > anchor + r - window; the shared bidirectional mask gives every
        // row the same preceding keys, so `window - 1` matches row 0 exactly and row r sees r older keys (r < block).
        device_ops::attend(c_.heads, c_.kv_heads, c_.window - 1U, anchor, R, *rings_[l], q_.data(), k_.data(), v_.data(),
                           att_.data(), zero_.data(), failure_.data(), stream_);
        device_ops::transform(att_.span().size(), core::row_transforms::Bf16Round{att_.data()}, stream_, "round DFlash attention");
        const std::array o{Projection{weights_.layer(l, "self_attn.o_proj.weight"), c_.hidden, branch_.data(), true}};
        dense(att_.data(), qw_, R, o);
        if (c_.conv_taps) conv(l, "attention_conv.", 1U, branch_.data());
        device_ops::transform(x_.span().size(), device_ops::AddRound{x_.data(), branch_.data()}, stream_, "DFlash residual");
        device_ops::rms(x_.data(), weights_.layer(l, "post_attention_layernorm.weight"), n_.data(), R, c_.hidden, c_.eps, stream_);
        if (c_.conv_taps) conv(l, "mlp_conv.", 0U, n_.data());
        const std::array gate_up{Projection{weights_.layer(l, "mlp.gate_proj.weight"), c_.intermediate, gate_.data(), true},
                                 Projection{weights_.layer(l, "mlp.up_proj.weight"), c_.intermediate, up_.data(), true}};
        dense(n_.data(), c_.hidden, R, gate_up);
        device_ops::transform(gate_.span().size(), ops::glu::Bf16SwiGluProduct{gate_.data(), up_.data()}, stream_, "DFlash SwiGLU");
        const std::array down{Projection{weights_.layer(l, "mlp.down_proj.weight"), c_.hidden, branch_.data(), true}};
        dense(gate_.data(), c_.intermediate, R, down);
        if (c_.conv_taps) conv(l, "mlp_conv.", 1U, branch_.data());
        device_ops::transform(x_.span().size(), device_ops::AddRound{x_.data(), branch_.data()}, stream_, "DFlash residual");
    }

    // Top-k of each proposal row's target logits (canonical order) -> cand_ / unary_.
    void top_k(unsigned k) {
        using namespace device_ops;
        const auto K = top_k_;
        transform(std::size_t{k} * kTopChunks, TopChunks{logits_.data(), vocab_out_, K, top_values_.data(), top_ids_.data(),
                  failure_.data()}, stream_, "DFlash top-k chunks");
        // 1024 lists -> 64 -> 4 -> 1 per row.
        transform(std::size_t{k} * 64U, MergeTop{top_values_.data(), top_ids_.data(), kTopChunks, 16U, K, mid_values_.data(),
                  mid_ids_.data()}, stream_, "DFlash top-k merge");
        transform(std::size_t{k} * 4U, MergeTop{mid_values_.data(), mid_ids_.data(), 64U, 16U, K, top_values_.data(),
                  top_ids_.data()}, stream_, "DFlash top-k merge");
        transform(k, MergeTop{top_values_.data(), top_ids_.data(), 4U, 4U, K, unary_.data(), cand_.data()}, stream_,
                  "DFlash top-k merge");
    }

    void forward(const BlockRequest& request, std::vector<std::uint32_t>& proposals, std::vector<float>& confidence) {
        const auto taps = target_.committed_taps();
        if (request.anchor_pos == 0U || request.anchor_pos != taps.size() / c_.tap_width() || request.ids.size() != c_.block ||
            request.prefix != 0U || request.k == 0U || request.k > max_drafts() || request.ids.front() >= c_.vocab)
            throw std::invalid_argument("invalid resident DFlash block");
        const auto anchor = static_cast<unsigned>(request.anchor_pos);
        const auto k = request.k;
        device_ops::gpu::clear(failure_.data(), sizeof(unsigned), stream_);
        refresh(taps);
        const auto R = c_.block;
        device_ops::transform(std::size_t{R} * c_.hidden, device_ops::EmbedBlock{target_.embed_table(), request.ids.front(),
            c_.mask_token, c_.hidden, x_.data()}, stream_, "DFlash block embedding");
        if (const auto* gain = target_.embed_norm())
            device_ops::rms(x_.data(), gain, x_.data(), R, c_.hidden, target_.embed_eps(), stream_);
        for (unsigned l = 0; l < c_.layers; ++l) run_layer(l, anchor);
        device_ops::rms(x_.data(), weights_.get("norm.weight"), n_.data(), R, c_.hidden, c_.eps, stream_);
        // Proposal j reads block row j + 1.
        const float* rows = n_.data() + c_.hidden;
        target_.head(rows, k, logits_.data());
        top_k(k);
        if (c_.selector_rank) {
            const std::array h{Projection{weights_.get("candidate_selector.hidden_projection.weight"), c_.selector_rank,
                                          hsel_.data(), false}};
            dense(rows, c_.hidden, k, h);
            device_ops::transform(std::size_t{k} * top_k_ * top_k_, device_ops::EdgeScores{unary_.data(), cand_.data(),
                hsel_.data(), weights_.get("candidate_selector.predecessor_codebook"),
                weights_.get("candidate_selector.successor_codebook"), request.ids.front(), c_.vocab, top_k_,
                c_.selector_rank, c_.softcap, !c_.softcap_after_correction, edges_.data()}, stream_, "DFlash selector scores");
        } else {
            device_ops::gpu::copy(edges_.data(), unary_.data(), std::size_t{k} * sizeof(float), device_ops::gpu::device_to_device,
                                  stream_);
        }
        device_ops::transform(1U, device_ops::GreedyPath{edges_.data(), cand_.data(), k, top_k_, picks_.data()}, stream_,
                              "DFlash selector path");
        proposals.resize(k);
        confidence.clear();
        device_ops::gpu::copy(proposals.data(), picks_.data(), std::size_t{k} * sizeof(std::uint32_t),
                              device_ops::gpu::device_to_host, stream_);
        device_ops::gpu::copy(&completed_failure_, failure_.data(), sizeof(completed_failure_), device_ops::gpu::device_to_host,
                              stream_);
        device_ops::gpu::synchronize(stream_);
        if (completed_failure_ || std::ranges::any_of(proposals, [&](std::uint32_t id) { return id >= vocab_out_; }))
            throw std::runtime_error("DFlash rejected non-finite attention or proposal logits");
    }

    Target target_;
    device_ops::gpu::Stream stream_;
    ResidentWeights weights_;
    const Config& c_;
    unsigned qw_, kvw_, dw_, top_k_, vocab_out_;
    ForwardBinding forward_binding_;
    ParallelBlockDrafter controller_;
    std::vector<std::unique_ptr<device_ops::KvRing>> rings_;
    device_ops::gpu::Buffer<float> x_, n_, t_, branch_, q_, att_, k_, v_, gate_, up_, delta_;
    device_ops::gpu::Buffer<std::uint16_t> zero_;
    device_ops::gpu::Buffer<unsigned> failure_;
    device_ops::gpu::Buffer<float> wide_, fused_, ck_, cv_, logits_, top_values_;
    device_ops::gpu::Buffer<std::uint32_t> top_ids_;
    device_ops::gpu::Buffer<float> mid_values_;
    device_ops::gpu::Buffer<std::uint32_t> mid_ids_;
    device_ops::gpu::Buffer<float> unary_;
    device_ops::gpu::Buffer<std::uint32_t> cand_;
    device_ops::gpu::Buffer<float> hsel_, edges_;
    device_ops::gpu::Buffer<std::uint32_t> picks_;
    unsigned projected_ = 0U;
    unsigned completed_failure_ = 0U;
    bool failed_ = false;
};

}  // namespace strix::spec::dflash
