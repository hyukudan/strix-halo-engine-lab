#pragma once

#include "core/markov_head_device.hpp"
#include "inkling2_mini_dspark_context.hpp"
#include "inkling2_mini_resident_decoder.hpp"
#include "ops/glu/swiglu_rows.hpp"

namespace strix::inkling2 {

namespace mini_dspark_device {

#if defined(__HIPCC__)
struct HipAttentionPolicy
    : attn_core::dense::PreparedBf16Policy<kHeadWidth, kQueryHeads / kKvHeads, false, false, 1U, 16U, true> {
    static constexpr float kScaleLog2 = 0.08838834764831845F * 1.4426950408889634F;
};
#endif

inline void attend(unsigned window, unsigned anchor, const KvCache& cache, const float* q, const float* k,
                   const float* v, float* output, const std::uint16_t* zeros, unsigned* failure,
                   mini_gpu::Stream stream) {
    constexpr auto block = MiniDSparkCheckpoint::block;
#if defined(__HIPCC__)
    const attn_core::PreparedGqaAttentionRows rows{q, k, v, nullptr, nullptr, output, anchor, block};
    const attn_core::dense::PreparedBf16Cache view{cache.keys.data(), cache.values.data(), zeros, cache.slots, cache.wrapped};
    mini_gpu::check(attn_core::dense::append_prepared_bf16<HipAttentionPolicy, true>(
        kKvHeads, window, 0U, rows, view, failure, stream, {}, 1U), "attend DSpark block");
#else
    (void)zeros;
    const cuda_core::AttentionDims dims{kQueryHeads, kKvHeads, kHeadWidth, 0.08838834764831845F, cache.slots};
    const cuda_core::PreparedBf16Flash policy{nullptr, nullptr, 0U, window, cache.wrapped, true};
    mini_gpu::check(cuda_core::append_attention_prepared_bf16(dims, policy, q, k, v,
        reinterpret_cast<__nv_bfloat16*>(cache.keys.data()), reinterpret_cast<__nv_bfloat16*>(cache.values.data()),
        anchor, block, output, stream, cuda_core::FlashPipe::pad2, failure), "attend DSpark block");
#endif
}

}  // namespace mini_dspark_device

// Request-owned binding. The verifier controls target publication; the shared
// controller supplies the next anchor and the number of proposals to read.
class MiniResidentDSpark final : public spec::ChainDrafter {
    using D = MiniDSparkCheckpoint;
    static constexpr unsigned rank = D::markov_rank;
    static constexpr unsigned recurrent_width = 2U * rank;
    static constexpr unsigned context_width = 2U * kHiddenWidth;
    static constexpr unsigned affine_width = recurrent_width + context_width;
    static constexpr unsigned affine_outputs = 3U * rank;
    static constexpr unsigned max_drafts = D::block - 1U;

public:
    MiniResidentDSpark(MiniResidentDecoder& target, unsigned positions, const std::string& path)
        : target_(target), stream_(target.stream_.get()), weights_(path, stream_), context_(positions, weights_),
          forward_binding_(*this), controller_(D::policy(), forward_binding_), hidden_(D::block * kHiddenWidth),
          normalized_(hidden_.span().size()),
          branch_(hidden_.span().size()), query_(D::block * kQueryWidth), attended_(query_.span().size()),
          key_(D::block * kKvWidth), value_(key_.span().size()), gate_(D::block * D::intermediate),
          up_(gate_.span().size()), zero_(kHeadWidth), failure_(1U), features_(max_drafts * context_width),
          context_affine_(max_drafts * affine_outputs), recurrent_affine_(affine_outputs), recurrent_(recurrent_width),
          latent_(rank), context_matrix_(affine_outputs * context_width), recurrent_matrix_(affine_outputs * recurrent_width),
          folded_(std::size_t{kOutputVocabulary} * rank), bias_(kOutputVocabulary), scores_(std::size_t{max_drafts} * kOutputVocabulary),
          correction_(kOutputVocabulary),
#if defined(__HIPCC__)
          pick_scratch_(static_cast<std::size_t>(hip_greedy_rows_scratch_bytes(1U, kOutputVocabulary))),
          pick_reply_(1U),
#else
          pick_scratch_(head_argmax::default_chunks(kOutputVocabulary)),
#endif
          picks_(max_drafts) {
        mini_gpu::clear(zero_.data(), zero_.span().size_bytes(), stream_);
        const auto* affine = weights_.get("markov_head.rnn_proj.weight");
        transform(context_matrix_.span().size(),
                  core::markov_device::CopyAffineSlice{affine, context_matrix_.data(), affine_width, context_width,
                                                       recurrent_width});
        transform(recurrent_matrix_.span().size(),
                  core::markov_device::CopyAffineSlice{affine, recurrent_matrix_.data(), affine_width, recurrent_width, 0U});
        // The folded head depends only on weights: build it here, at load, not in the first draft round (it is a
        // 201K x (rank + 1) x 2,304 FP64 dot product, seconds on gfx1151, which landed inside the timed decode).
        fold();
        // Set up on the same stream as the target. Failed setup ends this request.
        mini_gpu::synchronize(stream_);
    }

    void start(std::uint32_t anchor, std::uint32_t position) {
        if (target_.committed_taps().size() != MiniDSparkTapPlan::elements(position))
            throw std::logic_error("invalid resident DSpark start frontier");
        controller_.start(anchor, position);
    }

    void draft(std::span<const std::uint32_t> prefix, std::uint32_t count, std::vector<std::uint32_t>& out) override {
        if (failed_) throw std::logic_error("invalid resident DSpark session");
        if (controller_.anchor_pos() > kMaxContext - D::block) {
            out.clear();
            return;
        }
        try {
            controller_.draft(prefix, count, out);
        } catch (...) {
            failed_ = true;
            try { mini_gpu::synchronize(stream_); } catch (...) {}
            throw;
        }
    }

    void accept(std::span<const std::uint32_t> emitted) override { controller_.accept(emitted); }

private:
    // Context is caught up lazily from the target, including lookup-only and
    // serial rounds. This callback runs after the target commits.
    class ForwardBinding final : public spec::BlockForward {
    public:
        explicit ForwardBinding(MiniResidentDSpark& owner) : owner_(owner) {}
        void forward(const spec::BlockRequest& request, std::vector<std::uint32_t>& proposals,
                     std::vector<float>& confidence) override {
            owner_.forward(request, proposals, confidence);
        }
        void accept(std::span<const std::uint32_t>) override {}
    private:
        MiniResidentDSpark& owner_;
    };

    template <class Transform>
    void transform(std::size_t count, Transform operation) {
        mini_gpu::check(core::pointwise::elements(count, operation, stream_), "enqueue DSpark transform");
    }

    void project(const float* input, unsigned width, unsigned rows, const std::uint16_t* weights,
                 unsigned outputs, float* output, bool round = true) {
        const std::array view{core::Bf16RowProjection{weights, outputs, output, round}};
        target_.backend_.dense(input, width, rows, view);
    }

    void residual_norm(const std::uint16_t* gain) {
        transform(hidden_.span().size(), core::pointwise::Add{hidden_.data(), branch_.data(), normalized_.data()});
        transform(hidden_.span().size(), core::row_transforms::CopyBf16Round{normalized_.data(), hidden_.data()});
        mini_dspark_device::rms(normalized_.data(), gain, normalized_.data(), D::block, kHiddenWidth, stream_);
    }

    void run_layer(unsigned layer, unsigned anchor) {
        const auto prefix = "layers." + std::to_string(layer) + ".";
        const auto get = [&](std::string_view suffix) { return weights_.get(prefix + std::string(suffix)); };
        const std::array qkv{
            core::Bf16RowProjection{get("self_attn.q_proj.weight"), kQueryWidth, query_.data(), true},
            core::Bf16RowProjection{get("self_attn.k_proj.weight"), kKvWidth, key_.data(), true},
            core::Bf16RowProjection{get("self_attn.v_proj.weight"), kKvWidth, value_.data(), true}};
        target_.backend_.dense(normalized_.data(), kHiddenWidth, D::block, qkv);
        mini_dspark_device::norm_rotary(query_.data(), get("self_attn.q_norm.weight"), D::block, kQueryHeads, anchor, stream_);
        mini_dspark_device::norm_rotary(key_.data(), get("self_attn.k_norm.weight"), D::block, kKvHeads, anchor, stream_);
        mini_dspark_device::attend(D::window(layer), anchor, context_.cache(layer), query_.data(), key_.data(),
                                  value_.data(), attended_.data(), zero_.data(), failure_.data(), stream_);
        transform(attended_.span().size(), core::row_transforms::Bf16Round{attended_.data()});
        project(attended_.data(), kQueryWidth, D::block, get("self_attn.o_proj.weight"), kHiddenWidth, branch_.data());
        residual_norm(get("post_attention_layernorm.weight"));
        const std::array gate_up{
            core::Bf16RowProjection{get("mlp.gate_proj.weight"), D::intermediate, gate_.data(), true},
            core::Bf16RowProjection{get("mlp.up_proj.weight"), D::intermediate, up_.data(), true}};
        target_.backend_.dense(normalized_.data(), kHiddenWidth, D::block, gate_up);
        transform(gate_.span().size(), ops::glu::Bf16SwiGluProduct{gate_.data(), up_.data()});
        project(gate_.data(), D::intermediate, D::block, get("mlp.down_proj.weight"), kHiddenWidth, branch_.data());
        const auto* next = layer + 1U < D::layers
            ? weights_.get("layers." + std::to_string(layer + 1U) + ".input_layernorm.weight") : weights_.get("norm.weight");
        residual_norm(next);
    }

    void fold() {
        using namespace core::markov_device;
        const auto* weight = weights_.get("markov_head.hidden_out.weight");
        const auto* bias = weights_.get("markov_head.hidden_out.bias");
        const auto run = [&](auto head) {
            transform(std::size_t{kOutputVocabulary} * (rank + 1U),
                Fold{head, weight, bias, folded_.data(), bias_.data(), kHiddenWidth, rank});
        };
        if (target_.weights_.compact_head_bound()) {
            const auto& head = target_.weights_.compact_head();
            const auto stride = static_cast<std::size_t>(head.row_bytes);
            if (head.storage == weight::kQ4G64WeightStorage)
                run(LowBitMatrix<weight::Q4G64Storage>{head.bytes.data(), stride});
            else if (head.storage == weight::kQ6G64WeightStorage)
                run(LowBitMatrix<weight::Q6G64Storage>{head.bytes.data(), stride});
            else
                run(Q8Matrix{head.bytes.data(), stride});
        } else {
            run(Bf16Matrix{target_.weights_.bf16("unembed.weight").data(), kHiddenWidth});
        }
    }

    void pick(float* scores, std::uint32_t* token) {
#if defined(__HIPCC__)
        const ExecutionBufferAddress logits{{}, reinterpret_cast<std::uintptr_t>(scores), kOutputVocabulary * sizeof(float), 0U};
        const ExecutionBufferAddress scratch{{}, reinterpret_cast<std::uintptr_t>(pick_scratch_.data()),
                                             pick_scratch_.span().size_bytes(), 0U};
        enqueue_hip_greedy_rows_device(std::span(&logits, 1U), kOutputVocabulary, scratch, pick_reply_.data(), stream_);
        transform(1U, GreedyToken{pick_reply_.data(), token});
#else
        namespace argmax = strix::cuda::core::head_argmax;
        const auto chunks = head_argmax::default_chunks(kOutputVocabulary);
        mini_gpu::check(argmax::pick_rows({scores, 0U, kOutputVocabulary, 1U, 0U, chunks, pick_scratch_.data()},
            argmax::EmitIndexOrNone{token}, stream_), "select DSpark proposal");
#endif
    }

    void enqueue_proposals(unsigned count, std::uint32_t anchor) {
        using namespace core::markov_device;
        constexpr auto hidden_bytes = kHiddenWidth * sizeof(float);
        for (unsigned j = 0; j < count; ++j) {
            auto* feature_row = features_.data() + j * context_width;
            mini_gpu::copy(feature_row, normalized_.data() + (j + 1U) * kHiddenWidth,
                           hidden_bytes, mini_gpu::device_to_device, stream_);
            mini_gpu::copy(feature_row + kHiddenWidth, context_.markov(), hidden_bytes, mini_gpu::device_to_device, stream_);
        }
        project(features_.data(), context_width, count, context_matrix_.data(), affine_outputs, context_affine_.data(), false);
        // The target head reads only the draft rows, not the picks: one multi-row pass reads the head once for all
        // candidates instead of once per candidate inside the serial Markov chain.
        target_.project_head(normalized_.data() + kHiddenWidth, scores_.span(), false, count);
        mini_gpu::clear(recurrent_.data(), recurrent_.span().size_bytes(), stream_);
        const auto* table = weights_.get("markov_head.markov_w1.weight");
        const auto* affine_bias = weights_.get("markov_head.rnn_proj.bias");
        for (unsigned j = 0; j < count; ++j) {
            auto* next = picks_.data() + j;
            transform(rank, PreviousEmbedding{table, j == 0U ? nullptr : next - 1U, anchor,
                                              kVocabulary, rank, recurrent_.data() + rank});
            project(recurrent_.data(), recurrent_width, 1U, recurrent_matrix_.data(),
                    affine_outputs, recurrent_affine_.data(), false);
            transform(rank, Gate{context_affine_.data() + j * affine_outputs, recurrent_affine_.data(), affine_bias,
                                 recurrent_.data(), latent_.data(), rank});
            auto* scores = scores_.data() + std::size_t{j} * kOutputVocabulary;
            project(latent_.data(), rank, 1U, folded_.data(), kOutputVocabulary, correction_.data(), false);
            transform(kOutputVocabulary, ComposeHead{scores, correction_.data(), bias_.data(), kLogitSoftcap});
            pick(scores, next);
        }
    }

    void forward(const spec::BlockRequest& request, std::vector<std::uint32_t>& proposals,
                 std::vector<float>& confidence) {
        const auto taps = target_.committed_taps();
        if (request.anchor_pos == 0U || request.anchor_pos != taps.size() / MiniDSparkTapPlan::width ||
            request.anchor_pos > kMaxContext - D::block || request.ids.size() != D::block || request.prefix != 0U ||
            request.k == 0U || request.k > max_drafts || request.ids.front() >= kVocabulary)
            throw std::invalid_argument("invalid resident DSpark block");
        const auto anchor = static_cast<unsigned>(request.anchor_pos);
        mini_gpu::clear(failure_.data(), sizeof(unsigned), stream_);
        context_.refresh(taps, target_.backend_, target_.max_rows_, failure_.data(), stream_);
        const auto token = request.ids.front();
        const auto embed = target_.weights_.bf16("embed.weight");
        transform(kHiddenWidth, core::row_transforms::WidenRepeatedBf16Row{
            embed.data() + std::size_t{token} * kHiddenWidth, hidden_.data(), kHiddenWidth});
        transform(max_drafts * kHiddenWidth, core::row_transforms::WidenRepeatedBf16Row{
            weights_.get("mask_embed"), hidden_.data() + kHiddenWidth, kHiddenWidth});
        mini_dspark_device::rms(hidden_.data(), target_.weights_.bf16("embed_norm.weight").data(), hidden_.data(),
                                1U, kHiddenWidth, stream_);
        mini_dspark_device::rms(hidden_.data(), weights_.get("layers.0.input_layernorm.weight"), normalized_.data(),
                                D::block, kHiddenWidth, stream_);
        for (unsigned layer = 0; layer < D::layers; ++layer) run_layer(layer, anchor);
        enqueue_proposals(request.k, token);
        proposals.resize(request.k);
        confidence.clear();
        mini_gpu::copy(proposals.data(), picks_.data(), request.k * sizeof(std::uint32_t), mini_gpu::device_to_host, stream_);
        mini_gpu::copy(&completed_failure_, failure_.data(), sizeof(completed_failure_), mini_gpu::device_to_host, stream_);
        mini_gpu::synchronize(stream_);
        if (completed_failure_ || std::ranges::any_of(proposals, [](unsigned id) { return id >= kOutputVocabulary; }))
            throw std::runtime_error("DSpark rejected non-finite attention or proposal logits");
    }

#if defined(__HIPCC__)
    struct GreedyToken {
        const HipGreedyRowsReply* reply;
        std::uint32_t* result;
        __device__ void operator()(std::size_t) const { *result = reply->status ? head_argmax::kNone : reply->token; }
    };
#endif

    MiniResidentDecoder& target_;
    mini_gpu::Stream stream_;
    MiniDSparkDeviceWeights weights_;
    MiniDSparkDeviceContext context_;
    ForwardBinding forward_binding_;
    spec::ParallelBlockDrafter controller_;
    mini_gpu::Buffer<float> hidden_, normalized_, branch_, query_, attended_, key_, value_, gate_, up_;
    mini_gpu::Buffer<std::uint16_t> zero_;
    mini_gpu::Buffer<unsigned> failure_;
    mini_gpu::Buffer<float> features_, context_affine_, recurrent_affine_, recurrent_, latent_;
    mini_gpu::Buffer<std::uint16_t> context_matrix_, recurrent_matrix_, folded_, bias_;
    mini_gpu::Buffer<float> scores_, correction_;
#if defined(__HIPCC__)
    mini_gpu::Buffer<std::uint8_t> pick_scratch_;
    mini_gpu::Buffer<HipGreedyRowsReply> pick_reply_;
#else
    mini_gpu::Buffer<head_argmax::Pick> pick_scratch_;
#endif
    mini_gpu::Buffer<std::uint32_t> picks_;
    unsigned completed_failure_{};
    bool failed_{};
};

}  // namespace strix::inkling2
