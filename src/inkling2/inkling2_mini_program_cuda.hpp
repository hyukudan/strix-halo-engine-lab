#pragma once

#include "exec/catalog_cuda_wr1.hpp"
#include "exec/cuda_decode_lowering.hpp"
#include "exec/cuda_decode_wr2.hpp"
#include "inkling2_mini_attention_prepare_device.hpp"
#include "inkling2_mini_expert_layout.hpp"
#include "inkling2_mini_resident_forward.hpp"
#include "inkling2_mini_verify.hpp"
#include "strix/graph/launch_plan.hpp"
#include "strix/serve/serial_generation_cli.hpp"
#include "strix/session/step_replay.hpp"
#include "strix/spec/engine_target.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if defined(STRIX_INKLING2_STEP_PROGRAM)
#include "inkling2_mini_resident_decoder.hpp"
#include <iostream>
#else
#include "exec/scheduled_stage_graph.hpp"
#endif

namespace strix::inkling2 {

#if !defined(STRIX_INKLING2_STEP_PROGRAM)
// Semantic input to the CUDA lowering. Native launches, including the
// resident copies and replies, need explicit exact bindings before execution.
class MiniCudaDecodePlan {
public:
    explicit MiniCudaDecodePlan(const graph::ModelDecl& declaration) : scheduled_(declaration) {}

    [[nodiscard]] const graph::StageGraph& stages() const noexcept { return scheduled_.stages(); }

    [[nodiscard]] graph::LaunchPlan at(std::uint32_t position) const {
        return graph::launch_plan(stages(), position, 1U, {.mode = graph::Fusion::today});
    }

private:
    exec::ScheduledStageGraph scheduled_;
};
#endif

struct MiniCudaStepStorage {
    unsigned max_rows;
    std::span<float> controller, attention;
    const float* lag_weights;
    std::span<const moe_core::device_grouped::ExpertWeights> matrices;
    unsigned* route_ids;
    float* route_logits;
    float* route_weights;
    std::uint16_t* expert_activation;
    std::span<unsigned> flags;
};

struct MiniCudaStepRows {
    std::span<float> embeddings, hidden, logits;
    head_argmax::Pick* partials;
    std::uint32_t* picked;
    std::uint32_t* reply;
};

struct MiniCudaVerifyEmbedding {
    const exec::StepBlock* step;
    const std::uint16_t* table;
    float* output;
    unsigned rows;
};

struct MiniCudaVerifyState {
    static constexpr unsigned entries = 6U * kLogicalVisits;
    state::StateTable directory;
    unsigned stride;
    std::uint16_t* pending_keys;
    std::uint16_t* pending_values;
    float* pending_history;
    const MiniSessionStatePlan& plan;

    static constexpr unsigned key_instance(LayerVisit visit) { return 6U * visit.logical_layer; }
    static constexpr unsigned history_instance(LayerVisit visit, MiniConvolutionStream path) {
        return key_instance(visit) + 2U + static_cast<unsigned>(path);
    }

    [[nodiscard]] std::size_t kv_offset(LayerVisit visit) const {
        return std::size_t{visit.logical_layer} * stride * kKvWidth;
    }
    [[nodiscard]] float* history(LayerVisit visit, MiniConvolutionStream path) const {
        return pending_history + plan.visit(visit).convolution_offsets[static_cast<unsigned>(path)] * stride;
    }
    [[nodiscard]] attn_core::PreparedStepCache cache(LayerVisit visit, MiniVisitKvCache kv) const {
        const auto ring = kv.wrapped ? kv.slots : 0U;
        constexpr auto bytes = kKvWidth * sizeof(std::uint16_t);
        const auto key = key_instance(visit);
        return {{{directory, key}, bytes, ring, false}, {{directory, key + 1U}, bytes, ring, false},
                pending_keys + kv_offset(visit), pending_values + kv_offset(visit)};
    }
};

struct MiniCudaVerifyRows {
    head_argmax::Pick2* partials;
    std::span<exec::RowOut> picked, reply;
    std::span<float> logits_reply;
};

template <class Args, class BindOther>
bool append_mini_verify_leaf(exec::CudaProgramRecipe& recipe, const Args& args, const exec::StepBlock* descriptor,
                             unsigned rows, const exec::cuda_decode::wr1::Thunks& thunks, const BindOther& other) {
    const exec::cuda_decode::wr1::Bindings binding(descriptor, rows, thunks);
    if constexpr (requires { binding.bind(args); }) {
        binding.append(recipe, args);
        return true;
    } else {
        return other(recipe, args, rows);
    }
}

[[nodiscard]] inline exec::ProgramKey mini_cuda_step_key(unsigned rows) {
    require_mini_verify_width(rows);
    return rows == 1U ? exec::ProgramKey{} : exec::ProgramKey{.phase = exec::Phase::verify, .rows = rows};
}

struct MiniCudaStepRecipe {
    exec::CudaProgramRecipe leaves;
    unsigned missing = 0U;

    [[nodiscard]] exec::CudaRecipeProgram compile(exec::ProgramKey key) const {
        require_mini_verify_leaves(key.rows, missing);
        return leaves.compile(key);
    }
};

// Bind the resident traversal's exact native leaves. The same traversal
// supplies visits, residual banks, copies and visit-local state to both routes.
template <class Weights, class Bind, class Dispatch, class Finish, class BindVerify>
class MiniCudaStepLowering {
    using Shape = MiniPreparedGqaShape;
    using Mode = exec::cuda_decode::Bf16DenseMode;
    static constexpr float epsilon = 1.0e-6F;

public:
    MiniCudaStepLowering(const Weights& weights, MiniCudaStepStorage storage, const exec::StepBlock* descriptor,
                        const Bind& bind, const Dispatch& dispatch, const Finish& finish,
                        const BindVerify& bind_verify, const MiniCudaVerifyState* verify_state = nullptr)
        : weights_(weights), storage_(storage), descriptor_(descriptor), bind_(bind), dispatch_(dispatch),
          finish_(finish), bind_verify_(bind_verify), verify_state_(verify_state) {}

    bool try_step_graph(MiniResidentChunk chunk, auto& work) {
        if (!mini_verify_width(chunk.rows) || chunk.rows > storage_.max_rows)
            throw std::invalid_argument("invalid Inkling CUDA step program width");
        rows_ = chunk.rows;
        return dispatch_(chunk, [&] {
            recipe_.leaves.stage("flags");
            recipe_.leaves.clear({storage_.flags.data(), storage_.flags.size_bytes(), 0}, rows_);
            work();
            return std::move(recipe_);
        });
    }
    void finish() { finish_(); }

    void embed(std::span<float> image) {
        using namespace exec::cuda_decode;
        recipe_.leaves.stage("embed");
        const auto* table = weights_.bf16("embed.weight").data();
        if (rows_ == 1U)
            add_leaf(Embedding{table, &descriptor_->row[0].token, image.data(), kVocabulary, kHiddenWidth}, true);
        else
            add_verify(MiniCudaVerifyEmbedding{descriptor_, table, image.data(), rows_}, MiniVerifyMissing::embedding);
        norm(image, "embed_norm.weight", image);
    }

    void head(MiniConstRows hidden, MiniCudaStepRows rows, MiniCudaVerifyRows verify = {}) {
        using namespace exec::cuda_decode;
        recipe_.leaves.visit(-1);
        recipe_.leaves.stage("head");
        const std::array projection{core::Bf16RowProjection{weights_.bf16("unembed.weight").data(),
                                                          kOutputVocabulary, rows.logits.data()}};
        dense(hidden.data(), kHiddenWidth, projection);
        add_leaf(Bf16Softcap{rows.logits.data(), rows.logits.size() / rows_, kLogitSoftcap});
        if (rows_ == 1U) {
            const GreedyPartial reduction{rows.logits.data(), rows.partials, kOutputVocabulary,
                                         head_argmax::default_chunks(kOutputVocabulary)};
            add_leaf(reduction);
            add_leaf(GreedyFinal{reduction, rows.picked});
            recipe_.leaves.copy(exec::CudaNodeKind::copy_d2h_pinned,
                                {rows.reply, rows.picked, sizeof(*rows.reply)}, rows_);
        } else {
            require(rows.logits.size() == std::size_t{rows_} * kOutputVocabulary &&
                    verify.logits_reply.size() == rows.logits.size() && verify.picked.size() == rows_ &&
                    verify.reply.size() == rows_);
            add_verify(exec::cuda_decode::wr1::Greedy{rows.logits.data(), verify.partials, verify.picked.data(),
                       kOutputVocabulary, head_argmax::default_chunks(kOutputVocabulary)}, MiniVerifyMissing::argmax);
            recipe_.leaves.copy(exec::CudaNodeKind::copy_d2h_pinned,
                                {verify.reply.data(), verify.picked.data(), verify.picked.size_bytes()}, rows_);
            recipe_.leaves.copy(exec::CudaNodeKind::copy_d2h_pinned,
                                {verify.logits_reply.data(), rows.logits.data(), rows.logits.size_bytes()}, rows_);
        }
    }

    template <class T>
    void copy(std::span<const T> input, std::span<T> output) {
        if (input.data() != output.data())
            recipe_.leaves.copy(exec::CudaNodeKind::copy_d2d, {output.data(), input.data(), input.size_bytes()}, rows_);
    }

    void norm(MiniConstRows input, std::string_view name, MiniMutableRows output) {
        rms(input.data(), weights_.bf16(name).data(), output.data());
    }

    void layer_input(LayerVisit visit, MessagePath path, MiniConstRows raw,
                     std::span<const MiniConstRows> previous, MiniMutableRows output) {
        recipe_.leaves.visit(static_cast<std::int32_t>(visit.logical_layer));
        recipe_.leaves.stage(mini_controller_path(path));
        const auto layer = visit.physical_layer;
        const auto* scale = weights_.bf16(parameter(layer, mini_input_norm_suffix(visit, path))).data();
        if (visit.pass == 0U) {
            rms(raw.data(), scale, output.data());
            return;
        }
        const bool key_value = path == MessagePath::key || path == MessagePath::value;
        const auto queries = weights_.f32(key_value ? "middle_cycle_cross_pass_kv_queries"
                                                    : "middle_cycle_cross_pass_queries");
        const auto query = 2U * (layer - kCoreFirst) + (path == MessagePath::value || path == MessagePath::moe ? 1U : 0U);
        const auto stem = parameter(layer, std::string("controllers.") + std::string(mini_controller_path(path)) + ".");
        exec::cuda_decode::lower_low_rank_controller(kHiddenWidth, kControllerRank, epsilon,
            queries.subspan(std::size_t{query} * kHiddenWidth, kHiddenWidth), raw, previous, output, storage_.controller,
            weights_.bf16(stem + "down_weight").data(), weights_.bf16(stem + "forget_weight").data(),
            weights_.bf16(stem + "gain_weight").data(), scale, add(), rows_);
    }

    void residual_read(std::span<const MiniConstRows> blocks, MiniConstRows current, std::string_view name,
                       std::span<const MiniMutableRows> output) {
        recipe_.leaves.stage("residual_read");
        core::residual_mix::MixArgs args{};
        args.queries = weights_.f32(name).data();
        args.rows = rows_;
        args.width = kHiddenWidth;
        args.epsilon = epsilon;
        args.bank_count = static_cast<unsigned>(blocks.size());
        args.query_count = static_cast<unsigned>(output.size());
        for (std::size_t i = 0; i < blocks.size(); ++i) args.banks[i] = blocks[i].data();
        args.banks[args.bank_count++] = current.data();
        for (std::size_t i = 0; i < output.size(); ++i) args.output[i] = output[i].data();
        add_leaf(args);
    }

    void capture_message_bank(const std::array<MiniConstRows, 2>& source,
                              const std::array<MiniMutableRows, 2>& normalized) {
        recipe_.leaves.stage("message_bank");
        for (std::size_t i = 0; i < source.size(); ++i)
            add_leaf(core::residual_mix::NormalizeArgs{source[i].data(), normalized[i].data(),
                                                       rows_, kHiddenWidth, epsilon});
    }

    void add(MiniConstRows lhs, MiniConstRows rhs, MiniMutableRows output, bool round_bf16) {
        recipe_.leaves.stage("residual_add");
        add_leaf(core::residual_mix::AddArgs{lhs.data(), rhs.data(), output.data(), output.size() / rows_, round_bf16});
    }

    void attention(LayerVisit visit, MiniResidentChunk chunk, MiniConstRows query, MiniConstRows key,
                   MiniConstRows value, const MiniResidentVisitState& state, MiniMutableRows output) {
        using namespace exec::cuda_decode;
        recipe_.leaves.stage("attention");
        auto arena = storage_.attention;
        const auto take = [&](unsigned width) {
            auto* ptr = arena.data();
            arena = arena.subspan(std::size_t{storage_.max_rows} * width);
            return ptr;
        };
        const MiniProjectionInputs inputs{query.data(), key.data(), value.data()};
        const MiniProjectionWorkspace work{
            take(kKvWidth), take(kKvWidth), take(kKvHeads * kConvTaps), take(kKvHeads * kConvTaps),
            take(kQueryHeads * kRelativeWidth), state.history[0].data(), state.history[1].data(),
        };
        const MiniPreparedAttention prep{
            take(kQueryWidth), take(kKvWidth), take(kKvWidth), take(kQueryHeads * kRelativeExtent),
            take(kQueryHeads), take(kQueryWidth),
        };
        auto* attended = take(kQueryWidth);
        auto* gated = take(kQueryWidth);
        const bool full = is_full_attention(visit.physical_layer);
        const auto weights = mini_device_projection_weights(weights_, visit,
            mini_convolution_layer(storage_.lag_weights, visit));
        const auto* position = &descriptor_->row[0].position;
        const Attention lowering{*this, visit, MiniConvolutionStream::key, work.value_history};
        require(attn_core::enqueue_prepared_gqa<Shape>(lowering, full, chunk.first_position, rows_,
                                                       inputs, weights, work, prep, nullptr, position));
        const attn_core::PreparedGqaAttentionRows prepared{prep.query, prep.key, prep.value, prep.relative_logits,
            prep.sinks, attended, rows_ == 1U ? chunk.first_position : 0U, rows_};
        if (rows_ == 1U) {
            add_leaf(AppendKvBf16{prep.key, prep.value, state.kv.keys, state.kv.values, position,
                            kQueryHeads, kKvHeads, kHeadWidth, state.kv.slots, chunk.first_position,
                            Shape::kScoreScale, state.kv.wrapped});
            add_leaf(PreparedFlashBf16{prep.query, prep.relative_logits, prep.sinks, state.kv.keys, state.kv.values,
                                  attended, position, kQueryHeads, kKvHeads, state.kv.slots, chunk.first_position,
                                  kRelativeExtent, full ? 0U : kSlidingWindow, Shape::kScoreScale, state.kv.wrapped});
            add_leaf(PreparedFinite{prepared, kQueryHeads, kKvHeads, kHeadWidth,
                               kRelativeExtent, storage_.flags.data() + visit.logical_layer});
        } else {
            if (verify_state_) {
                const auto cache = verify_state_->cache(visit, state.kv);
                add_verify(attn_core::PreparedStepKv{descriptor_, cache, prep.key, prep.value, kKvHeads, kHeadWidth},
                           MiniVerifyMissing::kv);
                add_verify(attn_core::PreparedStepFlash{descriptor_, cache, prep.query, prep.relative_logits, prep.sinks,
                           attended, kQueryHeads, kKvHeads, kRelativeExtent, full ? 0U : kSlidingWindow, Shape::kScoreScale},
                           MiniVerifyMissing::attention);
            } else {
                refuse(MiniVerifyMissing::kv);
                refuse(MiniVerifyMissing::attention);
            }
            add_verify(attn_core::PreparedStepFinite{descriptor_, prepared, kQueryHeads, kKvHeads, kHeadWidth,
                       kRelativeExtent, storage_.flags.data() + std::size_t{visit.logical_layer} * rows_},
                       MiniVerifyMissing::finite);
        }
        require(attn_core::enqueue_prepared_gqa_output<Shape>(lowering, rows_,
            weights.projection(MiniAttentionProjection::output), attended, prep.gate_logits, gated, output.data()));
    }

    void convolve(LayerVisit visit, MiniConvolutionStream path, MiniConstRows input, MiniMutableRows history,
                  MiniMutableRows output) {
        recipe_.leaves.stage("post_convolution");
        const auto* weights = mini_post_convolution_weights(storage_.lag_weights, visit, path == MiniConvolutionStream::moe);
        require(attn_core::enqueue_prepared_gqa_post_convolution<Shape>(
             Attention{*this, visit, path}, rows_, weights, input.data(), history.data(), output.data()));
    }

    void moe(LayerVisit visit, MiniConstRows input, MiniMutableRows output) {
        using namespace exec::cuda_decode;
        recipe_.leaves.stage("moe");
        const auto layer = visit.physical_layer;
        const std::array router{core::Bf16RowProjection{weights_.bf16(parameter(layer, "mlp.gate.weight")).data(),
                                                       kExperts, storage_.route_logits}};
        lower_dense(input.data(), kHiddenWidth, router, true, add());
        auto* failure = storage_.flags.data() + std::size_t{kLogicalVisits + visit.logical_layer} * rows_;
        const auto* bias = weights_.f32(parameter(layer, "mlp.gate.bias")).data();
        static_assert(kExpertsPerToken == kPackedExpertsPerToken);
        const moe_core::microscaled_r1::Args args{
            storage_.matrices.data() + std::size_t{layer} * kExperts, storage_.route_ids, storage_.route_weights,
            failure, input.data(), storage_.expert_activation, output.data(), kExperts, kHiddenWidth, kExpertWidth,
            kMlpGateSoftcap, kMlpUpSoftcap,
        };
        const auto format = mini_expert_format(layer);
        if (rows_ == 1U) {
            add_leaf(CheckedWaveRoute{storage_.route_logits, bias, storage_.route_ids, storage_.route_weights, failure});
            add_leaf(PackedGateUp{args, format});
            add_leaf(PackedDownCombine{args, format});
        } else {
            add_verify(StepCheckedWaveRoute{descriptor_, storage_.route_logits, bias, storage_.route_ids,
                       storage_.route_weights, failure}, MiniVerifyMissing::route);
            add_verify(StepPackedGateUp{{descriptor_, args}, format}, MiniVerifyMissing::gate_up);
            add_verify(StepPackedDownCombine{{descriptor_, args}, format}, MiniVerifyMissing::down_combine);
        }
    }

private:
    static std::string parameter(unsigned layer, std::string_view suffix) {
        return "layers." + std::to_string(layer) + "." + std::string(suffix);
    }

    static void require(bool valid) {
        if (!valid) throw std::invalid_argument("invalid Inkling CUDA step program binding");
    }

    void rms(const float* input, const std::uint16_t* scale, float* output) {
        rms(exec::cuda_decode::RmsBf16{input, scale, output, 1U, kHiddenWidth, epsilon});
    }

    void rms(exec::cuda_decode::RmsBf16 args) { add_leaf(args, true); }

    void refuse(MiniVerifyMissing reason) { recipe_.missing |= mini_verify_bit(reason); }

    template <class Args>
    void add_verify(const Args& args, MiniVerifyMissing reason) {
        try {
            if (!bind_verify_(recipe_.leaves, args, descriptor_, rows_)) refuse(reason);
        } catch (const std::invalid_argument& error) {
            throw std::invalid_argument("Inkling verify R=" + std::to_string(rows_) + " cannot bind " +
                std::string(kMiniVerifyReasons[static_cast<unsigned>(reason)]) + ": " + error.what());
        }
    }

    template <class Args>
    void add_leaf(const Args& args, bool pdl = false) {
        using namespace exec::cuda_decode;
        constexpr auto wide = [] {
            if constexpr (std::is_same_v<Args, Bf16DenseGroup>) return MiniVerifyMissing::dense;
            if constexpr (std::is_same_v<Args, core::residual_mix::MixArgs>) return MiniVerifyMissing::residual_mix;
            if constexpr (std::is_same_v<Args, core::residual_mix::NormalizeArgs>) return MiniVerifyMissing::bank_norm;
            if constexpr (std::is_same_v<Args, core::residual_mix::NormalizePairArgs> ||
                          std::is_same_v<Args, core::residual_mix::ControllerArgs>) return MiniVerifyMissing::controller;
            if constexpr (std::is_same_v<Args, HeadClipBf16>) return MiniVerifyMissing::head_clip;
            if constexpr (std::is_same_v<Args, Bf16RelativeProjection16>) return MiniVerifyMissing::relative;
            if constexpr (std::is_same_v<Args, RmsBf16>) return MiniVerifyMissing::rms;
            if constexpr (std::is_same_v<Args, core::residual_mix::AddArgs>) return MiniVerifyMissing::add;
            if constexpr (std::is_same_v<Args, Bf16Round>) return MiniVerifyMissing::round;
            if constexpr (std::is_same_v<Args, Bf16SigmoidProduct>) return MiniVerifyMissing::sigmoid;
            if constexpr (std::is_same_v<Args, Bf16Softcap>) return MiniVerifyMissing::softcap;
            return MiniVerifyMissing::count;
        }();
        if (rows_ == 1U) {
            recipe_.leaves.kernel(bind_(args), pdl, rows_);
        } else if constexpr (wide != MiniVerifyMissing::count) {
            add_verify(args, wide);
        } else {
            throw std::logic_error("Inkling verify received a scalar-only leaf");
        }
    }
    auto add() { return [this](const auto& args, bool pdl) { add_leaf(args, pdl); }; }

    void dense(const float* input, unsigned width, std::span<const core::Bf16RowProjection> projections) {
        exec::cuda_decode::lower_dense(input, width, projections, false, add());
    }

    struct Attention {
        using Status = bool;
        static constexpr bool success = true, invalid = false;
        MiniCudaStepLowering& lower;
        LayerVisit visit;
        MiniConvolutionStream path;
        const float* value_history = nullptr;

        bool linear(const float* input, unsigned width, unsigned, std::span<const MiniAttentionLinear> views) const {
            return core::for_each_bf16_projection_group<4>(views, success, [&](auto group) {
                lower.dense(input, width, group);
                return success;
            });
        }
        template <bool Round = false>
        bool convolve(const core::ShortConvolution& plan, unsigned, const float* input, const float* weights,
                      const float* dynamic, float* history, float* output) const {
            if (lower.rows_ == 1U) {
                lower.add_leaf(exec::cuda_decode::CausalConvolution{plan, input, weights, dynamic, history, output, Round});
            } else if (const auto* state = lower.verify_state_) {
                const auto stream = history == value_history ? MiniConvolutionStream::value : path;
                const core::decode_state::SlotView slot{state->directory, state->history_instance(visit, stream)};
                lower.add_verify(core::causal_conv::StepArgs{lower.descriptor_, slot, plan, input, weights, dynamic,
                                 state->history(visit, stream), output, Round}, MiniVerifyMissing::convolution);
            } else {
                lower.refuse(MiniVerifyMissing::convolution);
            }
            return success;
        }
        bool clip(float* row, const std::uint16_t* gain, unsigned, unsigned heads, unsigned width, float threshold,
                  bool position_scale, unsigned first, const std::uint32_t* position = nullptr) const {
            const bool scalar_position = lower.rows_ == 1U && position_scale;
            lower.add_leaf(exec::cuda_decode::HeadClipBf16{row, gain, row, position, heads, width,
                scalar_position ? first : 0U, position_scale ? Shape::kPositionFloor : 0U,
                Shape::kAttentionEpsilon, threshold, position_scale ? Shape::kPositionAlpha : 0.0F});
            return success;
        }
        bool round(float* row, std::size_t elements, unsigned* = nullptr) const {
            lower.add_leaf(exec::cuda_decode::Bf16Round{row, elements / lower.rows_});
            return success;
        }
        bool rms_values(float* row, unsigned) const {
            lower.rms(exec::cuda_decode::RmsBf16{row, nullptr, row, kKvHeads, kHeadWidth, Shape::kAttentionEpsilon});
            return success;
        }
        bool relative_logits(const float* input, const std::uint16_t* weights, float* output, unsigned, unsigned*) const {
            lower.add_leaf(exec::cuda_decode::Bf16RelativeProjection16{input, weights, output, kQueryHeads, kRelativeExtent});
            return success;
        }
        bool gate(const float* input, const float* logits, float* output, std::size_t elements) const {
            lower.add_leaf(exec::cuda_decode::Bf16SigmoidProduct{input, logits, output, elements / lower.rows_});
            return success;
        }
    };

    const Weights& weights_;
    MiniCudaStepStorage storage_;
    const exec::StepBlock* descriptor_;
    const Bind& bind_;
    const Dispatch& dispatch_;
    const Finish& finish_;
    const BindVerify& bind_verify_;
    const MiniCudaVerifyState* verify_state_;
    unsigned rows_ = 1U;
    MiniCudaStepRecipe recipe_;
};

class MiniResidentDecoder;

class MiniCudaProgramDecoder {
public:
    explicit MiniCudaProgramDecoder(MiniResidentDecoder&);
    ~MiniCudaProgramDecoder();
    MiniCudaProgramDecoder(const MiniCudaProgramDecoder&) = delete;
    MiniCudaProgramDecoder& operator=(const MiniCudaProgramDecoder&) = delete;

    unsigned prefill(std::span<const unsigned> ids, std::span<float> logits);
    unsigned decode(unsigned token, std::span<float> logits);
    void verify(std::span<const unsigned> ids, std::span<unsigned> picks, std::span<float> logits);

    // spec::EngineVerifyTarget binding (strix/spec/engine_target.hpp). A window of 2..16 rows runs on the
    // next graph width; padded rows repeat the last id and are never kept (rows are causal: row r does
    // not read rows after it, and every row of a width-W graph is bitwise the R=1 step). The window is
    // pending until commit(keep) publishes its kept prefix; a one-row window runs the decode step.
    using Rows = std::span<const unsigned>;
    enum class LogitRows { all };
    Rows forward_rows(const unsigned* ids, unsigned rows, LogitRows);
    void read_top(Rows top, unsigned rows, unsigned* output) const;
    void read_logits_rows(Rows top, unsigned rows, float* output);
    void commit(unsigned keep);
    void sync() const {}
    [[nodiscard]] unsigned committed() const;
    [[nodiscard]] unsigned vocab() const { return vocab_; }
    [[nodiscard]] static constexpr unsigned max_verify_rows() { return kMiniVerifyMaxRows; }

    void reset_gate_session();
    void prefill_gate_rows(std::span<const unsigned> ids, std::span<float> logits);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    unsigned vocab_ = kOutputVocabulary;
};

std::unique_ptr<serve::SerialGenerationDecoder> load_mini_cuda_program(const serve::SerialGenerationLoad&);

#if defined(STRIX_INKLING2_STEP_PROGRAM)
namespace detail {

exec::KernelBinding bind_mini_verify(const MiniCudaVerifyEmbedding&);

struct MiniCudaScalarBinding {
    template <class Args>
    exec::KernelBinding operator()(const Args& args) const {
        using namespace exec::cuda_decode;
        if constexpr (std::is_same_v<Args, CheckedWaveRoute>)
            return bind_checked_wave<MiniRoutePolicy, MiniMoeMath, moe::RouteWaveTraits<true>>(
                args, "cuda-sm120-checked-wave-mini-route-r1-v1");
        else
            return exec::cuda_decode::bind(args);
    }
};

struct MiniCudaVerifyBinding {
    template <class Args>
    bool operator()(exec::CudaProgramRecipe& recipe, const Args& args, const exec::StepBlock* descriptor,
                    unsigned rows) const {
        using namespace exec::cuda_decode;
        const auto other = [](exec::CudaProgramRecipe& leaves, const auto& a, unsigned width) {
            using Leaf = std::remove_cvref_t<decltype(a)>;
            if constexpr (std::is_same_v<Leaf, MiniCudaVerifyEmbedding>) {
                leaves.kernel(bind_mini_verify(a), false, width);
            } else if constexpr (std::is_same_v<Leaf, StepCheckedWaveRoute>) {
                bind_step_route<MiniRoutePolicy, MiniMoeMath, moe::RouteWaveTraits<true>>(
                    a, width, "cuda-sm120-checked-wave-mini-route").append_to(leaves);
            } else if constexpr (requires { bind_step(a, width); }) {
                bind_step(a, width).append_to(leaves);
            } else {
                (void)leaves;
                (void)a;
                (void)width;
                return false;
            }
            return true;
        };
        return append_mini_verify_leaf(recipe, args, descriptor, rows, wr1::native_thunks(), other);
    }
};

template <class T>
struct MiniCudaVerifyReply {
    explicit MiniCudaVerifyReply(std::size_t count) : size(count) {
        void* address{};
        mini_gpu::check(cudaMallocHost(&address, size * sizeof(T)), "allocate Inkling verify reply");
        data = static_cast<T*>(address);
    }
    ~MiniCudaVerifyReply() { (void)cudaFreeHost(data); }
    MiniCudaVerifyReply(const MiniCudaVerifyReply&) = delete;
    MiniCudaVerifyReply& operator=(const MiniCudaVerifyReply&) = delete;
    std::span<T> span() const { return {data, size}; }
    T* data{};
    std::size_t size;
};

struct MiniCudaVerifyWorkspace {
    static constexpr unsigned rows = kMiniVerifyMaxRows;
    static constexpr std::size_t logits_count = std::size_t{rows} * kOutputVocabulary;
    static constexpr unsigned chunks = head_argmax::default_chunks(kOutputVocabulary);

    explicit MiniCudaVerifyWorkspace(const MiniSessionStatePlan& plan)
        : pending_history(session::cache_buffer_elements<float>(plan.convolution_elements() * rows)) {}

    mini_gpu::Buffer<float> activations{mini_resident_activation_elements(rows)};
    mini_gpu::Buffer<float> embeddings{std::size_t{rows} * kHiddenWidth}, hidden{embeddings.span().size()};
    mini_gpu::Buffer<float> logits{logits_count};
    mini_gpu::Buffer<float> controller{exec::cuda_decode::controller_scratch_floats(kHiddenWidth, kControllerRank, rows)};
    mini_gpu::Buffer<float> attention{mini_resident_attention_elements(rows)};
    mini_gpu::Buffer<unsigned> route_ids{rows * kExpertsPerToken};
    mini_gpu::Buffer<float> route_logits{rows * kExperts}, route_weights{rows * kExpertsPerToken};
    mini_gpu::Buffer<std::uint16_t> expert_activation{std::size_t{rows} * kExpertsPerToken * kExpertWidth};
    mini_gpu::Buffer<unsigned> flags{2U * kLogicalVisits * rows};
    std::array<unsigned, 2U * kLogicalVisits * rows> failures{};
    mini_gpu::Buffer<head_argmax::Pick2> partials{head_argmax::workspace2_bytes(rows, chunks) / sizeof(head_argmax::Pick2)};
    mini_gpu::Buffer<exec::RowOut> picked{rows};
    MiniCudaVerifyReply<exec::RowOut> reply{rows};
    MiniCudaVerifyReply<float> logits_reply{logits_count};
    mini_gpu::Buffer<state::SlotEntry> directory{MiniCudaVerifyState::entries};
    std::array<state::SlotEntry, MiniCudaVerifyState::entries> upload{};
    mini_gpu::Buffer<std::uint16_t> pending_keys{std::size_t{kLogicalVisits} * rows * kKvWidth};
    mini_gpu::Buffer<std::uint16_t> pending_values{pending_keys.span().size()};
    mini_gpu::Buffer<float> pending_history;

    MiniCudaVerifyState state_view(const MiniSessionStatePlan& plan) const {
        return {{directory.data(), MiniCudaVerifyState::entries, 1U}, rows,
                pending_keys.data(), pending_values.data(), pending_history.data(), plan};
    }

    void upload_directory(const MiniResidentSessionState& session, MiniResidentChunk chunk, mini_gpu::Stream stream) {
        constexpr auto full = mini_schedule().back();
        static_assert(is_full_attention(full.physical_layer));
        const auto capacity = session.plan().visit(full).kv_rows;
        const auto entry = [chunk, capacity](const void* base) {
            return state::SlotEntry{reinterpret_cast<std::uintptr_t>(base), chunk.first_position, chunk.rows, capacity, 0U};
        };
        for (const auto visit : mini_schedule()) {
            const auto resident = session.visit(visit);
            const auto key = MiniCudaVerifyState::key_instance(visit);
            upload[key] = entry(resident.kv.keys);
            upload[key + 1U] = entry(resident.kv.values);
            for (unsigned path = 0; path < resident.history.size(); ++path)
                upload[key + 2U + path] = entry(resident.history[path].data());
        }
        mini_gpu::copy(directory.data(), upload.data(), sizeof(upload), mini_gpu::host_to_device, stream);
    }

    // This CLI accepts the whole forced chain. Retain only its final recurrent
    // image and copy the KV tail into the incumbent resident layout.
    void publish(const MiniResidentSessionState& session, MiniResidentChunk chunk, mini_gpu::Stream stream) const {
        const auto view = state_view(session.plan());
        const auto copy = [stream](auto* destination, const auto* source, std::size_t elements) {
            mini_gpu::copy(destination, source, elements * sizeof(*source), mini_gpu::device_to_device, stream);
        };
        for (const auto visit : mini_schedule()) {
            const auto resident = session.visit(visit);
            const auto& kv = resident.kv;
            const auto cache = view.cache(visit, kv);
            const auto first = kv.wrapped ? chunk.first_position % kv.slots : chunk.first_position;
            const auto contiguous = std::min(chunk.rows, kv.slots - first);
            const auto offset = std::size_t{first} * kKvWidth;
            const auto elements = std::size_t{contiguous} * kKvWidth;
            copy(kv.keys + offset, cache.pending_keys, elements);
            copy(kv.values + offset, cache.pending_values, elements);
            if (contiguous != chunk.rows) {
                const auto tail = std::size_t{chunk.rows - contiguous} * kKvWidth;
                copy(kv.keys, cache.pending_keys + elements, tail);
                copy(kv.values, cache.pending_values + elements, tail);
            }
            for (unsigned path = 0; path < resident.history.size(); ++path) {
                const auto image = resident.history[path];
                copy(image.data(), view.history(visit, static_cast<MiniConvolutionStream>(path)) +
                    (chunk.rows - 1U) * image.size(), image.size());
            }
        }
        mini_gpu::synchronize(stream);
    }
};

}  // namespace detail

struct MiniCudaProgramDecoder::Impl {
    explicit Impl(MiniResidentDecoder& resident) : decoder(resident) {
        if (!decoder.step_program_backend().step_program_buffers().packed_decode)
            throw std::invalid_argument("--step-program requires packed decode");
    }

    unsigned decode(unsigned token, std::span<float> host_logits) {
        if (token >= kVocabulary || (!host_logits.empty() && host_logits.size() != kOutputVocabulary))
            throw std::invalid_argument("invalid Inkling step program input or logits row");
        auto& backend = decoder.step_program_backend();
        auto& state = decoder.step_program_state();
        const MiniResidentChunk chunk{state.committed(), 1U};
        state.require_append(chunk);
        if (!host_logits.empty() && !staged_logits) {
            void* address{};
            mini_gpu::check(cudaMallocHost(&address, host_logits.size_bytes()), "allocate Mini program logits reply");
            staged_logits.reset(static_cast<float*>(address));
        }
        auto& greedy = decoder.step_program_greedy();
        const MiniCudaStepRows rows{decoder.step_program_embeddings(), decoder.step_program_hidden(),
            decoder.step_program_logits(), greedy.step_program_partials(), greedy.step_program_device_token(),
            greedy.step_program_reply()};
        const auto raw = backend.step_program_buffers();
        const MiniCudaStepStorage storage{raw.max_rows, raw.controller, raw.attention, raw.lag_weights,
            raw.matrices, raw.route_ids, raw.route_logits, raw.route_weights, raw.expert_activation, raw.flags};
        const detail::MiniCudaScalarBinding bind{};
        exec::CudaTicket ticket{};
        const auto dispatch = [&](MiniResidentChunk step, const auto& build) {
            const bool install = !cached.has_value();
            if (install) cached.emplace(build().compile(key));
            const auto input = mini_verify_descriptor(step.rows, std::span(&token, 1U), step.first_position);
            ticket = executor.submit_x3(cached->program, input, {&cached->binding, backend.stream()});
            if (install) {
                const auto stats = executor.graph_stats(key);
                if (stats.captures != 1U || stats.instantiations != 1U ||
                    stats.nodes != cached->program.launches.size() || stats.kernels != cached->program.predicted.launches)
                    throw std::runtime_error("Inkling step program capture did not install the compiled plan");
                std::cerr << "inkling2 step graph route=program captures=1 nodes=" << stats.nodes
                          << " launches=" << stats.kernels << '\n';
            }
            return true;
        };
        const auto finish = [&] {
            backend.finish();
            if (!executor.poll(ticket, {})) throw std::logic_error("Inkling program stream handoff did not complete");
        };
        const detail::MiniCudaVerifyBinding bind_verify{};
        MiniCudaStepLowering lowering(decoder.step_program_weights(), storage, executor.step_descriptor(),
                                     bind, dispatch, finish, bind_verify);
        MiniResidentForward forward(lowering, state, decoder.step_program_workspace());
        unsigned picked = head_argmax::kNone;
        try {
            forward.append(chunk, rows.embeddings, rows.hidden,
                [&](MiniResidentChunk, MiniConstRows hidden) { lowering.head(hidden, rows); },
                [&] {
                    picked = greedy.token();
                    if (picked >= kOutputVocabulary) throw std::runtime_error("Inkling 2 Mini: non-finite output logits");
                },
                [&] { lowering.embed(rows.embeddings); },
                [&] {
                    if (!host_logits.empty())
                        mini_gpu::copy(staged_logits.get(), rows.logits.data(), host_logits.size_bytes(),
                                       mini_gpu::device_to_host, backend.stream());
                });
        } catch (...) {
            state.drain_or_poison([&] { mini_gpu::synchronize(backend.stream()); });
            throw;
        }
        if (!host_logits.empty()) std::copy_n(staged_logits.get(), host_logits.size(), host_logits.data());
        return picked;
    }

    // Stage a verify window of a graph width (2/4/8/16). The graph writes only the verify workspace
    // (pending K/V rows, one history image per row, picks, logits reply); nothing reaches the resident
    // session until commit(keep) publishes the kept prefix.
    void stage(std::span<const unsigned> ids, std::span<unsigned> picks) {
        const auto logits_count = ids.size() * kOutputVocabulary;
        auto& backend = decoder.step_program_backend();
        auto& state = decoder.step_program_state();
        const MiniResidentChunk chunk{state.committed(), static_cast<unsigned>(ids.size())};
        state.require_append(chunk);
        const auto input = mini_verify_descriptor(chunk.rows, ids, chunk.first_position);
        if (!verify_workspace) verify_workspace.emplace(state.plan());
        auto& work = *verify_workspace;
        const auto resident = backend.step_program_buffers();
        const MiniCudaStepStorage storage{work.rows, work.controller.span(), work.attention.span(), resident.lag_weights,
            resident.matrices, work.route_ids.data(), work.route_logits.data(), work.route_weights.data(),
            work.expert_activation.data(), work.flags.span().first(2U * kLogicalVisits * chunk.rows)};
        const auto hidden_count = std::size_t{chunk.rows} * kHiddenWidth;
        const MiniCudaStepRows rows{work.embeddings.span().first(hidden_count), work.hidden.span().first(hidden_count),
            work.logits.span().first(logits_count), nullptr, nullptr, nullptr};
        const MiniCudaVerifyRows replies{work.partials.data(), work.picked.span().first(chunk.rows),
            work.reply.span().first(chunk.rows), work.logits_reply.span().first(logits_count)};
        const auto view = work.state_view(state.plan());
        const auto verify_key = mini_cuda_step_key(chunk.rows);
        auto& recipe = verify_programs[std::bit_width(chunk.rows) - 2U];
        exec::CudaTicket ticket{};
        const auto queue = backend.stream();
        const auto dispatch = [&](MiniResidentChunk step, const auto& build) {
            const bool install = !recipe;
            if (install) recipe.emplace(build().compile(verify_key));
            work.upload_directory(state, step, queue);
            ticket = executor.submit_x3(recipe->program, input, {&recipe->binding, queue});
            const auto stats = executor.graph_stats(verify_key);
            if (stats.captures != 1U || stats.instantiations != 1U ||
                stats.nodes != recipe->program.launches.size() || stats.kernels != recipe->program.predicted.launches)
                throw std::runtime_error("Inkling verify program did not retain one graph for its width");
            if (install)
                std::cerr << "inkling2 verify graph R=" << step.rows << " captures=1 nodes=" << stats.nodes
                          << " launches=" << stats.kernels << '\n';
            return true;
        };
        const auto failures = std::span(work.failures).first(storage.flags.size());
        const auto finish = [&] {
            mini_gpu::copy(failures.data(), storage.flags.data(), storage.flags.size_bytes(),
                           mini_gpu::device_to_host, queue);
            mini_gpu::synchronize(queue);
            if (!executor.poll(ticket, {})) throw std::logic_error("Inkling verify stream handoff did not complete");
            if (std::ranges::any_of(failures, [](unsigned flag) { return flag != 0U; }))
                throw std::runtime_error("Inkling 2 Mini: device rejected a verify route or attention row");
        };
        const detail::MiniCudaScalarBinding bind{};
        const detail::MiniCudaVerifyBinding bind_verify{};
        MiniCudaStepLowering lowering(decoder.step_program_weights(), storage, executor.step_descriptor(),
                                     bind, dispatch, finish, bind_verify, &view);
        auto workspace = decoder.step_program_workspace();
        workspace.activations = work.activations.span();
        MiniResidentForward forward(lowering, state, workspace);
        try {
            forward.append_pending(chunk, rows.embeddings, rows.hidden,
                [&](MiniResidentChunk, MiniConstRows hidden) { lowering.head(hidden, rows, replies); },
                [&] {
                    for (const auto& row : replies.reply) {
                        if (row.status != exec::RowStatus::picked || row.id >= kOutputVocabulary)
                            throw std::runtime_error("Inkling 2 Mini: invalid verify head reply");
                    }
                },
                [&] { lowering.embed(rows.embeddings); });
        } catch (...) {
            state.drain_or_poison([&] { mini_gpu::synchronize(queue); });
            throw;
        }
        std::ranges::transform(replies.reply, picks.begin(), [](const exec::RowOut& row) { return row.id; });
    }

    // Publish rows [0, keep) of the staged window (KV rows + history image keep - 1) and move the
    // frontier by keep; keep = 0 discards the window.
    void commit(unsigned keep) {
        auto& state = decoder.step_program_state();
        const auto queue = decoder.step_program_backend().stream();
        const MiniResidentChunk kept{state.committed(), keep};
        state.commit_pending(keep, [&](unsigned) { verify_workspace->publish(state, kept, queue); });
    }

    // Host logits of row r of the staged window (pinned verify reply).
    [[nodiscard]] std::span<const float> window_logits(unsigned rows) {
        return verify_workspace->logits_reply.span().first(std::size_t{rows} * kOutputVocabulary);
    }

    void verify(std::span<const unsigned> ids, std::span<unsigned> picks, std::span<float> logits) {
        stage(ids, picks);
        std::ranges::copy(window_logits(static_cast<unsigned>(ids.size())), logits.begin());
        commit(static_cast<unsigned>(ids.size()));
    }

    struct Window {
        std::array<unsigned, kMiniVerifyMaxRows> ids{};
        std::array<unsigned, kMiniVerifyMaxRows> picks{};
        std::vector<float> one_row_logits = std::vector<float>(kOutputVocabulary);
        unsigned rows = 0;
        unsigned width = 0;
        bool decoded = false;
    };

    MiniResidentDecoder& decoder;
    Window window;
    // Prepared R1 attention reads position from X3 and keeps its grid fixed.
    // There is one key for this resident packed, untapped decode route.
    const exec::ProgramKey key{};
    std::unique_ptr<float, detail::MiniPinnedLogitsDeleter> staged_logits;
    std::optional<exec::CudaRecipeProgram> cached;
    std::optional<detail::MiniCudaVerifyWorkspace> verify_workspace;
    std::array<std::optional<exec::CudaRecipeProgram>, exec::cuda_decode::wr1::widths.size() - 1U> verify_programs;
    exec::CudaExecutor executor{true, true, exec::CudaDescriptorLayout::x3};
};

inline MiniCudaProgramDecoder::MiniCudaProgramDecoder(MiniResidentDecoder& decoder)
    : impl_(std::make_unique<Impl>(decoder)) {}
inline MiniCudaProgramDecoder::~MiniCudaProgramDecoder() = default;

inline unsigned MiniCudaProgramDecoder::prefill(std::span<const unsigned> ids, std::span<float> logits) {
    return impl_->decoder.prefill(ids, logits);
}
inline unsigned MiniCudaProgramDecoder::decode(unsigned token, std::span<float> logits) {
    return impl_->decode(token, logits);
}
inline void MiniCudaProgramDecoder::verify(std::span<const unsigned> ids, std::span<unsigned> picks,
                                          std::span<float> logits) {
    if (ids.size() > kMiniVerifyMaxRows || picks.size() != ids.size() ||
        logits.size() != ids.size() * kOutputVocabulary)
        throw std::invalid_argument("invalid Inkling verify output geometry");
    require_mini_verify_width(static_cast<unsigned>(ids.size()));
    if (ids.size() == 1U) picks.front() = decode(ids.front(), logits);
    else impl_->verify(ids, picks, logits);
}
inline MiniCudaProgramDecoder::Rows MiniCudaProgramDecoder::forward_rows(const unsigned* ids, unsigned rows, LogitRows) {
    if (ids == nullptr || rows == 0U || rows > kMiniVerifyMaxRows)
        throw std::invalid_argument("invalid Inkling program verify window");
    auto& window = impl_->window;
    window.rows = rows;
    window.width = session::padded_step_replay_width(rows);
    // The padded rows are staged too: the session plan must admit the whole width (load geometry).
    if (!impl_->decoder.step_program_state().plan().admits(committed(), window.width))
        throw std::invalid_argument("Inkling program verify window of " + std::to_string(rows) + " rows (width " +
                                    std::to_string(window.width) + ") at position " + std::to_string(committed()) +
                                    " exceeds the session chunk or context");
    std::copy_n(ids, rows, window.ids.begin());
    std::fill(window.ids.begin() + rows, window.ids.begin() + window.width, ids[rows - 1U]);
    if (window.width == 1U) {
        // One row: the plain decode step publishes it; commit(0) cannot retract it, so a one-row window
        // keeps its row (the speculator always keeps the anchor row of a window).
        window.picks.front() = impl_->decode(ids[0], window.one_row_logits);
        window.decoded = true;
    } else {
        require_mini_verify_width(window.width);
        impl_->stage(std::span(window.ids).first(window.width), std::span(window.picks).first(window.width));
        window.decoded = false;
    }
    return std::span(window.picks).first(rows);
}

inline void MiniCudaProgramDecoder::read_top(Rows top, unsigned rows, unsigned* output) const {
    const auto& window = impl_->window;
    if (top.data() != window.picks.data() || top.size() != rows || rows != window.rows)
        throw std::logic_error("invalid Inkling program verify reply");
    std::copy_n(top.data(), rows, output);
}

inline void MiniCudaProgramDecoder::read_logits_rows(Rows top, unsigned rows, float* output) {
    auto& window = impl_->window;
    if (top.data() != window.picks.data() || top.size() != rows || rows != window.rows)
        throw std::logic_error("invalid Inkling program verify reply");
    const auto logits = window.decoded ? std::span<const float>(window.one_row_logits) : impl_->window_logits(rows);
    std::copy_n(logits.data(), std::size_t{rows} * kOutputVocabulary, output);
}

inline void MiniCudaProgramDecoder::commit(unsigned keep) {
    auto& window = impl_->window;
    if (keep > window.rows) throw std::invalid_argument("invalid Inkling program verify prefix");
    if (window.decoded) {
        if (keep != 1U) throw std::logic_error("a one-row Inkling program window keeps its row");
        window.decoded = false;
        return;
    }
    impl_->commit(keep);
}

inline unsigned MiniCudaProgramDecoder::committed() const { return impl_->decoder.step_program_state().committed(); }

inline void MiniCudaProgramDecoder::reset_gate_session() { impl_->decoder.reset_gate_session(); }
inline void MiniCudaProgramDecoder::prefill_gate_rows(std::span<const unsigned> ids, std::span<float> logits) {
    impl_->decoder.prefill_gate_rows(ids, logits);
}

// Drafted window rows of the program route without a lookup source (DSpark block = 7 drafts + anchor).
inline constexpr std::uint32_t kMiniProgramDraftRows = 8U;

inline std::unique_ptr<serve::SerialGenerationDecoder> load_mini_cuda_program(const serve::SerialGenerationLoad& load) {
    struct Decoder final : serve::SerialGenerationDecoder {
        explicit Decoder(const serve::SerialGenerationLoad& input)
            : Decoder(input, session::step_replay_session_geometry(input.positions, input.prefill_chunk,
                  std::min(input.verify_rows, std::uint32_t{kMiniVerifyMaxRows}), kMaxContext)) {}
        // A verify window of R rows stages all padded_step_replay_width(R) rows as one pending append:
        // the session plan admits the widest window (chunk) and its padding overhang (context). Without
        // speculation (verify_rows 0) the geometry is the request's.
        Decoder(const serve::SerialGenerationLoad& input, session::StepReplaySessionGeometry geometry)
            : resident(input.checkpoint, geometry.positions, input.prefill_chunk, input.packed_decode, false,
                       geometry.append_rows),
              program(resident) {}
        unsigned prefill(std::span<const unsigned> ids, std::span<float> logits) override {
            return program.prefill(ids, logits);
        }
        unsigned decode(unsigned token, std::span<float> logits) override { return program.decode(token, logits); }
        // Speculation on the step program: verify windows on the width graphs, kept prefix published from
        // the pending verify workspace (no checkpoint, no replay). Drafters: lookup (shared BlockDrafter);
        // the DSpark block drafter needs a tap-store leaf first (chain_drafter stays null here).
        spec::VerifyTarget* verify_target() override { return &target; }
        std::uint32_t max_verify_rows(serve::CliSpeculation mode) const override {
            return mode == serve::CliSpeculation::lookup ? MiniCudaProgramDecoder::max_verify_rows()
                                                         : kMiniProgramDraftRows;
        }
        MiniResidentDecoder resident;
        MiniCudaProgramDecoder program;
        spec::EngineVerifyTarget<MiniCudaProgramDecoder> target{program};
    };
    return std::make_unique<Decoder>(load);
}
#endif

}  // namespace strix::inkling2
