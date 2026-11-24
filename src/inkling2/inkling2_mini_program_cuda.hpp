#pragma once

#include "exec/catalog_cuda_wr1.hpp"
#include "exec/cuda_decode_epilogue.hpp"
#include "core/screened_head_plan.hpp"
#include "exec/cuda_decode_lowering.hpp"
#include "exec/cuda_decode_wr2.hpp"
#include "inkling2_mini_attention_prepare_device.hpp"
#include "inkling2_mini_dspark_taps.hpp"
#include "inkling2_mini_expert_layout.hpp"
#include "inkling2_mini_moe_dispatch.hpp"
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
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#if defined(STRIX_INKLING2_STEP_PROGRAM)
#include "device/cuda_cost_identity.hpp"
#include "inkling2_mini_resident_decoder.hpp"
#include "inkling2_mini_resident_dspark.hpp"
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
    // Drafter feature taps, position-major [position][tap][hidden] BF16 (MiniDSparkTapPlan), or null: every
    // width writes the rows of the positions it runs (step descriptor), so captured graphs stay valid.
    std::uint16_t* draft_taps = nullptr;
    unsigned draft_tap_positions = 0U;
};

struct MiniCudaStepRows {
    std::span<float> embeddings, hidden, logits;
    head_argmax::Pick* partials;
    std::uint32_t* picked;
    std::uint32_t* reply;
    // EXACT screened greedy head (one lane; x is bound to the final hidden row): replaces the projection, the softcap pass
    // and the two greedy leaves with the five screened stages, same picked word. screen_stats (optional, pinned) receives
    // the pick's State for diagnostics.
    const core::screened_head_plan::Launch* screen = nullptr;
    core::screened_head_plan::State* screen_stats = nullptr;
};

struct MiniCudaStepMode {
    exec::cuda_decode::DecodeRecipeKey key;
    exec::cuda_decode::GroupedDecodeScratch grouped;
    moe::RouteTap* tap = nullptr;
    void* stream = nullptr;
    void (*copy_eager)(void*, const void*, std::size_t, void*) = nullptr;
    bool epilogue_fusion = true;
    const device::Profile* projection_profile = nullptr;
    decltype(&exec::cuda_decode::bind_projection_batch) bind_projections = nullptr;
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
// A verify window (2..16 rows) lowers the same traversal on the width-bound step leaves (descriptor
// positions, StateTable slots, pending K/V rows and one history image per row); the R1-only latency
// forms, epilogue fusion, projection batches and route taps stay on the one-row step.
template <class Weights, class Bind, class Dispatch, class Finish, class BindVerify>
class MiniCudaStepLowering {
    using Shape = MiniPreparedGqaShape;
    using Mode = exec::cuda_decode::Bf16DenseMode;
    static constexpr float epsilon = 1.0e-6F;

public:
    MiniCudaStepLowering(const Weights& weights, MiniCudaStepStorage storage, const exec::StepBlock* descriptor,
                        const Bind& bind, const Dispatch& dispatch, const Finish& finish,
                        const BindVerify& bind_verify, MiniCudaStepMode mode = {},
                        const MiniCudaVerifyState* verify_state = nullptr)
        : weights_(weights), storage_(storage), descriptor_(descriptor), bind_(bind), dispatch_(dispatch),
          finish_(finish), bind_verify_(bind_verify), verify_state_(verify_state), mode_(mode),
          recipe_{exec::CudaProgramRecipe(mode.key.program.tap), 0U} {}

    bool try_step_graph(MiniResidentChunk chunk, auto& work) {
        if (mode_.key.program.tap) return false;
        if (!mini_verify_width(chunk.rows) || chunk.rows > storage_.max_rows ||
            (chunk.rows != 1U && mode_.key.grouped))
            throw std::invalid_argument("invalid Inkling CUDA step program width");
        rows_ = chunk.rows;
        return dispatch_(chunk, [&] {
            begin();
            work();
            return std::move(recipe_);
        });
    }
    bool abandon_capture() noexcept {
        recording_ = false;
        return false;
    }
    void finish() { finish_(); }

    void begin_eager() {
        if (mode_.key.program.tap) begin();
    }

    void submit_eager(MiniResidentChunk chunk) {
        if (!mode_.key.program.tap) return;
        recording_ = false;
        (void)dispatch_(chunk, [&] { return std::move(recipe_); });
    }

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

    // Drafter feature tap `tap` of every row of this step, stored at its descriptor position (shared
    // position-indexed row store): one captured graph per width serves every position.
    void store_draft_tap_rows(MiniConstRows input, unsigned tap) {
        if (!storage_.draft_taps || tap >= MiniDSparkTapPlan::count)
            throw std::logic_error("Inkling step program has no drafter tap store");
        recipe_.leaves.stage("draft_tap");
        add_verify(cuda::core::StepRowStoreBf16{descriptor_, input.data(),
            storage_.draft_taps + std::size_t{tap} * kHiddenWidth, kHiddenWidth, MiniDSparkTapPlan::width,
            storage_.draft_tap_positions}, MiniVerifyMissing::draft_tap);
    }

    void head(MiniConstRows hidden, MiniCudaStepRows rows, MiniCudaVerifyRows verify = {}) {
        using namespace exec::cuda_decode;
        recipe_.leaves.visit(-1);
        recipe_.leaves.stage("head");
        if (rows.screen) {
            auto launch = *rows.screen;
            launch.x = hidden.data();
            launch.lanes = 1U;
            launch.x_stride = kHiddenWidth;
            add_leaf(ScreenedHeadPrep{launch});
            add_leaf(ScreenedHeadPass1{launch});
            add_leaf(ScreenedHeadCompact{launch});
            add_leaf(ScreenedHeadPass2{launch});
            add_leaf(ScreenedHeadPick{launch, rows.picked});
            if (rows.screen_stats)
                recipe_.leaves.copy(exec::CudaNodeKind::copy_d2h_pinned,
                             {rows.screen_stats, launch.scratch, sizeof(*rows.screen_stats)});
            recipe_.leaves.copy(exec::CudaNodeKind::copy_d2h_pinned, {rows.reply, rows.picked, sizeof(*rows.reply)});
            return;
        }
        const std::array projection{core::Bf16RowProjection{weights_.bf16("unembed.weight").data(),
                                                          kOutputVocabulary, rows.logits.data()}};
        if (rows_ != 1U) {
            require(rows.logits.size() == std::size_t{rows_} * kOutputVocabulary &&
                    verify.logits_reply.size() == rows.logits.size() && verify.picked.size() == rows_ &&
                    verify.reply.size() == rows_);
            lower_dense(hidden.data(), kHiddenWidth, projection, false, add());
            add_leaf(Bf16Softcap{rows.logits.data(), kOutputVocabulary, kLogitSoftcap});
            add_verify(exec::cuda_decode::wr1::Greedy{rows.logits.data(), verify.partials, verify.picked.data(),
                       kOutputVocabulary, head_argmax::default_chunks(kOutputVocabulary)}, MiniVerifyMissing::argmax);
            recipe_.leaves.copy(exec::CudaNodeKind::copy_d2h_pinned,
                                {verify.reply.data(), verify.picked.data(), verify.picked.size_bytes()}, rows_);
            recipe_.leaves.copy(exec::CudaNodeKind::copy_d2h_pinned,
                                {verify.logits_reply.data(), rows.logits.data(), rows.logits.size_bytes()}, rows_);
            return;
        }
        const Bf16Softcap tail{rows.logits.data(), rows.logits.size(), kLogitSoftcap};
        lower_dense(hidden.data(), kHiddenWidth, projection, false, [&](const auto& head, bool pdl) {
            auto producer = bind_(head);
            if (epilogue_fusion() && bind_epilogue(producer.identity, graph::Op::linear, graph::Form::softcap,
                                                   head, tail, pdl)) return;
            recipe_.leaves.kernel(std::move(producer), pdl);
            add_leaf(tail);
        });
        const GreedyPartial reduction{rows.logits.data(), rows.partials, kOutputVocabulary,
                                     head_argmax::default_chunks(kOutputVocabulary)};
        add_leaf(reduction);
        add_leaf(GreedyFinal{reduction, rows.picked});
        recipe_.leaves.copy(exec::CudaNodeKind::copy_d2h_pinned, {rows.reply, rows.picked, sizeof(*rows.reply)});
    }

    template <class T>
    void copy(std::span<const T> input, std::span<T> output) {
        if (input.data() == output.data()) return;
        if (!recording_ && mode_.key.program.tap) {
            mode_.copy_eager(output.data(), input.data(), input.size_bytes(), mode_.stream);
            return;
        }
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
        // The R1 latency forms are one-row bodies: R = 1 lowers the form overload, a wide step the rows overload.
        const auto q = queries.subspan(std::size_t{query} * kHiddenWidth, kHiddenWidth);
        const auto* const down = weights_.bf16(stem + "down_weight").data();
        const auto* const forget = weights_.bf16(stem + "forget_weight").data();
        const auto* const gain = weights_.bf16(stem + "gain_weight").data();
        if (rows_ == 1U)
            exec::cuda_decode::lower_low_rank_controller(kHiddenWidth, kControllerRank, epsilon, q, raw, previous,
                output, storage_.controller, down, forget, gain, scale, add(), forms());
        else
            exec::cuda_decode::lower_low_rank_controller(kHiddenWidth, kControllerRank, epsilon, q, raw, previous,
                output, storage_.controller, down, forget, gain, scale, add(), rows_);
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
        exec::cuda_decode::lower_bank_readout(args, forms(), add());
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
        if (rows_ != 1U) {
            attention_rows(visit, chunk, inputs, weights, work, prep, attended, gated, full, state, output);
            return;
        }
        const auto* position = mode_.key.program.tap ? nullptr : &descriptor_->row[0].position;
        // These scratch images have no readers before their in-place round in prepared GQA.
        // Only full-attention visits consume the last two images, the dynamic taps.
        std::array round_sites{
            RoundSite{{work.raw_keys, kKvWidth}}, RoundSite{{work.raw_values, kKvWidth}},
            RoundSite{{prep.value, kKvWidth}}, RoundSite{{prep.sinks, kQueryHeads}},
            RoundSite{{output.data(), kHiddenWidth}},
            RoundSite{{work.dynamic_keys, kKvHeads * kConvTaps}},
            RoundSite{{work.dynamic_values, kKvHeads * kConvTaps}},
        };
        const auto count = epilogue_fusion() ? round_sites.size() - (full ? 0U : 2U) : 0U;
        const Attention lowering{*this, std::span(round_sites).first(count), visit, MiniConvolutionStream::key,
                                 work.value_history};
        const auto prepare = [&](const auto& backend) {
            require(attn_core::enqueue_prepared_gqa<Shape>(backend, full, chunk.first_position, 1U,
                                                           inputs, weights, work, prep, nullptr, position));
        };
        if (mode_.projection_profile && !mode_.key.program.tap) {
            require(mode_.bind_projections != nullptr);
            prepare(ProjectionAttention{lowering});
        } else {
            prepare(lowering);
        }
        const AppendKvBf16 append{prep.key, prep.value, state.kv.keys, state.kv.values, position,
                                  kQueryHeads, kKvHeads, kHeadWidth, state.kv.slots, chunk.first_position,
                                  Shape::kScoreScale, state.kv.wrapped};
        const PreparedFlashBf16 flash{prep.query, prep.relative_logits, prep.sinks, state.kv.keys, state.kv.values,
                                     attended, position, kQueryHeads, kKvHeads, state.kv.slots, chunk.first_position,
                                     kRelativeExtent, full ? 0U : kSlidingWindow, Shape::kScoreScale, state.kv.wrapped};
        if (mode_.key.program.tap) {
            add_leaf(AppendKvHostBf16{append});
            add_leaf(PreparedFlashHostBf16{flash});
        } else {
            add_leaf(append);
            add_leaf(flash);
        }
        add_leaf(PreparedFinite{{prep.query, prep.key, prep.value, prep.relative_logits, prep.sinks,
                                attended, chunk.first_position, 1U}, kQueryHeads, kKvHeads, kHeadWidth,
                               kRelativeExtent, storage_.flags.data() + visit.logical_layer});
        require(attn_core::enqueue_prepared_gqa_output<Shape>(lowering, 1U,
            weights.projection(MiniAttentionProjection::output), attended, prep.gate_logits, gated, output.data()));
    }

    void convolve(LayerVisit visit, MiniConvolutionStream path, MiniConstRows input, MiniMutableRows history,
                  MiniMutableRows output) {
        recipe_.leaves.stage("post_convolution");
        const auto* weights = mini_post_convolution_weights(storage_.lag_weights, visit, path == MiniConvolutionStream::moe);
        require(attn_core::enqueue_prepared_gqa_post_convolution<Shape>(
            Attention{*this, {}, visit, path}, rows_, weights, input.data(), history.data(), output.data()));
    }

    void moe(LayerVisit visit, MiniConstRows input, MiniMutableRows output) {
        using namespace exec::cuda_decode;
        recipe_.leaves.stage("moe");
        const auto layer = visit.physical_layer;
        const std::array router{core::Bf16RowProjection{weights_.bf16(parameter(layer, "mlp.gate.weight")).data(),
                                                       kExperts, storage_.route_logits}};
        lower_dense(input.data(), kHiddenWidth, router, true, add());
        auto* failure = storage_.flags.data() + std::size_t{kLogicalVisits + visit.logical_layer} * rows_;
        const auto* matrices = storage_.matrices.data() + std::size_t{layer} * kExperts;
        if (rows_ != 1U) {
            static_assert(kExpertsPerToken == kPackedExpertsPerToken);
            const moe_core::microscaled_r1::Args args{
                matrices, storage_.route_ids, storage_.route_weights, failure, input.data(), storage_.expert_activation,
                output.data(), kExperts, kHiddenWidth, kExpertWidth, kMlpGateSoftcap, kMlpUpSoftcap,
            };
            const auto format = mini_expert_format(layer);
            add_verify(StepCheckedWaveRoute{descriptor_, storage_.route_logits,
                       weights_.f32(parameter(layer, "mlp.gate.bias")).data(), storage_.route_ids,
                       storage_.route_weights, failure}, MiniVerifyMissing::route);
            add_verify(StepPackedGateUp{{descriptor_, args}, format}, MiniVerifyMissing::gate_up);
            add_verify(StepPackedDownCombine{{descriptor_, args}, format}, MiniVerifyMissing::down_combine);
            return;
        }
        const auto select = [&](unsigned* status) {
            add_leaf(CheckedWaveRoute{storage_.route_logits, weights_.f32(parameter(layer, "mlp.gate.bias")).data(),
                                      storage_.route_ids, storage_.route_weights, status});
            if (mode_.key.program.tap)
                recipe_.leaves.host_action(bind_(RouteTapExchange{mode_.tap, layer, 1U, kExpertsPerToken, kExperts,
                                                           storage_.route_ids, storage_.route_weights}));
        };
        const auto format = mini_expert_format(layer);
        if (mode_.key.grouped) {
            const auto [plan, scratch, status] = mode_.grouped.bind(kExperts, kExpertsPerToken,
                mini_grouped::CudaBackend::rows_per_tile, matrices, storage_.route_ids, storage_.route_weights,
                failure, storage_.expert_activation, kHiddenWidth, kExpertWidth);
            select(status);
            lower_grouped_decode(plan, scratch, format, input.data(), output.data(), kMlpGateSoftcap, kMlpUpSoftcap,
                                 add(), [this](auto clear) { recipe_.leaves.clear(clear); });
            return;
        }
        select(failure);
        static_assert(kExpertsPerToken == kPackedExpertsPerToken);
        const moe_core::microscaled_r1::Args args{
            matrices, storage_.route_ids, storage_.route_weights,
            failure, input.data(), storage_.expert_activation, output.data(), kExperts, kHiddenWidth, kExpertWidth,
            kMlpGateSoftcap, kMlpUpSoftcap,
        };
        if (mode_.key.forms.moe_pipe) {
            add_leaf(PackedGateUpPipe{args, format});
            add_leaf(PackedDownCombinePipe{args, format});
            return;
        }
        add_leaf(PackedGateUp{args, format});
        add_leaf(PackedDownCombine{args, format});
    }

private:
    void begin() {
        recording_ = true;
        recipe_.leaves.stage("flags");
        recipe_.leaves.clear({storage_.flags.data(), storage_.flags.size_bytes(), 0}, rows_);
    }

    [[nodiscard]] exec::cuda_decode::R1Forms forms() const noexcept {
        return rows_ == 1U ? mode_.key.forms : exec::cuda_decode::R1Forms{};
    }

    template <class Inputs, class Projections, class Work, class Prep>
    void attention_rows(LayerVisit visit, MiniResidentChunk chunk, const Inputs& inputs, const Projections& weights,
                        const Work& work, const Prep& prep, float* attended, float* gated, bool full,
                        const MiniResidentVisitState& state, MiniMutableRows output) {
        const auto* position = &descriptor_->row[0].position;
        const Attention lowering{*this, {}, visit, MiniConvolutionStream::key, work.value_history};
        require(attn_core::enqueue_prepared_gqa<Shape>(lowering, full, chunk.first_position, rows_,
                                                       inputs, weights, work, prep, nullptr, position));
        const attn_core::PreparedGqaAttentionRows prepared{prep.query, prep.key, prep.value, prep.relative_logits,
            prep.sinks, attended, 0U, rows_};
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
        require(attn_core::enqueue_prepared_gqa_output<Shape>(lowering, rows_,
            weights.projection(MiniAttentionProjection::output), attended, prep.gate_logits, gated, output.data()));
    }

    static std::string parameter(unsigned layer, std::string_view suffix) {
        return "layers." + std::to_string(layer) + "." + std::string(suffix);
    }

    static void require(bool valid) {
        if (!valid) throw std::invalid_argument("invalid Inkling CUDA step program binding");
    }

    void rms(const float* input, const std::uint16_t* scale, float* output) {
        add_leaf(exec::cuda_decode::RmsBf16{input, scale, output, 1U, kHiddenWidth, epsilon}, true);
    }

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
            recipe_.leaves.kernel(bind_(args), pdl);
        } else if constexpr (wide != MiniVerifyMissing::count) {
            add_verify(args, wide);
        } else {
            throw std::logic_error("Inkling verify received a one-row-only leaf");
        }
    }
    auto add() { return [this](const auto& args, bool pdl) { add_leaf(args, pdl); }; }

    [[nodiscard]] bool epilogue_fusion() const noexcept {
        return mode_.epilogue_fusion && !mode_.key.program.tap;
    }

    bool bind_epilogue(std::string_view producer, graph::Op op, graph::Form form,
                       const auto& head, const auto& tail, bool pdl) {
        // D3 sees a single raw image for the whole native projection group.
        // The rounded image reuses that storage after the tail or fused store.
        const std::array stages{
            graph::StageDecl{.id = "producer", .op = op, .entry = producer, .out = "raw"},
            graph::StageDecl{.id = "round", .in = {"raw"}, .out = "rounded"},
        };
        graph::PlanStage binding{.stage = &stages.back()};
        exec::cuda_decode::project_exact_epilogue(binding, exec::Catalog::builtin(exec::Backend::cuda),
                                                   producer, form, true);
        if (binding.exact_epilogue_entry.empty()) return false;
        const auto plan = graph::launch_plan(graph::StageGraph{.before = stages},
                                            graph::PlanRequest{.rows = 1, .stages = std::span(&binding, 1)});
        return exec::cuda_decode::lower_exact_epilogue(recipe_.leaves, plan, plan.launches.front(), head, tail, pdl, bind_);
    }

    struct RoundSite {
        exec::cuda_decode::Bf16Round tail;
        bool bound = false;
    };

    struct Attention {
        using Status = bool;
        static constexpr bool success = true, invalid = false;
        MiniCudaStepLowering& lower;
        std::span<RoundSite> rounds{};
        LayerVisit visit{};
        MiniConvolutionStream path{};
        const float* value_history = nullptr;

        auto find_round(float* row, std::size_t elements) const {
            return std::ranges::find_if(rounds, [=](const RoundSite& site) {
                return site.tail.values == row && site.tail.elements == elements;
            });
        }

        bool bind_rounds(std::string_view producer, const exec::cuda_decode::Bf16DenseGroup& head, bool pdl) const {
            using namespace exec::cuda_decode;
            if (rounds.empty()) return false;
            std::array<RoundSite*, std::tuple_size_v<decltype(head.projections)>> sites{};
            std::array<Bf16Round, sites.size()> tails{};
            for (unsigned i = 0; i < head.count; ++i) {
                const auto& projection = head.projections[i];
                const auto site = find_round(projection.output, projection.outputs);
                if (site == rounds.end() || site->bound) return false;
                sites[i] = &*site;
                tails[i] = site->tail;
            }
            constexpr auto op = graph::Op::linear;
            constexpr auto form = graph::Form::store_bf16;
            const bool bound = head.count == 1U
                ? lower.bind_epilogue(producer, op, form, head, tails.front(), pdl)
                : lower.bind_epilogue(producer, op, form, head, std::span<const Bf16Round>(tails).first(head.count), pdl);
            if (bound)
                for (auto* site : std::span(sites).first(head.count)) site->bound = true;
            return bound;
        }

        bool linear(const float* input, unsigned width, unsigned, std::span<const MiniAttentionLinear> views) const {
            return core::for_each_bf16_projection_group<4>(views, success, [&](auto group) {
                exec::cuda_decode::lower_dense(input, width, group, false, [&](const auto& head, bool pdl) {
                    auto producer = lower.bind_(head);
                    if (lower.rows_ != 1U) {
                        lower.add_leaf(head, pdl);
                        return;
                    }
                    if (!bind_rounds(producer.identity, head, pdl)) lower.recipe_.leaves.kernel(std::move(producer), pdl);
                });
                return success;
            });
        }
        template <bool Round = false>
        bool convolve(const core::ShortConvolution& plan, unsigned, const float* input, const float* weights,
                      const float* dynamic, float* history, float* output) const {
            if (lower.rows_ != 1U) {
                if (const auto* state = lower.verify_state_) {
                    const auto stream = history == value_history ? MiniConvolutionStream::value : path;
                    const core::decode_state::SlotView slot{state->directory, state->history_instance(visit, stream)};
                    lower.add_verify(core::causal_conv::StepArgs{lower.descriptor_, slot, plan, input, weights, dynamic,
                                     state->history(visit, stream), output, Round}, MiniVerifyMissing::convolution);
                } else {
                    lower.refuse(MiniVerifyMissing::convolution);
                }
                return success;
            }
            const exec::cuda_decode::CausalConvolution head{plan, input, weights, dynamic, history, output, Round};
            auto producer = lower.bind_(head);
            if constexpr (!Round) {
                const auto site = find_round(output, plan.channels);
                if (site != rounds.end() && !site->bound &&
                    lower.bind_epilogue(producer.identity, graph::Op::elementwise, graph::Form::store_bf16,
                                        head, site->tail, false)) {
                    site->bound = true;
                    return success;
                }
            }
            lower.recipe_.leaves.kernel(std::move(producer), false);
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
        bool round(float* row, std::size_t elements, unsigned* failure = nullptr) const {
            const auto site = find_round(row, elements);
            if (!failure && site != rounds.end() && site->bound) {
                site->bound = false;
                return success;
            }
            lower.add_leaf(exec::cuda_decode::Bf16Round{row, elements / lower.rows_});
            return success;
        }
        bool rms_values(float* row, unsigned) const {
            lower.add_leaf(exec::cuda_decode::RmsBf16{row, nullptr, row, kKvHeads, kHeadWidth,
                                                     Shape::kAttentionEpsilon}, true);
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

    struct ProjectionAttention : Attention {
        mutable core::Bf16ProjectionBatch projections{};

        bool linear(const float* input, unsigned width, unsigned rows, std::span<const MiniAttentionLinear> views) const {
            return rows == 1 && projections.append(input, width, views);
        }

        bool finish_projections() const {
            // Only prepare owns these unconsumed tails. The output-projection
            // and convolution producers keep using the ordinary Attention binding.
            for (auto& site : this->rounds)
                if (projections.try_round_store(site.tail.values, site.tail.elements)) site.bound = true;
            const auto& mode = this->lower.mode_;
            auto leaves = mode.bind_projections(projections, *mode.projection_profile);
            for (auto& leaf : leaves) {
                // The untiled sink remains a validated incumbent FP32 group.
                // Let the existing exact-epilogue resolver consume its round.
                if (!this->rounds.empty() && leaf.identity == "cuda-sm120-bf16-r1-fp32-group-v1") {
                    exec::cuda_decode::Bf16DenseGroup scalar{};
                    require(leaf.args.size() == sizeof(scalar));
                    std::memcpy(&scalar, leaf.args.data(), sizeof(scalar));
                    if (this->bind_rounds(leaf.identity, scalar, true)) continue;
                }
                this->lower.recipe_.leaves.kernel(std::move(leaf), true);
            }
            return true;
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
    MiniCudaStepMode mode_;
    unsigned rows_ = 1U;
    MiniCudaStepRecipe recipe_;
    bool recording_ = false;
};

class MiniResidentDecoder;

class MiniCudaProgramDecoder {
public:
    explicit MiniCudaProgramDecoder(MiniResidentDecoder&, bool epilogue_fusion = true,
                                    exec::cuda_decode::R1Forms forms = {});
    ~MiniCudaProgramDecoder();
    MiniCudaProgramDecoder(const MiniCudaProgramDecoder&) = delete;
    MiniCudaProgramDecoder& operator=(const MiniCudaProgramDecoder&) = delete;

    unsigned prefill(std::span<const unsigned> ids, std::span<float> logits);
    unsigned decode(unsigned token, std::span<float> logits);
    void verify(std::span<const unsigned> ids, std::span<unsigned> picks, std::span<float> logits);
    void reset_gate_session();
    void prefill_gate_rows(std::span<const unsigned> ids, std::span<float> logits);
    void set_gate_route_tap(moe::RouteTap*);

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
    // One serial step of the speculator: the plain decode step (committed at once).
    unsigned decode_committed_row(unsigned token, std::span<float> logits) { return decode(token, logits); }
    [[nodiscard]] unsigned committed() const;
    [[nodiscard]] unsigned admissible_verify_rows(unsigned limit) const;
    [[nodiscard]] unsigned vocab() const { return kOutputVocabulary; }
    [[nodiscard]] static constexpr unsigned max_verify_rows() { return kMiniVerifyMaxRows; }
    // Verify windows need the packed expert route (the grouped tile has no width-bound leaves).
    [[nodiscard]] bool supports_verify() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Exact R1 latency forms (bitwise against the incumbents), opt-in until the RTX A/B selects them:
// mask bit 1 = tiled bank readouts, 2 = controller output split, 4 = packed MoE pipe.
inline constexpr unsigned kMiniFormsMix = 3U, kMiniFormsMoe = 4U, kMiniFormsAll = 7U;
[[nodiscard]] constexpr exec::cuda_decode::R1Forms mini_latency_forms(unsigned mask) noexcept {
    return {(mask & 1U) != 0U, (mask & 2U) != 0U, (mask & 4U) != 0U};
}
inline constexpr exec::cuda_decode::R1Forms kMiniLatencyForms = mini_latency_forms(kMiniFormsAll);
// RTX r1 (11-22, ink_rtx_dec kit out/r1, bitwise ids / logits / 31 decode rows): greedy 256 tok/s base 102.8,
// mix 109.9, mix + MoE pipe 108.4 -> the default binds mix; the pipe stays opt-in (--latency-forms).
inline constexpr unsigned kMiniDefaultForms = kMiniFormsMix;

// moe_w4a4: resolved operand class of the W4A4 expert lever (mini_moe_w4a4_class; bf16 = off), applied to the resident
// prefill before the program binds.
// dspark: the block drafter checkpoint (mtp.safetensors) or null; it is loaded only when the request may draft.
template <bool EpilogueFusion, unsigned LatencyForms = kMiniDefaultForms>
std::unique_ptr<serve::SerialGenerationDecoder> load_mini_cuda_program(const serve::SerialGenerationLoad&,
                                                                       moe_core::grouped_w4a4::Activations moe_w4a4,
                                                                       const char* dspark = nullptr);

#if defined(STRIX_INKLING2_STEP_PROGRAM)
namespace detail {

exec::KernelBinding bind_mini_verify(const MiniCudaVerifyEmbedding&);

// Scalar (R = 1) leaf binding of the step program: the Mini route policy, grouped expert plans and route taps
// bind here; every other leaf takes its catalogue binding.
struct MiniCudaScalarBinding {
    template <class Args>
    exec::KernelBinding operator()(const Args& args) const {
        using namespace exec::cuda_decode;
        if constexpr (std::is_same_v<Args, CheckedWaveRoute>)
            return bind_checked_wave<MiniRoutePolicy, MiniMoeMath, moe::RouteWaveTraits<true>>(
                args, "cuda-sm120-checked-wave-mini-route-r1-v1");
        else if constexpr (std::is_same_v<Args, GroupedPlanLeaf>)
            return bind_grouped_plan<kExperts, kExpertsPerToken>(args);
        else if constexpr (std::is_same_v<Args, RouteTapExchange>)
            return bind_route_tap(args);
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

    // Publish the kept prefix of a staged window: its K/V rows into the resident layout (two segments
    // when a ring wraps) and history image rows - 1 of every convolution stream.
    void publish(const MiniResidentSessionState& session, MiniResidentChunk chunk, mini_gpu::Stream stream) const {
        const auto view = state_view(session.plan());
        const auto copy = [stream](auto* destination, const auto* source, std::size_t elements) {
            mini_gpu::copy(destination, source, elements * sizeof(*source), mini_gpu::device_to_device, stream);
        };
        for (const auto visit : mini_schedule()) {
            const auto resident = session.visit(visit);
            const auto& kv = resident.kv;
            const auto cache = view.cache(visit, kv);
            const auto plan = session::kept_prefix_copy_geometry(chunk, kv, kKvWidth);
            copy(kv.keys + plan.kv_offset, cache.pending_keys, plan.contiguous_elements);
            copy(kv.values + plan.kv_offset, cache.pending_values, plan.contiguous_elements);
            if (plan.wrapped_elements != 0U) {
                copy(kv.keys, cache.pending_keys + plan.contiguous_elements, plan.wrapped_elements);
                copy(kv.values, cache.pending_values + plan.contiguous_elements, plan.wrapped_elements);
            }
            for (unsigned path = 0; path < resident.history.size(); ++path) {
                const auto image = resident.history[path];
                copy(image.data(), view.history(visit, static_cast<MiniConvolutionStream>(path)) +
                    plan.history_row * image.size(), image.size());
            }
        }
        mini_gpu::synchronize(stream);
    }
};

}  // namespace detail

struct MiniCudaProgramDecoder::Impl {
    using GroupedScratch = exec::cuda_decode::GroupedDecodeScratch;
    Impl(MiniResidentDecoder& resident, bool fuse, exec::cuda_decode::R1Forms selected)
        : decoder(resident), grouped(!decoder.step_program_backend().step_program_buffers().packed_decode),
           epilogue_fusion(fuse), forms(selected),
          grouped_words(grouped ? GroupedScratch::word_count(kExperts, kExpertsPerToken) : 0U),
          grouped_values(grouped ? GroupedScratch::value_count(kHiddenWidth, kExpertWidth, kExpertsPerToken) : 0U),
          grouped_descriptors(grouped ? 2U * kExpertsPerToken : 0U) {}

    // The drafter tap store of the resident session (empty without a drafter).
    void bind_taps(MiniCudaStepStorage& storage) {
        const auto taps = decoder.step_program_draft_taps();
        if (taps.empty()) return;
        storage.draft_taps = taps.data();
        storage.draft_tap_positions = static_cast<unsigned>(taps.size() / MiniDSparkTapPlan::width);
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
        // A greedy step without host logits runs the screened head (its own program, graph_id 1); a step that returns
        // the logits row keeps the full head program (graph_id 0, the incumbent key).
        auto& screen = decoder.step_program_screen();
        const bool screened = screen.get() != nullptr && host_logits.empty();
        std::optional<core::screened_head_plan::Launch> launch;
        if (screened) launch = screen.get()->launch(nullptr);
        const MiniCudaStepRows rows{decoder.step_program_embeddings(), decoder.step_program_hidden(),
            decoder.step_program_logits(), greedy.step_program_partials(), greedy.step_program_device_token(),
            greedy.step_program_reply(), launch ? &*launch : nullptr, screened ? screen.stats_reply() : nullptr};
        auto& cached = programs[screened ? 1U : 0U];
        const auto raw = backend.step_program_buffers();
        MiniCudaStepStorage storage{raw.max_rows, raw.controller, raw.attention, raw.lag_weights,
            raw.matrices, raw.route_ids, raw.route_logits, raw.route_weights, raw.expert_activation, raw.flags};
        bind_taps(storage);
        const bool tapped = gate_tap ? gate_tap->active() : cuda::core::route_tap_active();
        const MiniCudaStepMode mode{{grouped, {.tap = tapped}, forms},
            {grouped_words.span(), grouped_values.span(), grouped_descriptors.span()}, gate_tap, backend.stream(),
            +[](void* to, const void* from, std::size_t bytes, void* stream) {
                mini_gpu::copy(to, from, bytes, mini_gpu::device_to_device, static_cast<cudaStream_t>(stream));
            }, epilogue_fusion, backend.step_projection_profile(), exec::cuda_decode::bind_projection_batch};
        auto key = mode.key.program;
        key.graph_id = screened ? std::uint8_t{1} : std::uint8_t{0};
        const detail::MiniCudaScalarBinding bind{};
        exec::CudaTicket ticket{};
        const auto dispatch = [&](MiniResidentChunk step, const auto& build) {
            const bool install = !tapped && !cached.has_value();
            std::optional<exec::CudaRecipeProgram> eager;
            if (tapped) eager.emplace(build().compile(key));
            else if (install) cached.emplace(mode.key, build().compile(key));
            if (!tapped && cached->first != mode.key)
                throw std::logic_error("Inkling resident expert mode changed");
            const auto& plan = tapped ? *eager : cached->second;
            exec::StepBlock input{};
            input.h.rows = input.h.valid_mask = 1U;
            input.h.step = step.first_position;
            input.row[0] = {token, step.first_position, 0U, exec::kStepNoParent, 1U, exec::step_valid, 0U, 0U};
            ticket = executor.submit_x3(plan.program, input, {&plan.binding, backend.stream()});
            if (install) {
                const auto stats = executor.graph_stats(key);
                if (stats.captures != 1U || stats.instantiations != 1U ||
                    stats.nodes != plan.program.launches.size() || stats.kernels != plan.program.predicted.launches)
                    throw std::runtime_error("Inkling step program capture did not install the compiled plan");
                std::cerr << "inkling2 step graph route=program head=" << (screened ? "screened" : "full")
                          << " captures=1 nodes=" << stats.nodes
                          << " launches=" << stats.kernels
                          << " epilogue_fusion=" << (epilogue_fusion ? "on" : "off")
                          << " latency_forms=" << (forms == exec::cuda_decode::R1Forms{} ? "off" : "")
                          << (forms.bank_tiled ? "B" : "") << (forms.output_split ? "S" : "")
                          << (forms.moe_pipe ? "P" : "") << " draft_taps=" << (storage.draft_taps ? "on" : "off")
                          << '\n';
            }
            if (tapped && !reported_tap) {
                std::cerr << "inkling2 step graph route=program captures=0 nodes=" << plan.program.launches.size()
                          << " launches=" << plan.program.predicted.launches << " epilogue_fusion=off\n";
                reported_tap = true;
            }
            return true;
        };
        const auto finish = [&] {
            backend.finish();
            if (ticket.serial != 0U) {
                if (!executor.poll(ticket, {})) throw std::logic_error("Inkling program stream handoff did not complete");
                ticket = {};
            }
        };
        const detail::MiniCudaVerifyBinding bind_verify{};
        MiniCudaStepLowering lowering(decoder.step_program_weights(), storage, executor.step_descriptor(),
                                     bind, dispatch, finish, bind_verify, mode);
        MiniResidentForward forward(lowering, state, decoder.step_program_workspace());
        unsigned picked = head_argmax::kNone;
        try {
            forward.append(chunk, rows.embeddings, rows.hidden,
                [&](MiniResidentChunk step, MiniConstRows hidden) {
                    lowering.head(hidden, rows);
                    lowering.submit_eager(step);
                },
                [&] {
                    if (screened) screen.record();
                    picked = greedy.token();
                    if (picked >= kOutputVocabulary) throw std::runtime_error("Inkling 2 Mini: non-finite output logits");
                },
                [&] {
                    lowering.begin_eager();
                    lowering.embed(rows.embeddings);
                },
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
    // (pending K/V rows, one history image per row, picks, logits reply) and the drafter taps of its
    // positions; nothing reaches the resident session until commit(keep) publishes the kept prefix.
    void stage(std::span<const unsigned> ids, std::span<unsigned> picks) {
        if (grouped) throw std::invalid_argument("Inkling verify windows require packed decode");
        const auto logits_count = ids.size() * kOutputVocabulary;
        auto& backend = decoder.step_program_backend();
        auto& state = decoder.step_program_state();
        const MiniResidentChunk chunk{state.committed(), static_cast<unsigned>(ids.size())};
        state.require_append(chunk);
        const auto input = mini_verify_descriptor(chunk.rows, ids, chunk.first_position);
        if (!verify_workspace) verify_workspace.emplace(state.plan());
        auto& work = *verify_workspace;
        const auto resident = backend.step_program_buffers();
        MiniCudaStepStorage storage{work.rows, work.controller.span(), work.attention.span(), resident.lag_weights,
            resident.matrices, work.route_ids.data(), work.route_logits.data(), work.route_weights.data(),
            work.expert_activation.data(), work.flags.span().first(2U * kLogicalVisits * chunk.rows)};
        bind_taps(storage);
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
                          << " launches=" << stats.kernels << " draft_taps=" << (storage.draft_taps ? "on" : "off")
                          << '\n';
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
                                     bind, dispatch, finish, bind_verify, MiniCudaStepMode{}, &view);
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
    // frontier by keep; keep = 0 discards the window. The drafter taps need nothing: they are stored
    // by position and read only up to the committed frontier.
    void commit(unsigned keep) {
        auto& state = decoder.step_program_state();
        const auto queue = decoder.step_program_backend().stream();
        const MiniResidentChunk kept{state.committed(), keep};
        state.commit_pending(keep, [&](unsigned) { verify_workspace->publish(state, kept, queue); });
    }

    // Host logits of the rows of the staged window (pinned verify reply).
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
    const bool grouped;
    const bool epilogue_fusion;
    const exec::cuda_decode::R1Forms forms;
    mini_gpu::Buffer<unsigned> grouped_words;
    mini_gpu::Buffer<float> grouped_values;
    mini_gpu::Buffer<moe_core::device_grouped::TileRows> grouped_descriptors;
    moe::RouteTap* gate_tap{};
    bool reported_tap = false;
    Window window;
    std::unique_ptr<float, detail::MiniPinnedLogitsDeleter> staged_logits;
    // One compiled program per head of the untapped route, keyed by the recipe: [0] the full head (logits row,
    // graph_id 0, the incumbent key) and [1] the screened head (graph_id 1).
    std::array<std::optional<std::pair<exec::cuda_decode::DecodeRecipeKey, exec::CudaRecipeProgram>>, 2> programs;
    std::optional<detail::MiniCudaVerifyWorkspace> verify_workspace;
    std::array<std::optional<exec::CudaRecipeProgram>, exec::cuda_decode::wr1::widths.size() - 1U> verify_programs;
    exec::CudaExecutor executor{true, true, exec::CudaDescriptorLayout::x3};
};

inline MiniCudaProgramDecoder::MiniCudaProgramDecoder(MiniResidentDecoder& decoder, bool epilogue_fusion,
                                                      exec::cuda_decode::R1Forms forms)
    : impl_(std::make_unique<Impl>(decoder, epilogue_fusion, forms)) {}
inline MiniCudaProgramDecoder::~MiniCudaProgramDecoder() = default;

inline unsigned MiniCudaProgramDecoder::prefill(std::span<const unsigned> ids, std::span<float> logits) {
    return impl_->decoder.prefill(ids, logits);
}
inline unsigned MiniCudaProgramDecoder::decode(unsigned token, std::span<float> logits) {
    return impl_->decode(token, logits);
}
inline bool MiniCudaProgramDecoder::supports_verify() const { return !impl_->grouped; }
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

inline unsigned MiniCudaProgramDecoder::admissible_verify_rows(unsigned limit) const {
    const auto& state = impl_->decoder.step_program_state();
    return session::step_replay_rows_that_fit(state.plan(), state.committed(), std::min(limit, kMiniVerifyMaxRows));
}

inline void MiniCudaProgramDecoder::reset_gate_session() { impl_->decoder.reset_gate_session(); }
inline void MiniCudaProgramDecoder::set_gate_route_tap(moe::RouteTap* tap) {
    impl_->decoder.set_gate_route_tap(tap);
    impl_->gate_tap = tap;
}
inline void MiniCudaProgramDecoder::prefill_gate_rows(std::span<const unsigned> ids, std::span<float> logits) {
    impl_->decoder.prefill_gate_rows(ids, logits);
}

// Rows of a drafted window of the program route: the block drafter proposes up to 7 tokens after the
// anchor (mtp.safetensors block of 8); lookup proposals use every verify width.
inline constexpr std::uint32_t kMiniProgramDraftRows = MiniDSparkCheckpoint::block;

// Shared speculator settings of the program route (spec::BlockConfig, never changes emitted ids). Windows
// run on padded graph widths 2/4/8/16, so a window of R rows costs the graph of its padded width: the
// arbiter considers only the graph widths (a 3-row window would pay the 4-row graph for fewer tokens).
inline void tune_mini_program_speculation(spec::BlockConfig& config, std::uint32_t max_rows) {
    config.arbiter.windows.clear();
    for (const auto width : exec::cuda_decode::wr1::widths)
        if (width <= std::max<std::uint32_t>(max_rows, 1U)) config.arbiter.windows.push_back(width);
    if (max_rows > config.arbiter.windows.back()) config.arbiter.windows.push_back(max_rows);
    config.chain_max_k = kMiniProgramDraftRows - 1U;
}

template <bool EpilogueFusion, unsigned LatencyForms>
inline std::unique_ptr<serve::SerialGenerationDecoder> load_mini_cuda_program(const serve::SerialGenerationLoad& load,
                                                                              moe_core::grouped_w4a4::Activations moe_w4a4,
                                                                              const char* dspark) {
    using Activations = moe_core::grouped_w4a4::Activations;
    struct Decoder final : serve::SerialGenerationDecoder {
        Decoder(const serve::SerialGenerationLoad& input, Activations w4a4, const char* drafter)
            : Decoder(input, w4a4, drafter,
                      mini_step_session_geometry(input.positions, input.prefill_chunk, input.verify_rows)) {}
        // A verify window of R rows stages all padded_step_replay_width(R) rows as one pending append:
        // the session plan admits the widest window (chunk) and its padding overhang (context). Without
        // speculation (verify_rows 0) the geometry is the request's. The drafter (block of 8, its
        // checkpoint loaded only for a drafting request) reads the feature taps the program stores.
        Decoder(const serve::SerialGenerationLoad& input, Activations w4a4, const char* drafter,
                session::StepReplaySessionGeometry geometry)
            : resident(input.checkpoint, geometry.positions, input.prefill_chunk, input.packed_decode, false,
                       weight::kBf16WeightStorage, 0U, drafter != nullptr, 1U, 0U, geometry.append_rows),
              program(with_w4a4(resident, w4a4), EpilogueFusion, mini_latency_forms(LatencyForms)) {
            if (drafter) block.emplace(resident, geometry.positions, drafter);
        }
        // Same order as the gate path: the lever (and its NVFP4 activation image) is set before the program binds.
        static MiniResidentDecoder& with_w4a4(MiniResidentDecoder& decoder, Activations w4a4) {
            decoder.set_moe_w4a4(w4a4);
            return decoder;
        }
        unsigned prefill(std::span<const unsigned> ids, std::span<float> logits) override {
            return program.prefill(ids, logits);
        }
        unsigned decode(unsigned token, std::span<float> logits) override { return program.decode(token, logits); }
        // Speculation on the step program: verify windows on the width graphs, kept prefix published from
        // the pending verify workspace (no checkpoint, no replay). Drafters: lookup (shared BlockDrafter)
        // and the block drafter (chain_drafter) on the taps the program stores in every graph.
        spec::VerifyTarget* verify_target() override { return program.supports_verify() ? &target : nullptr; }
        std::uint32_t max_verify_rows(serve::CliSpeculation mode) const override {
            return mode == serve::CliSpeculation::lookup ? MiniCudaProgramDecoder::max_verify_rows()
                                                         : kMiniProgramDraftRows;
        }
        spec::ChainDrafter* chain_drafter(std::uint32_t anchor, std::uint32_t position) override {
            if (!block) return nullptr;
            block->start(anchor, position);
            return &*block;
        }
        void tune_speculation(spec::BlockConfig& config) const override {
            tune_mini_program_speculation(config, MiniCudaProgramDecoder::max_verify_rows());
        }
        [[nodiscard]] std::string verify_cost_device() const override { return device::cuda_cost_device_identity(); }
        [[nodiscard]] std::string_view verify_cost_route() const noexcept override { return "program"; }
        MiniResidentDecoder resident;
        MiniCudaProgramDecoder program;
        spec::EngineVerifyTarget<MiniCudaProgramDecoder> target{program};
        std::optional<MiniResidentDSpark> block;
    };
    // The drafter is loaded only when the request may draft with it (auto, more than one generated row).
    const bool drafts = dspark && load.verify_rows > 1U && load.speculation == serve::CliSpeculation::automatic;
    return std::make_unique<Decoder>(load, moe_w4a4, drafts ? dspark : nullptr);
}
#endif

}  // namespace strix::inkling2
