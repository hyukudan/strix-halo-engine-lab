#pragma once

#include "exec/cuda_decode_epilogue.hpp"
#include "exec/cuda_decode_lowering.hpp"
#include "exec/cuda_decode_transition.hpp"
#include "inkling2_mini_attention_prepare_device.hpp"
#include "inkling2_mini_controller_device.hpp"
#include "inkling2_mini_expert_layout.hpp"
#include "inkling2_mini_moe_dispatch.hpp"
#include "inkling2_mini_resident_forward.hpp"
#include "strix/graph/launch_plan.hpp"
#include "strix/serve/serial_generation_cli.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <optional>

#if defined(STRIX_INKLING2_STEP_PROGRAM)
#include "inkling2_mini_resident_decoder.hpp"
#include "inkling2_mini_weight_options.hpp"
#include "strix/env_flags.hpp"
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
    std::span<float> messages{};
};

struct MiniCudaStepRows {
    std::span<float> embeddings, hidden, logits;
    head_argmax::Pick* partials;
    std::uint32_t* picked;
    std::uint32_t* reply;
    // EXACT screened greedy head (screened-head lever; one lane, x bound to the final hidden row): its five stages replace
    // the projection, the softcap pass and the two greedy leaves, writing the same picked word. nullptr: the full head.
    const core::screened_head_plan::Launch* screen = nullptr;
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
    bool moe_tail = false;
    bool moe_route_inline = false;
};

// Bind the resident traversal's exact native leaves. The same traversal
// supplies visits, residual banks, copies and visit-local state to both routes.
template <class Weights, class Bind, class Dispatch, class Finish>
class MiniCudaStepLowering {
    using Shape = MiniPreparedGqaShape;
    using Mode = exec::cuda_decode::Bf16DenseMode;
    using MoeTail = moe_core::r1_tail::ConvResidualTail<exec::cuda_decode::kPackedConvTailTaps>;
    static constexpr float epsilon = 1.0e-6F;

public:
    MiniCudaStepLowering(const Weights& weights, MiniCudaStepStorage storage, const exec::StepBlock* descriptor,
                        const Bind& bind, const Dispatch& dispatch, const Finish& finish, MiniCudaStepMode mode = {})
        : weights_(weights), storage_(storage), descriptor_(descriptor), bind_(bind), dispatch_(dispatch),
          finish_(finish), mode_(mode), recipe_(mode.key.program.tap), transition_(transition_forms(mode)) {}

    bool try_step_graph(MiniResidentChunk chunk, auto& work) {
        if (mode_.key.program.tap) return false;
        return dispatch_(chunk, [&] {
            begin();
            work();
            flush();
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
        flush();
        recording_ = false;
        (void)dispatch_(chunk, [&] { return std::move(recipe_); });
    }

    void embed(std::span<float> image) {
        using namespace exec::cuda_decode;
        recipe_.stage("embed");
        add_leaf(Embedding{weights_.bf16("embed.weight").data(), &descriptor_->row[0].token,
                            image.data(), kVocabulary, kHiddenWidth}, true);
        norm(image, "embed_norm.weight", image);
    }

    void head(MiniConstRows hidden, MiniCudaStepRows rows) {
        using namespace exec::cuda_decode;
        flush();
        recipe_.visit(-1);
        recipe_.stage("head");
        if (rows.screen) {
            auto launch = *rows.screen;  // the one-lane launch record of the resident screen (ResidentScreen::launch)
            launch.x = hidden.data();
            add_leaf(ScreenedHeadPrep{launch});
            add_leaf(ScreenedHeadPass1{launch});
            add_leaf(ScreenedHeadCompact{launch});
            add_leaf(ScreenedHeadPass2{launch});
            add_leaf(ScreenedHeadPick{launch, rows.picked});
            recipe_.copy(exec::CudaNodeKind::copy_d2h_pinned, {rows.reply, rows.picked, sizeof(*rows.reply)});
            return;
        }
        const std::array projection{core::Bf16RowProjection{weights_.bf16("unembed.weight").data(),
                                                          kOutputVocabulary, rows.logits.data()}};
        const Bf16Softcap tail{rows.logits.data(), rows.logits.size(), kLogitSoftcap};
        lower_dense(hidden.data(), kHiddenWidth, projection, false, [&](const auto& head, bool pdl) {
            auto producer = bind_(head);
            if (epilogue_fusion() && bind_epilogue(producer.identity, graph::Op::linear, graph::Form::softcap,
                                                   head, tail, pdl)) return;
            recipe_.kernel(std::move(producer), pdl);
            add_leaf(tail);
        });
        const GreedyPartial reduction{rows.logits.data(), rows.partials, kOutputVocabulary,
                                     head_argmax::default_chunks(kOutputVocabulary)};
        add_leaf(reduction);
        add_leaf(GreedyFinal{reduction, rows.picked});
        recipe_.copy(exec::CudaNodeKind::copy_d2h_pinned, {rows.reply, rows.picked, sizeof(*rows.reply)});
    }

    template <class T>
    void copy(std::span<const T> input, std::span<T> output) {
        if (input.data() == output.data()) return;
        flush();
        if (!recording_ && mode_.key.program.tap) {
            mode_.copy_eager(output.data(), input.data(), input.size_bytes(), mode_.stream);
            return;
        }
        recipe_.copy(exec::CudaNodeKind::copy_d2d, {output.data(), input.data(), input.size_bytes()});
    }

    void norm(MiniConstRows input, std::string_view name, MiniMutableRows output) {
        rms(input.data(), weights_.bf16(name).data(), output.data());
    }

    void layer_input(LayerVisit visit, MessagePath path, MiniConstRows raw,
                     std::span<const MiniConstRows> previous, MiniMutableRows output) {
        recipe_.visit(static_cast<std::int32_t>(visit.logical_layer));
        recipe_.stage(mini_controller_path(path));
        const auto layer = visit.physical_layer;
        const auto* scale = weights_.bf16(parameter(layer, mini_input_norm_suffix(visit, path))).data();
        if (visit.pass == 0U) {
            // The input norm of a raw readout row joins the held residual transition (same bits, one launch).
            rms(raw.data(), scale, output.data());
            return;
        }
        flush();
        const bool key_value = path == MessagePath::key || path == MessagePath::value;
        const auto queries = weights_.f32(key_value ? "middle_cycle_cross_pass_kv_queries"
                                                    : "middle_cycle_cross_pass_queries");
        const auto query = 2U * (layer - kCoreFirst) + (path == MessagePath::value || path == MessagePath::moe ? 1U : 0U);
        const auto stem = parameter(layer, std::string("controllers.") + std::string(mini_controller_path(path)) + ".");
        if (message_batch_) {
            // The pass's bank batch already holds this controller's message row (same bits as its own readout).
            const auto* message = storage_.messages.data() + (key_value ? kMiniMessageQueryRows : 0U) * std::size_t{kHiddenWidth} +
                                  std::size_t{query} * kHiddenWidth;
            exec::cuda_decode::lower_low_rank_controller_tail(kHiddenWidth, kControllerRank, epsilon, raw, message,
                output, storage_.controller, weights_.bf16(stem + "down_weight").data(),
                weights_.bf16(stem + "forget_weight").data(), weights_.bf16(stem + "gain_weight").data(), scale, add(),
                mode_.key.forms);
            return;
        }
        exec::cuda_decode::lower_low_rank_controller(kHiddenWidth, kControllerRank, epsilon,
            queries.subspan(std::size_t{query} * kHiddenWidth, kHiddenWidth), raw, previous, output, storage_.controller,
            weights_.bf16(stem + "down_weight").data(), weights_.bf16(stem + "forget_weight").data(),
            weights_.bf16(stem + "gain_weight").data(), scale, add(), mode_.key.forms);
    }

    void residual_read(std::span<const MiniConstRows> blocks, MiniConstRows current, std::string_view name,
                       std::span<const MiniMutableRows> output) {
        recipe_.stage("residual_read");
        core::residual_mix::MixArgs args{};
        args.queries = weights_.f32(name).data();
        args.rows = 1U;
        args.width = kHiddenWidth;
        args.epsilon = epsilon;
        args.bank_count = static_cast<unsigned>(blocks.size());
        args.query_count = static_cast<unsigned>(output.size());
        for (std::size_t i = 0; i < blocks.size(); ++i) args.banks[i] = blocks[i].data();
        args.banks[args.bank_count++] = current.data();
        for (std::size_t i = 0; i < output.size(); ++i) args.output[i] = output[i].data();
        // Held: a pending residual add of its current row and the input norms of its rows join it (R1Forms.transition).
        transition_.readout(args, held());
    }

    void capture_message_bank(const std::array<MiniConstRows, 2>& source,
                              const std::array<MiniMutableRows, 2>& normalized) {
        flush();
        recipe_.stage("message_bank");
        for (std::size_t i = 0; i < source.size(); ++i)
            add_leaf(core::residual_mix::NormalizeArgs{source[i].data(), normalized[i].data(),
                                                      1U, kHiddenWidth, epsilon});
        message_batch_ = !mode_.key.program.tap && mode_.key.forms.bank_batch && bind_message_batch(normalized);
    }

    void add(MiniConstRows lhs, MiniConstRows rhs, MiniMutableRows output, bool round_bf16) {
        recipe_.stage("residual_add");
        // Held: the next bank readout of this row takes it as its prologue (R1Forms.transition).
        transition_.add(core::residual_mix::AddArgs{lhs.data(), rhs.data(), output.data(), output.size(), round_bf16},
                        held());
    }

    void attention(LayerVisit visit, MiniResidentChunk chunk, MiniConstRows query, MiniConstRows key,
                   MiniConstRows value, const MiniResidentVisitState& state, MiniMutableRows output) {
        using namespace exec::cuda_decode;
        flush();
        recipe_.stage("attention");
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
        const Attention lowering{*this, std::span(round_sites).first(count)};
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
        flush();
        recipe_.stage("post_convolution");
        const auto* weights = mini_post_convolution_weights(storage_.lag_weights, visit, path == MiniConvolutionStream::moe);
        require(attn_core::enqueue_prepared_gqa_post_convolution<Shape>(
            Attention{*this}, 1U, weights, input.data(), history.data(), output.data()));
    }

    void moe(LayerVisit visit, MiniConstRows input, MiniMutableRows output) {
        lower_moe(visit, input, output, nullptr);
    }

    [[nodiscard]] bool moe_tail_enabled() const noexcept {
        return mode_.moe_tail && !mode_.key.grouped && !mode_.key.forms.moe_pipe && !mode_.key.program.tap;
    }

    void moe_tail(LayerVisit visit, MiniConstRows input, MiniMutableRows output, MiniMutableRows history,
                  MiniConstRows residual, MiniMutableRows convolution, MiniMutableRows sum) {
        static_assert(Shape::kConvTaps == exec::cuda_decode::kPackedConvTailTaps);
        const MoeTail tail{
            attn_core::prepared_gqa_post_convolution_plan<Shape>(),
            mini_post_convolution_weights(storage_.lag_weights, visit, true), history.data(), convolution.data(),
            residual.data(), sum.data(), attn_core::kPreparedGqaPostConvolutionRound, true,
        };
        lower_moe(visit, input, output, &tail);
    }

private:
    void lower_moe(LayerVisit visit, MiniConstRows input, MiniMutableRows output, const MoeTail* tail) {
        using namespace exec::cuda_decode;
        flush();
        recipe_.stage("moe");
        const auto layer = visit.physical_layer;
        const std::array router{core::Bf16RowProjection{weights_.bf16(parameter(layer, "mlp.gate.weight")).data(),
                                                       kExperts, storage_.route_logits}};
        lower_dense(input.data(), kHiddenWidth, router, true, add());
        auto* failure = storage_.flags.data() + kLogicalVisits + visit.logical_layer;
        const auto* matrices = storage_.matrices.data() + std::size_t{layer} * kExperts;
        const auto* bias = weights_.f32(parameter(layer, "mlp.gate.bias")).data();
        const auto select = [&](unsigned* status) {
            add_leaf(CheckedWaveRoute{storage_.route_logits, bias, storage_.route_ids, storage_.route_weights, status});
            if (mode_.key.program.tap)
                recipe_.host_action(bind_(RouteTapExchange{mode_.tap, layer, 1U, kExpertsPerToken, kExperts,
                                                           storage_.route_ids, storage_.route_weights}));
        };
        // The loaded image says the layer's format (narrow expert image); weights without one keep the checkpoint's.
        const auto format = [&] {
            if constexpr (requires { weights_.expert_format(layer); }) return weights_.expert_format(layer);
            else return mini_expert_format(layer);
        }();
        if (mode_.key.grouped) {
            const auto [plan, scratch, status] = mode_.grouped.bind(kExperts, kExpertsPerToken,
                mini_grouped::CudaBackend::rows_per_tile, matrices, storage_.route_ids, storage_.route_weights,
                failure, storage_.expert_activation, kHiddenWidth, kExpertWidth);
            select(status);
            lower_grouped_decode(plan, scratch, format, input.data(), output.data(), kMlpGateSoftcap, kMlpUpSoftcap,
                                 add(), [this](auto clear) { recipe_.clear(clear); });
            return;
        }
        static_assert(kExpertsPerToken == kPackedExpertsPerToken);
        const moe_core::microscaled_r1::Args args{
            matrices, storage_.route_ids, storage_.route_weights,
            failure, input.data(), storage_.expert_activation, output.data(), kExperts, kHiddenWidth, kExpertWidth,
            kMlpGateSoftcap, kMlpUpSoftcap,
        };
        if (mode_.key.forms.moe_pipe) {
            select(failure);
            add_leaf(PackedGateUpPipe{args, format});
            add_leaf(PackedDownCombinePipe{args, format});
            return;
        }
        if (mode_.moe_route_inline && !mode_.key.program.tap) {
            add_leaf(RoutedPackedGateUp{args, format, storage_.route_logits, bias,
                                        storage_.route_ids, storage_.route_weights, failure}, true);
        } else {
            select(failure);
            add_leaf(PackedGateUp{args, format});
        }
        if (tail) {
            require(moe_core::microscaled_r1::admits_tail(args, *tail, 1U, kPackedExpertsPerToken));
            add_leaf(PackedDownCombineTail{args, format, *tail}, true);
        } else {
            add_leaf(PackedDownCombine{args, format});
        }
    }

    void begin() {
        recording_ = true;
        message_batch_ = false;
        recipe_.stage("flags");
        recipe_.clear({storage_.flags.data(), storage_.flags.size_bytes(), 0});
    }

    static std::string parameter(unsigned layer, std::string_view suffix) {
        return "layers." + std::to_string(layer) + "." + std::string(suffix);
    }

    static void require(bool valid) {
        if (!valid) throw std::invalid_argument("invalid Inkling CUDA step program binding");
    }

    void rms(const float* input, const std::uint16_t* scale, float* output) {
        transition_.norm(exec::cuda_decode::RmsBf16{input, scale, output, 1U, kHiddenWidth, epsilon}, held());
    }

    void add_leaf(const auto& args, bool pdl = false) {
        flush();
        recipe_.kernel(bind_(args), pdl);
    }

    // Emission of the leaves the transition chain held: a held add / readout / transition carries its own stage
    // label; the labels of the caller that triggered the emission are restored after it.
    auto held() {
        return [this](const auto& args, bool pdl) {
            using Args = std::remove_cvref_t<decltype(args)>;
            const std::string stage(recipe_.stage_label());
            if constexpr (std::is_same_v<Args, core::residual_mix::AddArgs>) recipe_.stage("residual_add");
            else if constexpr (std::is_same_v<Args, core::residual_mix::TransitionArgs>) recipe_.stage("residual_transition");
            else if constexpr (!std::is_same_v<Args, exec::cuda_decode::RmsBf16>) recipe_.stage("residual_read");
            recipe_.kernel(bind_(args), pdl);
            recipe_.stage(stage);
        };
    }

    // Emits what the transition chain holds, in program order; called before any other leaf.
    void flush() { transition_.flush(held()); }

    static exec::cuda_decode::R1Forms transition_forms(const MiniCudaStepMode& mode) {
        auto forms = mode.key.forms;
        // The route-tapped (gate) recipe keeps every incumbent leaf, like the epilogue fusion.
        if (mode.key.program.tap) forms.transition = false;
        return forms;
    }

    // Every controller of the next pass reads the bank just captured: one residual bank batch issues all of their
    // message readouts (each job = that controller's single-query readout, same bits). False keeps the
    // per-controller readout when the scratch or the query tensors do not have the declared shape.
    bool bind_message_batch(const std::array<MiniMutableRows, 2>& bank) {
        const auto queries = weights_.f32("middle_cycle_cross_pass_queries");
        const auto kv_queries = weights_.f32("middle_cycle_cross_pass_kv_queries");
        const auto rows = std::size_t{kMiniMessageQueryRows} * kHiddenWidth;
        if (storage_.messages.size() < kMiniMessageBatchElements || queries.size() != rows || kv_queries.size() != rows)
            return false;
        core::residual_mix::BankBatchArgs batch{};
        batch.banks[0] = bank[0].data();
        batch.banks[1] = bank[1].data();
        batch.bank_count = 2U;
        batch.width = kHiddenWidth;
        batch.epsilon = epsilon;
        batch.normalized = true;
        batch.sets[0] = {queries.data(), storage_.messages.data(), kMiniMessageQueryRows};
        batch.sets[1] = {kv_queries.data(), storage_.messages.data() + rows, kMiniMessageQueryRows};
        batch.set_count = 2U;
        if (!core::residual_mix::fits(batch)) return false;
        add_leaf(batch, true);
        return true;
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
        return exec::cuda_decode::lower_exact_epilogue(recipe_, plan, plan.launches.front(), head, tail, pdl, bind_);
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
                    if (!bind_rounds(producer.identity, head, pdl)) lower.recipe_.kernel(std::move(producer), pdl);
                });
                return success;
            });
        }
        template <bool Round = false>
        bool convolve(const core::ShortConvolution& plan, unsigned, const float* input, const float* weights,
                      const float* dynamic, float* history, float* output) const {
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
            lower.recipe_.kernel(std::move(producer), false);
            return success;
        }
        bool clip(float* row, const std::uint16_t* gain, unsigned, unsigned heads, unsigned width, float threshold,
                  bool position_scale, unsigned first, const std::uint32_t* position = nullptr) const {
            lower.add_leaf(exec::cuda_decode::HeadClipBf16{row, gain, row, position, heads, width,
                position_scale ? first : 0U, position_scale ? Shape::kPositionFloor : 0U,
                Shape::kAttentionEpsilon, threshold, position_scale ? Shape::kPositionAlpha : 0.0F});
            return success;
        }
        bool round(float* row, std::size_t elements, unsigned* failure = nullptr) const {
            const auto site = find_round(row, elements);
            if (!failure && site != rounds.end() && site->bound) {
                site->bound = false;
                return success;
            }
            lower.add_leaf(exec::cuda_decode::Bf16Round{row, elements});
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
            lower.add_leaf(exec::cuda_decode::Bf16SigmoidProduct{input, logits, output, elements});
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
                this->lower.recipe_.kernel(std::move(leaf), true);
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
    MiniCudaStepMode mode_;
    exec::CudaProgramRecipe recipe_;
    exec::cuda_decode::TransitionChain transition_;
    bool recording_ = false;
    bool message_batch_ = false;
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
    void reset_gate_session();
    void prefill_gate_rows(std::span<const unsigned> ids, std::span<float> logits);
    void set_gate_route_tap(moe::RouteTap*);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Exact R1 latency forms (bitwise against the incumbents), opt-in until the RTX A/B selects them:
// mask bit 1 = tiled bank readouts, 2 = controller output split, 4 = packed MoE pipe, 8 = per-pass controller
// message readouts in one residual bank batch, 16 = residual transition (add + bank readout + input norms in one
// row-owner launch). The controller batch and residual transition are explicit CUDA step-program options.
inline constexpr unsigned kMiniFormsMix = 3U, kMiniFormsMoe = 4U, kMiniFormsAll = 7U;
inline constexpr unsigned kMiniFormsBatch = kMiniFormsMix | 8U, kMiniFormsB3 = kMiniFormsBatch | 16U;
[[nodiscard]] constexpr exec::cuda_decode::R1Forms mini_latency_forms(unsigned mask) noexcept {
    return {.bank_tiled = (mask & 1U) != 0U, .output_split = (mask & 2U) != 0U, .moe_pipe = (mask & 4U) != 0U,
            .transition = (mask & 16U) != 0U, .bank_batch = (mask & 8U) != 0U};
}
inline constexpr exec::cuda_decode::R1Forms kMiniLatencyForms = mini_latency_forms(kMiniFormsAll);
// RTX r1 (11-22, ink_rtx_dec kit out/r1, bitwise ids / logits / 31 decode rows): greedy 256 tok/s base 102.8,
// tiled + split 109.9, + MoE pipe 108.4 -> the default binds mix; the pipe stays opt-in (--latency-forms).
inline constexpr unsigned kMiniDefaultForms = kMiniFormsMix;

// moe_w4a4: resolved operand class of the W4A4 expert lever (mini_moe_w4a4_class; bf16 = off), applied to the resident
// prefill before the program binds. attn_projected_bias: the resolved projected-bias attention lever. screened_head: the
// resolved screened-head lever (a greedy step without host logits runs the screened head program, graph_id 1).
template <bool EpilogueFusion, unsigned LatencyForms = kMiniDefaultForms>
std::unique_ptr<serve::SerialGenerationDecoder> load_mini_cuda_program(const serve::SerialGenerationLoad&,
                                                                       moe_core::grouped_w4a4::Activations moe_w4a4,
                                                                       bool controller_one_plane, bool attn_projected_bias,
                                                                       bool screened_head);

#if defined(STRIX_INKLING2_STEP_PROGRAM)
struct MiniCudaProgramDecoder::Impl {
    using GroupedScratch = exec::cuda_decode::GroupedDecodeScratch;
    Impl(MiniResidentDecoder& resident, bool fuse, exec::cuda_decode::R1Forms selected)
        : decoder(resident), grouped(!decoder.step_program_backend().step_program_buffers().packed_decode),
           epilogue_fusion(fuse), forms(selected),
          grouped_words(grouped ? GroupedScratch::word_count(kExperts, kExpertsPerToken) : 0U),
          grouped_values(grouped ? GroupedScratch::value_count(kHiddenWidth, kExpertWidth, kExpertsPerToken) : 0U),
          grouped_descriptors(grouped ? 2U * kExpertsPerToken : 0U),
          messages(forms.bank_batch ? kMiniMessageBatchElements : 0U) {}

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
        // A greedy step without host logits runs the lever's screened head in its own program (graph_id 1); a step that
        // returns the logits row, and every step without the lever, keeps the full-head program (graph_id 0, the
        // incumbent key).
        const auto* screen = decoder.step_program_screen();
        const bool screened = screen != nullptr && host_logits.empty();
        std::optional<core::screened_head_plan::Launch> launch;
        if (screened) launch = screen->launch(nullptr);
        const MiniCudaStepRows rows{decoder.step_program_embeddings(), decoder.step_program_hidden(),
            decoder.step_program_logits(), greedy.step_program_partials(), greedy.step_program_device_token(),
            greedy.step_program_reply(), launch ? &*launch : nullptr};
        const auto raw = backend.step_program_buffers();
        const MiniCudaStepStorage storage{raw.max_rows, raw.controller, raw.attention, raw.lag_weights,
            raw.matrices, raw.route_ids, raw.route_logits, raw.route_weights, raw.expert_activation, raw.flags, messages.span()};
        const bool tapped = gate_tap ? gate_tap->active() : cuda::core::route_tap_active();
        const bool packed_r1 = !grouped && !forms.moe_pipe && !tapped;
        const MiniCudaStepMode mode{{grouped, {.tap = tapped, .graph_id = static_cast<std::uint8_t>(screened ? 1U : 0U)},
                                     forms},
            {grouped_words.span(), grouped_values.span(), grouped_descriptors.span()}, gate_tap, backend.stream(),
            +[](void* to, const void* from, std::size_t bytes, void* stream) {
                mini_gpu::copy(to, from, bytes, mini_gpu::device_to_device, static_cast<cudaStream_t>(stream));
            }, epilogue_fusion, backend.step_projection_profile(), exec::cuda_decode::bind_projection_batch,
            packed_r1 && moe_tail, packed_r1 && moe_route_inline};
        const auto key = mode.key.program;
        const auto bind = [](const auto& args) {
            using namespace exec::cuda_decode;
            using Args = std::remove_cvref_t<decltype(args)>;
            if constexpr (std::is_same_v<Args, CheckedWaveRoute>)
                return bind_checked_wave<MiniRoutePolicy, detail::MiniMoeMath, moe::RouteWaveTraits<true>>(
                    args, "cuda-sm120-checked-wave-mini-route-r1-v1");
            else if constexpr (std::is_same_v<Args, RoutedPackedGateUp>)
                return bind_routed_gate_up<MiniRoutePolicy, detail::MiniMoeMath, moe::RouteWaveTraits<true>>(
                    args, "mini-route");
            else if constexpr (std::is_same_v<Args, GroupedPlanLeaf>)
                return bind_grouped_plan<kExperts, kExpertsPerToken>(args);
            else if constexpr (std::is_same_v<Args, RouteTapExchange>)
                return bind_route_tap(args);
            else
                return exec::cuda_decode::bind(args);
        };
        exec::CudaTicket ticket{};
        auto& cached = programs[screened ? 1U : 0U];
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
                std::cerr << "inkling2 step graph route=program captures=1 nodes=" << stats.nodes
                          << " launches=" << stats.kernels
                          << " epilogue_fusion=" << (epilogue_fusion ? "on" : "off")
                          << " latency_forms=" << (forms == exec::cuda_decode::R1Forms{} ? "off" : "")
                          << (forms.bank_tiled ? "B" : "") << (forms.output_split ? "S" : "")
                          << (forms.moe_pipe ? "P" : "") << (forms.bank_batch ? "B3-batch" : "")
                          << (forms.transition ? "B3-transition" : "") << (screened ? " head=screened" : "")
                          << (moe_tail ? (mode.moe_tail ? " moe_tail=on" : " moe_tail=declined") : "")
                          << (moe_route_inline ? (mode.moe_route_inline ? " moe_route=inline" : " moe_route=declined") : "")
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
        MiniCudaStepLowering lowering(decoder.step_program_weights(), storage, executor.step_descriptor(),
                                     bind, dispatch, finish, mode);
        MiniResidentForward forward(lowering, state, decoder.step_program_workspace());
        unsigned picked = head_argmax::kNone;
        try {
            forward.append(chunk, rows.embeddings, rows.hidden,
                [&](MiniResidentChunk step, MiniConstRows hidden) {
                    lowering.head(hidden, rows);
                    lowering.submit_eager(step);
                },
                [&] {
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
        if (screened) decoder.step_program_screened_pick();
        return picked;
    }

    MiniResidentDecoder& decoder;
    const bool grouped;
    const bool epilogue_fusion;
    const exec::cuda_decode::R1Forms forms;
    const bool moe_tail = mini_moe_tail_switch();
    const bool moe_route_inline = strix::envcfg::flag("STRIX_MOE_ROUTE_INLINE", false);
    mini_gpu::Buffer<unsigned> grouped_words;
    mini_gpu::Buffer<float> grouped_values;
    mini_gpu::Buffer<moe_core::device_grouped::TileRows> grouped_descriptors;
    mini_gpu::Buffer<float> messages;
    moe::RouteTap* gate_tap{};
    bool reported_tap = false;
    std::unique_ptr<float, detail::MiniPinnedLogitsDeleter> staged_logits;
    // One installed program per head: [0] the full head (the incumbent key), [1] the screened head (graph_id 1).
    std::array<std::optional<std::pair<exec::cuda_decode::DecodeRecipeKey, exec::CudaRecipeProgram>>, 2> programs;
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
inline void MiniCudaProgramDecoder::reset_gate_session() { impl_->decoder.reset_gate_session(); }
inline void MiniCudaProgramDecoder::set_gate_route_tap(moe::RouteTap* tap) {
    impl_->decoder.set_gate_route_tap(tap);
    impl_->gate_tap = tap;
}
inline void MiniCudaProgramDecoder::prefill_gate_rows(std::span<const unsigned> ids, std::span<float> logits) {
    impl_->decoder.prefill_gate_rows(ids, logits);
}

template <bool EpilogueFusion, unsigned LatencyForms>
inline std::unique_ptr<serve::SerialGenerationDecoder> load_mini_cuda_program(const serve::SerialGenerationLoad& load,
                                                                              moe_core::grouped_w4a4::Activations moe_w4a4,
                                                                              bool controller_one_plane,
                                                                              bool attn_projected_bias,
                                                                              bool screened_head) {
    using Activations = moe_core::grouped_w4a4::Activations;
    struct Decoder final : serve::SerialGenerationDecoder {
        Decoder(const serve::SerialGenerationLoad& input, Activations w4a4, bool one_plane, bool projected_bias,
                bool screened)
            : resident(input.checkpoint, input.positions, input.prefill_chunk, input.packed_decode, false,
                       weight::kBf16WeightStorage, 0U, false, 1U, 0U, false, one_plane),
              program(with_levers(resident, w4a4, projected_bias, screened), EpilogueFusion,
                      mini_latency_forms(LatencyForms)),
              projected_bias_(projected_bias) {}
        // Same order as the gate path: the levers (and the NVFP4 activation image) are set before the program binds.
        static MiniResidentDecoder& with_levers(MiniResidentDecoder& decoder, Activations w4a4, bool projected_bias,
                                                bool screened) {
            decoder.set_moe_w4a4(w4a4);
            decoder.set_attention_projected_bias(projected_bias);
            static_cast<void>(decoder.set_screened_head(screened));
            return decoder;
        }
        unsigned prefill(std::span<const unsigned> ids, std::span<float> logits) override {
            const auto next = program.prefill(ids, logits);
            // The same observed-route line as the resident command line and the server (lever selected only).
            if (projected_bias_)
                std::cerr << mini_projected_bias_dispatches_line(resident.projected_attention_dispatches()) << '\n';
            return next;
        }
        unsigned decode(unsigned token, std::span<float> logits) override { return program.decode(token, logits); }
        MiniResidentDecoder resident;
        MiniCudaProgramDecoder program;
        bool projected_bias_;
    };
    return std::make_unique<Decoder>(load, moe_w4a4, controller_one_plane, attn_projected_bias, screened_head);
}
#endif

}  // namespace strix::inkling2
