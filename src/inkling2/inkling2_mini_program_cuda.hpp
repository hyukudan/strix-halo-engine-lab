#pragma once

#include "exec/cuda_decode_lowering.hpp"
#include "inkling2_mini_attention_prepare_device.hpp"
#include "inkling2_mini_expert_layout.hpp"
#include "inkling2_mini_resident_forward.hpp"
#include "strix/graph/launch_plan.hpp"
#include "strix/serve/serial_generation_cli.hpp"

#include <memory>

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

// Bind the resident traversal's exact native leaves. The same traversal
// supplies visits, residual banks, copies and visit-local state to both routes.
template <class Weights, class Bind, class Dispatch, class Finish>
class MiniCudaStepLowering {
    using Shape = MiniPreparedGqaShape;
    using Mode = exec::cuda_decode::Bf16DenseMode;
    static constexpr float epsilon = 1.0e-6F;

public:
    MiniCudaStepLowering(const Weights& weights, MiniCudaStepStorage storage, const exec::StepBlock* descriptor,
                        const Bind& bind, const Dispatch& dispatch, const Finish& finish)
        : weights_(weights), storage_(storage), descriptor_(descriptor), bind_(bind), dispatch_(dispatch),
          finish_(finish) {}

    bool try_step_graph(MiniResidentChunk chunk, auto& work) {
        return dispatch_(chunk, [&] {
            recipe_.stage("flags");
            recipe_.clear({storage_.flags.data(), storage_.flags.size_bytes(), 0});
            work();
            return std::move(recipe_);
        });
    }
    void finish() { finish_(); }

    void embed(std::span<float> image) {
        using namespace exec::cuda_decode;
        recipe_.stage("embed");
        add_leaf(Embedding{weights_.bf16("embed.weight").data(), &descriptor_->row[0].token,
                            image.data(), kVocabulary, kHiddenWidth}, true);
        norm(image, "embed_norm.weight", image);
    }

    void head(MiniConstRows hidden, MiniCudaStepRows rows) {
        using namespace exec::cuda_decode;
        recipe_.visit(-1);
        recipe_.stage("head");
        const std::array projection{core::Bf16RowProjection{weights_.bf16("unembed.weight").data(),
                                                          kOutputVocabulary, rows.logits.data()}};
        dense(hidden.data(), kHiddenWidth, projection);
        add_leaf(Bf16Softcap{rows.logits.data(), rows.logits.size(), kLogitSoftcap});
        const GreedyPartial reduction{rows.logits.data(), rows.partials, kOutputVocabulary,
                                     head_argmax::default_chunks(kOutputVocabulary)};
        add_leaf(reduction);
        add_leaf(GreedyFinal{reduction, rows.picked});
        recipe_.copy(exec::CudaNodeKind::copy_d2h_pinned, {rows.reply, rows.picked, sizeof(*rows.reply)});
    }

    template <class T>
    void copy(std::span<const T> input, std::span<T> output) {
        if (input.data() != output.data())
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
            weights_.bf16(stem + "gain_weight").data(), scale, add());
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
        add_leaf(args);
    }

    void capture_message_bank(const std::array<MiniConstRows, 2>& source,
                              const std::array<MiniMutableRows, 2>& normalized) {
        recipe_.stage("message_bank");
        for (std::size_t i = 0; i < source.size(); ++i)
            add_leaf(core::residual_mix::NormalizeArgs{source[i].data(), normalized[i].data(),
                                                      1U, kHiddenWidth, epsilon});
    }

    void add(MiniConstRows lhs, MiniConstRows rhs, MiniMutableRows output, bool round_bf16) {
        recipe_.stage("residual_add");
        add_leaf(core::residual_mix::AddArgs{lhs.data(), rhs.data(), output.data(), output.size(), round_bf16});
    }

    void attention(LayerVisit visit, MiniResidentChunk chunk, MiniConstRows query, MiniConstRows key,
                   MiniConstRows value, const MiniResidentVisitState& state, MiniMutableRows output) {
        using namespace exec::cuda_decode;
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
        const auto* position = &descriptor_->row[0].position;
        const Attention lowering{*this};
        require(attn_core::enqueue_prepared_gqa<Shape>(lowering, full, chunk.first_position, 1U,
                                                       inputs, weights, work, prep, nullptr, position));
        add_leaf(AppendKvBf16{prep.key, prep.value, state.kv.keys, state.kv.values, position,
                            kQueryHeads, kKvHeads, kHeadWidth, state.kv.slots, chunk.first_position,
                            Shape::kScoreScale, state.kv.wrapped});
        add_leaf(PreparedFlashBf16{prep.query, prep.relative_logits, prep.sinks, state.kv.keys, state.kv.values,
                                  attended, position, kQueryHeads, kKvHeads, state.kv.slots, chunk.first_position,
                                  kRelativeExtent, full ? 0U : kSlidingWindow, Shape::kScoreScale, state.kv.wrapped});
        add_leaf(PreparedFinite{{prep.query, prep.key, prep.value, prep.relative_logits, prep.sinks,
                                attended, chunk.first_position, 1U}, kQueryHeads, kKvHeads, kHeadWidth,
                               kRelativeExtent, storage_.flags.data() + visit.logical_layer});
        require(attn_core::enqueue_prepared_gqa_output<Shape>(lowering, 1U,
            weights.projection(MiniAttentionProjection::output), attended, prep.gate_logits, gated, output.data()));
    }

    void convolve(LayerVisit visit, MiniConvolutionStream path, MiniConstRows input, MiniMutableRows history,
                  MiniMutableRows output) {
        recipe_.stage("post_convolution");
        const auto* weights = mini_post_convolution_weights(storage_.lag_weights, visit, path == MiniConvolutionStream::moe);
        require(attn_core::enqueue_prepared_gqa_post_convolution<Shape>(
            Attention{*this}, 1U, weights, input.data(), history.data(), output.data()));
    }

    void moe(LayerVisit visit, MiniConstRows input, MiniMutableRows output) {
        using namespace exec::cuda_decode;
        recipe_.stage("moe");
        const auto layer = visit.physical_layer;
        const std::array router{core::Bf16RowProjection{weights_.bf16(parameter(layer, "mlp.gate.weight")).data(),
                                                       kExperts, storage_.route_logits}};
        lower_dense(input.data(), kHiddenWidth, router, true, add());
        auto* failure = storage_.flags.data() + kLogicalVisits + visit.logical_layer;
        add_leaf(CheckedWaveRoute{storage_.route_logits, weights_.f32(parameter(layer, "mlp.gate.bias")).data(),
                                 storage_.route_ids, storage_.route_weights, failure});
        static_assert(kExpertsPerToken == kPackedExpertsPerToken);
        const moe_core::microscaled_r1::Args args{
            storage_.matrices.data() + std::size_t{layer} * kExperts, storage_.route_ids, storage_.route_weights,
            failure, input.data(), storage_.expert_activation, output.data(), kExperts, kHiddenWidth, kExpertWidth,
            kMlpGateSoftcap, kMlpUpSoftcap,
        };
        const auto format = mini_expert_format(layer);
        add_leaf(PackedGateUp{args, format});
        add_leaf(PackedDownCombine{args, format});
    }

private:
    static std::string parameter(unsigned layer, std::string_view suffix) {
        return "layers." + std::to_string(layer) + "." + std::string(suffix);
    }

    static void require(bool valid) {
        if (!valid) throw std::invalid_argument("invalid Inkling CUDA step program binding");
    }

    void rms(const float* input, const std::uint16_t* scale, float* output) {
        add_leaf(exec::cuda_decode::RmsBf16{input, scale, output, 1U, kHiddenWidth, epsilon}, true);
    }

    void add_leaf(const auto& args, bool pdl = false) { recipe_.kernel(bind_(args), pdl); }
    auto add() { return [this](const auto& args, bool pdl) { add_leaf(args, pdl); }; }

    void dense(const float* input, unsigned width, std::span<const core::Bf16RowProjection> projections) {
        exec::cuda_decode::lower_dense(input, width, projections, false, add());
    }

    struct Attention {
        using Status = bool;
        static constexpr bool success = true, invalid = false;
        MiniCudaStepLowering& lower;

        bool linear(const float* input, unsigned width, unsigned, std::span<const MiniAttentionLinear> views) const {
            return core::for_each_bf16_projection_group<4>(views, success, [&](auto group) {
                lower.dense(input, width, group);
                return success;
            });
        }
        template <bool Round = false>
        bool convolve(const core::ShortConvolution& plan, unsigned, const float* input, const float* weights,
                      const float* dynamic, float* history, float* output) const {
            lower.add_leaf(exec::cuda_decode::CausalConvolution{plan, input, weights, dynamic, history, output, Round});
            return success;
        }
        bool clip(float* row, const std::uint16_t* gain, unsigned, unsigned heads, unsigned width, float threshold,
                  bool position_scale, unsigned first, const std::uint32_t* position = nullptr) const {
            lower.add_leaf(exec::cuda_decode::HeadClipBf16{row, gain, row, position, heads, width,
                position_scale ? first : 0U, position_scale ? Shape::kPositionFloor : 0U,
                Shape::kAttentionEpsilon, threshold, position_scale ? Shape::kPositionAlpha : 0.0F});
            return success;
        }
        bool round(float* row, std::size_t elements, unsigned* = nullptr) const {
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

    const Weights& weights_;
    MiniCudaStepStorage storage_;
    const exec::StepBlock* descriptor_;
    const Bind& bind_;
    const Dispatch& dispatch_;
    const Finish& finish_;
    exec::CudaProgramRecipe recipe_;
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
    void reset_gate_session();
    void prefill_gate_rows(std::span<const unsigned> ids, std::span<float> logits);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<serve::SerialGenerationDecoder> load_mini_cuda_program(const serve::SerialGenerationLoad&);

#if defined(STRIX_INKLING2_STEP_PROGRAM)
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
        const auto bind = [](const auto& args) {
            using namespace exec::cuda_decode;
            if constexpr (std::is_same_v<std::remove_cvref_t<decltype(args)>, CheckedWaveRoute>)
                return bind_checked_wave<MiniRoutePolicy, detail::MiniMoeMath, moe::RouteWaveTraits<true>>(
                    args, "cuda-sm120-checked-wave-mini-route-r1-v1");
            else
                return exec::cuda_decode::bind(args);
        };
        exec::CudaTicket ticket{};
        const auto dispatch = [&](MiniResidentChunk step, const auto& build) {
            const bool install = !cached.has_value();
            if (install) cached.emplace(build().compile(key));
            exec::StepBlock input{};
            input.h.rows = input.h.valid_mask = 1U;
            input.h.step = step.first_position;
            input.row[0] = {token, step.first_position, 0U, exec::kStepNoParent, 1U, exec::step_valid, 0U, 0U};
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
        MiniCudaStepLowering lowering(decoder.step_program_weights(), storage, executor.step_descriptor(),
                                     bind, dispatch, finish);
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

    MiniResidentDecoder& decoder;
    // Prepared R1 attention reads position from X3 and keeps its grid fixed.
    // There is one key for this resident packed, untapped decode route.
    const exec::ProgramKey key{};
    std::unique_ptr<float, detail::MiniPinnedLogitsDeleter> staged_logits;
    std::optional<exec::CudaRecipeProgram> cached;
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
inline void MiniCudaProgramDecoder::reset_gate_session() { impl_->decoder.reset_gate_session(); }
inline void MiniCudaProgramDecoder::prefill_gate_rows(std::span<const unsigned> ids, std::span<float> logits) {
    impl_->decoder.prefill_gate_rows(ids, logits);
}

inline std::unique_ptr<serve::SerialGenerationDecoder> load_mini_cuda_program(const serve::SerialGenerationLoad& load) {
    struct Decoder final : serve::SerialGenerationDecoder {
        explicit Decoder(const serve::SerialGenerationLoad& input)
            : resident(input.checkpoint, input.positions, input.prefill_chunk, input.packed_decode, false), program(resident) {}
        unsigned prefill(std::span<const unsigned> ids, std::span<float> logits) override {
            return program.prefill(ids, logits);
        }
        unsigned decode(unsigned token, std::span<float> logits) override { return program.decode(token, logits); }
        MiniResidentDecoder resident;
        MiniCudaProgramDecoder program;
    };
    return std::make_unique<Decoder>(load);
}
#endif

}  // namespace strix::inkling2
