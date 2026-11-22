#pragma once

#include "core/row_transforms.hpp"
#include "strix/device/runtime.hpp"
#include "strix/env_flags.hpp"
#include "inkling2_mini_controller_device.hpp"
#include "inkling2_mini_residual_device.hpp"
#include "moe_core/microscaled_r1.hpp"

#include <memory>
#include <optional>
#include <string_view>

#if defined(__HIPCC__)
#include "inkling2_mini_attention_hip.hpp"
#include "inkling2_mini_attention_prepare_hip.hpp"
#include "inkling2_mini_moe_hip.hpp"
#else
#include "inkling2_mini_attention_cuda.hpp"
#include "inkling2_mini_attention_prepare_cuda.hpp"
#include "inkling2_mini_moe_cuda.hpp"
#endif

namespace strix::inkling2 {

namespace mini_gpu = device::native;

class MiniResidentBackend {
public:
#if defined(__HIPCC__)
    using Projection = MiniHipDenseProjection;
    using Moe = MiniHipMoeBackend;
#else
    using Projection = MiniCudaDenseProjection;
    using Moe = MiniCudaMoeBackend;
#endif
    using Tile = moe_core::device_grouped::TileRows;
    using Planner = moe_core::device_grouped::Planner<kExperts, kExpertsPerToken>;

    static void require_device() {
        const auto device = mini_gpu::current_device();
#if defined(__HIPCC__)
        hipDeviceProp_t properties{};
        mini_gpu::check(hipGetDeviceProperties(&properties, device), "read HIP device properties");
        const std::string_view target(properties.gcnArchName);
        const auto arch = target.substr(0, target.find(':'));
        if (arch != "gfx1151" && arch != "gfx1150" && arch != "gfx1100" &&
            arch != "gfx1101" && arch != "gfx1102")
            throw std::runtime_error("Inkling 2 Mini: selected HIP device lacks the admitted BF16 WMMA path");
#else
        int major{};
        mini_gpu::check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device), "read CUDA architecture");
        if (major < 9)
            throw std::runtime_error("Inkling 2 Mini: shared CUDA runtime requires SM90 or newer");
#endif
    }

    static unsigned checked_chunk_rows(unsigned rows) {
        const auto tiles = moe_core::device_grouped::tile_bound(rows, kExpertsPerToken, kExperts, Moe::rows_per_tile);
        if (rows == 0U || rows > 65535U || tiles > moe_core::grouped_microscaled::kMaxTiles)
            throw std::invalid_argument("Inkling 2 Mini: unsupported device chunk size");
        return rows;
    }

    static constexpr bool default_step_graphs =
#if defined(__HIPCC__)
        false;
#else
        true;
#endif

    MiniResidentBackend(const MiniResidentWeights& weights, unsigned max_rows, mini_gpu::Stream stream,
                        bool packed_decode = false, bool step_graphs = default_step_graphs)
        : weights_(weights), stream_(stream), max_rows_(checked_chunk_rows(max_rows)), packed_decode_(packed_decode),
          step_graphs_(step_graphs),
          tiles_(moe_core::device_grouped::tile_bound(max_rows_, kExpertsPerToken, kExperts, Moe::rows_per_tile)),
          controller_(mini_controller_elements(max_rows_)), attention_(mini_resident_attention_elements(max_rows_)),
#if defined(__HIPCC__)
          attention_partials_(attn_core::dense::key_split_workspace_floats(max_rows_, kQueryHeads, kHeadWidth)),
#endif
          lag_weights_(kPhysicalLayers * kMiniConvLayerElements),
          matrices_(kPhysicalLayers * kExperts), route_data_(route_elements(max_rows_, tiles_)),
          route_floats_(std::size_t{max_rows_} * (kExperts + 2U * kExpertsPerToken)),
          route_scratch_(Planner::scratch_bytes(max_rows_) / sizeof(unsigned)),
          descriptors_(2U * tiles_), moe_bf16_(mini_moe_workspace_elements(max_rows_)),
          moe_fp32_(mini_moe_workspace_elements(max_rows_)),
#if defined(__HIPCC__)
          moe_activation_(mini_moe_activation_elements(max_rows_)),
#endif
          flags_(2U * kLogicalVisits),
          zero_padding_(kHeadWidth) {
#if !defined(__HIPCC__)
        if (const char* enable = strix::envcfg::text("STRIX_INKLING_K1BIND"); enable && std::string_view(enable) == "1") {
            int units = 0;
            check(cudaDeviceGetAttribute(&units, cudaDevAttrMultiProcessorCount, mini_gpu::current_device()));
            if (units <= 0) throw std::runtime_error("Inkling 2 Mini: invalid CUDA SM count");
            rows_profile_.emplace();
            rows_profile_->units = static_cast<unsigned>(units);
        }
        if (max_rows_ > cuda_core::kMaxRows) {
            big_context_.reset(cuda_core::dense_big_ctx_create());
            if (!big_context_) throw std::runtime_error("Inkling 2 Mini: could not allocate CUDA dense context");
            check(cuda_core::dense_big_reserve(max_rows_, 2U * kHiddenWidth, 0U, 3, big_context_.get()));
        }
#endif
        try {
            for (unsigned layer = 0; layer < kPhysicalLayers; ++layer) {
                const auto matrices = weights_.experts(layer);
                mini_gpu::copy(matrices_.data() + layer * kExperts, matrices.data(), matrices.size_bytes(),
                              mini_gpu::host_to_device, stream_);
            }
            check(bind_mini_convolution_weights(weights_, lag_weights_.data(), stream_));
            mini_gpu::clear(zero_padding_.data(), zero_padding_.span().size_bytes(), stream_);
            clear_flags();
            mini_gpu::synchronize(stream_);
        } catch (...) {
            // A previously enqueued copy may still borrow descriptor storage.
            try { mini_gpu::synchronize(stream_); } catch (...) {}
            throw;
        }
    }

    [[nodiscard]] mini_gpu::Stream stream() const { return stream_; }
#if !defined(__HIPCC__)
    struct StepProgramBuffers {
        unsigned max_rows;
        bool packed_decode;
        std::span<float> controller, attention;
        const float* lag_weights;
        std::span<const MiniMoeExpert> matrices;
        unsigned* route_ids;
        float* route_logits;
        float* route_weights;
        std::uint16_t* expert_activation;
        std::span<unsigned> flags;
    };

    [[nodiscard]] StepProgramBuffers step_program_buffers() noexcept {
        return {max_rows_, packed_decode_, controller_.span(), attention_.span(), lag_weights_.data(),
                matrices_.span(), route_data_.data(), route_floats_.data(),
                route_floats_.data() + std::size_t{max_rows_} * kExperts, moe_bf16_.data(), flags_.span()};
    }
#endif
    void check(mini_gpu::Status status) const { mini_gpu::check(status, "Inkling 2 Mini stage"); }
    void set_gate_route_tap(strix::moe::RouteTap* tap) {
        graph_.invalidate();
        gate_route_tap_ = tap;
    }

    void set_step_token(std::optional<unsigned> token) { step_token_ = token; }
    void reset_step_graph() noexcept { graph_.reset(); }
    [[nodiscard]] const session::StepDescriptor* capture_descriptor() const noexcept {
        return graph_.capture_descriptor();
    }

    template <class Work>
    bool try_step_graph(MiniResidentChunk chunk, Work& work) {
        // Layout and workspace addresses are fixed for this backend's lifetime.
        // A route tap performs host-synchronous reads and writes.
        if (graph_.try_launch(stream_, step_graphs_, chunk.rows, step_token_.has_value(), !gate_route_tap_,
                             {chunk.first_position, step_token_.value_or(0U)}, [&] {
            clear_flags();
            work();
        })) return true;
        clear_flags();
        return false;
    }

    bool abandon_capture() noexcept {
        graph_.invalidate();
        // try_launch closes even a failed capture before it returns or throws.
        return false;
    }

    void finish() {
        std::array<unsigned, 2U * kLogicalVisits> flags{};
        mini_gpu::copy(flags.data(), flags_.data(), sizeof(flags), mini_gpu::device_to_host, stream_);
        mini_gpu::synchronize(stream_);
        if (std::ranges::any_of(flags, [](unsigned flag) { return flag != 0U; }))
            throw std::runtime_error("Inkling 2 Mini: device rejected a route or an attention row");
    }

    template <class T>
    void copy(std::span<const T> input, std::span<T> output) {
        if (input.size() != output.size()) throw std::invalid_argument("Inkling 2 Mini: incompatible copy rows");
        if (input.data() == output.data()) return;
        mini_gpu::copy(output.data(), input.data(), input.size_bytes(), mini_gpu::device_to_device, stream_);
    }

    void dense(const float* input, unsigned width, unsigned rows, std::span<const Projection> projections) {
#if defined(__HIPCC__)
        check(enqueue_mini_hip_dense_rows(input, width, rows, projections, stream_));
#else
        check(enqueue_mini_cuda_dense_rows(input, width, rows, projections, stream_,
                                          cuda_core::DenseArith::mma3, big_context_.get()));
#endif
    }

#if defined(__HIPCC__)
    // Low-rank controller gate pair with the controller as the shared tile's epilogue; false = not admitted.
    bool dense_controller(const float* input, unsigned width, unsigned rows, std::span<const Projection> gates,
                          const core::residual_mix::ControllerArgs& control) {
        const auto status = core::fma_rows::enqueue_bf16_controller_rows(input, width, rows, gates, control.current,
                                                                         control.message, control.output, stream_);
        if (status == hipErrorNotSupported) return false;
        check(status);
        return true;
    }
#endif

    void rms(const float* input, const std::uint16_t* scale, float* output, unsigned rows) {
#if defined(__HIPCC__)
        check(enqueue_mini_hip_rms({input, scale, output, rows, kHiddenWidth, kHiddenWidth, kHiddenWidth}, stream_));
#else
        check(enqueue_mini_cuda_rmsnorm(input, scale, output, rows, stream_));
#endif
    }

    void norm(MiniConstRows input, std::string_view scale_name, MiniMutableRows output) {
        rms(input.data(), weights_.bf16(scale_name).data(), output.data(), rows_of(input));
    }

    void layer_input(LayerVisit visit, MessagePath path, MiniConstRows input,
                     std::span<const MiniConstRows> previous_bank, MiniMutableRows output) {
        enqueue_mini_layer_input(weights_, visit, path, rows_of(input), input, previous_bank,
                                 output, controller_.span(), *this);
    }

    void residual_read(std::span<const MiniConstRows> blocks, MiniConstRows current, std::string_view name,
                       std::span<const MiniMutableRows> output) {
        check(launch_mini_residual_read(weights_, blocks, current, name, output, stream_));
    }

    void capture_message_bank(const std::array<MiniConstRows, 2>& bank,
                              const std::array<MiniMutableRows, 2>& normalized) {
        check(launch_mini_capture_message_bank(bank, normalized, stream_));
    }

    void add(MiniConstRows lhs, MiniConstRows rhs, MiniMutableRows output, bool round_bf16) {
        check(launch_mini_residual_add(lhs, rhs, output, round_bf16, stream_));
    }

    void attention(LayerVisit visit, MiniResidentChunk chunk, MiniConstRows query,
                   MiniConstRows key, MiniConstRows value, const MiniResidentVisitState& state, MiniMutableRows output) {
        if (chunk.rows != rows_of(query) || key.size() != query.size() ||
            value.size() != query.size() || output.size() != query.size())
            throw std::invalid_argument("Inkling 2 Mini: incompatible attention rows");
        auto arena = attention_.span();
        const auto take = [&](unsigned width) {
            const auto elems = std::size_t{max_rows_} * width;
            auto* ptr = arena.data();
            arena = arena.subspan(elems);
            return ptr;
        };
        const MiniProjectionInputs inputs{query.data(), key.data(), value.data()};
        const MiniProjectionWorkspace work{
            take(kKvWidth), take(kKvWidth), take(kKvHeads * kConvTaps), take(kKvHeads * kConvTaps),
            take(kQueryHeads * kRelativeWidth),
            state.history[0].data(), state.history[1].data(),
        };
        const MiniPreparedAttention prep{
            take(kQueryWidth), take(kKvWidth), take(kKvWidth), take(kQueryHeads * kRelativeExtent),
            take(kQueryHeads), take(kQueryWidth),
        };
        auto* attended = take(kQueryWidth);
        auto* gated = take(kQueryWidth);
        const auto weights = mini_device_projection_weights(weights_, visit, mini_convolution_layer(lag_weights_.data(), visit));
        const auto* step = capture_descriptor();
        const auto* device_position = step ? &step->position : nullptr;
#if defined(__HIPCC__)
        const auto splits = attn_core::dense::small_row_key_splits(
            chunk.first_position, chunk.rows, mini_prepared_gqa_window(visit));
        const std::span<float> partials = splits > 1U || step ? attention_partials_.span() : std::span<float>{};
        check(enqueue_mini_hip_attention_forward(visit, chunk.first_position, chunk.rows, inputs, weights, work, prep,
            {attended, gated, output.data()},
            {state.kv.keys, state.kv.values, zero_padding_.data(), state.kv.slots, state.kv.wrapped},
            flags_.data() + visit.logical_layer, stream_, partials, splits, device_position));
#else
        check(enqueue_mini_cuda_attention_prepare(visit, chunk.first_position, chunk.rows, inputs, weights, work, prep,
                                                 stream_, cuda_core::DenseArith::mma3, big_context_.get(),
                                                 device_position, rows_profile_ ? &*rows_profile_ : nullptr));
        check(enqueue_mini_cuda_attention(visit,
            {prep.query, prep.key, prep.value, prep.relative_logits, prep.sinks, attended, chunk.first_position, chunk.rows},
            state.kv, stream_, cuda_core::FlashPipe::pad2, flags_.data() + visit.logical_layer, device_position));
        check(enqueue_mini_cuda_attention_output(chunk.rows, weights.projection(MiniAttentionProjection::output),
            attended, prep.gate_logits, gated, output.data(), stream_, cuda_core::DenseArith::mma3, big_context_.get()));
#endif
    }

    void convolve(LayerVisit visit, MiniConvolutionStream path, MiniConstRows input,
                  MiniMutableRows history, MiniMutableRows output) {
        if ((path != MiniConvolutionStream::attention && path != MiniConvolutionStream::moe) ||
            output.size() != input.size() || history.size() != (kConvTaps - 1U) * kHiddenWidth)
            throw std::invalid_argument("Inkling 2 Mini: incompatible residual convolution rows");
        const auto* weights = mini_post_convolution_weights(lag_weights_.data(), visit, path == MiniConvolutionStream::moe);
        const auto rows = rows_of(input);
#if defined(__HIPCC__)
        check(enqueue_mini_hip_post_convolution(rows, weights, input.data(),
                                                history.data(), output.data(), stream_));
#else
        check(enqueue_mini_cuda_post_convolution(rows, weights, input.data(),
                                                 history.data(), output.data(), stream_));
#endif
    }

    void moe(LayerVisit visit, MiniConstRows input, MiniMutableRows output) {
        namespace route = moe_core::device_grouped;
        const auto rows = rows_of(input);
        const auto count = route::tile_bound(rows, kExpertsPerToken, kExperts, Moe::rows_per_tile);
        const auto assignments = std::size_t{max_rows_} * kExpertsPerToken;
        auto bits = route_data_.span();
        const auto take = [&](std::size_t size) {
            const auto ptr = bits.data();
            bits = bits.subspan(size);
            return ptr;
        };
        auto* ids = take(assignments);
        auto* status = take(max_rows_);
        route::CsrTables csr{
            take(kExperts + 1U), take(kExperts), take(tiles_), take(tiles_ + 1U),
            take(assignments), take(assignments),
        };
        auto* combine = take(assignments);
        auto* tile_count = take(1U);
        auto values = route_floats_.span();
        auto logits = values.first(std::size_t{rows} * kExperts);
        auto route_weights = values.subspan(std::size_t{max_rows_} * kExperts, assignments);
        route::Output planned{
            values.last(assignments).data(), combine, descriptors_.data(), descriptors_.data() + tiles_,
            tile_count, flags_.data() + kLogicalVisits + visit.logical_layer, count,
        };
        const auto layer = visit.physical_layer;
        const auto router = weights_.bf16(mini_layer_parameter(layer, "mlp.gate.weight"));
        const auto bias = weights_.f32(mini_layer_parameter(layer, "mlp.gate.bias"));
#if defined(__HIPCC__)
        const std::array projection{Projection{router.data(), kExperts, logits.data()}};
        dense(input.data(), kHiddenWidth, rows, projection);
#else
        check(enqueue_mini_cuda_router_logits({router.data(), cuda_core::WeightFormat::bf16,
                                                    kExperts, kHiddenWidth}, rows, input, logits, stream_));
#endif
#if defined(__HIPCC__)
        const MiniMoeWorkspace workspace{moe_bf16_.span(), moe_fp32_.span(), moe_activation_.span()};
#else
        const MiniMoeWorkspace workspace{moe_bf16_.span(), moe_fp32_.span()};
#endif
        const MiniMoeResidentRouteBuffers buffers{{ids, assignments}, {status, max_rows_},
                                                  route_scratch_.span(), route_weights};
        const auto matrices = matrices_.span().subspan(std::size_t{layer} * kExperts, kExperts);
        if (rows == 1U && packed_decode_) {
            enqueue_packed_decode(layer, input, logits, bias, output, buffers, matrices, planned.failure);
            return;
        }
#if defined(__HIPCC__)
        check(enqueue_mini_hip_resident_moe(layer, rows, input, logits, bias, output, buffers,
                                            matrices, csr, planned, workspace, stream_, gate_route_tap_));
#else
        check(enqueue_mini_cuda_resident_moe(layer, rows, input, logits, bias, output, buffers,
                                             matrices, csr, planned, workspace, stream_, gate_route_tap_));
#endif
    }

private:
    void enqueue_packed_decode(unsigned layer, MiniConstRows input, MiniConstRows logits, MiniConstRows bias,
                               MiniMutableRows output, MiniMoeResidentRouteBuffers route,
                               std::span<const MiniMoeExpert> matrices, unsigned* failure) {
        namespace decode = moe_core::microscaled_r1;
#if defined(__HIPCC__)
        check(moe_core::enqueue_route_wave<MiniRoutePolicy, detail::MiniMoeMath, moe::RouteWaveTraits<true, true>>(
            logits.data(), bias.data(), route.ids.data(), route.weights.data(), failure, 1U, stream_));
        moe_core::route_tap(layer, 1U, kExpertsPerToken, kExperts, route.ids.data(),
                            route.weights.data(), stream_, gate_route_tap_);
#else
        check(moe_core::enqueue_route_wave<MiniRoutePolicy, detail::MiniMoeMath, moe::RouteWaveTraits<true>>(
            logits.data(), bias.data(), route.ids.data(), route.weights.data(), failure, 1U, stream_));
        cuda::core::route_tap(layer, 1U, kExpertsPerToken, kExperts, route.ids.data(),
                              route.weights.data(), stream_, gate_route_tap_);
#endif
        const decode::Args args{
            matrices.data(), route.ids.data(), route.weights.data(), failure, input.data(), moe_bf16_.data(),
            output.data(), kExperts, kHiddenWidth, kExpertWidth, kMlpGateSoftcap, kMlpUpSoftcap,
        };
        using weight::microscaled::MatrixFormat;
        if (mini_expert_format(layer) == MatrixFormat::nvfp4)
            check(decode::launch<MatrixFormat::nvfp4, kExpertsPerToken>(args, stream_));
        else
            check(decode::launch<MatrixFormat::mxfp8, kExpertsPerToken>(args, stream_));
    }

    [[nodiscard]] unsigned rows_of(MiniConstRows input) const {
        const auto rows = input.size() / kHiddenWidth;
        if (rows == 0U || rows > max_rows_ || input.size() % kHiddenWidth != 0U)
            throw std::invalid_argument("Inkling 2 Mini: rows exceed the resident workspace");
        return static_cast<unsigned>(rows);
    }

    static constexpr std::size_t route_elements(unsigned rows, unsigned tiles) {
        return std::size_t{rows} * (4U * kExpertsPerToken + 1U) + 2U * kExperts + 2U * tiles + 3U;
    }

    const MiniResidentWeights& weights_;
    mini_gpu::Stream stream_;
    unsigned max_rows_;
    bool packed_decode_;
    bool step_graphs_;
    void clear_flags() {
        mini_gpu::clear(flags_.data(), flags_.span().size_bytes(), stream_);
    }

    std::optional<unsigned> step_token_;
    strix::moe::RouteTap* gate_route_tap_{};
    unsigned tiles_;
    mini_gpu::Buffer<float> controller_, attention_;
#if defined(__HIPCC__)
    mini_gpu::Buffer<float> attention_partials_;
#endif
    mini_gpu::Buffer<float> lag_weights_;
    mini_gpu::Buffer<MiniMoeExpert> matrices_;
    mini_gpu::Buffer<unsigned> route_data_;
    mini_gpu::Buffer<float> route_floats_;
    mini_gpu::Buffer<unsigned> route_scratch_;
    mini_gpu::Buffer<Tile> descriptors_;
    mini_gpu::Buffer<std::uint16_t> moe_bf16_;
    mini_gpu::Buffer<float> moe_fp32_;
#if defined(__HIPCC__)
    mini_gpu::Buffer<std::uint16_t> moe_activation_;  // gate-up epilogue activations (grouped_dispatch)
#endif
    mini_gpu::Buffer<unsigned> flags_;
    mini_gpu::Buffer<std::uint16_t> zero_padding_;
#if !defined(__HIPCC__)
    std::optional<device::Profile> rows_profile_;
    std::unique_ptr<cuda_core::BigCtx, decltype(&cuda_core::dense_big_ctx_destroy)>
        big_context_{nullptr, cuda_core::dense_big_ctx_destroy};
#endif
    mini_gpu::StepGraph graph_;
};

}  // namespace strix::inkling2
