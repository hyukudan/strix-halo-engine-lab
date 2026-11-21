#pragma once

#include "core/prepared_rows_device.hpp"
#include "core/resident_device.hpp"
#include "inkling2_mini_controller_device.hpp"
#include "inkling2_mini_residual_device.hpp"
#include "moe_core/microscaled_r1.hpp"

#include <memory>

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

namespace mini_gpu = core::resident_device;

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

    MiniResidentBackend(const MiniResidentWeights& weights, unsigned max_rows, mini_gpu::Stream stream,
                        bool packed_decode = false)
        : weights_(weights), stream_(stream), max_rows_(checked_chunk_rows(max_rows)), packed_decode_(packed_decode),
          tiles_(moe_core::device_grouped::tile_bound(max_rows_, kExpertsPerToken, kExperts, Moe::rows_per_tile)),
          controller_(mini_controller_elements(max_rows_)), attention_(attention_elements(max_rows_)),
          lag_weights_(kHiddenWidth * kConvTaps),
          matrices_(kPhysicalLayers * kExperts), route_data_(route_elements(max_rows_, tiles_)),
          route_floats_(std::size_t{max_rows_} * (kExperts + 2U * kExpertsPerToken)),
          route_scratch_(Planner::scratch_bytes(max_rows_) / sizeof(unsigned)),
          descriptors_(2U * tiles_), moe_bf16_(mini_moe_workspace_elements(max_rows_)),
          moe_fp32_(mini_moe_workspace_elements(max_rows_)), flags_(2U * kLogicalVisits),
          zero_padding_(kHeadWidth) {
#if !defined(__HIPCC__)
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
            mini_gpu::clear(zero_padding_.data(), zero_padding_.span().size_bytes(), stream_);
            start_chunk();
            mini_gpu::synchronize(stream_);
        } catch (...) {
            // A previously enqueued copy may still borrow descriptor storage.
            try { mini_gpu::synchronize(stream_); } catch (...) {}
            throw;
        }
    }

    [[nodiscard]] mini_gpu::Stream stream() const { return stream_; }
    void check(mini_gpu::Status status) const { mini_gpu::check(status, "Inkling 2 Mini stage"); }

    void start_chunk() { mini_gpu::clear(flags_.data(), flags_.span().size_bytes(), stream_); }

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
            take(kQueryHeads * kRelativeWidth), lag_weights_.data(),
            state.history[0].data(), state.history[1].data(),
        };
        const MiniPreparedAttention prep{
            take(kQueryWidth), take(kKvWidth), take(kKvWidth), take(kQueryHeads * kRelativeExtent),
            take(kQueryHeads), take(kQueryWidth),
        };
        auto* attended = take(kQueryWidth);
        auto* gated = take(kQueryWidth);
        const auto weights = mini_device_projection_weights(weights_, visit);
#if defined(__HIPCC__)
        check(enqueue_mini_hip_attention_prepare(visit, chunk.first_position, chunk.rows, inputs, weights, work, prep, stream_));
        check(enqueue_mini_hip_attention(visit,
            {prep.query, prep.key, prep.value, prep.relative_logits, prep.sinks, attended, chunk.first_position, chunk.rows},
            {state.kv.keys, state.kv.values, zero_padding_.data(), state.kv.slots, state.kv.wrapped},
            flags_.data() + visit.logical_layer, stream_));
        check(enqueue_mini_hip_attention_output(chunk.rows, weights.projection(MiniAttentionProjection::output),
            attended, prep.gate_logits, gated, output.data(), stream_));
#else
        check(enqueue_mini_cuda_attention_prepare(visit, chunk.first_position, chunk.rows, inputs, weights, work, prep,
                                                 stream_, cuda_core::DenseArith::mma3, big_context_.get()));
        check(enqueue_mini_cuda_attention(visit,
            {prep.query, prep.key, prep.value, prep.relative_logits, prep.sinks, attended, chunk.first_position, chunk.rows},
            state.kv, stream_));
        check(enqueue_mini_cuda_attention_output(chunk.rows, weights.projection(MiniAttentionProjection::output),
            attended, prep.gate_logits, gated, output.data(), stream_, cuda_core::DenseArith::mma3, big_context_.get()));
#endif
    }

    void convolve(LayerVisit visit, MiniConvolutionStream path, MiniConstRows input,
                  MiniMutableRows history, MiniMutableRows output) {
        if ((path != MiniConvolutionStream::attention && path != MiniConvolutionStream::moe) ||
            output.size() != input.size() || history.size() != (kConvTaps - 1U) * kHiddenWidth)
            throw std::invalid_argument("Inkling 2 Mini: incompatible residual convolution rows");
        const auto suffix = path == MiniConvolutionStream::moe ? "mlp_sconv.weight" : "attn_sconv.weight";
        const auto weights = weights_.bf16(mini_layer_parameter(visit.physical_layer, suffix));
        const auto rows = rows_of(input);
#if defined(__HIPCC__)
        check(enqueue_mini_hip_post_convolution(rows, weights.data(), input.data(), lag_weights_.data(),
                                                history.data(), output.data(), stream_));
#else
        check(enqueue_mini_cuda_post_convolution(rows, weights.data(), input.data(), lag_weights_.data(),
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
        const MiniMoeWorkspace workspace{moe_bf16_.span(), moe_fp32_.span()};
        const MiniMoeResidentRouteBuffers buffers{{ids, assignments}, {status, max_rows_},
                                                  route_scratch_.span(), route_weights};
        const auto matrices = matrices_.span().subspan(std::size_t{layer} * kExperts, kExperts);
        if (rows == 1U && packed_decode_) {
            enqueue_packed_decode(layer, input, logits, bias, output, buffers, matrices, planned.failure);
            return;
        }
#if defined(__HIPCC__)
        check(enqueue_mini_hip_resident_moe(layer, rows, input, logits, bias, output, buffers,
                                            matrices, csr, planned, workspace, stream_));
#else
        check(enqueue_mini_cuda_resident_moe(layer, rows, input, logits, bias, output, buffers,
                                             matrices, csr, planned, workspace, stream_));
#endif
    }

private:
    void enqueue_packed_decode(unsigned layer, MiniConstRows input, MiniConstRows logits, MiniConstRows bias,
                               MiniMutableRows output, MiniMoeResidentRouteBuffers route,
                               std::span<const MiniMoeExpert> matrices, unsigned* failure) {
        namespace decode = moe_core::microscaled_r1;
#if defined(__HIPCC__)
        check(moe_core::enqueue_route_wave<kMiniRoute, detail::MiniMoeMath, moe::RouteWaveTraits<true, true>>(
            logits.data(), bias.data(), route.ids.data(), route.weights.data(), failure, 1U, stream_));
#else
        check(cuda::core::route_wave::launch<mini_cuda_moe_detail::Router>(
            logits.data(), bias.data(), route.ids.data(), route.weights.data(), failure, 1U, stream_));
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

    static constexpr std::size_t attention_elements(unsigned rows) {
        return std::size_t{rows} * (4U * kKvWidth + 2U * kKvHeads * kConvTaps +
            kQueryHeads * kRelativeWidth + 4U * kQueryWidth + kQueryHeads * kRelativeExtent + kQueryHeads);
    }

    static constexpr std::size_t route_elements(unsigned rows, unsigned tiles) {
        return std::size_t{rows} * (4U * kExpertsPerToken + 1U) + 2U * kExperts + 2U * tiles + 3U;
    }

    const MiniResidentWeights& weights_;
    mini_gpu::Stream stream_;
    unsigned max_rows_;
    bool packed_decode_;
    unsigned tiles_;
    mini_gpu::Buffer<float> controller_, attention_, lag_weights_;
    mini_gpu::Buffer<MiniMoeExpert> matrices_;
    mini_gpu::Buffer<unsigned> route_data_;
    mini_gpu::Buffer<float> route_floats_;
    mini_gpu::Buffer<unsigned> route_scratch_;
    mini_gpu::Buffer<Tile> descriptors_;
    mini_gpu::Buffer<std::uint16_t> moe_bf16_;
    mini_gpu::Buffer<float> moe_fp32_;
    mini_gpu::Buffer<unsigned> flags_;
    mini_gpu::Buffer<std::uint16_t> zero_padding_;
#if !defined(__HIPCC__)
    std::unique_ptr<cuda_core::BigCtx, decltype(&cuda_core::dense_big_ctx_destroy)>
        big_context_{nullptr, cuda_core::dense_big_ctx_destroy};
#endif
};

}  // namespace strix::inkling2
