#pragma once

#include "core/resident_q8_rows.hpp"
#include "core/row_transforms.hpp"
#include "strix/device/runtime.hpp"
#include "strix/env_flags.hpp"
#include "inkling2_mini_calibration.hpp"
#include "inkling2_mini_controller_device.hpp"
#include "inkling2_mini_resident_route.hpp"
#include "inkling2_mini_residual_device.hpp"
#if !defined(__HIPCC__)
#include "inkling2_mini_controller_operands_cuda.hpp"
#endif
#include "inkling2_mini_weight_options.hpp"
#include "moe_core/microscaled_r1.hpp"

#include <algorithm>
#include <array>
#include <deque>
#include <cstdio>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#if defined(__HIPCC__)
#include "inkling2_mini_attention_hip.hpp"
#include "inkling2_mini_attention_prepare_hip.hpp"
#include "inkling2_mini_moe_hip.hpp"
#else
#include "inkling2_mini_attention_cuda.hpp"
#include "inkling2_mini_attention_prepare_cuda.hpp"
#include "inkling2_mini_moe_cuda.hpp"
#endif
#include "strix/exec/step_lanes.hpp"
#include "strix/exec/width_graphs.hpp"

namespace strix::inkling2 {

// Slot directory states of the cohort decode programs (strix/exec/step_lanes.hpp): every slot's K plane,
// V plane and convolution histories, in that order.
inline constexpr std::uint32_t kMiniLaneKeys = 0U, kMiniLaneValues = 1U, kMiniLaneHistory = 2U, kMiniLaneStates = 3U;

// The device lanes of a cohort decode program (step block + slot directory) and the host base addresses of
// lane 0's slot for each directory state: a visit's pointers of lane 0 minus these bases are the plane
// offsets, the same for every slot of the uniform session layout.
struct MiniCohortLanes {
    state::LaneView lanes;
    std::array<std::uintptr_t, kMiniLaneStates> base;

    [[nodiscard]] state::LanePlane plane(std::uint32_t which, const void* address) const {
        const auto at = reinterpret_cast<std::uintptr_t>(address);
        if (which >= kMiniLaneStates || at < base[which]) throw std::logic_error("Inkling 2 Mini: lane plane outside its slot");
        return {which, static_cast<std::uint64_t>(at - base[which])};
    }
};

namespace mini_gpu = device::native;

namespace detail {

#if defined(__HIPCC__)
using MiniNativeGqaBackend = MiniPreparedHipBackend;
#else
using MiniNativeGqaBackend = MiniPreparedCudaBackend;
#endif

// Base is the native preparation backend; on CUDA one-row decode with a rows
// profile it is the projection batch (BF16 groups deferred to the first round).
template <class Base = MiniNativeGqaBackend>
struct MiniResidentGqaBackend : Base {
    using Status = typename Base::Status;
    using Base::linear;
    using Base::success;
    using Base::invalid;
    std::uint64_t* null_applications{};
    core::LowBitArithmetic lowbit_arithmetic = kMiniLowBitArithmetic;

    Status linear(const float* input, unsigned width, unsigned rows,
                  std::span<const core::ResidentRowProjection> views) const {
        return core::dispatch_resident_rows<MiniProjectionWeights::role_count>(input, width, rows, views, success, invalid,
            [&](std::span<const MiniAttentionLinear> group) {
                return Base::linear(input, width, rows, group);
            },
            [&](std::span<const core::ResidentRowProjection> group) {
                return core::enqueue_resident_q8_projections<MiniProjectionWeights::role_count>(
                    input, width, rows, group, this->stream
#if !defined(__HIPCC__)
                    , this->big_context
#endif
                    , lowbit_arithmetic);
            },
            [&](std::span<const MiniAttentionLinear> group) {
                return Base::linear_admitted(input, width, rows, group);
            },
            [&](std::span<const core::ResidentRowProjection> group) {
                return core::resident_q8_projections_admitted<MiniProjectionWeights::role_count>(
                    input, width, rows, group, lowbit_arithmetic);
            });
    }

    Status relative_logits(const float* input, const std::uint16_t* weights, float* output, unsigned rows,
                           unsigned* failure) const {
        if (!null_applications) return Base::relative_logits(input, weights, output, rows, failure);
        const auto status = Base::template relative_logits<true>(input, weights, output, rows, failure);
        if (status == success) *null_applications += rows;
        return status;
    }
};

}  // namespace detail

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

    static weight::RowPitchRule resident_row_pitch_rule() {
        const auto device = mini_gpu::current_device();
#if defined(__HIPCC__)
        hipDeviceProp_t properties{};
        mini_gpu::check(hipGetDeviceProperties(&properties, device), "read HIP device properties");
        const std::string_view target(properties.gcnArchName);
        const auto arch = target.substr(0, target.find(':'));
        if (arch != "gfx1151" && arch != "gfx1150" && arch != "gfx1100" &&
            arch != "gfx1101" && arch != "gfx1102")
            throw std::runtime_error("Inkling 2 Mini: selected HIP device lacks the admitted BF16 WMMA path");
        return weight::row_pitch_rule_for_arch(arch);
#else
        int major{};
        mini_gpu::check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device), "read CUDA architecture");
        if (major < 9)
            throw std::runtime_error("Inkling 2 Mini: shared CUDA runtime requires SM90 or newer");
        return {};
#endif
    }

    static bool admits_chunk_rows(unsigned rows) noexcept {
        const auto tiles = moe_core::device_grouped::tile_bound(rows, kExpertsPerToken, kExperts, Moe::rows_per_tile);
        return rows != 0U && rows <= 65535U && tiles <= moe_core::grouped_microscaled::kMaxTiles;
    }

    static unsigned checked_chunk_rows(unsigned rows) {
        if (!admits_chunk_rows(rows)) throw std::invalid_argument("Inkling 2 Mini: unsupported device chunk size");
        return rows;
    }

    static constexpr bool default_step_graphs =
#if defined(__HIPCC__)
        false;
#else
        true;
#endif

    static constexpr bool kBackendHip =
#if defined(__HIPCC__)
        true;
#else
        false;
#endif

    MiniResidentBackend(const MiniResidentWeights& weights, unsigned max_rows, mini_gpu::Stream stream,
                        bool packed_decode = false, bool step_graphs = default_step_graphs, bool cohort_rows = false,
                        core::LowBitArithmetic lowbit_arithmetic = kMiniLowBitArithmetic)
        : weights_(weights), stream_(stream), max_rows_(checked_chunk_rows(max_rows)), packed_decode_(packed_decode),
          cohort_rows_(mini_packed_cohort_rows(kBackendHip, packed_decode, cohort_rows)),
          lowbit_arithmetic_(lowbit_arithmetic), step_graphs_(step_graphs),
          tiles_(moe_core::device_grouped::tile_bound(max_rows_, kExpertsPerToken, kExperts, Moe::rows_per_tile)),
          controller_(std::max(mini_controller_elements(max_rows_), kMiniStepControllerElements)), attention_(attention_elements(max_rows_)),
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
          // Packed decode keeps one route status per routed row: up to kMiniMaxVerifyRows verify rows, or, with packed
          // cohort rows (the served route lever; every packed CUDA decoder), a cohort decode pass of up to max_rows_ lanes.
          flags_((1U + (packed_decode ? (mini_packed_cohort_rows(kBackendHip, packed_decode, cohort_rows)
                                             ? max_rows_ : std::min(max_rows_, kMiniMaxVerifyRows)) : 1U)) * kLogicalVisits),
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
            // Row-scaled device workspaces (cudaMalloc): an out-of-memory refusal is the typed device allocation
            // failure, as for the Buffer members above (the implicit chunk's pressure fallback keys on it).
            mini_gpu::check_device_allocation(
                cuda_core::dense_big_reserve(max_rows_, 2U * kHiddenWidth, 0U, 3, big_context_.get()),
                "Inkling 2 Mini: reserve CUDA dense rows");
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
            start_chunk(0U, MiniAppendKind::prefill);
            mini_gpu::synchronize(stream_);
        } catch (...) {
            // A previously enqueued copy may still borrow descriptor storage.
            try { mini_gpu::synchronize(stream_); } catch (...) {}
            throw;
        }
    }

    [[nodiscard]] mini_gpu::Stream stream() const { return stream_; }
#if !defined(__HIPCC__)
    // The attention linears' backend (what attention() prepares with): the operand class the staged attention weights'
    // certificate declares, on this backend's stream and dense context. `planes_report` observes each dispatch.
    [[nodiscard]] detail::MiniPreparedCudaBackend attention_operand_backend(int* planes_report = nullptr) const {
        return detail::mini_attention_backend(stream_, big_context_.get(), weights_.attention_certificate(), planes_report);
    }
#endif
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
    [[nodiscard]] const device::Profile* step_projection_profile() const noexcept {
        return rows_profile_ ? &*rows_profile_ : nullptr;
    }
#endif
    void check(mini_gpu::Status status) const { mini_gpu::check(status, "Inkling 2 Mini stage"); }
    void set_gate_route_tap(strix::moe::RouteTap* tap) {
        graph_.invalidate();
        invalidate_cohort_graphs();
        gate_route_tap_ = tap;
    }
    // Calibration input statistics (strix/weight/calib_moments.hpp, off when null): every eager chunk run while a
    // registered calibration window is open hands its attention / output-projection / head input rows to the set,
    // attributed to that window (host-synchronous; step graphs stay off). Rows outside an open window (for example
    // decode steps of generated tokens) belong to no declared window and are not captured.
    void set_calibration_moments(weight::calib::MomentSet* moments) {
        graph_.invalidate();
        invalidate_cohort_graphs();
        calibration_ = moments;
        calibration_window_.reset();
    }
    void set_calibration_window(std::optional<std::size_t> window) { calibration_window_ = window; }
    [[nodiscard]] weight::calib::MomentSet* calibration_moments() const noexcept {
        return calibration_window_ ? calibration_ : nullptr;
    }
    void capture_calibration(const float* rows_device, std::size_t rows, std::span<weight::calib::InputMoments* const> targets) {
        if (!calibration_ || !calibration_window_ || targets.empty() || rows == 0U) return;
        const std::size_t width = targets.front()->width;
        calibration_rows_.resize(rows * width);
        mini_gpu::copy(calibration_rows_.data(), rows_device, calibration_rows_.size() * sizeof(float), mini_gpu::device_to_host,
                       stream_);
        mini_gpu::synchronize(stream_);
        calibration_->add(*calibration_window_, targets, calibration_rows_.data(), rows);
    }
    void set_gate_order_null(bool enabled) {
        graph_.invalidate();
        invalidate_cohort_graphs();
        gate_order_null_ = enabled;
        null_applications_ = 0;
    }
    [[nodiscard]] std::uint64_t gate_null_applications() const { return null_applications_; }
    // Resolved prepared-attention round fold (mini_prepared_round_fold): admitted on the qualified HIP target
    // (mini_prepared_round_fold_admitted), declined on any other target and on CUDA, whose preparation has no fold. Set
    // once after construction; a captured step graph made under another effective setting is dropped. Returns the
    // effective state (false = off or declined); every entry point reports that, never the request.
    bool set_prepared_round_fold(bool selected) {
        bool effective = false;
#if defined(__HIPCC__)
        if (selected) {
            hipDeviceProp_t properties{};
            mini_gpu::check(hipGetDeviceProperties(&properties, mini_gpu::current_device()), "read HIP device properties");
            const std::string_view target(properties.gcnArchName);
            effective = mini_prepared_round_fold_admitted(true, target.substr(0, target.find(':')));
        }
        if (effective != prepared_round_fold_) {
            graph_.invalidate();
            invalidate_cohort_graphs();
            prepared_round_fold_ = effective;
        }
#else
        static_cast<void>(selected);
#endif
        return effective;
    }

    // Lever inkling2-mini-attn-projected-bias@1 (bit-identical to the image route): single multi-row prompt chunks project the
    // relative bias inside the flash. Every entry point (gate, generation CLI, step program, server) passes the resolved value;
    // HIP ignores it (its projected route is selected by its own binding).
    void set_attention_projected_bias(bool enabled) {
#if defined(__HIPCC__)
        static_cast<void>(enabled);
#else
        attention_projected_bias_ = enabled;
#endif
    }
    // CUDA projected-bias attention submissions so far (0 on HIP and with the lever off): what a test observes to tell the
    // projected route from the image route, whose bits are equal.
    [[nodiscard]] std::uint64_t projected_attention_dispatches() const {
#if defined(__HIPCC__)
        return 0U;
#else
        return projected_dispatches_;
#endif
    }

    // FAST W4A4 expert lever (inkling2-mini-moe-w4a4@1): multi-row prefill chunks run admitted NVFP4 expert layers through
    // the shared grouped W4A4 operand class. The NVFP4 image of the assignment rows is allocated only when the lever is on.
    // `activations` is the shared operand class the lever value selects (moe_core/grouped_w4a4.hpp: nvfp4 = nearest@1 scale
    // rule, nvfp4_mse = mse@1, nvfp4_mse_gate_up = mse@1 gate-up with the down projection on BF16 activations,
    // nvfp4_mse_r2_gate_up / nvfp4_mse_r2 = two-term activations on gate-up / on both projections, nvfp4_r2 = two nearest@1
    // terms on both projections).
    // bf16 = lever off. Every entry point (gate, generation CLI, step program, server) passes the resolved class.
    // HIP: the rotated integer class (Activations::iu4_rot, moe_core/grouped_iu4a4.hpp, grouped-iu4a4-rot-v1): every layer
    // whose experts pass admission gets a rotated image of both projections here (load time; ~13 GB at Inkling 2 Mini
    // shapes) and multi-row prefill chunks run it; the NVFP4 classes are CUDA values (ignored). CUDA declines iu4_rot.
    void set_moe_w4a4(moe_core::grouped_w4a4::Activations activations) {
        // Admission for this build's backend: a class the backend cannot run is refused (CUDA: the W4A4 classes; HIP: iu4_rot).
        static_cast<void>(admit_mini_moe_w4a4(activations, kMiniBuildBackend));
#if defined(__HIPCC__)
        // Images are built on the first iu4 selection and kept; each selection (per prompt under the ladder) turns the
        // class on or off for the chunks that follow. Narrow (NVFP4-transcoded) experts keep the BF16 tile: the images
        // are built from the checkpoint's own matrices.
        const bool iu4 = activations == moe_core::grouped_w4a4::Activations::iu4_rot && !weights_.narrow_experts();
        if (iu4 && !moe_iu4_) build_iu4_images();
        iu4_on_ = iu4 && moe_iu4_;
#else
        moe_w4a4_ = moe_core::grouped_w4a4::w4a4_gate_up(activations);
        moe_w4a4_class_ = activations;
        if (!moe_w4a4_) return;
        for (unsigned layer = 0; layer < kPhysicalLayers; ++layer)
            moe_w4a4_layer_[layer] = mini_grouped::admit_w4a4<MiniMoeConfig>(layer, weights_.experts(layer));
        // The shared dispatch refuses an image too small for the selected class.
        const auto bytes = mini_moe_nvfp4_image_bytes(max_rows_, activations);
        if (!moe_nvfp4_ || moe_nvfp4_->span().size() < bytes) moe_nvfp4_.emplace(bytes);
#endif
    }

#if !defined(__HIPCC__)
    [[nodiscard]] moe_core::grouped_w4a4::Activations moe_w4a4_class(unsigned layer) const noexcept {
        return moe_w4a4_ && moe_w4a4_layer_[layer] ? moe_w4a4_class_ : moe_core::grouped_w4a4::Activations::bf16;
    }
#endif

    // True while a W4A4 class is bound (CUDA only; HIP refuses a selected class in set_moe_w4a4).
    [[nodiscard]] bool moe_w4a4_bound() const noexcept {
#if defined(__HIPCC__)
        return false;
#else
        return moe_w4a4_;
#endif
    }

    // Exact down-image lever (inkling2-mini-combine-down-image@1; mini_combine_image resolves it): on the qualified HIP
    // target (mini_combine_image_arch) the grouped MoE down stores a BF16 image the combine reads, with the NaN side-channel
    // word allocated here, zeroed once. Unselected, or on any other target, no word exists and the dispatch keeps the
    // FP32 down rows (null pointer). Returns whether the image route is active. CUDA ignores it.
    // Verify rows lever (MiniResidentDecoder::set_verify_rows): HIP verify windows attend their rows in one lanes launch
    // set. Verify windows run eagerly, so no captured graph depends on it. CUDA ignores it.
    void set_verify_lanes(bool enabled) noexcept {
#if defined(__HIPCC__)
        verify_lanes_ = enabled;
#else
        static_cast<void>(enabled);
#endif
    }
    bool set_combine_image(bool selected) {
#if defined(__HIPCC__)
        if (!selected || moe_nan_seen_) return moe_nan_seen_.has_value();
        hipDeviceProp_t properties{};
        mini_gpu::check(hipGetDeviceProperties(&properties, mini_gpu::current_device()), "read HIP device properties");
        const std::string_view target(properties.gcnArchName);
        if (!mini_combine_image_arch(target.substr(0, target.find(':')))) return false;
        moe_nan_seen_.emplace(1U);
        mini_gpu::clear(moe_nan_seen_->data(), moe_nan_seen_->span().size_bytes(), stream_);
        mini_gpu::synchronize(stream_);
        return true;
#else
        static_cast<void>(selected);
        return false;
#endif
    }

    [[nodiscard]] bool lowbit_applies(MiniAppendKind kind, unsigned rows) const noexcept {
        if (!weights_.compact() && !weights_.compact_head_bound()) return false;
        if (!weights_.bf16_bound()) return true;
        return kind == MiniAppendKind::decode && mini_lowbit_receipt_width(rows);
    }
    // The current chunk reads the compact roles (attention projections here, the head in the decoder).
    [[nodiscard]] bool lowbit_chunk() const noexcept { return lowbit_chunk_; }
    [[nodiscard]] core::LowBitArithmetic lowbit_arithmetic() const noexcept { return lowbit_arithmetic_; }
    void set_lowbit_arithmetic(core::LowBitArithmetic arithmetic) noexcept {
        if (lowbit_arithmetic_ == arithmetic) return;
        reset_step_graph();
        lowbit_arithmetic_ = arithmetic;
    }

    void start_chunk(unsigned rows, MiniAppendKind kind, unsigned decode_cohort_width = 0U) {
        // Route capture/replay reads or writes host data synchronously. The step graph itself is
        // decided by try_step_graph; the route only picks the row geometry here.
        select_route(rows, kind, decode_cohort_width);
        clear_flags();
    }

    void set_step_token(std::optional<unsigned> token) { step_token_ = token; }
    void reset_step_graph() noexcept {
        graph_.reset();
#if !defined(__HIPCC__)
        cohort_graphs_.reset();
#endif
    }

    void invalidate_cohort_graphs() noexcept {
#if !defined(__HIPCC__)
        cohort_graphs_.invalidate();
#endif
    }

    [[nodiscard]] exec::StepBlock* cohort_block() { return cohort_graphs_.block(); }
    [[nodiscard]] bool cohort_lanes_active() const noexcept { return cohort_lanes_.has_value(); }

#if defined(__HIPCC__)
    [[nodiscard]] bool admits_eager_lanes() const { return host_taps_off(); }

    template <class Work>
    void run_eager_lanes(const exec::StepBlock& block, const MiniCohortLanes& lanes, Work&& work) {
        cohort_graphs_.upload(stream_, block);
        struct ClearLanes {
            std::optional<MiniCohortLanes>& active;
            ~ClearLanes() { active.reset(); }
        } clear{cohort_lanes_};
        cohort_lanes_ = lanes;
        work();
    }
#else
    // Cohort decode programs: one captured graph per program width over one device step block. While a
    // program is captured the backend is in lane mode: every lane-stateful stage (K/V and post convolutions,
    // position-scaled clips, cache store + attention) is one launch over the lanes, reading each lane's slot,
    // position and id from the step block (no host position or slot address is baked into the graph).
    [[nodiscard]] const exec::WidthStepGraphs& cohort_graphs() const noexcept { return cohort_graphs_; }

    // The host route of a chunk whose device work replays (the captured start_chunk also clears the flags).
    void prepare_route(unsigned rows, MiniAppendKind kind) {
        select_route(rows, kind);
    }

    // False: nothing ran (programs off, a gate tap or order-null arm, or this width declined); run eagerly.
    template <class MakeLanes, class Work>
    bool try_cohort_graph(bool enabled, const exec::StepBlock& block, MakeLanes make_lanes, Work& work) {
        return cohort_graphs_.try_launch(stream_, enabled && step_graphs_, host_taps_off(), block,
            [&] {
                struct Leave {
                    std::optional<MiniCohortLanes>& active;
                    ~Leave() { active.reset(); }
                } leave{cohort_lanes_};
                cohort_lanes_ = make_lanes();
                clear_flags();
                work();
            });
    }
#endif
    [[nodiscard]] const session::StepDescriptor* capture_descriptor() const noexcept {
        return graph_.capture_descriptor();
    }

    template <class Work>
    bool try_step_graph(MiniResidentChunk chunk, Work& work) {
        // Layout and workspace addresses are fixed for this backend's lifetime. A route tap and the
        // order-null gate perform host-synchronous reads and writes.
        // An append that is not the one-row decode step (a prompt chunk, the kept rows a partial verify commit replays)
        // runs eagerly and leaves the captured step as it is: it binds no other address, and every setting a capture
        // depends on drops the graph where it changes (levers, low-bit arithmetic, the screened head, slot resets).
        // Without this, every verify replay of a drafting run made the next plain step capture again.
        if (!step_token_) {
            clear_flags();
            return false;
        }
        if (graph_.try_launch(stream_, step_graphs_, chunk.rows, step_token_.has_value(),
                              host_taps_off(),
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

    // Decode expert form this backend was built with (packed one-row form or the grouped tile kernel).
    [[nodiscard]] bool packed_decode() const noexcept { return packed_decode_; }
    [[nodiscard]] bool step_graphs() const noexcept { return step_graphs_; }

    // Served held-back route on a chunk-invariant backend (CUDA): every prefill row keeps the grouped tile, a one-row
    // prefill chunk included (mini_resident_chunk_route grouped_prefill), so a served prompt row's bits do not depend
    // on the admitted chunk or the pieces. Set once before any append; the command line keeps its one-row route.
    void keep_prefill_grouped() noexcept { grouped_prefill_ = !kBackendHip; }
    [[nodiscard]] bool prefill_grouped() const noexcept { return grouped_prefill_; }

    void finish() {
        std::array<unsigned, (1U + kMiniMaxVerifyRows) * kLogicalVisits> storage{};
        const auto active = active_flags().size();
        // Packed cohort rows (served route lever) can route more rows than the verify window: a host copy that grows
        // once to the widest routed call.
        if (active > storage.size() && host_flags_.size() < active) host_flags_.resize(active);
        const auto flags = active > storage.size() ? std::span(host_flags_).first(active) : std::span(storage).first(active);
        mini_gpu::copy(flags.data(), flags_.data(), flags.size_bytes(), mini_gpu::device_to_host, stream_);
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

#if !defined(__HIPCC__)
    // The opt-in CUDA schedule uses one plane for declared BF16 operands with certified resident weights.
    void dense(const float* input, unsigned width, unsigned rows, std::span<const Projection> projections,
               core::Bf16Operand operand) {
        check(enqueue_controller_dense(weights_.controller_certificate(), input, width, rows, projections, stream_,
                                       big_context_.get(), operand).status);
    }
#endif

#if defined(__HIPCC__)
    // Projections on a BF16 operand image (the shared tile's image rows: same bits as dense on the FP32 rows).
    static bool admits_dense_bf16_image(unsigned width, unsigned rows, std::span<const Projection> projections) {
        return core::fma_rows::admits_bf16_image_projections(width, rows, projections, kVocabulary);
    }
    void dense_bf16_image(const std::uint16_t* input, unsigned width, unsigned rows,
                          std::span<const Projection> projections) {
        check(core::fma_rows::enqueue_bf16_image_rows(input, width, rows, projections, kVocabulary, stream_));
    }

    // Low-rank controller gate pair with the controller as the shared tile's epilogue; false = not admitted.
    bool dense_controller(const float* input, unsigned width, unsigned rows, std::span<const Projection> gates,
                          const core::residual_mix::ControllerArgs& control) {
        const auto status = core::fma_rows::enqueue_bf16_controller_rows(input, width, rows, gates, control.current,
                                                                         control.message, control.output, stream_);
        if (status == hipErrorNotSupported) return false;
        check(status);
        return true;
    }

    // The gate pair, the controller and rms() below (the shared BF16-rows norm, kMiniHipRmsEpsilon) as one row-owner
    // launch writing the normalized rows to control.output; false = not admitted (the caller keeps the launches).
    bool dense_controller_rms(const float* input, unsigned width, unsigned rows, std::span<const Projection> gates,
                              const core::residual_mix::ControllerArgs& control, const std::uint16_t* scale) {
        if (rows > kMaxContext) return false;
        const auto status = core::fma_rows::enqueue_bf16_controller_rms_rows(
            input, width, rows, gates, control.current, control.message, scale, kMiniHipRmsEpsilon, control.output,
            stream_);
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
#if defined(__HIPCC__)
        // The message row of this controller was produced by the pass's residual bank batch (capture_message_bank).
        if (visit.pass != 0U && batched_bank_[0] != nullptr && previous_bank.size() == 2U &&
            previous_bank[0].data() == batched_bank_[0] && previous_bank[1].data() == batched_bank_[1] &&
            rows_of(input) == 1U) {
            const auto controller = controller_.span();
            const auto row = mini_message_slot(visit, path).row();
            enqueue_mini_layer_input_message(
                weights_, visit, path, 1U, input,
                MiniConstRows(controller.subspan(kMiniController.scratch_elements_per_row() +
                                                     std::size_t{row} * kHiddenWidth, kHiddenWidth)),
                output, controller.first(kMiniController.scratch_elements_per_row()), *this);
            return;
        }
#endif
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
#if defined(__HIPCC__)
        enqueue_message_batch(normalized);
#endif
    }

    void add(MiniConstRows lhs, MiniConstRows rhs, MiniMutableRows output, bool round_bf16) {
        check(launch_mini_residual_add(lhs, rhs, output, round_bf16, stream_));
    }

    // `output` starts at the tap's column of the chunk's first position. During a one-row step capture it starts at
    // the tap's column of position 0 instead and the row comes from the step descriptor, so the replayed graph stores
    // every step at its own position (same bytes, same place as the eager store).
    void store_draft_tap(MiniConstRows input, std::span<std::uint16_t> output) {
        namespace rows = core::row_transforms;
        if (const auto* step = capture_descriptor()) {
            check(rows::elements(input.size(), rows::StoreBf16SliceAtRow{
                input.data(), output.data(), kHiddenWidth, MiniDSparkTapPlan::width, &step->position}, stream_));
            return;
        }
        check(rows::elements(input.size(), rows::StoreBf16Slice{
            input.data(), output.data(), kHiddenWidth, MiniDSparkTapPlan::width}, stream_));
    }

    void attention(LayerVisit visit, MiniResidentChunk chunk, MiniConstRows query,
                   MiniConstRows key, MiniConstRows value, const MiniResidentVisitState& state, MiniMutableRows output) {
        const MiniResidentAttentionGroup group{chunk, state};
        attention_grouped(visit, &group, {}, query, key, value, output);
    }

    void attention_batch(LayerVisit visit, std::span<const MiniResidentAttentionGroup> groups,
                         MiniConstRows query, MiniConstRows key, MiniConstRows value, MiniMutableRows output) {
        if (groups.empty()) throw std::invalid_argument("Inkling 2 Mini: empty attention batch");
        if (groups.size() == 1U) {
            attention_grouped(visit, &groups.front(), {}, query, key, value, output);
        } else {
            attention_grouped(visit, nullptr, groups, query, key, value, output);
        }
    }

private:
    // Key / value convolution tape of a staged append (bound only for verify windows), or none.
    static float* tape_rows(std::span<float> tape) { return tape.empty() ? nullptr : tape.data(); }

    void attention_grouped(LayerVisit visit, const MiniResidentAttentionGroup* single,
                           std::span<const MiniResidentAttentionGroup> groups,
                           MiniConstRows query, MiniConstRows key, MiniConstRows value, MiniMutableRows output) {
        const auto rows = rows_of(query);
        if (single && single->chunk.rows != rows)
            throw std::invalid_argument("Inkling 2 Mini: incompatible attention chunk");
        if (key.size() != query.size() ||
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
            single ? single->state.history[0].data() : nullptr,
            single ? single->state.history[1].data() : nullptr,
            single ? tape_rows(single->state.tape[0]) : nullptr,
            single ? tape_rows(single->state.tape[1]) : nullptr,
        };
        const MiniPreparedAttention prep{
            take(kQueryWidth), take(kKvWidth), take(kKvWidth), take(kQueryHeads * kRelativeExtent),
            take(kQueryHeads), take(kQueryWidth),
        };
        auto* attended = take(kQueryWidth);
        auto* gated = take(kQueryWidth);
#if defined(__HIPCC__)
        // The prefill projections' BF16 operand image: rows x the widest projection input (u16).
        const std::span<std::uint16_t> operand_image(reinterpret_cast<std::uint16_t*>(take(kQueryWidth / 2U)),
                                                     std::size_t{max_rows_} * kQueryWidth);
#endif
        const auto* lag = mini_convolution_layer(lag_weights_.data(), visit);
        auto* failure = flags_.data() + visit.logical_layer;
        const auto* step = capture_descriptor();
        const auto* device_position = step ? &step->position : nullptr;
        const std::span<const MiniResidentAttentionGroup> visits =
            single ? std::span<const MiniResidentAttentionGroup>(single, 1U) : groups;
        const auto image_of = [&](const MiniResidentAttentionGroup& group, std::size_t base) {
            return attn_core::PreparedGqaAttentionRows{
                prep.query + base * kQueryWidth, prep.key + base * kKvWidth, prep.value + base * kKvWidth,
                prep.relative_logits + base * kQueryHeads * kRelativeExtent, prep.sinks + base * kQueryHeads,
                attended + base * kQueryWidth, group.chunk.first_position, group.chunk.rows,
            };
        };
#if defined(__HIPCC__)
        const auto cache_of = [&](const MiniResidentAttentionGroup& group) {
            const auto& kv = group.state.kv;
            return MiniHipAttentionCache{kv.keys, kv.values, zero_padding_.data(), kv.slots, kv.wrapped};
        };
        const auto key_splits = [&](const auto& window) {
            return attn_core::dense::small_row_key_splits(
                window.first_position, window.count, mini_prepared_gqa_window(visit));
        };
        const auto workspace = [&](unsigned splits) {
            return splits > 1U || step ? attention_partials_.span() : std::span<float>{};
        };
        // Admit every lane before any preparation writes.
        const auto admit = [&](const MiniResidentAttentionGroup& group, std::size_t base) {
            const auto attention = image_of(group, base);
            const auto splits = key_splits(attention);
            if (!can_enqueue_mini_hip_attention(visit, attention, cache_of(group), failure,
                                                workspace(splits).size(), splits))
                check(hipErrorInvalidValue);
        };
        const detail::MiniResidentGqaBackend<detail::MiniPreparedHipImageBackend> adapter{
            {{{stream_}, prepared_round_fold_}, operand_image}, gate_order_null_ ? &null_applications_ : nullptr, lowbit_arithmetic_};
        const auto prepare_with = [&](const auto& body) { return body(adapter); };
        auto* operand_failure = failure;
        const auto launch_group = [&](const MiniResidentAttentionGroup& group, std::size_t base) {
            const auto cache = cache_of(group);
            const auto launch = [&](const auto& window) {
                const auto window_splits = key_splits(window);
                return enqueue_mini_hip_attention<true>(visit, window, cache, failure, stream_,
                                                        workspace(window_splits), window_splits, device_position);
            };
            // A window changes the key tile limits and, on HIP, the Split-K
            // partition. Use each query's decode geometry in verification.
            return attn_core::for_each_prepared_gqa_window<MiniPreparedGqaShape>(
                image_of(group, base), route_.attention_rows, adapter.success, launch);
        };
        // A cohort decode step (one row per lane, eager): one store, flash and combine launch for every lane instead of
        // one launch_group per lane. Lane bits = its launch_group bits (attn_core/prepared_bf16_lanes_plan.hpp).
        const bool decode_lanes = !single && !step && std::ranges::all_of(groups, [](const auto& group) {
            return group.chunk.rows == 1U;
        });
        const auto launch_lanes = [&] {
            lanes_.clear();
            for (const auto& group : visits) {
                const auto& kv = group.state.kv;
                lanes_.push_back({kv.keys, kv.values, group.chunk.first_position, kv.slots, kv.wrapped ? 1U : 0U});
            }
            auto image = image_of(visits.front(), 0U);
            image.count = static_cast<unsigned>(visits.size());
            return attn_core::dense::append_prepared_bf16_lanes<detail::MiniHipAttentionPolicy>(
                kKvHeads, mini_prepared_gqa_window(visit), kRelativeExtent, image, lanes_, zero_padding_.data(),
                failure, stream_, attention_partials_.span());
        };
        // Verify rows lever (set_verify_lanes): a verify window (one sequence, one-row decode windows, eager) puts its
        // rows into one lanes launch set on the sequence's cache, each lane at its own position with its one-row decode
        // geometry, instead of one launch set per row. Lane bits = the per-window launch bits; the shared admission
        // (sequence_lane_caches) holds because the window ring reserves window - 1 + the largest chunk rows
        // (tests/inkling2/mini_schedule_test.cpp).
        const bool verify_lanes = verify_lanes_ && single && !step && route_.attention_rows == 1U &&
                                  single->chunk.rows > 1U && single->chunk.rows <= attn_core::dense::kMaxPreparedBf16Lanes;
        const auto launch_verify_lanes = [&] {
            lanes_.clear();
            const auto& kv = single->state.kv;
            for (unsigned row = 0; row < single->chunk.rows; ++row)
                lanes_.push_back({kv.keys, kv.values, single->chunk.first_position + row, kv.slots,
                                  kv.wrapped ? 1U : 0U});
            return attn_core::dense::append_prepared_bf16_lanes<detail::MiniHipAttentionPolicy>(
                kKvHeads, mini_prepared_gqa_window(visit), kRelativeExtent, image_of(*single, 0U), lanes_,
                zero_padding_.data(), failure, stream_, attention_partials_.span());
        };
        const auto launch_visits = [&](std::span<const MiniResidentAttentionGroup> lanes) {
            if (decode_lanes) {
                check(launch_lanes());
                return;
            }
            if (verify_lanes) {
                check(launch_verify_lanes());
                return;
            }
            std::size_t base = 0;
            for (const auto& group : lanes) {
                check(launch_group(group, base));
                base += group.chunk.rows;
            }
        };
#else
        const auto admit = [](const MiniResidentAttentionGroup&, std::size_t) {};
        // Producer-backed operands: the attention inputs are rmsnorm_bf16 rows and the output projection reads the
        // BF16 gated product, so the linears declare bf16_exact (one plane, same bits) where the staged attention
        // weights are certified (finite, ranges kept finite); otherwise they keep FP32 operands (three planes).
        const detail::MiniPreparedCudaBackend native = attention_operand_backend();
        const detail::MiniResidentGqaBackend<> adapter{native, gate_order_null_ ? &null_applications_ : nullptr};
        // One-row decode over native BF16 weights with a rows profile issues the
        // projection groups as one grouped rows call at the first round (the
        // incumbent CUDA prepare); resident Q8_0 weights keep the plain adapter.
        const auto prepare_with = [&](const auto& body) {
            if (rows_profile_ && rows == 1U && !weights_.compact()) {
                const detail::MiniResidentGqaBackend<detail::MiniPreparedCudaProjectionBatch> batch{
                    {native, rows, *rows_profile_}, adapter.null_applications};
                return body(batch);
            }
            return body(adapter);
        };
        unsigned* operand_failure = nullptr;  // CUDA validates with the append.
        const MiniCohortLanes* cohort = cohort_lanes_ ? &*cohort_lanes_ : nullptr;
        if (cohort && device_position) throw std::logic_error("Inkling 2 Mini: lane mode inside a one-row step capture");
        const auto launch_group = [&](const MiniResidentAttentionGroup& group, std::size_t base) {
            const auto launch = [&](const auto& window) {
                return enqueue_mini_cuda_attention(visit, window, group.state.kv, stream_,
                                                   cuda_core::FlashPipe::pad2, failure, device_position);
            };
            return attn_core::for_each_prepared_gqa_window<MiniPreparedGqaShape>(
                image_of(group, base), route_.attention_rows, adapter.success, launch);
        };
        // Several one-row windows at host positions (cohort decode): every lane's window goes into one
        // lane-batched append with exactly the arguments of its own launch above (same bits per row,
        // independent of the other lanes). Multi-row windows and device-position steps keep the
        // per-window launch.
        const auto launch_visits = [&](std::span<const MiniResidentAttentionGroup> lanes) {
            if (cohort) {
                // Lane mode: lane 0's visit caches give the plane offsets; the device resolves every lane.
                auto packed = image_of(lanes.front(), 0U);
                packed.count = rows;
                const auto& kv = lanes.front().state.kv;
                const cuda_core::PreparedBf16StepLanes step_lanes{cohort->lanes, cohort->plane(kMiniLaneKeys, kv.keys),
                    cohort->plane(kMiniLaneValues, kv.values), kv.slots, kv.wrapped ? 1U : 0U};
                check(enqueue_mini_cuda_attention_step_lanes(visit, packed, step_lanes, stream_, failure));
                return;
            }
            lane_scratch_.clear();
            bool one_row = device_position == nullptr;
            std::size_t base = 0;
            for (const auto& group : lanes) {
                check(attn_core::for_each_prepared_gqa_window<MiniPreparedGqaShape>(
                    image_of(group, base), route_.attention_rows, adapter.success, [&](const auto& window) {
                        one_row = one_row && window.count == 1U;
                        const auto& kv = group.state.kv;
                        lane_scratch_.push_back({kv.keys, kv.values, window.first_position, kv.slots,
                                                 kv.wrapped ? 1U : 0U});
                        return adapter.success;
                    }));
                base += group.chunk.rows;
            }
            if (!one_row || lane_scratch_.size() < 2U) {
                base = 0;
                for (const auto& group : lanes) {
                    check(launch_group(group, base));
                    base += group.chunk.rows;
                }
                return;
            }
            auto packed = image_of(lanes.front(), 0U);
            packed.count = static_cast<unsigned>(base);
            check(enqueue_mini_cuda_attention_lanes(visit, packed, lane_scratch_, stream_, failure));
        };
#endif
        prep_groups_.clear();
        if (!single) {
            for (const auto& group : groups)
                prep_groups_.push_back({group.chunk.first_position, group.chunk.rows,
                                        group.state.history[0].data(), group.state.history[1].data(),
                                        tape_rows(group.state.tape[0]), tape_rows(group.state.tape[1])});
        }
#if defined(__HIPCC__)
        const auto* cohort = cohort_lanes_ ? &*cohort_lanes_ : nullptr;
#endif
        std::optional<attn_core::PreparedGqaStepLanes> prep_step;
        if (cohort) {
            const auto& front = visits.front().state;
            prep_step = attn_core::PreparedGqaStepLanes{cohort->lanes,
                cohort->plane(kMiniLaneHistory, front.history[static_cast<unsigned>(MiniConvolutionStream::key)].data()),
                cohort->plane(kMiniLaneHistory, front.history[static_cast<unsigned>(MiniConvolutionStream::value)].data())};
        }
        const attn_core::PreparedGqaStepLanes* prep_lanes = prep_step ? &*prep_step : nullptr;
        const auto enqueue = [&](const auto& weights) {
            {
                std::size_t base = 0;
                for (const auto& group : visits) {
                    admit(group, base);
                    base += group.chunk.rows;
                }
            }
            const auto prepare = [&](const auto& backend) {
                return attn_core::enqueue_prepared_gqa<MiniPreparedGqaShape>(
                    backend, is_full_attention(visit.physical_layer), single ? single->chunk.first_position : 0U,
                    rows, inputs, weights, work, prep, operand_failure, device_position, prep_groups_, prep_lanes);
            };
#if defined(__HIPCC__)
            // A single prefill chunk in one window (the route mini_hip_bias_in_attention admits) projects the relative
            // bias inside the flash: the prepare stage skips the [row][head][distance] image and its producer, the
            // flash computes the same logits and flags (shared DenseProjectedBias trait). The null-order count needs
            // the image producer, so it keeps the image route.
            const auto* bias_weights =
                attn_core::detail::bf16_projection(weights.projection(MiniAttentionProjection::relative_bias));
            if (single && !adapter.null_applications && route_.attention_rows >= rows &&
                mini_hip_bias_in_attention(rows, key_splits(image_of(*single, 0U)), device_position,
                                           work.relative_features, bias_weights)) {
                const detail::MiniResidentGqaBackend<detail::MiniPreparedHipImageProjectedBackend> projected{
                    {{{{stream_}, prepared_round_fold_}, operand_image}}, nullptr, lowbit_arithmetic_};
                check(prepare(projected));
                check(enqueue_mini_hip_attention_projected<true>(visit, image_of(*single, 0U), cache_of(*single),
                                                                 work.relative_features, bias_weights, failure, stream_));
            } else
#else
            // A single multi-row prompt chunk in one window projects the relative bias inside the CUDA flash (lever
            // inkling2-mini-attn-projected-bias@1, off by default until its identity receipt): the prepare stage skips
            // the image and its producer, the flash computes the same logits and raises the same flag
            // (cuda_core::PreparedBf16Projection). The null-order count needs the image producer and lane mode
            // resolves positions on the device, so both keep the image route. The whole projected append is prepared
            // ONCE, BEFORE the image producer is dropped (prepare_mini_cuda_attention_projected): a declined value
            // (e.g. the dynamic shared memory of the 1024-column projection above the device opt-in) keeps the image
            // route, any other status is a real error, and the admitted value is the one submitted after the
            // preparation of q / k / v, with no second admission.
            const auto* bias_weights =
                attn_core::detail::bf16_projection(weights.projection(MiniAttentionProjection::relative_bias));
            bool projected_route = attention_projected_bias_ && single && !cohort && !adapter.null_applications &&
                                   route_.attention_rows >= rows &&
                                   mini_cuda_bias_in_attention(rows, device_position, work.relative_features, bias_weights);
            if (projected_route) {
                auto ran = MiniCudaAttentionRoute::image;
                check(dispatch_mini_cuda_attention(
                    visit, image_of(*single, 0U), single->state.kv, work.relative_features, bias_weights, failure, stream_,
                    [&] {
                        const detail::MiniResidentGqaBackend<detail::MiniPreparedCudaProjectedBackend> projected{{native},
                                                                                                             nullptr};
                        return prepare(projected);
                    },
                    [&] {
                        check(prepare_with(prepare));
                        launch_visits(visits);
                        return cudaSuccess;
                    },
                    ran));
                if (ran == MiniCudaAttentionRoute::projected) ++projected_dispatches_;
            } else
#endif
            {
                check(prepare_with(prepare));
                launch_visits(visits);
            }
            if (single) {
                // A taped (staged verify) chunk keeps its raw K/V rows, the inputs of the two KV convolutions
                // (verify rows lever: the shared session tape is bound only then).
                const auto kv_rows = std::size_t{rows} * kKvWidth;
                if (const auto key_tape = single->state.tape[0]; !key_tape.empty())
                    copy(std::span<const float>(work.raw_keys, kv_rows), key_tape);
                if (const auto value_tape = single->state.tape[1]; !value_tape.empty())
                    copy(std::span<const float>(work.raw_values, kv_rows), value_tape);
#if defined(__HIPCC__)
                // The history stores the BF16-rounded K/V inputs. Without the round fold the stage rounds raw_keys /
                // raw_values in place before the convolution (the tape copied rounded rows); with the fold the
                // convolution rounds its inputs on the fly and the raw rows stay FP32, so the tape takes the same
                // round here and a partial commit restores the history the replayed rows would leave.
                if (prepared_round_fold_) {
                    for (const auto tape : {single->state.tape[0], single->state.tape[1]})
                        if (!tape.empty()) check(adapter.round(tape.data(), kv_rows));
                }
#endif
            }
            check(attn_core::enqueue_prepared_gqa_output<MiniPreparedGqaShape>(
                adapter, rows, weights.projection(MiniAttentionProjection::output), attended, prep.gate_logits,
                gated, output.data(), weights.output_row_bytes));
            if (auto* moments = calibration_moments(); moments && single) {
                weight::calib::InputMoments* const target =
                    &mini_calibration_input(*moments, MiniCalibrationInput::output, visit.physical_layer);
                capture_calibration(gated, rows, std::span(&target, 1U));
            }
        };
        if (weights_.compact() && lowbit_chunk_) {
            auto weights = weights_.compact_attention(visit.physical_layer);
            weights.key_convolution = lag;
            weights.value_convolution = lag + kKvWidth * kConvTaps;
            enqueue(weights);
        } else {
            enqueue(mini_device_projection_weights(weights_, visit, lag));
        }
    }

public:
    void convolve(LayerVisit visit, MiniConvolutionStream path, MiniConstRows input,
                  MiniMutableRows history, MiniMutableRows output) {
        if ((path != MiniConvolutionStream::attention && path != MiniConvolutionStream::moe) ||
            output.size() != input.size() || history.size() != (kConvTaps - 1U) * kHiddenWidth)
            throw std::invalid_argument("Inkling 2 Mini: incompatible residual convolution rows");
        const auto* weights = mini_post_convolution_weights(lag_weights_.data(), visit, path == MiniConvolutionStream::moe);
        const auto rows = rows_of(input);
#if defined(__HIPCC__)
        check(enqueue_mini_hip_post_convolution(rows, weights, input.data(),
                                                history.data(), output.data(), stream_, prepared_round_fold_));
#else
        check(enqueue_mini_cuda_post_convolution(rows, weights, input.data(),
                                                 history.data(), output.data(), stream_));
#endif
    }

    // Lane mode: one post convolution launch over every lane; `history` is lane 0's visit history of this
    // stream (its plane offset), each lane's own history is resolved on the device.
    void convolve_lanes(LayerVisit visit, MiniConvolutionStream path, MiniConstRows input,
                        MiniMutableRows history, MiniMutableRows output) {
        if (!cohort_lanes_ || (path != MiniConvolutionStream::attention && path != MiniConvolutionStream::moe) ||
            output.size() != input.size() || history.size() != (kConvTaps - 1U) * kHiddenWidth ||
            rows_of(input) != cohort_lanes_->lanes.lanes)
            throw std::invalid_argument("Inkling 2 Mini: incompatible lane convolution rows");
        const auto* weights = mini_post_convolution_weights(lag_weights_.data(), visit, path == MiniConvolutionStream::moe);
        const state::LaneState<float> lanes{cohort_lanes_->lanes, cohort_lanes_->plane(kMiniLaneHistory, history.data())};
#if defined(__HIPCC__)
        check(attn_core::enqueue_prepared_gqa_post_convolution_lanes<MiniPreparedGqaShape>(
            detail::MiniPreparedHipBackend{{stream_}}, rows_of(input), weights, input.data(), lanes, output.data()));
#else
        check(enqueue_mini_cuda_post_convolution_lanes(rows_of(input), weights, input.data(), lanes, output.data(), stream_));
#endif
    }

    void moe(LayerVisit visit, MiniConstRows input, MiniMutableRows output) {
        moe_with(visit, input, output, moe_core::r1_tail::NoTail{});
    }

#if defined(__HIPCC__)
    // MoE post-convolution and residual add inside the R1 down-combine launch (moe_core/r1_tail.hpp): one routed row on
    // the packed decode route. Read once at construction; STRIX_MOE_TAIL=0 keeps the separate convolution and add.
    [[nodiscard]] bool moe_tail_enabled() const noexcept {
        return moe_tail_ && route_.packed_moe && chunk_rows_ == 1U;
    }

    // moe(), then convolve(moe stream) and add(residual, conv, sum, BF16 round) as one down-combine launch: the tail
    // carries the plan, weights, history and rounding of those two nodes.
    void moe_tail(LayerVisit visit, MiniConstRows input, MiniMutableRows moe_output, MiniMutableRows history,
                  MiniConstRows residual, MiniMutableRows conv_output, MiniMutableRows sum) {
        if (!moe_tail_enabled() || rows_of(input) != 1U || history.size() != (kConvTaps - 1U) * kHiddenWidth ||
            residual.size() != kHiddenWidth || conv_output.size() != kHiddenWidth || sum.size() != kHiddenWidth)
            throw std::invalid_argument("Inkling 2 Mini: incompatible MoE tail rows");
        static_assert(MiniPreparedGqaShape::kConvTaps == kConvTaps);
        const moe_core::r1_tail::ConvResidualTail<kConvTaps> tail{
            attn_core::prepared_gqa_post_convolution_plan<MiniPreparedGqaShape>(),
            mini_post_convolution_weights(lag_weights_.data(), visit, true), history.data(), conv_output.data(),
            residual.data(), sum.data(), attn_core::kPreparedGqaPostConvolutionRound, true,
        };
        moe_with(visit, input, moe_output, tail);
    }
#endif

private:
    template <class Tail>
    void moe_with(LayerVisit visit, MiniConstRows input, MiniMutableRows output, const Tail& tail) {
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
            tile_count, flags_.data() + kLogicalVisits + visit.logical_layer * route_flag_rows_, count,
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
        const bool iu4 = chunk_iu4_ && iu4_images_[layer].gate_up.codes != nullptr;
        const MiniMoeWorkspace workspace{moe_bf16_.span(), moe_fp32_.span(), moe_activation_.span(),
                                         iu4 ? moe_nvfp4_->span() : std::span<std::uint8_t>{},
                                         moe_nan_seen_ ? moe_nan_seen_->data() : nullptr};
#else
        using A = moe_core::grouped_w4a4::Activations;
        const auto activations = chunk_w4a4_ ? moe_w4a4_class(layer) : A::bf16;
        const MiniMoeWorkspace workspace{moe_bf16_.span(), moe_fp32_.span(), {},
                                         activations != A::bf16 ? moe_nvfp4_->span() : std::span<std::uint8_t>{}};
#endif
        const MiniMoeResidentRouteBuffers buffers{{ids, assignments}, {status, max_rows_},
                                                  route_scratch_.span(), route_weights};
        const auto matrices = matrices_.span().subspan(std::size_t{layer} * kExperts, kExperts);
        if (route_.packed_moe) {
            enqueue_packed_decode(layer, rows, input, logits, bias, output, buffers, matrices, planned.failure, tail);
            return;
        }
        if constexpr (Tail::enabled) throw std::logic_error("Inkling 2 Mini: MoE tail without the packed route");
#if defined(__HIPCC__)
        if (weights_.narrow_experts())
            check(enqueue_mini_hip_resident_moe_for<MiniMoeNarrowConfig>(layer, rows, input, logits, bias, output, buffers,
                                                                         matrices, csr, planned, workspace, stream_,
                                                                         gate_route_tap_));
        else
            check(enqueue_mini_hip_resident_moe(layer, rows, input, logits, bias, output, buffers,
                                                matrices, csr, planned, workspace, stream_, gate_route_tap_,
                                                iu4 ? moe_core::grouped_iu4a4::Selection{&iu4_images_[layer]}
                                                    : moe_core::grouped_iu4a4::Selection{}));
#else
        if (weights_.narrow_experts())
            check(enqueue_mini_cuda_resident_moe_decoded<MiniMoeNarrowConfig>(
                moe_decode_, layer, rows, input, logits, bias, output, buffers, matrices, csr, planned, workspace, stream_,
                gate_route_tap_, activations, moe_schedule_));
        else
            check(enqueue_mini_cuda_resident_moe_decoded(moe_decode_, layer, rows, input, logits, bias, output, buffers,
                                                         matrices, csr, planned, workspace, stream_, gate_route_tap_,
                                                         activations, moe_schedule_));
#endif
    }

private:
#if defined(__HIPCC__)
    // Rotated images of both expert projections of every admitted layer (gate/up in the paired column order), built once
    // from the resident matrices; a layer whose matrices do not sit at one code stride keeps the BF16 tile.
    void build_iu4_images() {
        namespace gi = moe_core::grouped_iu4a4;
        using weight::microscaled::MatrixFormat;
        constexpr unsigned gate_width = 2U * kExpertWidth;
        const std::size_t gate_codes = gi::image_code_bytes(kHiddenWidth, gate_width);
        const std::size_t gate_steps = gi::image_step_floats(kHiddenWidth, gate_width);
        const std::size_t down_codes = gi::image_code_bytes(kExpertWidth, kHiddenWidth);
        const std::size_t down_steps = gi::image_step_floats(kExpertWidth, kHiddenWidth);
        unsigned built = 0U;
        for (unsigned layer = 0; layer < kPhysicalLayers; ++layer) {
            const auto experts = weights_.experts(layer);
            if (experts.size() != kExperts || kExperts < 2U) continue;
            std::array<weight::microscaled::PackedMatrix, kExperts> gate{}, down{};
            for (unsigned e = 0; e < kExperts; ++e) gate[e] = experts[e].gate_up, down[e] = experts[e].down;
            const auto stride = [](const auto& m) { return static_cast<std::size_t>(m[1].codes - m[0].codes); };
            if (!gi::expert_codes_strided(gate, gate[0].codes, stride(gate)) ||
                !gi::expert_codes_strided(down, down[0].codes, stride(down)))
                continue;
            auto& codes = iu4_codes_.emplace_back(std::size_t{kExperts} * (gate_codes + down_codes));
            auto& steps = iu4_steps_.emplace_back(std::size_t{kExperts} * (gate_steps + down_steps));
            gi::LayerImages& im = iu4_images_[layer];
            im.gate_up = {codes.data(), steps.data(), gate_codes, gate_steps, kExperts};
            im.down = {codes.data() + std::size_t{kExperts} * gate_codes, steps.data() + std::size_t{kExperts} * gate_steps,
                       down_codes, down_steps, kExperts};
            im.gate_up_codes = gate[0].codes;
            im.gate_up_stride = stride(gate);
            im.down_codes = down[0].codes;
            im.down_stride = stride(down);
            for (unsigned e = 0; e < kExperts; ++e) {
                const auto run = [&]<MatrixFormat F>() {
                    gi::enqueue_requantise_expert<F>(gate[e], kExpertWidth, kHiddenWidth, gate_width,
                                                     im.gate_up.codes + e * gate_codes, im.gate_up.steps + e * gate_steps, stream_);
                    gi::enqueue_requantise_expert<F>(down[e], 0U, kExpertWidth, kHiddenWidth,
                                                     im.down.codes + e * down_codes, im.down.steps + e * down_steps, stream_);
                };
                if (MiniMoeConfig::format(layer) == MatrixFormat::nvfp4) run.template operator()<MatrixFormat::nvfp4>();
                else run.template operator()<MatrixFormat::mxfp8>();
            }
            check(hipGetLastError());
            ++built;
        }
        mini_gpu::synchronize(stream_);
        moe_iu4_ = built > 0U;
        const auto act = gi::act_bytes(max_rows_, max_rows_ * kExpertsPerToken, kHiddenWidth, kExpertWidth);
        if (moe_iu4_ && !moe_nvfp4_) moe_nvfp4_.emplace(act);
        std::fprintf(stderr, "strix-moe-iu4a4 %s layers %u/%u (rotated expert images %.2f GB)\n", moe_core::grouped_iu4a4::kArithmeticId,
                     built, kPhysicalLayers, double(built) * double(kExperts) * double(gate_codes + down_codes + 4U * (gate_steps + down_steps)) / 1e9);
    }
#endif
    // The row geometry of a chunk and the prompt-row rule of the bound W4A4 class (mini_resident_chunk_route). The class
    // quantises each token row on its own and sums each (row, output) in a fixed K order, so a prompt row's bits do not
    // depend on how the prompt is chunked.
    void select_route(unsigned rows, MiniAppendKind kind, unsigned decode_cohort_width = 0U) {
        // Cohorts may schedule independent decode lanes with the verify kernel form. The receipt still names
        // the full decode cohort and the active width of this pass.
        const auto weight_kind = decode_cohort_width ? MiniAppendKind::decode : kind;
        lowbit_chunk_ = lowbit_applies(weight_kind, rows) &&
                        (!weights_.bf16_bound() || decode_cohort_width == 0U ||
                         mini_lowbit_receipt_width(decode_cohort_width));
        route_ = mini_resident_chunk_route(kind, rows, packed_decode_, moe_w4a4_bound(), cohort_rows_, grouped_prefill_);
#if defined(__HIPCC__)
        chunk_iu4_ = iu4_on_ && kind == MiniAppendKind::prefill && rows > 1U;
#else
        chunk_w4a4_ = mini_w4a4_rows(moe_w4a4_, kind);
#endif
        route_flag_rows_ = route_.packed_moe ? rows : 1U;
        chunk_rows_ = rows;
    }

    void clear_flags() {
        mini_gpu::clear(flags_.data(), active_flags().size_bytes(), stream_);
    }

    [[nodiscard]] std::span<unsigned> active_flags() {
        return flags_.span().first((1U + route_flag_rows_) * kLogicalVisits);
    }

    template <class Tail>
    void enqueue_packed_decode(unsigned layer, unsigned rows, MiniConstRows input, MiniConstRows logits, MiniConstRows bias,
                               MiniMutableRows output, MiniMoeResidentRouteBuffers route,
                               std::span<const MiniMoeExpert> matrices, unsigned* failure, const Tail& tail) {
        namespace decode = moe_core::microscaled_r1;
#if defined(__HIPCC__)
        check(moe_core::enqueue_route_wave<MiniRoutePolicy, detail::MiniMoeMath, moe::RouteWaveTraits<true, true>>(
            logits.data(), bias.data(), route.ids.data(), route.weights.data(), failure, rows, stream_));
        moe_core::route_tap(layer, rows, kExpertsPerToken, kExperts, route.ids.data(),
                            route.weights.data(), stream_, gate_route_tap_);
#else
        check(moe_core::enqueue_route_wave<MiniRoutePolicy, detail::MiniMoeMath, moe::RouteWaveTraits<true>>(
            logits.data(), bias.data(), route.ids.data(), route.weights.data(), failure, rows, stream_));
        cuda::core::route_tap(layer, rows, kExpertsPerToken, kExperts, route.ids.data(),
                              route.weights.data(), stream_, gate_route_tap_);
#endif
        using weight::microscaled::MatrixFormat;
        const auto format = weights_.expert_format(layer);
#if defined(__HIPCC__)
        if constexpr (Tail::enabled) {
            // One routed row: gate-up + the down-combine with the tail (moe_tail()).
            const decode::Args args{
                matrices.data(), route.ids.data(), route.weights.data(), failure, input.data(), moe_bf16_.data(),
                output.data(), kExperts, kHiddenWidth, kExpertWidth, kMlpGateSoftcap, kMlpUpSoftcap,
            };
            if (rows != 1U || !decode::admits_tail(args, tail, 1U, kExpertsPerToken))
                throw std::invalid_argument("Inkling 2 Mini: MoE tail not admitted");
            if (format == MatrixFormat::nvfp4)
                check(decode::launch_tail<MatrixFormat::nvfp4, kExpertsPerToken>(args, tail, stream_));
            else
                check(decode::launch_tail<MatrixFormat::mxfp8, kExpertsPerToken>(args, tail, stream_));
            return;
        }
        // HIP: one microscaled_r1 rows launch serves the routed rows (row r's bits are those of a one-row launch,
        // ink_hip_dec); the route status words stay one per row, contiguous from failure.
        if (!cohort_rows_ && rows > 1U && rows <= decode::kMaxRows &&
            moe_bf16_.span().size() >= std::size_t{rows} * kExpertsPerToken * kExpertWidth) {
            const decode::Args args{
                matrices.data(), route.ids.data(), route.weights.data(), failure, input.data(), moe_bf16_.data(),
                output.data(), kExperts, kHiddenWidth, kExpertWidth, kMlpGateSoftcap, kMlpUpSoftcap,
            };
            if (format == MatrixFormat::nvfp4)
                check(decode::launch<MatrixFormat::nvfp4, kExpertsPerToken>(args, stream_, rows));
            else
                check(decode::launch<MatrixFormat::mxfp8, kExpertsPerToken>(args, stream_, rows));
            return;
        }
#endif
        // Packed cohort rows (the HIP served route lever; every packed CUDA decoder, mini_packed_cohort_rows):
        // launch_rows runs chunks of at most rows_chunk_cap(scratch rows) rows, so a pass of up to kMaxLaunchRows (16)
        // rows is ONE launch when the activation scratch holds that many rows; row r keeps the bits of a one-row launch
        // (HIP and CUDA rows fixtures, mini_moe_device_fixture run_r1_rows; CUDA served lanes: cohort-lanes fixture).
        const auto scratch_rows = static_cast<unsigned>(
            std::min<std::size_t>(moe_bf16_.span().size() / (std::size_t{kExpertsPerToken} * kExpertWidth),
                                  decode::kMaxLaunchRows));
        if (cohort_rows_ && rows > 1U && scratch_rows != 0U) {
            const decode::Args args{
                matrices.data(), route.ids.data(), route.weights.data(), failure, input.data(), moe_bf16_.data(),
                output.data(), kExperts, kHiddenWidth, kExpertWidth, kMlpGateSoftcap, kMlpUpSoftcap,
            };
            if (format == MatrixFormat::nvfp4)
                check(decode::launch_rows<MatrixFormat::nvfp4, kExpertsPerToken>(args, stream_, rows, scratch_rows));
            else
                check(decode::launch_rows<MatrixFormat::mxfp8, kExpertsPerToken>(args, stream_, rows, scratch_rows));
            return;
        }
        // One route status per token remains live until finish(). Expert scratch
        // is reused in stream order after each decode form has consumed it.
        for (unsigned row = 0; row < rows; ++row) {
            const auto route_offset = std::size_t{row} * kExpertsPerToken;
            const auto hidden_offset = std::size_t{row} * kHiddenWidth;
            const decode::Args args{
                matrices.data(), route.ids.data() + route_offset, route.weights.data() + route_offset,
                failure + row, input.data() + hidden_offset, moe_bf16_.data(), output.data() + hidden_offset,
                kExperts, kHiddenWidth, kExpertWidth, kMlpGateSoftcap, kMlpUpSoftcap,
            };
            if (format == MatrixFormat::nvfp4)
                check(decode::launch<MatrixFormat::nvfp4, kExpertsPerToken>(args, stream_));
            else
                check(decode::launch<MatrixFormat::mxfp8, kExpertsPerToken>(args, stream_));
        }
    }

#if defined(__HIPCC__)
    // One launch per pass for every pass 1-2 controller message (2 x kMiniMessageQueryRows single-query readouts of
    // the captured bank, residual_mix::launch_bank_batch); each row is the readout its controller would issue. One
    // routed row only (decode); other shapes keep the per-controller readout.
    void enqueue_message_batch(const std::array<MiniMutableRows, 2>& normalized) {
        batched_bank_ = {};
        if (!bank_batch_ || normalized[0].size() != kHiddenWidth || normalized[1].size() != kHiddenWidth) return;
        const auto queries = weights_.f32("middle_cycle_cross_pass_queries");
        const auto kv_queries = weights_.f32("middle_cycle_cross_pass_kv_queries");
        constexpr auto rows = std::size_t{kMiniMessageQueryRows} * kHiddenWidth;
        const auto controller = controller_.span();
        if (queries.size() != rows || kv_queries.size() != rows ||
            controller.size() < kMiniController.scratch_elements_per_row() + 2U * rows)
            return;
        auto* messages = controller.data() + kMiniController.scratch_elements_per_row();
        core::residual_mix::BankBatchArgs args{};
        args.banks[0] = normalized[0].data();
        args.banks[1] = normalized[1].data();
        args.sets[0] = {queries.data(), messages, kMiniMessageQueryRows};
        args.sets[1] = {kv_queries.data(), messages + rows, kMiniMessageQueryRows};
        args.width = kHiddenWidth;
        args.bank_count = 2U;
        args.set_count = 2U;
        args.epsilon = kMiniController.epsilon;
        args.normalized = true;
        if (!core::residual_mix::fits(args)) return;
        check(core::residual_mix::launch_bank_batch(args, stream_));
        batched_bank_ = {normalized[0].data(), normalized[1].data()};
    }
#endif

    [[nodiscard]] unsigned rows_of(MiniConstRows input) const {
        const auto rows = input.size() / kHiddenWidth;
        if (rows == 0U || rows > max_rows_ || input.size() % kHiddenWidth != 0U)
            throw std::invalid_argument("Inkling 2 Mini: rows exceed the resident workspace");
        return static_cast<unsigned>(rows);
    }

    static constexpr std::size_t attention_elements(unsigned rows) {
        return std::size_t{rows} * (4U * kKvWidth + 2U * kKvHeads * kConvTaps +
            kQueryHeads * kRelativeWidth + 4U * kQueryWidth + kQueryHeads * kRelativeExtent + kQueryHeads
#if defined(__HIPCC__)
            + kQueryWidth / 2U  // the u16 operand image of the prefill projections
#endif
            );
    }

    static constexpr std::size_t route_elements(unsigned rows, unsigned tiles) {
        return std::size_t{rows} * (4U * kExpertsPerToken + 1U) + 2U * kExperts + 2U * tiles + 3U;
    }

    std::vector<attn_core::PreparedGqaGroup> prep_groups_;
#if defined(__HIPCC__)
    std::vector<attn_core::dense::PreparedBf16Lane> lanes_;
    bool verify_lanes_{false};  // verify rows lever (set_verify_lanes): verify windows attend in one lanes launch set
#endif
#if !defined(__HIPCC__)
    std::vector<attn::lanes::PreparedBf16Lane> lane_scratch_;  // cohort attention lanes of one visit
#endif
    exec::WidthStepGraphs cohort_graphs_;  // shared step descriptor, CUDA graph per width
    std::optional<MiniCohortLanes> cohort_lanes_;  // set while a cohort is captured or submitted eagerly
    const MiniResidentWeights& weights_;
    mini_gpu::Stream stream_;
    unsigned max_rows_;
    bool packed_decode_;
    bool cohort_rows_;  // packed cohort decode passes of up to max_rows_ lanes (mini_packed_cohort_rows)
    bool grouped_prefill_ = false;  // keep_prefill_grouped (served held-back route, CUDA)
#if !defined(__HIPCC__)
    moe_core::grouped_microscaled::Decode moe_decode_ = mini_cuda_moe_decode();
    moe_core::grouped_microscaled::Schedule moe_schedule_ = mini_cuda_moe_schedule();
    bool moe_w4a4_ = false, chunk_w4a4_ = false;
    bool attention_projected_bias_ = false;  // lever inkling2-mini-attn-projected-bias@1 (resolved; set once after load)
    std::uint64_t projected_dispatches_ = 0;  // projected-bias appends submitted (projected_attention_dispatches)
    moe_core::grouped_w4a4::Activations moe_w4a4_class_ = moe_core::grouped_w4a4::Activations::nvfp4;
    std::array<bool, kPhysicalLayers> moe_w4a4_layer_{};
    std::optional<mini_gpu::Buffer<std::uint8_t>> moe_nvfp4_;
#else
    bool moe_iu4_ = false, iu4_on_ = false, chunk_iu4_ = false;  // images built, class selected, this chunk
    std::array<moe_core::grouped_iu4a4::LayerImages, kPhysicalLayers> iu4_images_{};
    std::deque<mini_gpu::Buffer<std::uint8_t>> iu4_codes_;  // deque: Buffer is not movable
    std::deque<mini_gpu::Buffer<float>> iu4_steps_;
    std::optional<mini_gpu::Buffer<std::uint8_t>> moe_nvfp4_;  // A4 activation images of a pass (grouped_iu4a4::act_bytes)
#endif
    core::LowBitArithmetic lowbit_arithmetic_;
    bool step_graphs_;
    unsigned route_flag_rows_ = 1U;
    std::vector<unsigned> host_flags_;  // host copy of the route / attention status words of the last call
    unsigned chunk_rows_ = 0U;
#if defined(__HIPCC__)
    // Fused decode transitions (HIP), read once: the residual bank batch of each pass's controller messages
    // (STRIX_RESIDUAL_BANK_BATCH=0 keeps one readout per controller) and the MoE tail (STRIX_MOE_TAIL=0).
    bool bank_batch_ = env_flag_default_on("STRIX_RESIDUAL_BANK_BATCH");
    bool moe_tail_ = mini_moe_tail_switch();
    // The bank whose message rows the last batch produced (the controller scratch after one controller row).
    std::array<const float*, 2> batched_bank_{};
#endif
    MiniResidentRoute route_{};
    bool gate_order_null_{};
    bool prepared_round_fold_{};
    std::uint64_t null_applications_{};
    std::optional<unsigned> step_token_;
    strix::moe::RouteTap* gate_route_tap_{};
    bool lowbit_chunk_{};
    weight::calib::MomentSet* calibration_{};
    std::optional<std::size_t> calibration_window_;
    std::vector<float> calibration_rows_;
    [[nodiscard]] bool host_taps_off() const noexcept { return !gate_route_tap_ && !gate_order_null_ && !calibration_; }
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
    // grouped_dispatch::Workspace::nan_seen: zeroed once; raised (sticky) by a down that stored a NaN sum.
    // Allocated by set_combine_image; no word, no image route.
    std::optional<mini_gpu::Buffer<unsigned>> moe_nan_seen_;  // only while the down-image lever is active
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
