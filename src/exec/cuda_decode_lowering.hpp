#pragma once

#include "catalog_cuda_decode.hpp"
#include "cuda_program_recipe.hpp"
#include "moe_core/device_grouped_plan.hpp"
#include "strix/core/rows.hpp"

#include <algorithm>
#include <limits>

#if defined(__CUDACC__) && !defined(__HIPCC__)
#include "moe_core/route_wave_kernel.hpp"
#include "strix/cuda/core/moe_route_tap.hpp"
#include "strix/device/runtime.hpp"
#include <cstring>
#endif

namespace strix::moe { class RouteTap; }

namespace strix::exec::cuda_decode {

// Each resident executor fixes the expert path at load. The recipe key also
// distinguishes that path so a shared recipe cache can use it as its key.
struct DecodeRecipeKey {
    bool grouped = false;
    ProgramKey program{};
    R1Forms forms{};

    auto operator<=>(const DecodeRecipeKey&) const = default;
};

struct CheckedWaveRoute {
    const float* logits;
    const float* bias;
    std::uint32_t* ids;
    float* weights;
    unsigned* failure;
};

// R1 gate-up that routes its own row (moe_core/r1_route.hpp Inline): one launch in place of CheckedWaveRoute +
// PackedGateUp. The route is the checked route body on `logits` / `bias`; it is published to ids / weights / failure,
// which must be the args route buffers down_combine reads (microscaled_r1::admits_route).
struct RoutedPackedGateUp {
    moe_core::microscaled_r1::Args args;
    weight::microscaled::MatrixFormat format;
    const float* logits;
    const float* bias;
    std::uint32_t* ids;
    float* weights;
    unsigned* failure;
};

// Identity of a routed gate-up binding: format, route policy identity (the model's route name) and the PDL entry.
[[nodiscard]] inline std::string routed_gate_up_identity(weight::microscaled::MatrixFormat format,
                                                         std::string_view route) {
    std::string id = "cuda-sm120-packed-";
    id += format == weight::microscaled::MatrixFormat::nvfp4 ? "nvfp4" : "mxfp8";
    id += "-gate-up-routed-";
    id += route;
    id += "-r1-k8-pdl-v1";
    return id;
}

struct RouteTapExchange {
    moe::RouteTap* tap;
    unsigned layer, rows, top_k, experts;
    unsigned* ids;
    float* weights;
};

enum class GroupedPhase { count, scan, scatter, materialize };

inline constexpr std::array grouped_phase_names{
    "cuda-grouped-csr-count-v1", "cuda-grouped-csr-scan-v1", "cuda-grouped-csr-scatter-v1",
    "cuda-grouped-materialize-v1",
};

struct GroupedPlanLeaf {
    moe_core::device_grouped::Args args;
    unsigned* scratch;
    GroupedPhase phase;
};

struct GroupedDecodePlan {
    moe_core::device_grouped::Args args;
    std::span<unsigned> scratch;
    unsigned* route_status;
};

// Scratch added to the existing resident route and BF16 activation images.
// R1 needs at most one tile per selected expert.
struct GroupedDecodeScratch {
    std::span<unsigned> words;
    std::span<float> values;
    std::span<moe_core::device_grouped::TileRows> descriptors;

    [[nodiscard]] static constexpr std::size_t word_count(unsigned experts, unsigned top_k) {
        return 4U * std::size_t{experts} + 5U * std::size_t{top_k} + 6U;
    }

    [[nodiscard]] static constexpr std::size_t value_count(unsigned hidden, unsigned intermediate, unsigned top_k) {
        return std::size_t{top_k} * (1U + std::max(hidden, 2U * intermediate));
    }

    [[nodiscard]] GroupedDecodePlan bind(unsigned experts, unsigned top_k, unsigned rows_per_tile,
                                         const moe_core::device_grouped::ExpertWeights* matrices,
                                         unsigned* ids, float* weights, unsigned* failure,
                                         std::uint16_t* bf16, unsigned hidden, unsigned intermediate) const {
        if (words.empty() || words.size() != word_count(experts, top_k) ||
            values.size() != value_count(hidden, intermediate, top_k) || descriptors.size() != 2U * top_k)
            throw std::invalid_argument("invalid CUDA grouped decode scratch");
        auto bits = words;
        const auto take = [&](std::size_t count) {
            auto* ptr = bits.data();
            bits = bits.subspan(count);
            return ptr;
        };
        auto* route_status = take(1U);
        const moe_core::device_grouped::CsrTables csr{
            take(experts + 1U), take(experts), take(top_k), take(top_k + 1U), take(top_k), take(top_k),
        };
        auto* combine = take(top_k);
        auto* tile_count = take(1U);
        const moe_core::device_grouped::Args plan{
            ids, weights, route_status, 1U, top_k, experts, rows_per_tile, matrices, csr,
            {values.data(), combine, descriptors.data(), descriptors.data() + top_k, tile_count, failure, top_k},
            {bf16, values.data() + top_k, hidden, intermediate},
        };
        return {plan, bits, route_status};
    }
};

#if defined(__CUDACC__) && !defined(__HIPCC__)
template <class Policy, class Math, class Traits>
[[nodiscard]] KernelBinding bind_checked_wave(const CheckedWaveRoute& args, std::string_view identity) {
    KernelBinding binding;
    binding.fn = +[](std::span<const std::byte> bytes, const Grid&, void* stream) {
        if (bytes.size() != sizeof(CheckedWaveRoute)) throw std::invalid_argument("invalid checked wave route binding");
        CheckedWaveRoute route{};
        std::memcpy(&route, bytes.data(), sizeof(route));
        device::native::check(moe_core::enqueue_route_wave<Policy, Math, Traits>(
            route.logits, route.bias, route.ids, route.weights, route.failure, 1U,
            static_cast<cudaStream_t>(stream)), "enqueue checked CUDA wave route");
    };
    binding.identity = binding.exact_identity = identity;
    binding.grid = {{1U, 1U, 1U}, {moe_core::kCheckedRouteRowsPerBlock * 32U, 1U, 1U}, 0U};
    binding.pdl = {PdlWait::absent, PdlOptOut::kernel_without_wait};
    const auto bytes = std::as_bytes(std::span(&args, 1U));
    binding.args.assign(bytes.begin(), bytes.end());
    binding.predicted.launches = 1U;
    return binding;
}

template <class Policy, class Math, class Traits>
[[nodiscard]] KernelBinding bind_routed_gate_up(const RoutedPackedGateUp& args, std::string_view route) {
    namespace r1 = moe_core::microscaled_r1;
    using Format = weight::microscaled::MatrixFormat;
    using Route = moe_core::r1_route::Inline<Policy, Math, Traits>;
    constexpr unsigned top_k = kPackedExpertsPerToken;
    const Route source{args.logits, args.bias, args.ids, args.weights, args.failure};
    const bool fits = args.format == Format::nvfp4 ? r1::fits<Format::nvfp4, top_k>(args.args)
                                                    : args.format == Format::mxfp8 && r1::fits<Format::mxfp8, top_k>(args.args);
    if (route.empty() || !fits || !r1::admits_route<top_k>(args.args, source, 1U))
        throw std::invalid_argument("invalid CUDA R1 routed gate-up");
    KernelBinding binding;
    binding.fn = +[](std::span<const std::byte> bytes, const Grid&, void* stream) {
        if (bytes.size() != sizeof(RoutedPackedGateUp)) throw std::invalid_argument("invalid routed gate-up binding");
        RoutedPackedGateUp leaf{};
        std::memcpy(&leaf, bytes.data(), sizeof(leaf));
        const Route route_source{leaf.logits, leaf.bias, leaf.ids, leaf.weights, leaf.failure};
        const auto queue = static_cast<cudaStream_t>(stream);
        const auto status = leaf.format == Format::nvfp4
            ? r1::launch_gate_up_routed<Format::nvfp4, top_k, r1::EntryPdl>(leaf.args, route_source, queue)
            : r1::launch_gate_up_routed<Format::mxfp8, top_k, r1::EntryPdl>(leaf.args, route_source, queue);
        device::native::check(status, "enqueue CUDA R1 routed gate-up");
    };
    binding.identity = binding.exact_identity = routed_gate_up_identity(args.format, route);
    binding.grid = {{r1::gate_blocks(args.args.intermediate, r1::kCudaGeometry.gate_columns), top_k, 1U},
                    {r1::detail::kThreads, 1U, 1U}, 0U};
    // EntryPdl: every CTA waits before its first read (logits are predecessor output), then triggers.
    binding.pdl.wait = PdlWait::before_step_io;
    const auto bytes = std::as_bytes(std::span(&args, 1U));
    binding.args.assign(bytes.begin(), bytes.end());
    binding.predicted.launches = 1U;
    return binding;
}

[[nodiscard]] inline KernelBinding bind_route_tap(const RouteTapExchange& args) {
    KernelBinding binding;
    binding.fn = +[](std::span<const std::byte> bytes, const Grid&, void* stream) {
        if (bytes.size() != sizeof(RouteTapExchange)) throw std::invalid_argument("invalid CUDA route tap binding");
        RouteTapExchange tap{};
        std::memcpy(&tap, bytes.data(), sizeof(tap));
        cuda::core::route_tap(tap.layer, tap.rows, tap.top_k, tap.experts, tap.ids, tap.weights,
                              static_cast<cudaStream_t>(stream), tap.tap);
    };
    binding.identity = binding.exact_identity = "cuda-route-tap-exchange-v1";
    const auto bytes = std::as_bytes(std::span(&args, 1U));
    binding.args.assign(bytes.begin(), bytes.end());
    return binding;
}

template <unsigned Experts, unsigned TopK, GroupedPhase Phase>
void enqueue_grouped_plan(std::span<const std::byte> bytes, const Grid& grid, void* stream) {
    using Policy = moe_core::device_grouped::CsrPolicy<Experts, TopK>;
    if (bytes.size() != sizeof(GroupedPlanLeaf)) throw std::invalid_argument("invalid CUDA CSR binding");
    GroupedPlanLeaf leaf{};
    std::memcpy(&leaf, bytes.data(), sizeof(leaf));
    const auto& a = leaf.args;
    const auto queue = static_cast<cudaStream_t>(stream);
    const auto blocks = grid.blocks[0];
    const auto threads = grid.threads[0];
    if constexpr (Phase == GroupedPhase::count)
        moe_core::device_plan::csr_count_kernel<Policy><<<blocks, threads, 0, queue>>>(
            a, a.rows, Experts, TopK, leaf.scratch);
    if constexpr (Phase == GroupedPhase::scan)
        moe_core::device_plan::csr_scan_kernel<Policy><<<blocks, threads, 0, queue>>>(
            a, a.rows, Experts, TopK, a.rows_per_tile,
            moe_core::device_grouped::Planner<Experts, TopK>::blocks_for(a.rows, TopK), leaf.scratch);
    if constexpr (Phase == GroupedPhase::scatter)
        moe_core::device_plan::csr_scatter_kernel<Policy><<<blocks, threads, 0, queue>>>(
            a, a.rows, TopK, leaf.scratch);
    if constexpr (Phase == GroupedPhase::materialize)
        moe_core::device_grouped::materialize<TopK><<<blocks, threads, 0, queue>>>(a);
    device::native::check(cudaGetLastError(), "enqueue CUDA grouped plan leaf");
}

template <unsigned Experts, unsigned TopK>
[[nodiscard]] KernelBinding bind_grouped_plan(const GroupedPlanLeaf& leaf) {
    using Policy = moe_core::device_grouped::CsrPolicy<Experts, TopK>;
    using Planner = moe_core::device_grouped::Planner<Experts, TopK>;
    if (!moe_core::device_grouped::fits(leaf.args) || !leaf.scratch ||
        leaf.args.experts != Experts || leaf.args.top_k != TopK)
        throw std::invalid_argument("invalid CUDA grouped plan operands");
    constexpr std::array<LaunchFn, 4> launch{
        enqueue_grouped_plan<Experts, TopK, GroupedPhase::count>, enqueue_grouped_plan<Experts, TopK, GroupedPhase::scan>,
        enqueue_grouped_plan<Experts, TopK, GroupedPhase::scatter>, enqueue_grouped_plan<Experts, TopK, GroupedPhase::materialize>,
    };
    const auto phase = static_cast<std::size_t>(leaf.phase);
    const auto n = Planner::blocks_for(leaf.args.rows, TopK);
    const auto materialize = leaf.phase == GroupedPhase::materialize;
    const auto blocks = materialize ? (leaf.args.rows * TopK - 1U) / 256U + 1U :
        (leaf.phase == GroupedPhase::scan ? 1U : n);
    KernelBinding binding;
    binding.fn = launch.at(phase);
    binding.grid = {{blocks, 1, 1}, {materialize ? 256U : Policy::kThreads, 1, 1}, 0};
    binding.identity = binding.exact_identity = grouped_phase_names.at(phase);
    binding.pdl = {PdlWait::absent, PdlOptOut::kernel_without_wait};
    const auto bytes = std::as_bytes(std::span(&leaf, 1U));
    binding.args.assign(bytes.begin(), bytes.end());
    binding.predicted.launches = 1U;
    return binding;
}
#endif

template <class Add, class Clear>
void lower_grouped_decode(const moe_core::device_grouped::Args& plan, std::span<unsigned> scratch,
                          weight::microscaled::MatrixFormat format, const float* input, float* output,
                          float gate_cap, float up_cap, const Add& add, const Clear& clear) {
    if (plan.rows != 1U || scratch.size() != 2U * std::size_t{plan.experts} + 2U)
        throw std::invalid_argument("invalid CUDA R1 grouped plan scratch");
    // PackedPlan::enqueue clears only the fail/status/repeat header before CSR.
    clear(CudaMemsetArgs{scratch.data(), (2U + std::size_t{plan.experts}) * sizeof(unsigned), 0});
    for (const auto phase : {GroupedPhase::count, GroupedPhase::scan, GroupedPhase::scatter, GroupedPhase::materialize})
        add(GroupedPlanLeaf{plan, scratch.data(), phase}, false);

    const auto& w = plan.workspace;
    const auto& out = plan.out;
    const auto gate_width = 2U * w.intermediate;
    add(moe_core::grouped_rows::GatherArgs{input, w.hidden, plan.csr.rows, plan.top_k,
                                          w.hidden, w.bf16, w.hidden, out.failure}, false);
    add(GroupedProjection{{out.gate_up, out.tile_capacity, w.hidden, gate_width, out.tile_count, out.failure, 1U},
                          format}, false);
    add(GroupedActivation{{w.fp32, w.bf16, plan.top_k, w.intermediate, gate_width, w.intermediate, out.failure},
                          gate_cap, up_cap}, false);
    add(GroupedProjection{{out.down, out.tile_capacity, w.intermediate, w.hidden, out.tile_count, out.failure, 1U},
                          format}, false);
    add(moe_core::grouped_rows::CombineArgs{w.fp32, w.hidden, out.combine_rows, out.assignment_weights, plan.rows,
                                           plan.top_k, w.hidden, output, w.hidden, out.failure}, false);
}

// Lower an incumbent homogeneous BF16 projection call. A mixed group issues
// its FP32 stripes first, in declaration order, then the MMA3 stripes.
template <class Add>
void lower_dense(const float* input, unsigned width, std::span<const core::Bf16RowProjection> projections,
                 bool fp32, const Add& add) {
    Bf16DenseGroup scalar{input, width, {}, 0, Bf16DenseMode::fp32};
    Bf16DenseGroup mma{input, width, {}, 0, Bf16DenseMode::mma3};
    if (projections.empty() || projections.size() > scalar.projections.size())
        throw std::invalid_argument("invalid CUDA decode projection count");
    for (const auto& projection : projections) {
        if (!projection.contiguous(width)) throw std::invalid_argument("unsupported CUDA decode BF16 row pitch");
        const bool tiled = !fp32 && width % core::rows::kBlockIn == 0 &&
                           projection.outputs % core::rows::kTileOut == 0;
        auto& group = tiled ? mma : scalar;
        group.projections[group.count++] = projection;
    }
    if (scalar.count) add(scalar, true);
    if (mma.count) {
        if (mma.projections[0].round_bf16) mma.mode = Bf16DenseMode::mma3_round;
        add(mma, true);
    }
}

inline std::size_t controller_scratch_floats(unsigned width, unsigned rank, unsigned rows) {
    constexpr auto max_floats = std::numeric_limits<std::size_t>::max() / sizeof(float);
    const auto planes = rows == 1U ? 4U : 6U;
    if (!rows || !rank || !width || width > std::numeric_limits<unsigned>::max() / 2U ||
        rank > max_floats / rows || width > (max_floats / rows - rank) / planes)
        throw std::invalid_argument("invalid CUDA decode controller dimensions");
    return rows * (planes * std::size_t{width} + rank);
}

// One-row bank readout: the tiled latency form when selected and in shape, else the incumbent.
template <class Add>
void lower_bank_readout(const core::residual_mix::MixArgs& mix, R1Forms forms, const Add& add) {
    if (forms.bank_tiled && fits(BankReadoutTiled{mix})) add(BankReadoutTiled{mix}, true);
    else add(mix, false);
}

// One rounded BF16 projection: the output-split form when selected and in shape, else lower_dense.
template <class Add>
void lower_rounded_projection(const float* input, unsigned width, const core::Bf16RowProjection& projection,
                              R1Forms forms, const Add& add) {
    const Bf16OutputSplitRound split{projection.weights, input, projection.output, width, projection.outputs};
    if (forms.output_split && projection.round_bf16 && fits(split)) {
        add(split, true);
        return;
    }
    const std::array one{projection};
    lower_dense(input, width, one, false, add);
}

// Scratch floats of one controller row: message, forget, gain and conditioned rows of `width`, then the latent.
[[nodiscard]] constexpr std::size_t low_rank_controller_scratch(unsigned width, unsigned rank) {
    return 4U * std::size_t{width} + rank;
}

// The controller after its message readout: `message` is the row the bank readout of this controller produced
// (scratch[0, width) when the readout is lowered here, or a row of a residual bank batch issued once for every
// controller that reads the same bank). Scratch keeps the layout of the full lowering, so both forms bind the same
// leaves with the same operands after the readout.
template <class Add>
void lower_low_rank_controller_tail(unsigned width, unsigned rank, float epsilon, std::span<const float> raw,
                                    const float* message, std::span<float> output, std::span<float> scratch,
                                    const std::uint16_t* down, const std::uint16_t* forget_weights,
                                    const std::uint16_t* gain_weights, const std::uint16_t* norm, const Add& add,
                                    R1Forms forms = {}) {
    const auto stride = 2U * width;
    auto* const forget = scratch.data() + width;
    auto* const gain = forget + width;
    auto* const conditioned = gain + width;
    auto* const latent = conditioned + width;
    add(core::residual_mix::NormalizePairArgs{
        {raw.data(), forget, 1U, width, epsilon, width, stride, true},
        {message, gain, 1U, width, epsilon, width, stride, true}}, false);
    lower_rounded_projection(forget, stride, core::Bf16RowProjection{down, rank, latent, true}, forms, add);
    const std::array gates{core::Bf16RowProjection{forget_weights, width, forget},
                           core::Bf16RowProjection{gain_weights, width, gain}};
    lower_dense(latent, rank, gates, false, add);
    add(core::residual_mix::ControllerArgs{raw.data(), message, forget, gain, conditioned, width}, false);
    add(RmsBf16{conditioned, norm, output.data(), 1U, width, epsilon}, true);
}

template <class Add>
void lower_low_rank_controller(unsigned width, unsigned rank, float epsilon,
                               std::span<const float> query, std::span<const float> raw,
                               std::span<const std::span<const float>> bank, std::span<float> output,
                               std::span<float> scratch, const std::uint16_t* down,
                               const std::uint16_t* forget_weights, const std::uint16_t* gain_weights,
                               const std::uint16_t* norm, const Add& add, unsigned rows) {
    const auto stride = 2U * width;
    const auto required = controller_scratch_floats(width, rank, rows);
    const auto plane = std::size_t{rows} * width;
    if (query.size() < width || raw.size() < plane || output.size() < plane ||
        scratch.size() < required || bank.empty() || bank.size() > core::residual_mix::kMaxBanks ||
        std::ranges::any_of(bank, [plane](auto source) { return source.size() < plane; }))
        throw std::invalid_argument("invalid CUDA decode controller storage");
    auto* const message = scratch.data();
    auto* const forget = message + plane;
    auto* const gain = forget + plane;
    auto* const conditioned = gain + plane;
    auto* const latent = conditioned + plane;
    // Wide steps keep paired norm scratch separate from the packed gate
    // destinations so one row cannot write through another row's padding.
    auto* const paired = rows == 1U ? forget : latent + std::size_t{rows} * rank;
    core::residual_mix::MixArgs mix{};
    mix.queries = query.data();
    mix.output[0] = message;
    mix.rows = rows;
    mix.query_count = 1U;
    mix.width = width;
    mix.bank_count = static_cast<unsigned>(bank.size());
    mix.epsilon = epsilon;
    mix.normalized = true;
    for (std::size_t i = 0; i < bank.size(); ++i) mix.banks[i] = bank[i].data();
    add(mix, false);
    add(core::residual_mix::NormalizePairArgs{
        {raw.data(), paired, rows, width, epsilon, width, stride, true},
        {message, paired + width, rows, width, epsilon, width, stride, true}}, false);
    const std::array down_projection{core::Bf16RowProjection{down, rank, latent, true}};
    lower_dense(paired, stride, down_projection, false, add);
    const std::array gates{core::Bf16RowProjection{forget_weights, width, forget},
                           core::Bf16RowProjection{gain_weights, width, gain}};
    lower_dense(latent, rank, gates, false, add);
    add(core::residual_mix::ControllerArgs{raw.data(), message, forget, gain, conditioned, width}, false);
    add(RmsBf16{conditioned, norm, output.data(), 1U, width, epsilon}, true);
}

// One-row controller with the R1 latency forms (the form the R = 1 recipe binds). The `rows` overload above is
// the width-class (wr1 / wr2) lowering: its recipe has no thunks for the tiled / split forms, so it never selects them.
template <class Add>
void lower_low_rank_controller(unsigned width, unsigned rank, float epsilon,
                               std::span<const float> query, std::span<const float> raw,
                               std::span<const std::span<const float>> bank, std::span<float> output,
                               std::span<float> scratch, const std::uint16_t* down,
                               const std::uint16_t* forget_weights, const std::uint16_t* gain_weights,
                               const std::uint16_t* norm, const Add& add, R1Forms forms = {}) {
    auto* const message = scratch.data();
    core::residual_mix::MixArgs mix{};
    mix.queries = query.data();
    mix.output[0] = message;
    mix.rows = 1U;
    mix.query_count = 1U;
    mix.width = width;
    mix.bank_count = static_cast<unsigned>(bank.size());
    mix.epsilon = epsilon;
    mix.normalized = true;
    for (std::size_t i = 0; i < bank.size(); ++i) mix.banks[i] = bank[i].data();
    lower_bank_readout(mix, forms, add);
    lower_low_rank_controller_tail(width, rank, epsilon, raw, message, output, scratch, down, forget_weights,
                                   gain_weights, norm, add, forms);
}

}  // namespace strix::exec::cuda_decode
