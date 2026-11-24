#pragma once

#include "moe_core/device_grouped_plan.hpp"
#include "moe_core/expert_activation.hpp"
#include "moe_core/expert_major_plan.hpp"
#include "moe_core/grouped_rows.hpp"
#include "moe_core/grouped_w4a4.hpp"

#if !defined(__CUDACC__) || defined(__HIPCC__)
#include "moe_core/grouped_microscaled.hpp"
#endif

#if !defined(__HIPCC__)
#include "cuda/core/mma_tile.cuh"
#if defined(__CUDACC__)
#include "cuda/core/dense/row_plan.hpp"
#include "cuda/core/grouped_w4a4.hpp"
#include "strix/cuda_core/dense.hpp"
#endif
#endif

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strix::moe_core::grouped_dispatch {

using device_grouped::ExpertWeights;
using grouped_microscaled::TileRows;

#if !defined(__CUDACC__) || defined(__HIPCC__)
struct HipBackend {
    static constexpr unsigned rows_per_tile = grouped_microscaled::kRowsPerTile;
    static constexpr auto fits = grouped_microscaled::fits;
    template<weight::microscaled::MatrixFormat Format>
    static constexpr auto fits_matrix_tile = grouped_microscaled::fits_tile<Format>;

#if defined(__HIPCC__)
    using Status = hipError_t;
    using Stream = hipStream_t;
    static constexpr auto success = hipSuccess;
    static constexpr auto invalid_value = hipErrorInvalidValue;
    static constexpr auto gather = grouped_rows::gather;
    // Gate-up reads token rows in place (grouped_microscaled::IndirectArgs): one BF16 image instead of top_k copies.
    static constexpr auto token_rows = grouped_rows::token_rows;
    template<weight::microscaled::MatrixFormat Format>
    static constexpr auto launch_projection = grouped_microscaled::launch<Format>;
    template<weight::microscaled::MatrixFormat Format>
    static constexpr auto launch_activated = grouped_microscaled::launch_activated<Format>;
    template<weight::microscaled::MatrixFormat Format>
    static constexpr auto launch_indirect = grouped_microscaled::launch_indirect<Format>;
    template<weight::microscaled::MatrixFormat Format>
    static constexpr auto launch_activated_indirect = grouped_microscaled::launch_activated_indirect<Format>;
    template<unsigned Experts, unsigned TopK>
    static constexpr auto materialize = device_grouped::enqueue<Experts, TopK>;
#endif
};
#endif

#if !defined(__HIPCC__)
struct CudaBackend {
    static constexpr unsigned rows_per_tile = cuda::core::mma::microscaled::kRowsPerTile;
    static constexpr auto fits = cuda::core::mma::microscaled::fits;
    template<weight::microscaled::MatrixFormat Format>
    static constexpr auto fits_matrix_tile = cuda::core::mma::microscaled::fits_tile<Format>;

#if defined(__CUDACC__)
    using Status = cudaError_t;
    using Stream = cudaStream_t;
    static constexpr auto success = cudaSuccess;
    static constexpr auto invalid_value = cudaErrorInvalidValue;
    static constexpr auto gather = grouped_rows::gather;
    template<weight::microscaled::MatrixFormat Format>
    static constexpr auto launch_projection = cuda::core::mma::microscaled::launch<Format>;
    template<unsigned Experts, unsigned TopK>
    static constexpr auto materialize = device_grouped::enqueue<Experts, TopK>;
    // FAST W4A4 operand class (moe_core/grouped_w4a4.hpp; sm_120 block-scaled FP4).
    static cudaError_t quantize_w4a4(const grouped_w4a4::QuantizeArgs<float>& a, cudaStream_t s, grouped_w4a4::ScaleRule r) {
        return cuda::core::grouped_w4a4::quantize(a, s, r);
    }
    static cudaError_t quantize_w4a4(const grouped_w4a4::QuantizeArgs<std::uint16_t>& a, cudaStream_t s,
                                     grouped_w4a4::ScaleRule r) {
        return cuda::core::grouped_w4a4::quantize(a, s, r);
    }
    static cudaError_t launch_w4a4(const grouped_w4a4::Args& a, cudaStream_t s) { return cuda::core::grouped_w4a4::launch(a, s); }
#endif
};
#endif

#if defined(__CUDACC__) && !defined(__HIPCC__)
// Preserve FP32 router projection accumulation and row order.
template<class Config>
inline cudaError_t cuda_router_logits(const cuda_core::DenseWeight& router, unsigned rows,
                                      std::span<const float> input, std::span<float> logits, cudaStream_t stream) {
    const auto input_elements = std::size_t{rows} * Config::hidden;
    const auto output_elements = std::size_t{rows} * Config::experts;
    if (rows == 0U || rows > Config::max_rows || !router.data ||
        router.in != Config::hidden || router.out != Config::experts ||
        (router.format != cuda_core::WeightFormat::bf16 && router.format != cuda_core::WeightFormat::f32) ||
        input.size() < input_elements || logits.size() < output_elements) return cudaErrorInvalidValue;
    for (unsigned first = 0; first < rows;) {
        const unsigned count = std::min(cuda_core::kFp32GridMaxRows, rows - first);
        const auto status = cuda_core::dense_fp32_rows_grid(
            router, input.data() + std::size_t{first} * Config::hidden,
            logits.data() + std::size_t{first} * Config::experts, count, stream);
        if (status != cudaSuccess) return status;
        first += count;
    }
    return cudaSuccess;
}
#endif

struct Workspace {
    // Gather/activation reuse BF16, gate-up/down reuse FP32, in stream order.
    std::span<std::uint16_t> bf16;
    std::span<float> fp32;
    // Optional separate BF16 activation rows (assignments x intermediate, 16-byte aligned). Where the backend has an
    // activated gate-up and this holds the rows, gate-up stores the activations here as its epilogue (no FP32
    // [gate | up] image, no activation launch) and down reads them here; otherwise the two-step form runs.
    std::span<std::uint16_t> activation{};
    // Optional NVFP4 image of the assignment rows (256-byte aligned, WorkspaceElements::nvfp4 bytes) for the FAST W4A4 operand
    // class (DevicePlan::activations = nvfp4); both projections reuse it in stream order.
    std::span<std::uint8_t> nvfp4{};
};

struct WorkspaceElements {
    std::size_t bf16;
    std::size_t fp32;
    std::size_t activation{};  // Workspace::activation rows for the epilogue form.
    std::size_t nvfp4{};       // Workspace::nvfp4 bytes for the W4A4 operand class.
};

template<class Config>
constexpr WorkspaceElements workspace_requirement(unsigned rows) {
    static_assert(Config::hidden > 0U && Config::intermediate > 0U &&
                  Config::intermediate <= std::numeric_limits<unsigned>::max() / 2U);
    static_assert(Config::top_k > 0U && Config::top_k <= Config::experts);
    constexpr auto fp32_width = std::max(Config::hidden, 2U * Config::intermediate);
    const auto assignments = std::uint64_t{rows} * Config::top_k;
    if (rows == 0U || rows > Config::max_rows || assignments > std::numeric_limits<unsigned>::max() ||
        assignments > std::numeric_limits<std::size_t>::max() / sizeof(float) / fp32_width) {
        return {};
    }
    const auto count = static_cast<std::size_t>(assignments);
    return {count * std::max(Config::hidden, Config::intermediate), count * fp32_width, count * Config::intermediate,
            std::max(grouped_w4a4::image_layout(count, Config::hidden).bytes,
                     grouped_w4a4::image_layout(count, Config::intermediate).bytes)};
}

template<class Config>
inline WorkspaceElements workspace_elements(unsigned rows) {
    const auto need = workspace_requirement<Config>(rows);
    if (need.bf16 == 0U) throw std::invalid_argument("grouped MoE: invalid row count");
    return need;
}

template<class Config, class Backend>
inline bool fits_tile(unsigned layer, typename Config::Projection projection, const TileRows& tile) {
    using Projection = typename Config::Projection;
    using weight::microscaled::MatrixFormat;
    if (layer >= Config::layers || (projection != Projection::gate_up && projection != Projection::down) ||
        tile.weights.row_order != Config::row_order(projection)) return false;
    const bool gate_up = projection == Projection::gate_up;
    const unsigned inputs = gate_up ? Config::hidden : Config::intermediate;
    const unsigned outputs = gate_up ? 2U * Config::intermediate : Config::hidden;
    switch (Config::format(layer)) {
    case MatrixFormat::nvfp4: return Backend::template fits_matrix_tile<MatrixFormat::nvfp4>(tile, inputs, outputs);
    case MatrixFormat::mxfp8: return Backend::template fits_matrix_tile<MatrixFormat::mxfp8>(tile, inputs, outputs);
    }
    return false;
}

// IndirectArgs (input rows read in place) select the backend's indirect launchers.
template<class A>
inline constexpr bool in_place_rows = std::is_same_v<A, grouped_microscaled::IndirectArgs>;

template<class Config, class Backend, class A = grouped_microscaled::Args>
inline typename Backend::Status project(unsigned layer, const A& args, typename Backend::Stream stream) {
    using weight::microscaled::MatrixFormat;
    switch (Config::format(layer)) {
    case MatrixFormat::nvfp4:
        if constexpr (in_place_rows<A>) return Backend::template launch_indirect<MatrixFormat::nvfp4>(args, stream);
        else return Backend::template launch_projection<MatrixFormat::nvfp4>(args, stream);
    case MatrixFormat::mxfp8:
        if constexpr (in_place_rows<A>) return Backend::template launch_indirect<MatrixFormat::mxfp8>(args, stream);
        else return Backend::template launch_projection<MatrixFormat::mxfp8>(args, stream);
    }
    return Backend::invalid_value;
}

// A backend with an activated gate-up projection and a configuration whose activation exposes its caps.
template<class Config, class Backend>
inline constexpr bool activation_epilogue = requires {
    Backend::template launch_activated<weight::microscaled::MatrixFormat::nvfp4>;
    Config::Activation::gate_cap;
    Config::Activation::up_cap;
};

template<class Config, class Backend, class A = grouped_microscaled::Args>
inline typename Backend::Status project_activated(unsigned layer, const A& args, typename Backend::Stream stream) {
    using weight::microscaled::MatrixFormat;
    if constexpr (activation_epilogue<Config, Backend>) {
        switch (Config::format(layer)) {
        case MatrixFormat::nvfp4:
            if constexpr (in_place_rows<A>)
                return Backend::template launch_activated_indirect<MatrixFormat::nvfp4>(args, stream);
            else return Backend::template launch_activated<MatrixFormat::nvfp4>(args, stream);
        case MatrixFormat::mxfp8:
            if constexpr (in_place_rows<A>)
                return Backend::template launch_activated_indirect<MatrixFormat::mxfp8>(args, stream);
            else return Backend::template launch_activated<MatrixFormat::mxfp8>(args, stream);
        }
    } else {
        (void)layer, (void)args, (void)stream;
    }
    return Backend::invalid_value;
}

struct HostPlan {
    unsigned physical_layer;
    unsigned rows;
    ExpertMajorPlan routes;
    std::vector<TileRows> gate_up;
    std::vector<TileRows> down;
};

template<class Config, class Backend>
inline HostPlan prepare(unsigned layer, unsigned rows, std::span<const std::uint32_t> experts,
                        std::span<const float> weights, std::span<const ExpertWeights> matrices,
                        Workspace workspace) {
    using Projection = typename Config::Projection;
    const auto need = workspace_elements<Config>(rows);
    if (layer >= Config::layers || matrices.size() != Config::experts ||
        workspace.bf16.size() < need.bf16 || workspace.fp32.size() < need.fp32) {
        throw std::invalid_argument("grouped MoE: invalid layer, weights or workspace");
    }
    auto routes = build_expert_major_plan({rows, Config::top_k, Config::experts, 0, experts},
                                          weights, Backend::rows_per_tile);
    if (routes.tiles.size() > grouped_microscaled::kMaxTiles) {
        throw std::invalid_argument("grouped MoE: chunk exceeds the backend launch grid");
    }
    HostPlan plan{layer, rows, std::move(routes), {}, {}};
    plan.gate_up.reserve(plan.routes.tiles.size());
    plan.down.reserve(plan.routes.tiles.size());
    constexpr unsigned gate_width = 2U * Config::intermediate;
    for (const auto& tile : plan.routes.tiles) {
        const auto& matrix = matrices[tile.expert];
        const auto first = std::size_t{tile.first_assignment};
        const auto hidden_offset = first * Config::hidden;
        const TileRows gate_up{matrix.gate_up, workspace.bf16.data() + hidden_offset,
                               workspace.fp32.data() + first * gate_width,
                               Config::hidden, gate_width, tile.rows};
        const TileRows down{matrix.down, workspace.bf16.data() + first * Config::intermediate,
                            workspace.fp32.data() + hidden_offset, Config::intermediate, Config::hidden, tile.rows};
        if (!fits_tile<Config, Backend>(layer, Projection::gate_up, gate_up) ||
            !fits_tile<Config, Backend>(layer, Projection::down, down)) {
            throw std::invalid_argument("grouped MoE: incompatible expert descriptor");
        }
        plan.gate_up.push_back(gate_up);
        plan.down.push_back(down);
    }
    return plan;
}

struct DevicePlan {
    unsigned physical_layer;
    unsigned rows;
    std::span<const std::uint32_t> assignment_rows;
    std::span<const std::uint32_t> combine_rows;
    std::span<const float> assignment_weights;
    std::span<const TileRows> gate_up;
    std::span<const TileRows> down;
    const unsigned* tile_count = nullptr;
    const unsigned* failure = nullptr;
    grouped_microscaled::Decode decode = grouped_microscaled::Decode::reference;  // weight decode form of both projections
    // Activation operand class (FAST trait): nvfp4 runs NVFP4 layers through the W4A4 tile when the backend has it and
    // Workspace::nvfp4 holds the image; the caller admits the expert matrices once (admit_w4a4). Other layers: BF16 tile.
    grouped_w4a4::Activations activations = grouped_w4a4::Activations::bf16;
    // Schedule of both BF16 projections (never changes bits): automatic = the backend's tall slab schedule where it fits.
    grouped_microscaled::Schedule schedule = grouped_microscaled::Schedule::automatic;
};

// Load-time admission of the W4A4 operand class for one layer's experts (stage copies need 16-byte code rows and 8-byte scale
// rows; shapes need inputs % 128 and outputs % 128). Bit-changing (FAST): grouped-nvfp4-w4a4-f32-v1.
template<class Config>
inline bool admit_w4a4(unsigned layer, std::span<const ExpertWeights> matrices) {
    using weight::microscaled::MatrixFormat;
    if (layer >= Config::layers || Config::format(layer) != MatrixFormat::nvfp4 || matrices.size() != Config::experts)
        return false;
    constexpr unsigned gate_width = 2U * Config::intermediate;
    const auto fits = [](const weight::microscaled::PackedMatrix& w, unsigned inputs, unsigned outputs) {
        return w.rows == outputs && w.columns == inputs && outputs % grouped_w4a4::kTileOutputs == 0U &&
               inputs % grouped_w4a4::kInputBlock == 0U && w.codes && w.block_scales &&
               reinterpret_cast<std::uintptr_t>(w.codes) % 16U == 0U &&
               reinterpret_cast<std::uintptr_t>(w.block_scales) % 8U == 0U;
    };
    for (const auto& m : matrices)
        if (!fits(m.gate_up, Config::hidden, gate_width) || !fits(m.down, Config::intermediate, Config::hidden)) return false;
    return true;
}

struct RouteBuffers {
    std::span<unsigned> ids, status, scratch;
    std::span<float> weights;
};

enum class LogitExtent { exact, at_least };

template<class Config, class Backend>
inline bool prepare_resident(unsigned layer, unsigned rows, RouteBuffers route,
                             std::span<const ExpertWeights> matrices, device_grouped::CsrTables csr,
                             device_grouped::Output out, Workspace workspace, device_grouped::Args& args) {
    if (layer >= Config::layers || rows == 0U || rows > std::min(Config::max_rows, 65535U) ||
        matrices.size() != Config::experts) return false;
    const device_grouped::Args candidate{
        route.ids.data(), route.weights.data(), route.status.data(), rows, Config::top_k, Config::experts,
        Backend::rows_per_tile, matrices.data(), csr, out,
        {workspace.bf16.data(), workspace.fp32.data(), Config::hidden, Config::intermediate},
    };
    if (!device_grouped::fits(candidate)) return false;
    const auto assignments = std::size_t{rows} * Config::top_k;
    const auto need = workspace_elements<Config>(rows);
    if (route.ids.size() < assignments || route.status.size() < rows || route.weights.size() < assignments ||
        workspace.bf16.size() < need.bf16 || workspace.fp32.size() < need.fp32) return false;
    args = candidate;
    return true;
}

// Upload or build the plan on this stream before launching. Keep all buffers
// alive through completion and check the device failure word before consuming output.
template<class Config, class Backend>
inline typename Backend::Status enqueue(
    const DevicePlan& plan, std::span<const float> input, std::span<float> output,
    Workspace workspace, typename Backend::Stream stream) {
    const auto need = workspace_requirement<Config>(plan.rows);
    if (need.bf16 == 0U) return Backend::invalid_value;
    const unsigned assignments = plan.rows * Config::top_k;
    const auto row_elements = std::size_t{plan.rows} * Config::hidden;
    if (plan.physical_layer >= Config::layers || input.size() < row_elements || output.size() < row_elements ||
        workspace.bf16.size() < need.bf16 || workspace.fp32.size() < need.fp32 ||
        plan.assignment_rows.size() != assignments || plan.combine_rows.size() != assignments ||
        plan.assignment_weights.size() != assignments || plan.gate_up.size() != plan.down.size() ||
        plan.gate_up.empty() || plan.gate_up.size() > grouped_microscaled::kMaxTiles) return Backend::invalid_value;

    constexpr unsigned gate_width = 2U * Config::intermediate;
    const auto tiles = static_cast<unsigned>(plan.gate_up.size());
    const unsigned max_tile_rows = std::min(plan.rows, Backend::rows_per_tile);
    grouped_microscaled::Args gate_up{plan.gate_up.data(), tiles, Config::hidden, gate_width,
                                     plan.tile_count, plan.failure, max_tile_rows};
    grouped_microscaled::Args down{plan.down.data(), tiles, Config::intermediate, Config::hidden,
                                  plan.tile_count, plan.failure, max_tile_rows};
    gate_up.mean_tile_rows = down.mean_tile_rows = (assignments + Config::experts - 1U) / Config::experts;
    gate_up.decode = down.decode = plan.decode;
    gate_up.schedule = down.schedule = plan.schedule;
    if (!Backend::fits(gate_up) || !Backend::fits(down)) return Backend::invalid_value;
    // A backend whose gate-up reads token rows in place writes the rows x hidden BF16 image of the token rows at
    // workspace.bf16 (the base the plan's gathered addresses are relative to) instead of top_k gathered copies; each
    // tile row then reads bf16_encode of the FP32 element the gather would have copied (same operand bits).
    constexpr bool in_place = requires { Backend::token_rows; };
    [[maybe_unused]] const auto rows_in_place = [&](const grouped_microscaled::Args& args) {
        grouped_microscaled::IndirectArgs indirect{args, plan.assignment_rows.data(), workspace.bf16.data()};
        return indirect;
    };
    typename Backend::Status status;
    if constexpr (requires { &Backend::launch_w4a4; }) {
        // FAST W4A4 operand class: quantise the gathered rows once per projection input (FP32 token rows by assignment, then
        // the BF16 activation rows) with the class's scale rule, run the projection(s) of the class on the block-scaled FP4
        // tile (nvfp4_mse_gate_up: the down projection reads the BF16 activation rows as the bf16 class does); plan, tiles
        // and outputs unchanged.
        if (grouped_w4a4::w4a4_gate_up(plan.activations) &&
            Config::format(plan.physical_layer) == weight::microscaled::MatrixFormat::nvfp4 &&
            workspace.nvfp4.size() >= need.nvfp4 && reinterpret_cast<std::uintptr_t>(workspace.nvfp4.data()) % 256U == 0U) {
            const auto rule = grouped_w4a4::scale_rule(plan.activations);
            const auto hidden_image = grouped_w4a4::make_image(workspace.nvfp4.data(), assignments, Config::hidden);
            status = Backend::quantize_w4a4(grouped_w4a4::QuantizeArgs<float>{input.data(), Config::hidden,
                                                plan.assignment_rows.data(), assignments, Config::hidden, hidden_image,
                                                plan.failure}, stream, rule);
            if (status != Backend::success) return status;
            status = Backend::launch_w4a4({gate_up, workspace.bf16.data(), hidden_image}, stream);
            if (status != Backend::success) return status;
            status = Config::Activation::launch({workspace.fp32.data(), workspace.bf16.data(), assignments,
                                                Config::intermediate, gate_width, Config::intermediate, plan.failure}, stream);
            if (status != Backend::success) return status;
            if (!grouped_w4a4::w4a4_down(plan.activations)) {
                status = project<Config, Backend>(plan.physical_layer, down, stream);
                if (status != Backend::success) return status;
                return Config::Combine::launch(
                    {workspace.fp32.data(), Config::hidden, plan.combine_rows.data(), plan.assignment_weights.data(),
                     plan.rows, Config::top_k, Config::hidden, output.data(), Config::hidden, plan.failure}, stream);
            }
            const auto inner_image = grouped_w4a4::make_image(workspace.nvfp4.data(), assignments, Config::intermediate);
            status = Backend::quantize_w4a4(grouped_w4a4::QuantizeArgs<std::uint16_t>{workspace.bf16.data(),
                                                Config::intermediate, nullptr, assignments, Config::intermediate,
                                                inner_image, plan.failure}, stream, rule);
            if (status != Backend::success) return status;
            status = Backend::launch_w4a4({down, workspace.bf16.data(), inner_image}, stream);
            if (status != Backend::success) return status;
            return Config::Combine::launch(
                {workspace.fp32.data(), Config::hidden, plan.combine_rows.data(), plan.assignment_weights.data(),
                 plan.rows, Config::top_k, Config::hidden, output.data(), Config::hidden, plan.failure}, stream);
        }
    }
    if constexpr (in_place) {
        status = Backend::token_rows({input.data(), Config::hidden, nullptr, plan.rows, Config::hidden,
                                      workspace.bf16.data(), Config::hidden, plan.failure}, stream);
    } else {
        status = Backend::gather({input.data(), Config::hidden, plan.assignment_rows.data(), assignments,
                                  Config::hidden, workspace.bf16.data(), Config::hidden, plan.failure}, stream);
    }
    if (status != Backend::success) return status;
    bool epilogue = false;
    if constexpr (activation_epilogue<Config, Backend>)
        epilogue = workspace.activation.size() >= need.activation &&
                   reinterpret_cast<std::uintptr_t>(workspace.activation.data()) % 16U == 0U;
    if (epilogue) {
        // The plan's down inputs address BF16 activation rows in workspace.bf16: rebased into workspace.activation
        // (gate-up stores there, down reads there), so gate-up never writes over gathered rows still being read.
        auto activated = gate_up;
        activated.pair_rows = plan.down.data();
        activated.rebase_from = workspace.bf16.data();
        activated.rebase_to = workspace.activation.data();
        activated.gate_cap = Config::Activation::gate_cap;
        activated.up_cap = Config::Activation::up_cap;
        auto rebased = down;
        rebased.rebase_from = workspace.bf16.data();
        rebased.rebase_to = workspace.activation.data();
        if constexpr (in_place) status = project_activated<Config, Backend>(plan.physical_layer, rows_in_place(activated), stream);
        else status = project_activated<Config, Backend>(plan.physical_layer, activated, stream);
        if (status != Backend::success) return status;
        status = project<Config, Backend>(plan.physical_layer, rebased, stream);
        if (status != Backend::success) return status;
    } else {
        if constexpr (in_place) status = project<Config, Backend>(plan.physical_layer, rows_in_place(gate_up), stream);
        else status = project<Config, Backend>(plan.physical_layer, gate_up, stream);
        if (status != Backend::success) return status;
        status = Config::Activation::launch({workspace.fp32.data(), workspace.bf16.data(), assignments,
                                            Config::intermediate, gate_width, Config::intermediate, plan.failure}, stream);
        if (status != Backend::success) return status;
        status = project<Config, Backend>(plan.physical_layer, down, stream);
        if (status != Backend::success) return status;
    }
    return Config::Combine::launch(
        {workspace.fp32.data(), Config::hidden, plan.combine_rows.data(), plan.assignment_weights.data(),
         plan.rows, Config::top_k, Config::hidden, output.data(), Config::hidden, plan.failure}, stream);
}

#if defined(__HIPCC__) || defined(__CUDACC__)
template<class Caps>
struct SoftcappedSwigluBf16 {
    static constexpr float gate_cap = Caps::gate;
    static constexpr float up_cap = Caps::up;
    static constexpr auto launch = [](const expert_activation::Args& args, auto stream) {
        return expert_activation::launch(args, Caps::gate, Caps::up, stream);
    };
};

struct Bf16Combine {
    static constexpr auto launch = grouped_rows::combine;
};

// The route hook owns selection. Its outputs feed the common CSR planner and
// the same execution path used with an uploaded host plan.
template<class Config, class Backend, class Route>
inline typename Backend::Status enqueue_resident(
    unsigned layer, unsigned rows, std::span<const float> input, std::span<const float> logits,
    std::span<const float> bias, std::span<float> output, RouteBuffers route,
    std::span<const ExpertWeights> matrices, device_grouped::CsrTables csr, device_grouped::Output out,
    Workspace workspace, typename Backend::Stream stream, LogitExtent logit_extent, Route select,
    grouped_microscaled::Decode decode = grouped_microscaled::Decode::reference,
    grouped_w4a4::Activations activations = grouped_w4a4::Activations::bf16,
    grouped_microscaled::Schedule schedule = grouped_microscaled::Schedule::automatic) {
    using Planner = device_grouped::Planner<Config::experts, Config::top_k>;
    device_grouped::Args args{};
    if (!prepare_resident<Config, Backend>(layer, rows, route, matrices, csr, out, workspace, args)) {
        return Backend::invalid_value;
    }
    const auto logit_elements = std::size_t{rows} * Config::experts;
    if (logits.size() < logit_elements || (logit_extent == LogitExtent::exact && logits.size() != logit_elements) ||
        input.size() < std::size_t{rows} * Config::hidden || output.size() < std::size_t{rows} * Config::hidden ||
        bias.size() != Config::experts || route.scratch.size() < Planner::scratch_bytes(rows) / sizeof(unsigned)) {
        return Backend::invalid_value;
    }
    auto status = select(logits, bias, route, rows, stream);
    if (status != Backend::success) return status;
    status = Backend::template materialize<Config::experts, Config::top_k>(args, route.scratch, stream);
    if (status != Backend::success) return status;

    const auto assignments = std::size_t{rows} * Config::top_k;
    const auto tiles = args.out.tile_capacity;
    const DevicePlan plan{layer, rows, {args.csr.rows, assignments}, {args.out.combine_rows, assignments},
                          {args.out.assignment_weights, assignments}, {args.out.gate_up, tiles},
                          {args.out.down, tiles}, args.out.tile_count, args.out.failure, decode, activations, schedule};
    return enqueue<Config, Backend>(plan, input, output, workspace, stream);
}
#endif

}  // namespace strix::moe_core::grouped_dispatch
