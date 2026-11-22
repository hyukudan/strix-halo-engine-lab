#include "executor_cpu_binding.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strix::exec {
namespace {

constexpr std::uint32_t max_steps = 8;
constexpr RowOut masked{0, 0.0F, 0.0F, RowStatus::masked};
using Clock = std::chrono::steady_clock;

CpuReferenceFn reference_from(std::span<const std::byte> args) {
    CpuReferenceFn reference = nullptr;
    if (args.size() >= sizeof(reference)) std::memcpy(&reference, args.data(), sizeof(reference));
    if (!reference) throw std::invalid_argument("missing CPU reference body");
    return reference;
}

void reference_thunk(std::span<const std::byte> args, const Grid& grid, void* context) {
    const auto reference = reference_from(args);
    reference(args.subspan(sizeof(reference)), grid, *static_cast<const CpuLaunchContext*>(context));
}

std::span<const std::byte> launch_args(const Program& program, const BoundLaunch& launch) {
    return std::span(program.args).subspan(launch.args_offset, launch.args_bytes);
}

void validate_program(const Program& program, std::size_t workspace_bytes) {
    if (program.launches.empty() || program.arena.total_bytes > workspace_bytes)
        throw std::invalid_argument("empty CPU program or insufficient arena storage");
    for (std::size_t i = 0; i < program.launches.size(); ++i) {
        const auto& launch = program.launches[i];
        if (launch.fn != reference_thunk || launch.identity.empty() || launch.args_offset > program.args.size() ||
            launch.args_bytes > program.args.size() - launch.args_offset || launch.rows != program.key.rows ||
            (launch.after != no_launch && launch.after >= i))
            throw std::invalid_argument("invalid CPU program launch binding: " + launch.entry);
        (void)reference_from(launch_args(program, launch));
    }
}

bool advances(const ProgramKey& key) {
    return key.phase == Phase::decode && !key.tree;
}

StepBlock stage_input(const Program& program, const CpuStepIn& input, const StepStopTable& stops,
                      std::uint32_t steps) {
    const auto width = program.key.rows;
    if (width == 0 || width > kStepMaxRows || program.descriptor.rows != width || input.rows.size() > width ||
        input.stops.size() > kStepMaxStops || steps == 0 || steps > max_steps ||
        (advances(program.key) && !std::has_single_bit(width)) ||
        (steps > 1 && (!advances(program.key) || program.key.sampled || input.feed == 0)))
        throw std::invalid_argument("invalid CPU step dimensions");

    StepBlock block{};
    block.h.rows = width;
    block.h.step = input.step;
    block.h.sampling_policy = input.sampling_policy;
    block.h.stop_count = static_cast<std::uint32_t>(input.stops.size());
    if (block.h.stop_count != 0) set_step_stop_address(block.h, reinterpret_cast<std::uintptr_t>(&stops));
    for (std::uint32_t i = 0; i < input.rows.size(); ++i) {
        auto& row = block.row[i];
        row = input.rows[i];
        if ((row.flags & step_valid) == 0) continue;
        const auto bit = std::uint32_t{1} << i;
        if (advances(program.key) && (row.parent != kStepNoParent || (row.ancestors & ~bit) != 0 ||
            (row.flags & step_pending_state) != 0))
            throw std::invalid_argument("invalid independent CPU decode row");
        block.h.valid_mask |= bit;
    }
    if ((input.feed & ~block.h.valid_mask) != 0) throw std::invalid_argument("invalid CPU step feed mask");
    block.h.feed_mask = steps > 1 ? input.feed : 0;
    return block;
}

float elapsed_us(Clock::time_point begin) {
    return std::chrono::duration<float, std::micro>(Clock::now() - begin).count();
}

}  // namespace

KernelBinding bind_cpu_reference(KernelBinding binding, CpuReferenceFn reference) {
    if (!reference) throw std::invalid_argument("missing CPU reference body");
    std::vector<std::byte> args(sizeof(reference));
    std::memcpy(args.data(), &reference, sizeof(reference));
    args.insert(args.end(), binding.args.begin(), binding.args.end());
    binding.args = std::move(args);
    binding.fn = reference_thunk;
    binding.capturable = false;
    return binding;
}

std::span<std::byte> CpuLaunchContext::arena_row(BufferId id, std::uint32_t row) const {
    const auto& buffer = program_.buffers.at(id);
    const auto& layout = buffer.layout;
    const auto bytes = program_.arena.total_bytes;
    if (row >= step.h.rows || layout.storage != BufferStorage::arena || layout.row_bytes == 0 ||
        layout.row_stride < layout.row_bytes || layout.row_bytes > bytes ||
        buffer.arena_offset > bytes - layout.row_bytes ||
        row > (bytes - layout.row_bytes - buffer.arena_offset) / layout.row_stride)
        throw std::invalid_argument("invalid CPU arena row: " + buffer.symbol);
    const auto offset = buffer.arena_offset + layout.row_stride * row;
    return workspace_.subspan(static_cast<std::size_t>(offset), static_cast<std::size_t>(layout.row_bytes));
}

struct CpuExecutor::Impl {
    struct Completed {
        std::uint64_t serial = 0;
        std::size_t count = 0;
        std::array<RowOut, kStepMaxRows * max_steps> rows{};
    };

    explicit Impl(std::span<std::byte> arena) : workspace(arena) {}

    void prepare(const Program& program, const CpuStepIn& input, std::uint32_t steps) {
        validate_program(program, workspace.size());
        const auto staged = stage_input(program, input, stops, steps);
        StepStopTable staged_stops{};
        std::ranges::copy(input.stops, staged_stops.ids);
        stops = staged_stops;
        block = staged;
    }

    void run(const Program& program, CpuStepProfile* profile = nullptr) {
        const bool advance_decode = advances(program.key);
        committed = 0;
        if (advance_decode) {
            for (std::uint32_t row = 0; row < block.h.rows; ++row)
                if (step_row_valid(block, row) && block.row[row].position == std::numeric_limits<std::uint32_t>::max())
                    throw std::overflow_error("step frontier overflow");
        }
        auto picks = std::span(result).first(block.h.rows);
        std::ranges::fill(picks, masked);
        CpuLaunchContext context(block, picks, program, workspace);
        const auto begin = profile ? Clock::now() : Clock::time_point{};
        for (const auto& launch : program.launches) {
            const auto start = profile ? Clock::now() : Clock::time_point{};
            // CPU bindings use the opaque stream slot for their per-step host context.
            launch.fn(launch_args(program, launch), launch.grid, &context);
            if (profile) profile->launch_us.push_back(elapsed_us(start));
        }
        if (profile) profile->total_us = elapsed_us(begin);

        if (advance_decode) {
            const auto advance = advance_step(block, picks, stops);
            std::ranges::copy(std::span(advance.readback).first(block.h.rows), picks.begin());
            block = advance.next;
            committed = advance.commit_mask;
        } else {
            for (std::uint32_t row = 0; row < block.h.rows; ++row)
                if (!step_row_valid(block, row)) picks[row] = masked;
        }
    }

    std::span<std::byte> workspace;
    StepBlock block{};
    StepStopTable stops{};
    std::array<RowOut, kStepMaxRows> result{};
    std::uint32_t committed = 0;
    std::array<Completed, 8> completed{};
    std::uint64_t serial = 0;
};

CpuExecutor::CpuExecutor(std::span<std::byte> workspace) : impl_(std::make_unique<Impl>(workspace)) {}
CpuExecutor::~CpuExecutor() = default;

const StepBlock* CpuExecutor::descriptor() const noexcept { return &impl_->block; }
const RowOut* CpuExecutor::output() const noexcept { return impl_->result.data(); }
std::uint32_t CpuExecutor::commit_mask() const noexcept { return impl_->committed; }

CpuTicket CpuExecutor::submit(const Program& program, const CpuStepIn& input) {
    return submit_chain(program, input, 1);
}

CpuTicket CpuExecutor::submit_chain(const Program& program, const CpuStepIn& first, std::uint32_t steps) {
    auto& state = *impl_;
    const auto slot = std::ranges::find(state.completed, 0U, &Impl::Completed::serial);
    if (slot == state.completed.end() || state.serial == std::numeric_limits<std::uint64_t>::max())
        throw std::runtime_error("CPU executor has no available ticket");
    state.prepare(program, first, steps);
    const auto rows = program.key.rows;
    for (std::uint32_t i = 0; i < steps; ++i) {
        state.run(program);
        std::ranges::copy(std::span(state.result).first(rows), slot->rows.begin() + i * rows);
    }
    slot->count = steps * rows;
    slot->serial = ++state.serial;
    return {slot->serial};
}

bool CpuExecutor::poll(CpuTicket ticket, std::span<RowOut> out) {
    wait(ticket, out);
    return true;
}

void CpuExecutor::wait(CpuTicket ticket, std::span<RowOut> out) {
    auto& slots = impl_->completed;
    const auto slot = std::ranges::find(slots, ticket.serial, &Impl::Completed::serial);
    if (ticket.serial == 0 || slot == slots.end() || out.size() != slot->count)
        throw std::invalid_argument("invalid CPU ticket or readback size");
    std::ranges::copy(std::span(slot->rows).first(slot->count), out.begin());
    slot->serial = 0;
}

void CpuExecutor::invalidate(std::uint64_t) noexcept {}

CpuStepProfile CpuExecutor::profile(const Program& program, const CpuStepIn& input) {
    impl_->prepare(program, input, 1);
    CpuStepProfile result;
    result.launch_us.reserve(program.launches.size());
    impl_->run(program, &result);
    return result;
}

}  // namespace strix::exec
