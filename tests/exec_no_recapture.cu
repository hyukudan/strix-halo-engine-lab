#include "executor_cuda.hpp"
#include "cuda_core/common.cuh"
#include "strix/device/runtime.hpp"

#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace {

using namespace strix::exec;
using namespace strix::session;
namespace native = strix::device::native;
using strix::cuda_core::pdl_trigger;
using strix::cuda_core::pdl_wait;

constexpr std::uint32_t rows = 2;
constexpr std::uint32_t masked = 1;

struct Args {
    StepReplayBlock* step;
    CudaStepOut* out;
    std::uint32_t* arena;
    std::uint32_t layout_bias;
};

__global__ void seed(Args args) {
    const auto lane = threadIdx.x;
    pdl_trigger();
    const auto row = args.step->rows[lane];
    args.arena[lane] = row.input.position * 17U + row.input.token + 3U * row.slot + args.layout_bias;
}

__global__ void mix(Args args) {
    const auto lane = threadIdx.x;
    (void)pdl_wait();
    pdl_trigger();
    args.arena[lane] = (args.arena[lane] * 31U) ^ args.step->rows[lane].input.position;
}

__global__ void pick(Args args) {
    const auto lane = threadIdx.x;
    const auto bit = std::uint32_t{1} << lane;
    (void)pdl_wait();
    const auto feed = args.step->header.feed_mask;
    bool stop = false;
    if ((args.step->header.valid_mask & bit) == 0) {
        args.out[lane] = {0, 0.0F, 0.0F, masked};
    } else {
        const auto score = args.arena[lane] & 0xFFFFU;
        const auto id = (score ^ args.step->header.step) & 0xFFU;
        args.out[lane] = {id, static_cast<float>(score), static_cast<float>(score ^ 0xFFFFU), 0};
        if ((feed & bit) != 0) {
            auto& input = args.step->rows[lane].input;
            input.token = id;
            ++input.position;
            stop = id == args.step->header.sampling_policy;
        }
    }
    __syncthreads();
    if (stop) {
        atomicAnd(&args.step->header.valid_mask, ~bit);
        args.step->rows[lane].flags &= ~step_replay_valid;
    }
    if (lane == 0 && feed != 0) ++args.step->header.step;
}

template <unsigned Index>
void enqueue(std::span<const std::byte> bytes, const Grid& grid, void* stream) {
    Args args{};
    std::memcpy(&args, bytes.data(), sizeof(args));
    const auto native_stream = static_cast<cudaStream_t>(stream);
    if constexpr (Index == 0 || Index == 3) seed<<<grid.blocks[0], grid.threads[0], 0, native_stream>>>(args);
    if constexpr (Index == 1) mix<<<grid.blocks[0], grid.threads[0], 0, native_stream>>>(args);
    if constexpr (Index == 2) pick<<<grid.blocks[0], grid.threads[0], 0, native_stream>>>(args);
    if constexpr (Index == 3) {
        cudaStreamCaptureStatus status{};
        native::check(cudaStreamIsCapturing(native_stream, &status), "check fixture capture mode");
        if (status == cudaStreamCaptureStatusActive) throw std::runtime_error("fixture intentionally declines capture");
    }
}

Program make_program(CudaExecutor& exec, std::uint32_t* arena, std::uint32_t bias = 0, bool decline = false) {
    const Args args{exec.descriptor(), exec.output(), arena, bias};
    Program program;
    program.key.rows = program.descriptor.rows = rows;
    program.key.graph_id = decline ? std::uint8_t{1} : std::uint8_t{0};
    program.identity = decline ? 2 : 1;
    program.args.resize(sizeof(args));
    std::memcpy(program.args.data(), &args, sizeof(args));
    const std::array<LaunchFn, 3> thunks{decline ? enqueue<3> : enqueue<0>, enqueue<1>, enqueue<2>};
    for (std::uint32_t i = 0; i < thunks.size(); ++i) {
        BoundLaunch launch;
        launch.fn = thunks[i];
        launch.args_bytes = sizeof(args);
        launch.grid.threads[0] = rows;
        launch.after = i == 0 ? no_launch : i - 1;
        program.launches.push_back(std::move(launch));
    }
    return program;
}

void require(bool condition, const char* detail) {
    if (!condition) throw std::runtime_error(detail);
}

template <std::size_t Size>
void equal(const std::array<CudaStepOut, Size>& expected, const std::array<CudaStepOut, Size>& actual) {
    require(std::memcmp(expected.data(), actual.data(), Size * sizeof(CudaStepOut)) == 0, "output bits differ");
}

CudaStepOut expected_row(const StepReplayRow& row, std::uint32_t step, std::uint32_t bias = 0) {
    if ((row.flags & step_replay_valid) == 0) return {0, 0.0F, 0.0F, masked};
    const auto base = row.input.position * 17U + row.input.token + 3U * row.slot + bias;
    const auto score = ((base * 31U) ^ row.input.position) & 0xFFFFU;
    return {(score ^ step) & 0xFFU, static_cast<float>(score), static_cast<float>(score ^ 0xFFFFU), 0};
}

std::array<StepReplayRow, rows> input_rows(std::uint32_t step) {
    return {{{{step + 1U, 2U * step + 10U}, step % 7U, 0, 0, step_replay_valid, 0, 0},
             {{2U * step + 10U, step + 20U}, step % 13U, 0, 0, step_replay_valid, 0, 0}}};
}

void compare_outputs(CudaExecutor& eager, CudaTicket a, CudaExecutor& replay, CudaTicket b,
                     std::uint32_t step, std::uint32_t bias = 0) {
    const auto input = input_rows(step);
    std::array<CudaStepOut, rows> expected{}, actual{}, oracle{};
    for (std::size_t i = 0; i < rows; ++i)
        oracle[i] = i <= step % rows ? expected_row(input[i], step, bias) : CudaStepOut{0, 0.0F, 0.0F, masked};
    eager.wait(a, expected);
    if (!replay.poll(b, actual)) replay.wait(b, actual);
    equal(oracle, expected);
    equal(expected, actual);
}

void compare_step(CudaExecutor& eager, const Program& eager_program, CudaExecutor& replay,
                  const Program& replay_program, std::uint32_t step, std::uint32_t bias = 0) {
    const auto input = input_rows(step);
    const CudaStepIn in{std::span(input).first(step % rows + 1U), 0, step};
    compare_outputs(eager, eager.submit(eager_program, in), replay, replay.submit(replay_program, in), step, bias);
}

void run() {
    native::Buffer<std::uint32_t> eager_arena(rows), replay_arena(rows), moved_arena(rows);
    CudaExecutor eager(false, false), replay(true);
    const auto eager_program = make_program(eager, eager_arena.data());
    const auto replay_program = make_program(replay, replay_arena.data());
    std::array<CudaTicket, 3> eager_tickets{}, replay_tickets{};
    for (std::uint32_t step = 0; step < 3; ++step) {
        const auto input = input_rows(step);
        const CudaStepIn in{std::span(input).first(step % rows + 1U), 0, step};
        eager_tickets[step] = eager.submit(eager_program, in);
        replay_tickets[step] = replay.submit(replay_program, in);
    }
    for (std::uint32_t step = 0; step < 3; ++step)
        compare_outputs(eager, eager_tickets[step], replay, replay_tickets[step], step);
    auto stats = replay.graph_stats(replay_program.key);
    require(stats.captures == 1 && stats.instantiations == 1 && stats.pdl_edges == 2 && stats.nodes == 3,
            "unexpected steady replay graph stats");
    std::puts("exec_no_recapture PASS steps=3 nodes=3 captures=1 pdl_edges=2");

    auto input = input_rows(0);
    const auto stop = expected_row(input.front(), 0).id;
    const auto chain = replay.submit_chain(replay_program, {input, 3, 0, stop}, 8);
    const auto eager_chain = eager.submit_chain(eager_program, {input, 3, 0, stop}, 8);
    std::array<CudaStepOut, 8 * rows> expected{}, actual{}, eager_actual{};
    for (std::uint32_t step = 0; step < 8; ++step) {
        for (std::size_t row = 0; row < rows; ++row) {
            auto& out = expected[step * rows + row];
            out = expected_row(input[row], step);
            if (out.status != masked) {
                input[row].input.token = out.id;
                ++input[row].input.position;
                if (out.id == stop) input[row].flags &= ~step_replay_valid;
            }
        }
    }
    replay.wait(chain, actual);
    eager.wait(eager_chain, eager_actual);
    equal(expected, eager_actual);
    equal(eager_actual, actual);
    require(replay.graph_stats(replay_program.key).captures == 1, "chain recaptured its program");
    std::puts("exec_cuda_chain PASS steps=8 captures=1");

    for (std::uint32_t step = 3; step < 1000; ++step) compare_step(eager, eager_program, replay, replay_program, step);
    require(replay.graph_stats(replay_program.key).captures == 1, "mixed sessions recaptured their program");
    std::puts("exec_cuda_mixed PASS steps=1000 captures=1");

    replay.invalidate(1);
    compare_step(eager, eager_program, replay, replay_program, 7);
    require(replay.graph_stats(replay_program.key).captures == 1, "stable layout change recaptured its program");
    replay.invalidate(2);
    const auto moved = make_program(replay, moved_arena.data(), 29);
    const auto eager_moved = make_program(eager, eager_arena.data(), 29);
    compare_step(eager, eager_moved, replay, moved, 7, 29);
    stats = replay.graph_stats(replay_program.key);
    require(stats.captures == 2 && stats.rebinds == 1 && stats.instantiations == 1, "layout update replaced the executable");
    std::puts("exec_cuda_layout PASS stable_captures=1 direct_rebinds=1 instantiations=1");

    const auto declined = make_program(replay, replay_arena.data(), 0, true);
    compare_step(eager, eager_program, replay, declined, 11);
    compare_step(eager, eager_program, replay, declined, 12);
    stats = replay.graph_stats(declined.key);
    require(stats.captures == 1 && stats.instantiations == 0, "failed capture did not stay eager");
    std::puts("exec_cuda_fallback PASS captures=1 instantiations=0");

    input = input_rows(0);
    const auto profile = replay.profile(moved, {input});
    require(profile.launch_us.size() == 3 && profile.total_us >= 0 &&
            replay.graph_stats(moved.key).captures == 2, "eager profile changed the replay graph");
    std::puts("exec_cuda_profile PASS timed_launches=3");
}

}  // namespace

int main() {
    try {
        run();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "exec_no_recapture FAIL: %s\n", error.what());
        return 1;
    }
}
