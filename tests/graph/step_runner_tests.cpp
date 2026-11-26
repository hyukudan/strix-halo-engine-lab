#include "strix/exec/dry_capture.hpp"
#include "strix/exec/graph_runner.hpp"
#include "strix/exec/step_lanes.hpp"
#include "strix/session/linear_verify_step.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

// Select the CUDA API spelling and provide the one-node host runtime stand-in.
#define __CUDACC__ 1
#include "strix/device/graph_inspection.hpp"
#undef __CUDACC__

namespace {
namespace ex = strix::exec;

void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

void native_inventory() {
    namespace native = strix::device::native;
    std::size_t count = 0;
    StubGraphNode node{cudaGraphNodeTypeKernel, {}};
    node.kernel.func = &node;
    check(native::inspect_raw_step_graph(&node, count) && count == 1U, "valid native leaf refused");
    node.kernel.blockDim.y = 0;
    check(native::inspect_segment_graph(&node) == cudaErrorInvalidValue, "invalid kernel block accepted");
    node.kind = cudaGraphNodeTypeEventRecord;
    check(!native::inspect_raw_step_graph(&node, count) && native::inspect_segment_graph(&node) == cudaSuccess,
          "events belong to segmented capture");
    node.kind = cudaGraphNodeTypeHost;
    check(native::inspect_segment_graph(&node) == cudaErrorInvalidValue, "host callback in a device segment");
    check(!native::inspect_raw_step_graph(nullptr, count), "empty capture was accepted");
}

// Single-stream fake driver. Capture records addresses and commands only. The work at each
// launch reads a fresh input and copies the resulting rows into its captured tape destination.
struct Runtime {
    using Tape = std::array<std::uint32_t, 2>;
    std::array<Tape*, ex::kStepMaxRows> tapes{};
    Tape result{};
    ex::StepBlock device{};
    bool decline{}, throw_on_launch{};
    unsigned records{}, uploads{}, launches{};

    template <class Work>
    bool capture(int, unsigned at, Work& work) {
        ++records;
        if (decline) return false;
        tapes[at] = work();
        return true;
    }
    void discard(unsigned at) noexcept { tapes[at] = nullptr; }
    void upload(int, const ex::StepBlock& block) {
        ++uploads;
        device = block;
    }
    void launch(int, unsigned at) {
        if (throw_on_launch) throw std::runtime_error("uncertain fake submission");
        ++launches;
        for (unsigned i = 0; i < device.h.rows; ++i)
            result[i] = device.row[i].position + device.row[i].token;
        *tapes.at(at) = result;
    }
};

void replay_windows() {
    ex::StepGraphRunner<Runtime::Tape*> runner{ex::GraphRows::inclusive_window};
    Runtime rt;
    Runtime::Tape first{}, second{};
    auto* destination = &first;
    auto work = [&] { return destination; };
    auto block = strix::session::linear_verify_step(
        {2U, 8U, 0U, 0U, strix::session::LinearVerifyRows::inclusive_valid}, std::array{17U, 23U});
    const auto run = [&] { return runner.run(rt, 0, true, true, block, destination, work); };
    check(run() && first == Runtime::Tape{25, 32}, "initial window tape");
    block.row[0].position = 11;
    block.row[0].token = 4;
    block.row[1].position = 12;
    block.row[1].token = 7;
    check(run() && first == Runtime::Tape{15, 19} && rt.records == 1 && rt.uploads == 2,
          "replay must refresh tape and descriptor");
    destination = &second;
    block.row[0].token = 77;
    check(run() && second == Runtime::Tape{88, 19} && first == Runtime::Tape{15, 19} && rt.records == 2,
          "rebind moved tape without touching the old owner");
    block.h.valid_mask = 0;
    try {
        (void)run();
        throw std::runtime_error("invalid header reached replay");
    } catch (const std::invalid_argument&) {}
    check(rt.launches == 3 && rt.uploads == 3, "invalid header submitted work");
    block.h.valid_mask = 3;
    block.row[1].slot = 1;
    try {
        (void)run();
        throw std::runtime_error("cross-slot verify window reached replay");
    } catch (const std::invalid_argument&) {}
    block.row[1].slot = 0;
    runner.invalidate(rt);
    rt.decline = true;
    check(!run() && !run() && rt.records == 3 && rt.uploads == 3 && rt.launches == 3,
          "declined width should remain eager prior to submission");
    destination = &first;
    rt.decline = false;
    check(run() && rt.records == 4, "new baked binding should retry capture");
    rt.throw_on_launch = true;
    bool threw = false;
    try { (void)run(); } catch (const std::runtime_error&) { threw = true; }
    check(threw && runner.policy().failed(2) && !run(), "uncertain launch must propagate");
    runner.reset(rt);
    rt.throw_on_launch = false;
    check(run(), "reset should retry failed capture");
    runner.reset(rt);
    auto broken = [&]() -> Runtime::Tape* { throw std::logic_error("could not record"); };
    try {
        (void)runner.run(rt, 0, true, true, block, destination, broken);
        throw std::runtime_error("failed recording reached launch");
    } catch (const std::logic_error&) {}
    check(!rt.tapes[1] && runner.policy().failed(2), "partially recorded graph was retained");
}

void replay_stream_programs() {
    ex::StepGraphRunner<Runtime::Tape*> runner{ex::StreamGraphPrograms{{2U, 1U}}};
    Runtime rt;
    Runtime::Tape single_reply{}, two_reply{};
    auto* reply = &single_reply;
    auto work = [&] { return reply; };
    auto single = strix::session::linear_verify_step(
        {2U, 8U, 0U, 0U, strix::session::LinearVerifyRows::inclusive_valid}, std::array{17U, 23U});
    auto two = ex::make_cohort_step(std::array{ex::CohortLane{1U, 7U, 8U}, ex::CohortLane{2U, 10U, 20U}},
                                    2U, 3U, 100U);
    const auto run = [&](const ex::StepBlock& block, std::uint32_t streams) {
        return runner.run_for_streams(rt, 0, true, true, block, reply, work, streams);
    };
    check(!runner.run(rt, 0, true, true, single, reply, work), "width entry point used the stream table");
    check(!run(single, 0U) && !run(single, 3U) && !run(single, 2U) && !run(two, 1U),
          "unlisted count or wrong slot grouping reached capture");
    check(rt.records == 0 && rt.uploads == 0 && rt.launches == 0, "a declined shape reached the device");

    check(run(single, 1U) && single_reply == Runtime::Tape{25U, 32U}, "one stream, two rows was not recorded");
    reply = &two_reply;
    check(run(two, 2U) && two_reply == Runtime::Tape{15U, 30U}, "two streams, one row each was not recorded");
    reply = &single_reply;
    single.row[0].token = 31U;
    single.row[1].token = 32U;
    check(run(single, 1U) && single_reply == Runtime::Tape{39U, 41U} && rt.records == 2 && rt.uploads == 3,
          "equal physical widths aliased or replay did not read its new descriptor");
    single.h.valid_mask = 1U;
    single.row[1].flags = 0U;
    check(!run(single, 1U) && rt.uploads == 3, "a padded window fit a rectangular program");

    reply = &two_reply;
    rt.throw_on_launch = true;
    bool threw = false;
    try { (void)run(two, 2U); } catch (const std::runtime_error&) { threw = true; }
    rt.throw_on_launch = false;
    check(threw && runner.policy().failed(2U) && runner.policy().ready(1U) && !run(two, 2U),
          "failed launch did not stay isolated to its stream count");
    runner.reset(rt);
    check(run(two, 2U) && rt.records == 3, "reset did not restore stream program eligibility");

    bool invalid_table = false;
    try {
        const ex::StepGraphRunner invalid{ex::StreamGraphPrograms{{0U, ex::kStepMaxRows}}};
        (void)invalid;
    } catch (const std::invalid_argument&) { invalid_table = true; }
    check(invalid_table, "table accepted a physical width the descriptor cannot hold");
}

void dry_arguments() {
    ex::StepBlock descriptor{};
    Runtime::Tape tape{};
    struct Args {
        const ex::StepRow* row;
        Runtime::Tape* output;
    };
    const auto bind = [&](bool capture_token) {
        ex::Program program;
        program.key.rows = 2;
        const auto add = [&](const auto& value) {
            ex::BoundLaunch leaf;
            leaf.entry = "stage";
            leaf.args_offset = static_cast<std::uint32_t>(program.args.size());
            const auto bytes = std::as_bytes(std::span(&value, 1U));
            leaf.args_bytes = static_cast<std::uint32_t>(bytes.size());
            program.args.insert(program.args.end(), bytes.begin(), bytes.end());
            program.launches.push_back(std::move(leaf));
        };
        add(Args{descriptor.row, &tape});
        add(capture_token ? descriptor.row[1].token : 99U);
        return program;
    };
    for (const bool capture_token : {false, true}) {
        descriptor.row[1].token = 8;
        const auto before = bind(capture_token);
        descriptor.row[1].token = 77;
        bool different = false;
        try { ex::verify_dry_capture(before, bind(capture_token)); }
        catch (const std::logic_error&) { different = true; }
        check(different == capture_token, "dry walk must find a baked token");
    }
}
}  // namespace

int main() try {
    native_inventory();
    replay_windows();
    replay_stream_programs();
    dry_arguments();
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
