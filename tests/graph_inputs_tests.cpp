#include "strix/graph_inputs.hpp"

#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
int failures = 0;
void expect(bool value, const char* message) {
    if (!value) { std::cerr << message << '\n'; ++failures; }
}
template<class Error, class Function> void rejects(Function f) {
    try { f(); expect(false, "expected graph input rejection"); }
    catch (const Error&) {}
    catch (...) { expect(false, "unexpected graph input error"); }
}
}
int main() {
    const auto layout = strix::graph_input_layout(3);
    expect(layout.position_offset == 16 && layout.bytes == 40, "aligned position layout");
    const std::array<std::uint32_t, 3> tokens{9, 0, 8};
    const std::array<std::uint64_t, 3> positions{31, 32, 33};
    auto packet = strix::pack_graph_inputs(layout, tokens, positions, 10, 34);
    std::array<std::uint32_t, 3> recovered_tokens{};
    std::array<std::uint64_t, 3> recovered_positions{};
    std::memcpy(recovered_tokens.data(), packet.data(), sizeof(recovered_tokens));
    std::memcpy(recovered_positions.data(), packet.data() + layout.position_offset, sizeof(recovered_positions));
    expect(recovered_tokens == tokens && recovered_positions == positions, "input ABI exact roundtrip");
    expect(packet[12] == std::byte{0} && packet[15] == std::byte{0}, "padding initialized");
    const auto args = strix::bind_graph_inputs(layout, {7, 0x1000, 40, 3});
    expect(args.device_argument_abi_identity == strix::graph_input_abi_v1 && args.position_offset == 16,
           "exact stable argument binding");
    auto changed_tokens = tokens;
    changed_tokens[0] = 1;
    expect(strix::pack_graph_inputs(layout, changed_tokens, positions, 10, 34) != packet,
           "changed token is device data, not capture metadata");
    rejects<std::invalid_argument>([] { (void)strix::graph_input_layout(0); });
    rejects<std::invalid_argument>([] { (void)strix::graph_input_layout(65537); });
    rejects<std::invalid_argument>([&] { (void)strix::bind_graph_inputs(layout, {7, 0x1001, 40, 3}); });
    rejects<std::invalid_argument>([&] { (void)strix::bind_graph_inputs(layout, {7, 0x1000, 48, 3}); });
    rejects<std::invalid_argument>([&] { (void)strix::bind_graph_inputs(layout, {7, 0x1000, 40, 0}); });
    rejects<std::invalid_argument>([&] { (void)strix::bind_graph_inputs(layout,
        {7, std::numeric_limits<std::uintptr_t>::max() - 7, 40, 3}); });
    auto bad_layout = layout;
    bad_layout.position_offset = 8;
    rejects<std::invalid_argument>([&] { (void)strix::pack_graph_inputs(bad_layout, tokens, positions, 10, 34); });
    rejects<std::out_of_range>([&] { (void)strix::pack_graph_inputs(layout, tokens, positions, 9, 34); });
    rejects<std::out_of_range>([&] { (void)strix::pack_graph_inputs(layout, tokens, positions, 10, 33); });
    auto nonconsecutive = positions;
    nonconsecutive[1] = 31;
    rejects<std::invalid_argument>([&] { (void)strix::pack_graph_inputs(layout, tokens, nonconsecutive, 10, 34); });
    rejects<std::invalid_argument>([&] { (void)strix::pack_graph_inputs(layout,
        std::span<const std::uint32_t>{tokens}.first(2), positions, 10, 34); });
    const auto single = strix::graph_input_layout(1);
    const std::array<std::uint32_t, 1> zero{0};
    const std::array<std::uint64_t, 1> last{std::numeric_limits<std::uint64_t>::max() - 1};
    expect(strix::pack_graph_inputs(single, zero, last, 1, std::numeric_limits<std::uint64_t>::max()).size() == 16,
           "maximum valid absolute position has no overflow");
    strix::RuntimeTensorBinding weights{0, strix::RuntimeTensorStorage::resident, 0x10000, 8192, 3, 0};
    const auto c1_args = strix::bind_graph_inputs(single, {7, 0x20000, 16, 4});
    strix::ExecutionBufferAddress output;
    output.address = 0x30000; output.bytes = 1024; output.allocation_generation = 5; output.use.generation = 1;
    strix::ExecutionMemoryBinding status{7, 0x40000, 4, 6};
    const auto validate = [&](auto w, auto a, auto o, auto s, std::uint32_t type = 0) {
        strix::validate_graph_embedding_bindings(256, type, 8, 64, 4, w, a, o, s);
    };
    validate(weights, c1_args, output, status);
    auto overlapping = output; overlapping.address = weights.address;
    rejects<std::invalid_argument>([&] { validate(weights, c1_args, overlapping, status); });
    auto short_weight = weights; short_weight.bytes = 8191;
    rejects<std::invalid_argument>([&] { validate(short_weight, c1_args, output, status); });
    auto remote = weights; remote.storage = strix::RuntimeTensorStorage::remote;
    rejects<std::invalid_argument>([&] { validate(remote, c1_args, output, status); });
    auto wrong_abi = c1_args; ++wrong_abi.device_argument_abi_identity;
    rejects<std::invalid_argument>([&] { validate(weights, wrong_abi, output, status); });
    auto wrong_status = status; ++wrong_status.execution_identity;
    rejects<std::invalid_argument>([&] { validate(weights, c1_args, output, wrong_status); });
    wrong_status = status; wrong_status.arena_bytes = 8;
    rejects<std::invalid_argument>([&] { validate(weights, c1_args, output, wrong_status); });
    rejects<std::invalid_argument>([&] { validate(weights, c1_args, output, status, 99); });
    rejects<std::invalid_argument>([&] { strix::validate_graph_embedding_bindings(640, 23, 8, 64, 4,
        weights, c1_args, output, status); });
    return failures == 0 ? 0 : 1;
}
