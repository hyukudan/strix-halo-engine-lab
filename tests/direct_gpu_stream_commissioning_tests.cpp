#include "strix/direct_gpu_stream_commissioning.hpp"
#include "../lab/direct_gpu_stream/bounded_tcp_control.hpp"

#include <deque>
#include <iostream>
#include <type_traits>

namespace {
using namespace strix;
using namespace strix::transport;
int failures = 0;
void expect(bool condition, const char* message) { if (!condition) { ++failures; std::cerr << message << '\n'; } }
template<class Function> bool rejected(Function function) { try { function(); } catch (const std::exception&) { return true; } return false; }
static_assert(!std::is_base_of_v<DirectTransportDriverIo, LinuxDirectGpuStreamCommissioning>);
static_assert(!std::is_base_of_v<DirectTransportNativeBackend, LinuxDirectGpuStreamCommissioning>);
static_assert(!std::is_base_of_v<DirectGpuStreamQualification, LinuxDirectGpuStreamCommissioning>);
struct Io {
    CommissioningTcpIdentity identity{11, 12, 0x7f000001U, 0x7f000001U, 1234, 4321};
    DirectStreamClock::time_point time{};
    std::deque<std::byte> input;
    std::vector<std::byte> output;
    bool eof = false, block = false, late = false, drift = false;
    int closed = 0, reads = 0, waits = 0, inspections = 0, throw_inspection = 0, drift_inspection = 0;
    CommissioningTcpCalls calls() {
        CommissioningTcpCalls result;
        result.open = [](const CommissioningTcpPolicy&, DirectStreamClock::time_point) { return 40; };
        result.inspect = [this](int) {
            ++inspections;
            if (inspections == throw_inspection) throw std::runtime_error("synthetic socket identity unavailable");
            if (inspections == drift_inspection) ++identity.inode;
            return identity;
        };
        result.now = [this] { time += std::chrono::microseconds{1}; return time; };
        result.close = [this](int) { ++closed; };
        result.write = [this](int, std::span<const std::byte> bytes) { const auto count = std::min<std::size_t>(3, bytes.size()); output.insert(output.end(), bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(count)); return DirectStreamIoResult{DirectStreamIoStatus::transferred, count}; };
        result.read = [this](int, std::span<std::byte> bytes) {
            ++reads;
            if (block) return DirectStreamIoResult{DirectStreamIoStatus::would_block, 0};
            if (input.empty()) return DirectStreamIoResult{eof ? DirectStreamIoStatus::end_of_stream : DirectStreamIoStatus::would_block, 0};
            const auto count = std::min<std::size_t>({bytes.size(), input.size(), 2});
            for (std::size_t index = 0; index < count; ++index) { bytes[index] = input.front(); input.pop_front(); }
            return DirectStreamIoResult{DirectStreamIoStatus::transferred, count};
        };
        result.wait = [this](int, bool, DirectStreamClock::time_point limit) { ++waits; if (late) time = limit; if (drift) ++identity.inode; block = false; return DirectStreamWaitStatus::ready; };
        return result;
    }
    void frame(std::span<const std::byte> bytes) {
        for (unsigned index = 0; index < 4; ++index) input.push_back(static_cast<std::byte>((bytes.size() >> (8U * index)) & 255U));
        input.insert(input.end(), bytes.begin(), bytes.end());
    }
};
}
int main() {
    {
        DirectDriverBinding binding{1, "synthetic-device", 1, {1, "synthetic-kernel", "synthetic-build", "synthetic-abi", "synthetic-local", "synthetic-peer", "synthetic-link", 7}};
        DirectGpuCommissioningConfig config{"synthetic-campaign", "synthetic-run", binding, {1, 7, 2, 3, 4, 5, 1, 6, 1, 4097}, UINT64_MAX, false, std::chrono::milliseconds{1000}, 16};
        validate_direct_gpu_commissioning_config(config);
        config.expected.interface.physical_link_id.assign(257, 'x'); expect(rejected([&] { validate_direct_gpu_commissioning_config(config); }), "oversized link label rejected before import/ARM/native work");
        config.expected = binding; config.expected.interface.kernel_release = "synthetic\nkernel"; expect(rejected([&] { validate_direct_gpu_commissioning_config(config); }), "noncanonical expected identity rejected by configuration preflight");
        config.expected = binding; config.timeout = std::chrono::minutes{2}; expect(rejected([&] { validate_direct_gpu_commissioning_config(config); }), "excessive overall deadline rejected before execution");
    }
    {
        std::vector<std::byte> bytes(5120);
        for (std::size_t index = 0; index < bytes.size(); ++index) bytes[index] = static_cast<std::byte>(direct_gpu_commissioning_pattern_byte(index, 123, 1));
        expect(verify_direct_gpu_commissioning_pattern(bytes, 123, 1), "complete deterministic 5120-byte partial-frame pattern accepted");
        for (const auto offset : {std::size_t{0}, std::size_t{2559}, std::size_t{5119}}) {
            bytes[offset] ^= std::byte{1}; expect(!verify_direct_gpu_commissioning_pattern(bytes, 123, 1), "first/middle/final byte corruption rejected independently of aggregate digest"); bytes[offset] ^= std::byte{1};
        }
        expect(!verify_direct_gpu_commissioning_pattern(bytes, 123, 2) && !verify_direct_gpu_commissioning_pattern(bytes, 124, 1) && !verify_direct_gpu_commissioning_pattern({}, 123, 1), "stale sequence/foreign seed/empty pattern rejected");
    }
    {
        Io io; const std::array frame{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}, std::byte{5}}; io.frame(frame);
        { BoundedCommissioningTcpControl stream({false, 1234, 2048, 64, std::chrono::milliseconds{1}}, io.calls()); stream.send(frame); expect(stream.receive() == std::vector<std::byte>(frame.begin(), frame.end()), "partial bounded TCP fixture preserves existing U32 framing and exact bytes"); }
        expect(io.closed == 1 && io.output.size() == frame.size() + 4, "serialized TCP owner closes only its exact owned FD");
    }
    for (const bool late : {true, false}) {
        Io io; io.block = true; io.late = late; io.drift = !late;
        {
            BoundedCommissioningTcpControl stream({false, 1234, 2048, 64, std::chrono::milliseconds{1}}, io.calls());
            expect(rejected([&] { static_cast<void>(stream.receive()); }) && io.reads == 1 && io.waits == 1, "late wait or replaced FD cannot authorize another read");
            expect(rejected([&] { stream.send(std::array{std::byte{1}}); }), "partial/expired/foreign socket failure is sticky");
        }
        expect(io.closed == static_cast<int>(late), "teardown closes known expired socket once but never closes replaced FD");
    }
    {
        Io io;
        { BoundedCommissioningTcpControl stream({false, 1234}, io.calls()); io.throw_inspection = io.inspections + 1; }
        expect(io.closed == 0, "destructor retains ambiguous descriptor when current socket identity cannot be inspected");
    }
    {
        Io io; io.throw_inspection = 1;
        expect(rejected([&] { BoundedCommissioningTcpControl stream({false, 1234}, io.calls()); }) && io.closed == 0, "constructor cannot close a newly returned FD without an observed identity");
    }
    for (const bool unknown : {true, false}) {
        Io io; io.identity.peer_address = 0x0a000001U;
        if (unknown) io.throw_inspection = 2; else io.drift_inspection = 2;
        expect(rejected([&] { BoundedCommissioningTcpControl stream({false, 1234}, io.calls()); }) && io.closed == 0, "constructor rollback cannot close unknown or replaced captured descriptor");
    }
    {
        Io io; io.identity.peer_address = 0x0a000001U;
        expect(rejected([&] { BoundedCommissioningTcpControl stream({false, 1234}, io.calls()); }) && io.closed == 1, "unauthenticated nonloopback peer is never accepted");
    }
    {
        Io io; io.eof = true; io.input = {std::byte{5}, std::byte{0}};
        BoundedCommissioningTcpControl stream({false, 1234}, io.calls()); expect(rejected([&] { static_cast<void>(stream.receive()); }), "partial TCP length EOF cannot publish a frame");
    }
    {
        Io io; io.eof = true; io.frame(std::array{std::byte{1}}); io.input.front() = std::byte{255};
        BoundedCommissioningTcpControl stream({false, 1234, 128}, io.calls()); expect(rejected([&] { static_cast<void>(stream.receive()); }) && io.reads == 2, "oversized TCP payload length rejected before payload read/allocation");
    }
    {
        Io io; expect(rejected([&] { BoundedCommissioningTcpControl stream({false, 0}, io.calls()); }) && io.closed == 0, "invalid bounded TCP policy rejected before open");
    }
    std::cout << "commissioning pattern/isolation and injected bounded loopback control tests; no GPU/DMA/network execution\n";
    return failures ? 1 : 0;
}
