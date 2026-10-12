#include "../lab/direct_gpu_stream/observe_driver.hpp"
#include <cstring>
#include <iostream>

namespace {
using namespace strix;
using namespace strix::transport;
int failures = 0;
void expect(bool condition, const char* message) { if (!condition) { ++failures; std::cerr << message << '\n'; } }
#if defined(__linux__)
DirectStreamDeviceIdentity route() { return {"/synthetic/observed-driver", 11, 12, 13, true}; }
class Fd final : public DirectStreamControlIo {
public:
    DirectStreamFd open(const DirectStreamDeviceIdentity&) override { ++opens; return {40, 7}; }
    DirectStreamFdIdentity inspect(DirectStreamFd) override { return {route(), generation}; }
    DirectStreamIoResult read(DirectStreamFd, std::span<std::byte>) override { ++payload_io; throw std::runtime_error("unexpected read"); }
    DirectStreamIoResult write(DirectStreamFd, std::span<const std::byte>) override { ++payload_io; throw std::runtime_error("unexpected write"); }
    DirectStreamWaitStatus wait(DirectStreamFd, bool, DirectStreamClock::time_point) override { ++payload_io; throw std::runtime_error("unexpected wait"); }
    DirectStreamClock::time_point now() const override { return {}; }
    void close(DirectStreamFd) noexcept override { ++closes; }
    std::uint64_t generation = 7;
    int opens = 0, closes = 0, payload_io = 0;
};
sgu_query query() {
    sgu_query value{}; value.fence.abi_major = SGU_ABI_MAJOR; value.fence.abi_minor = SGU_ABI_MINOR; value.fence.abi_cookie = SGU_ABI_COOKIE;
    value.fence.device_instance[0] = 1; value.fence.local_uuid[0] = 2; value.fence.peer_uuid[0] = 3; value.fence.route = 4; value.fence.in_hopid = 5; value.fence.out_hopid = 6;
    value.frame_bytes = SGU_FRAME_BYTES; value.ring_frames = 32; value.maximum_sg_entries = SGU_MAX_SG_ENTRIES;
    value.maximum_window_bytes = 32 * SGU_FRAME_BYTES; value.maximum_imported_bytes = SGU_MAX_IMPORTED_BYTES; value.maximum_event_batch = SGU_EVENT_BATCH; return value;
}
struct Fixture {
    std::shared_ptr<Fd> fd = std::make_shared<Fd>();
    sgu_query response = query();
    bool replace_fd = false, replace_build = false, missing_build = false, unsupported = false;
    int calls = 0, build_calls = 0;
    LinuxDirectGpuStreamCalls io() {
        LinuxDirectGpuStreamCalls value; value.fd_owner = fd; value.kernel_release = [] { return "synthetic-kernel"; };
        value.driver_build_identity = [this]() -> std::optional<std::string> { ++build_calls; if (missing_build) return {}; return replace_build && build_calls > 1 ? "05060708" : "01020304"; };
        value.ioctl_call = [this](DirectStreamFd, unsigned long command, void* pointer) {
            ++calls; expect(command == SGU_IOC_QUERY, "observation invokes only actual QUERY command, never import/ARM/control");
            sgu_query expected{}; expected.fence.abi_major = SGU_ABI_MAJOR; expected.fence.abi_minor = SGU_ABI_MINOR;
            expect(std::memcmp(pointer, &expected, sizeof(expected)) == 0, "QUERY input contains exact ABI version and otherwise zero bytes");
            if (replace_fd) ++fd->generation;
            if (unsupported) return LinuxDirectGpuStreamSyscallResult{-1, ENOTTY};
            *static_cast<sgu_query*>(pointer) = response; return LinuxDirectGpuStreamSyscallResult{0, 0};
        }; return value;
    }
};
#endif
}
int main() {
#if defined(__linux__)
    {
        Fixture fixture; const auto value = observe_direct_gpu_stream_driver(route(), {true, "synthetic-kernel", "01020304"}, fixture.io());
        expect(value.query_observed && value.error.empty() && value.status == DirectNativeQueryStatus::unavailable && value.driver_build_id == "01020304" && value.fd_close_requested && !value.fd_retained,
               "synthetic exact raw QUERY observation never promotes native query/readiness qualification");
        expect(fixture.calls == 1 && fixture.fd->closes == 1 && fixture.fd->payload_io == 0, "observation has one bounded QUERY and exact FD cleanup with zero payload/control calls");
        const auto json = direct_gpu_stream_driver_observation_json(value);
        expect(json.find("\"native_ready\":false") != std::string::npos && json.find("\"executed_payload\":false") != std::string::npos && json.find("\"maximum_window_bytes\":131072") != std::string::npos, "inventory JSON preserves raw limits and explicit nonqualification");
    }
    for (const auto corrupt : {0, 1, 2, 3, 4, 5, 6}) {
        Fixture fixture;
        switch (corrupt) {
        case 0: ++fixture.response.fence.abi_minor; break;
        case 1: fixture.response.reserved = 1; break;
        case 2: ++fixture.response.maximum_window_bytes; break;
        case 3: fixture.response.maximum_imported_bytes = SGU_MAX_IMPORTED_BYTES + 1; break;
        case 4: fixture.response.fence.import_generation = 1; break;
        case 5: fixture.response.state = SGU_STATE_ARMED; break;
        default: fixture.response.state = 99; break;
        }
        const auto value = observe_direct_gpu_stream_driver(route(), {true, "synthetic-kernel", "01020304"}, fixture.io());
        expect(!value.query_observed && !value.error.empty() && value.status != DirectNativeQueryStatus::verified && fixture.fd->closes == 1, "malformed ABI/extent/state/foreign active import cannot become accepted inventory");
    }
    {
        Fixture fixture; fixture.replace_fd = true; const auto value = observe_direct_gpu_stream_driver(route(), {true, "synthetic-kernel", "01020304"}, fixture.io());
        expect(!value.query_observed && value.fd_retained && !value.fd_close_requested && fixture.fd->closes == 0, "FD replacement during QUERY rejects observation and cannot close replacement");
    }
    {
        Fixture fixture; fixture.replace_build = true; const auto value = observe_direct_gpu_stream_driver(route(), {true, "synthetic-kernel", "01020304"}, fixture.io());
        expect(!value.query_observed && !value.error.empty() && value.status == DirectNativeQueryStatus::incompatible && fixture.fd->closes == 1, "module-build replacement during QUERY invalidates exact observed identity");
    }
    {
        Fixture fixture; fixture.missing_build = true; const auto value = observe_direct_gpu_stream_driver(route(), {true, "synthetic-kernel", "01020304"}, fixture.io());
        expect(!value.query_observed && !value.error.empty() && value.driver_build_id.empty() && value.status == DirectNativeQueryStatus::unavailable && fixture.fd->opens == 0, "missing loaded GNU build note stops before opening unknown stream route");
    }
    {
        Fixture fixture; fixture.unsupported = true; const auto value = observe_direct_gpu_stream_driver(route(), {true, "synthetic-kernel", "01020304"}, fixture.io());
        expect(!value.query_observed && value.status == DirectNativeQueryStatus::unsupported && fixture.fd->closes == 1, "stock unsupported QUERY remains unsupported, not simulated native availability");
    }
    {
        // Real /dev/null syscall negative with explicitly synthetic preflight
        // identity. No stock/target stream is opened or inferred compatible.
        auto calls = LinuxDirectGpuStreamCalls::native(); calls.kernel_release = [] { return "synthetic-kernel"; }; calls.driver_build_identity = [] { return std::optional<std::string>{"01020304"}; };
        const auto value = observe_direct_gpu_stream_driver(inspect_direct_stream_device("/dev/null"), {true, "synthetic-kernel", "01020304"}, calls);
        expect(!value.query_observed && value.status == DirectNativeQueryStatus::unsupported && value.fd_close_requested && !value.fd_retained, "real /dev/null ENOTTY rejects unsupported ABI; synthetic mode binding is not stock probing proof");
    }
    for (const auto& policy : {DirectGpuStreamDriverObservationPolicy{}, DirectGpuStreamDriverObservationPolicy{true, "foreign-kernel", "01020304"}, DirectGpuStreamDriverObservationPolicy{true, "synthetic-kernel", "05060708"}}) {
        Fixture fixture; const auto value = observe_direct_gpu_stream_driver(route(), policy, fixture.io());
        expect(!value.query_observed && !value.error.empty() && fixture.fd->opens == 0 && fixture.calls == 0 && fixture.fd->closes == 0, "unacknowledged mode or mismatched kernel/module stops before any route open/QUERY");
    }
#endif
    std::cout << "driver observation injected checks and /dev/null unsupported negative; no GPU/driver DMA/network execution\n";
    return failures ? 1 : 0;
}
