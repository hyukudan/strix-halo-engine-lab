#pragma once

#include "strix/direct_stream_control.hpp"
#include <functional>

namespace strix::transport {
struct CommissioningTcpPolicy {
    bool listener = false;
    std::uint16_t port = 0;
    std::uint64_t maximum_frame_bytes = 2048;
    std::uint32_t maximum_io_calls = 4096;
    std::chrono::milliseconds timeout{10000};
};
struct CommissioningTcpIdentity {
    std::uint64_t device = 0, inode = 0;
    std::uint32_t local_address = 0, peer_address = 0;
    std::uint16_t local_port = 0, peer_port = 0;
    bool operator==(const CommissioningTcpIdentity&) const = default;
};
// Trusted injectable POSIX boundary. No IO call retains supplied buffers;
// opening itself must respect the supplied deadline and bounded syscall budget.
struct CommissioningTcpCalls {
    std::function<int(const CommissioningTcpPolicy&, DirectStreamClock::time_point)> open;
    std::function<CommissioningTcpIdentity(int)> inspect;
    std::function<DirectStreamIoResult(int, std::span<std::byte>)> read;
    std::function<DirectStreamIoResult(int, std::span<const std::byte>)> write;
    std::function<DirectStreamWaitStatus(int, bool, DirectStreamClock::time_point)> wait;
    std::function<DirectStreamClock::time_point()> now;
    std::function<void(int)> close;
    [[nodiscard]] static CommissioningTcpCalls native();
};
// Loopback IPv4 ONLY. Operator establishes authenticated SSH tunneling outside
// this program; there is no unauthenticated LAN listener or network security
// claim. The exact run/peer/epoch descriptor is additionally fenced by kind4.
// Cleanup closes only a freshly matching captured socket identity. An unknown
// or replaced numeric descriptor is abandoned rather than closing another
// owner's socket; this is control-FD retention, not GPU quiescence evidence.
class BoundedCommissioningTcpControl final : public DistributedControlByteStream {
public:
    BoundedCommissioningTcpControl(CommissioningTcpPolicy, CommissioningTcpCalls = CommissioningTcpCalls::native());
    ~BoundedCommissioningTcpControl() override;
    void send(std::span<const std::byte>) override;
    [[nodiscard]] std::vector<std::byte> receive() override;
private:
    void close_noexcept() noexcept;
    void require_current();
    void transfer(std::span<std::byte>, std::span<const std::byte>, bool, DirectStreamClock::time_point, DirectStreamClock::time_point&, std::uint32_t&);
    CommissioningTcpPolicy policy_;
    CommissioningTcpCalls calls_;
    CommissioningTcpIdentity identity_;
    int fd_ = -1;
    bool failed_ = false, identity_observed_ = false;
};
} // namespace strix::transport
