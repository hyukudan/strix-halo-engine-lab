#pragma once

#include "strix/distributed_control_wire.hpp"

#include <chrono>
#include <memory>
#include <string>

namespace strix {

// Explicit local route observation, not peer reconciliation or DMA proof.
struct DirectStreamDeviceIdentity {
    std::string canonical_path;
    std::uint64_t filesystem_device = 0;
    std::uint64_t inode = 0;
    std::uint64_t character_device_number = 0;
    bool character_device = false;
    bool operator==(const DirectStreamDeviceIdentity&) const = default;
};

// Read-only POSIX discovery. Rejects missing/non-character paths. The opener
// checks the opened object against this exact observation to close path races.
[[nodiscard]] DirectStreamDeviceIdentity inspect_direct_stream_device(const std::string& path);

struct DirectStreamFd {
    int descriptor = -1;
    std::uint64_t open_generation = 0;
    bool operator==(const DirectStreamFd&) const = default;
};
struct DirectStreamFdIdentity {
    DirectStreamDeviceIdentity device;
    std::uint64_t open_generation = 0;
};
enum class DirectStreamIoStatus { transferred, interrupted, would_block, end_of_stream, failed };
struct DirectStreamIoResult {
    DirectStreamIoStatus status = DirectStreamIoStatus::failed;
    std::uint64_t bytes = 0;
};
enum class DirectStreamWaitStatus { ready, interrupted, timed_out, end_of_stream, failed };
using DirectStreamClock = std::chrono::steady_clock;

// Trusted synchronous FD boundary. No operation may retain a supplied buffer
// after return. The owner outlives its streams and never exposes their FD for
// external close/dup2/reuse. Replacement invalidates the open generation.
class DirectStreamControlIo {
public:
    virtual ~DirectStreamControlIo() = default;
    [[nodiscard]] virtual DirectStreamFd open(const DirectStreamDeviceIdentity&) = 0;
    [[nodiscard]] virtual DirectStreamFdIdentity inspect(DirectStreamFd) = 0;
    [[nodiscard]] virtual DirectStreamIoResult read(DirectStreamFd, std::span<std::byte>) = 0;
    [[nodiscard]] virtual DirectStreamIoResult write(DirectStreamFd, std::span<const std::byte>) = 0;
    [[nodiscard]] virtual DirectStreamWaitStatus wait(DirectStreamFd, bool writing, DirectStreamClock::time_point deadline) = 0;
    [[nodiscard]] virtual DirectStreamClock::time_point now() const = 0;
    virtual void close(DirectStreamFd) noexcept = 0;
};

// Nonblocking O_CLOEXEC/O_NOFOLLOW character-device read/write/poll owner.
// It checks fstat identity on inspection; no import/export/ioctl is attempted.
class PosixDirectStreamControlIo final : public DirectStreamControlIo {
public:
    PosixDirectStreamControlIo();
    ~PosixDirectStreamControlIo() override;
    [[nodiscard]] DirectStreamFd open(const DirectStreamDeviceIdentity&) override;
    [[nodiscard]] DirectStreamFdIdentity inspect(DirectStreamFd) override;
    [[nodiscard]] DirectStreamIoResult read(DirectStreamFd, std::span<std::byte>) override;
    [[nodiscard]] DirectStreamIoResult write(DirectStreamFd, std::span<const std::byte>) override;
    [[nodiscard]] DirectStreamWaitStatus wait(DirectStreamFd, bool, DirectStreamClock::time_point) override;
    [[nodiscard]] DirectStreamClock::time_point now() const override;
    void close(DirectStreamFd) noexcept override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct DirectStreamControlPolicy {
    std::uint64_t maximum_frame_bytes = 64ULL * 1024ULL;
    std::uint32_t maximum_io_calls = 4096;
    std::chrono::milliseconds timeout{1000};
};

// CPU-copy control tier using the TCP-compatible little-endian U32 length
// prefix. Existing FramedDistributedControlWire v2 owns protocol/peer/request
// validation. This class does not supply tensor DMA or native qualification.
// One serialized owner; a partial/error/expired frame permanently fails it.
class DirectStreamControlByteStream final : public DistributedControlByteStream {
public:
    [[nodiscard]] static DirectStreamControlByteStream open_device(
        const DirectStreamDeviceIdentity&, DirectStreamControlPolicy policy = {});
    [[nodiscard]] static DirectStreamControlByteStream open_with_io(
        DirectStreamControlIo&, const DirectStreamDeviceIdentity&, DirectStreamControlPolicy policy = {});
    DirectStreamControlByteStream(const DirectStreamControlByteStream&) = delete;
    DirectStreamControlByteStream& operator=(const DirectStreamControlByteStream&) = delete;
    DirectStreamControlByteStream(DirectStreamControlByteStream&&) noexcept;
    DirectStreamControlByteStream& operator=(DirectStreamControlByteStream&&) noexcept;
    ~DirectStreamControlByteStream() override;

    void send(std::span<const std::byte>) override;
    [[nodiscard]] std::vector<std::byte> receive() override;
    [[nodiscard]] bool failed() const noexcept { return failed_; }
    [[nodiscard]] bool valid() const noexcept { return io_ != nullptr && fd_.descriptor >= 0; }

private:
    DirectStreamControlByteStream(DirectStreamControlIo&, DirectStreamDeviceIdentity, DirectStreamControlPolicy);
    void require_identity();
    void transfer(std::span<std::byte>, std::span<const std::byte>, bool writing,
                  DirectStreamClock::time_point deadline, DirectStreamClock::time_point& last_time,
                  std::uint32_t& calls);
    [[nodiscard]] DirectStreamClock::time_point deadline(DirectStreamClock::time_point& last_time);
    void check_time(DirectStreamClock::time_point deadline, DirectStreamClock::time_point& last_time);
    void close_noexcept() noexcept;

    std::unique_ptr<DirectStreamControlIo> owned_io_;
    DirectStreamControlIo* io_ = nullptr;
    DirectStreamDeviceIdentity expected_;
    DirectStreamControlPolicy policy_;
    DirectStreamFd fd_;
    bool failed_ = false;
};

}  // namespace strix
