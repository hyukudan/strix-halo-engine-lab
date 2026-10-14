#include "strix/tcp_copied_payload_stream.hpp"
#include "strix/copied_stream_transport.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <exception>
#include <iostream>
#include <fcntl.h>
#include <netinet/in.h>
#include <stdexcept>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace {
using namespace strix;
using namespace strix::transport;
int failures = 0;
void expect(bool condition, const char* message) {
    if (!condition) { ++failures; std::cerr << "FAIL: " << message << '\n'; }
}
template<class Function> bool rejected(Function&& function) {
    try { function(); } catch (const std::exception&) { return true; } return false;
}
struct OwnedFd {
    int value = -1;
    explicit OwnedFd(int fd) : value(fd) {}
    ~OwnedFd() { if (value >= 0) ::close(value); }
    int release() { const auto fd = value; value = -1; return fd; }
};
std::array<int, 2> connected_tcp_pair() {
    OwnedFd listener(::socket(AF_INET, SOCK_STREAM, 0));
    sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (listener.value < 0 || ::bind(listener.value, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
        ::listen(listener.value, 1) != 0) throw std::runtime_error("loopback test listener failed");
    socklen_t length = sizeof(address);
    if (::getsockname(listener.value, reinterpret_cast<sockaddr*>(&address), &length) != 0)
        throw std::runtime_error("loopback test port unavailable");
    OwnedFd client(::socket(AF_INET, SOCK_STREAM, 0));
    if (client.value < 0 || ::connect(client.value, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0)
        throw std::runtime_error("loopback test connection failed");
    OwnedFd server(::accept(listener.value, nullptr, nullptr));
    if (server.value < 0) throw std::runtime_error("loopback test accept failed");
    return {client.release(), server.release()};
}
TcpCopiedPayloadPolicy limits(std::uint64_t bytes = 2ULL * 1024ULL * 1024ULL + 4096ULL,
                              std::chrono::milliseconds timeout = std::chrono::milliseconds(1000)) {
    return {bytes, 4096, timeout};
}
void framing_tests() {
    {
        const auto sockets = connected_tcp_pair();
        auto sender = TcpCopiedPayloadByteStream::adopt_connected_loopback(sockets[0], limits(4096));
        auto receiver = TcpCopiedPayloadByteStream::adopt_connected_loopback(sockets[1], limits(4096));
        auto moved = std::move(sender);
        const std::vector<std::byte> frame(4096, std::byte{0x9b});
        moved.send(frame);
        expect(receiver.receive() == frame && !sender.valid(), "owned TCP framing retains exact bytes and move ownership");
        expect(rejected([&] { moved.send(std::vector<std::byte>(4097)); }) && moved.failed() && !moved.valid(),
               "oversized outbound frame permanently poisons connection");
    }
    for (const std::uint32_t announced : {0U, 4097U}) {
        const auto sockets = connected_tcp_pair(); OwnedFd raw(sockets[0]);
        auto receiver = TcpCopiedPayloadByteStream::adopt_connected_loopback(sockets[1], limits(4096));
        std::array<std::byte, 4> prefix{};
        for (unsigned shift = 0; shift < 32; shift += 8) prefix[shift / 8U] = static_cast<std::byte>((announced >> shift) & 255U);
        if (::send(raw.value, prefix.data(), prefix.size(), 0) != 4) throw std::runtime_error("raw prefix test send failed");
        expect(rejected([&] { (void)receiver.receive(); }) && receiver.failed(),
               "empty or oversized inbound length rejected before payload allocation");
        expect(rejected([&] { (void)receiver.receive(); }), "failed framed receive cannot resume on another message");
    }
    {
        const auto sockets = connected_tcp_pair(); OwnedFd raw(sockets[0]);
        auto receiver = TcpCopiedPayloadByteStream::adopt_connected_loopback(sockets[1], limits(4096));
        const std::array<std::byte, 6> truncated{std::byte{4}, {}, {}, {}, std::byte{1}, std::byte{2}};
        if (::send(raw.value, truncated.data(), truncated.size(), 0) != 6) throw std::runtime_error("truncated test send failed");
        ::shutdown(raw.value, SHUT_WR);
        expect(rejected([&] { (void)receiver.receive(); }) && receiver.failed(), "truncated framed body poisons connection");
    }
    {
        const auto sockets = connected_tcp_pair(); OwnedFd idle(sockets[0]);
        auto receiver = TcpCopiedPayloadByteStream::adopt_connected_loopback(sockets[1], limits(4096, std::chrono::milliseconds(30)));
        const auto before = std::chrono::steady_clock::now();
        expect(rejected([&] { (void)receiver.receive(); }) && receiver.failed(), "idle receive has an absolute deadline");
        expect(std::chrono::steady_clock::now() - before < std::chrono::seconds(1), "receive deadline is wall-clock bounded");
    }
    {
        const auto sockets = connected_tcp_pair(); OwnedFd peer(sockets[1]);
        const int small_buffer = 4096;
        if (::setsockopt(sockets[0], SOL_SOCKET, SO_SNDBUF, &small_buffer, sizeof(small_buffer)) != 0)
            throw std::runtime_error("send deadline fixture setup failed");
        auto sender = TcpCopiedPayloadByteStream::adopt_connected_loopback(sockets[0],
            limits(16ULL * 1024ULL * 1024ULL, std::chrono::milliseconds(30)));
        const std::vector<std::byte> frame(16ULL * 1024ULL * 1024ULL, std::byte{0x7d});
        const auto before = std::chrono::steady_clock::now();
        expect(rejected([&] { sender.send(frame); }) && sender.failed() && !sender.valid(),
               "peer that stops reading cannot extend partial-send deadline");
        expect(std::chrono::steady_clock::now() - before < std::chrono::seconds(1), "send deadline is wall-clock bounded");
    }
    {
        const auto sockets = connected_tcp_pair(); OwnedFd peer(sockets[0]);
        auto receiver = TcpCopiedPayloadByteStream::adopt_connected_loopback(sockets[1], limits(4096));
        ::shutdown(peer.value, SHUT_WR);
        expect(rejected([&] { (void)receiver.receive(); }) && receiver.failed(), "clean EOF before prefix fails closed");
    }
    {
        const auto sockets = connected_tcp_pair(); OwnedFd raw(sockets[0]);
        auto receiver = TcpCopiedPayloadByteStream::adopt_connected_loopback(sockets[1], {4096, 1, std::chrono::milliseconds(100)});
        const std::array<std::byte, 5> frame{std::byte{1}, {}, {}, {}, std::byte{3}};
        if (::send(raw.value, frame.data(), frame.size(), 0) != 5) throw std::runtime_error("budget test send failed");
        expect(rejected([&] { (void)receiver.receive(); }) && receiver.failed(), "IO call ceiling bounds progress independently of deadline");
    }
    {
        int sockets[2]{};
        if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) throw std::runtime_error("socketpair failed");
        OwnedFd other(sockets[1]);
        expect(rejected([&] { (void)TcpCopiedPayloadByteStream::adopt_connected_loopback(sockets[0]); }),
               "socket adoption rejects non-TCP endpoints rather than inventing tunnel identity");
        expect(::fcntl(sockets[0], F_GETFD) == -1 && errno == EBADF, "rejected adoption closes the consumed descriptor");
    }
    {
        OwnedFd unconnected(::socket(AF_INET, SOCK_STREAM, 0));
        const auto fd = unconnected.release();
        expect(rejected([&] { (void)TcpCopiedPayloadByteStream::adopt_connected_loopback(fd); }),
               "unconnected TCP socket has no acceptable connected peer");
        expect(::fcntl(fd, F_GETFD) == -1 && errno == EBADF, "unconnected adoption failure closes descriptor");
        expect(rejected([&] { (void)TcpCopiedPayloadByteStream::adopt_connected_loopback(-1); }),
               "invalid descriptor is rejected");
    }
    {
        OwnedFd datagram(::socket(AF_INET, SOCK_DGRAM, 0));
        const auto fd = datagram.release();
        expect(rejected([&] { (void)TcpCopiedPayloadByteStream::adopt_connected_loopback(fd); }) &&
               ::fcntl(fd, F_GETFD) == -1 && errno == EBADF, "UDP descriptor is rejected and closed");
    }
    {
        const auto sockets = connected_tcp_pair(); OwnedFd remote(sockets[1]);
        expect(rejected([&] { (void)TcpCopiedPayloadByteStream::adopt_connected_loopback(sockets[0], {0, 4096, std::chrono::milliseconds(100)}); }),
               "invalid policy rejects adoption before exposing a stream");
        expect(::fcntl(sockets[0], F_GETFD) == -1 && errno == EBADF, "invalid policy failure still consumes descriptor");
    }
    {
        const auto sockets = connected_tcp_pair();
        auto left = TcpCopiedPayloadByteStream::adopt_connected_loopback(sockets[0], limits(4096));
        OwnedFd peer(sockets[1]);
        CopiedStreamConnectionState state({"loopback-ceiling-test", 51, 0, 1, 0});
        expect(rejected([&] { CopiedStreamTransportBackend wrong(std::move(left), state, {16, 1024, 16}); }),
               "copied TCP rejects mismatched frame ceilings before taking stream ownership");
        expect(left.valid(), "failed backend construction preserves unconsumed stream");
        CopiedStreamTransportBackend correct(std::move(left), state, {16, 4096, 16});
        expect(!left.valid(), "failed constructor releases state attachment for the later correct owner");
    }
    {
        const auto sockets = connected_tcp_pair(); OwnedFd raw(sockets[0]);
        auto receiver = TcpCopiedPayloadByteStream::adopt_connected_loopback(sockets[1], limits(4096));
        std::exception_ptr error;
        std::jthread writer([&] {
            try {
                const std::array<std::byte, 9> frame{std::byte{5}, {}, {}, {}, std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}, std::byte{5}};
                for (const auto byte : frame) {
                    if (::send(raw.value, &byte, 1, 0) != 1) throw std::runtime_error("fragmented frame send failed");
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            } catch (...) { error = std::current_exception(); }
        });
        const auto actual = receiver.receive(); writer.join(); if (error) std::rethrow_exception(error);
        expect(actual == std::vector<std::byte>({std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}, std::byte{5}}),
               "fragmented prefix and body are assembled without numeric conversion");
    }
}
DistributedBoundaryTransfer message(std::uint64_t bytes) {
    DistributedPlacementPlan plan; plan.local_node = 0; plan.plan_identity = 31; plan.executable = true;
    DistributedExecutionBoundary boundary{91, 1, 2, 0, 1, bytes, {}};
    plan.boundaries = {boundary};
    DistributedBoundaryTransfer transfer;
    transfer.descriptor = make_distributed_boundary_descriptor(plan, boundary, 9, 11, 2, 1, 7);
    transfer.flow = classify_distributed_boundary_flow(plan, boundary);
    transfer.link_plan.kind = PlanKind::single_link; transfer.link_plan.slices = {{0, 0, bytes}};
    return transfer;
}
CopiedStreamOwnedBuffer buffer(std::uint64_t bytes, AllocationGenerationSource& generations, std::byte poison) {
    auto owner = std::make_shared<CopiedStreamHostBuffer>(bytes, generations);
    std::fill(owner->writable_bytes().begin(), owner->writable_bytes().end(), poison);
    return {owner, owner->view()};
}
void copied_roundtrip(std::uint64_t bytes, unsigned mismatch = 0) {
    const auto sockets = connected_tcp_pair();
    // Force large payloads through short nonblocking sendmsg writes instead
    // of relying on the kernel's default multi-megabyte send buffer.
    const int small_send_buffer = 4096;
    if (::setsockopt(sockets[0], SOL_SOCKET, SO_SNDBUF, &small_send_buffer, sizeof(small_send_buffer)) != 0)
        throw std::runtime_error("partial-send fixture setup failed");
    const auto timeout = mismatch ? std::chrono::milliseconds(50) : std::chrono::milliseconds(1000);
    auto left = TcpCopiedPayloadByteStream::adopt_connected_loopback(sockets[0], limits(bytes + 4096, timeout));
    auto right = TcpCopiedPayloadByteStream::adopt_connected_loopback(sockets[1], limits(bytes + 4096, timeout));
    CopiedStreamConnectionState left_state({"loopback-test-owned-channel", 51, 0, 1, 0});
    CopiedStreamConnectionState right_state({"loopback-test-owned-channel", 51, 1, 0, 0});
    const CopiedStreamTransportPolicy policy{bytes, bytes + 4096, 16};
    CopiedStreamTransportBackend sender(std::move(left), left_state, policy), receiver(std::move(right), right_state, policy);
    expect(!left.valid() && !right.valid(), "copied TCP backend exclusively owns both adopted streams");
    AllocationGenerationSource generations;
    auto source = buffer(bytes, generations, std::byte{0}), destination = buffer(bytes, generations, std::byte{0xa5});
    for (std::size_t index = 0; index < source.allocation->bytes().size(); ++index)
        source.allocation->writable_bytes()[index] = static_cast<std::byte>((index * 13U) % 251U);
    const auto sent = message(bytes);
    auto expected = sent;
    if (mismatch == 1) ++expected.descriptor.dtype_id;
    if (mismatch == 2) ++expected.descriptor.request_id;
    if (mismatch == 3) ++expected.descriptor.generation;
    if (mismatch == 4) ++expected.descriptor.connection_epoch;
    const auto tx = sender.prepare_send(sent, {source, make_copied_stream_synchronous_proof(
        {sent.descriptor, source.view, CopiedStreamCompletionScope::producer_released})});
    const auto rx = receiver.prepare_receive(expected, destination);
    std::exception_ptr sender_error;
    bool acknowledged = false;
    std::jthread sending([&] {
        try {
            sender.submit(tx);
            for (unsigned poll = 0; poll < 4; ++poll) {
                const auto completion = sender.poll();
                if (completion && completion->kind == CompletionKind::remote_consumer_complete) { acknowledged = true; break; }
            }
        } catch (...) { sender_error = std::current_exception(); }
    });
    if (mismatch) {
        expect(rejected([&] { receiver.submit(rx); }) && receiver.failed(), "foreign typed/request/generation/epoch metadata fails closed");
        expect(std::all_of(destination.allocation->bytes().begin(), destination.allocation->bytes().end(),
                           [](std::byte value) { return value == std::byte{0xa5}; }),
               "foreign envelope leaves entire receive destination unchanged");
        sending.join(); expect(sender_error && !acknowledged, "rejected metadata never receives consumer acknowledgement");
        return;
    }
    receiver.submit(rx);
    const auto receive_completion = receiver.poll();
    expect(receive_completion && receive_completion->kind == CompletionKind::receive_complete,
           "copied TCP envelope reaches validated receive completion");
    const auto received = receiver.receive_buffer(rx);
    expect(received.bytes == bytes && std::memcmp(reinterpret_cast<void*>(received.address),
        source.allocation->bytes().data(), static_cast<std::size_t>(bytes)) == 0, "copied TCP payload remains bit-exact at full bounds");
    receiver.finish_consumer(rx, make_copied_stream_synchronous_proof(
        {expected.descriptor, received, CopiedStreamCompletionScope::consumer_complete}));
    const auto consumer_completion = receiver.poll();
    expect(consumer_completion && consumer_completion->kind == CompletionKind::local_buffer_reusable,
           "CPU consumer proof retires receive ownership");
    sending.join(); if (sender_error) std::rethrow_exception(sender_error);
    expect(acknowledged && !sender.retains_buffer() && !receiver.retains_buffer(), "exact ACK/consumer credit retires both copied allocations");
    const auto capabilities = sender.capabilities();
    expect(capabilities.buffer_mode == TransportBufferMode::copy_stream,
           "TCP route advertises host copy rather than GPU/direct DMA capability");
}
} // namespace
int main() try {
    framing_tests();
    for (const std::uint64_t bytes : {20480ULL, 993280ULL, 2ULL * 1024ULL * 1024ULL}) copied_roundtrip(bytes);
    for (unsigned mismatch = 1; mismatch <= 4; ++mismatch) copied_roundtrip(20480, mismatch);
    if (!failures) std::cout << "TCP copied payload tests passed\n";
    return failures ? 1 : 0;
} catch (const std::exception& error) { std::cerr << "TCP copied payload tests: " << error.what() << '\n'; return 2; }
