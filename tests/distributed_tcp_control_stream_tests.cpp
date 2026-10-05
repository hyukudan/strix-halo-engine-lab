#include "strix/distributed_tcp_control_stream.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

std::array<int,2> socket_pair() {
    std::array<int,2> fds{-1,-1};
    if (::socketpair(
            AF_UNIX,
            SOCK_STREAM,
            0,
            fds.data()) != 0) {
        throw std::runtime_error(
            "socketpair failed");
    }
    return fds;
}

}  // namespace

int main() {
    {
        const auto fds =
            socket_pair();
        auto left =
            strix::TcpDistributedControlByteStream::
                adopt_connected(
                    fds[0],
                    1024);
        auto right =
            strix::TcpDistributedControlByteStream::
                adopt_connected(
                    fds[1],
                    1024);

        const std::vector<std::byte> payload{
            std::byte{0x01},
            std::byte{0x02},
            std::byte{0x7f},
        };
        left.send(payload);
        const auto received =
            right.receive();
        expect(
            received == payload,
            "TCP control stream round-trips length-prefixed frame");

        const auto command =
            strix::DistributedRemoteStepCommand{
                1,
                2,
                3,
                4,
                5,
                6,
                strix::SerialOp::moe_experts,
                1,
            };
        const auto encoded =
            strix::encode_distributed_remote_step_command(
                command);
        right.send(encoded);
        const auto decoded =
            strix::decode_distributed_remote_step_command(
                left.receive());
        expect(
            decoded.sequence == 5 &&
                decoded.op ==
                    strix::SerialOp::moe_experts,
            "TCP byte-stream carries distributed control wire frames unchanged");

        bool empty_rejected = false;
        try {
            left.send({});
        } catch (const std::invalid_argument&) {
            empty_rejected = true;
        }
        expect(
            empty_rejected,
            "TCP control stream rejects empty frame");
    }

    {
        const auto fds =
            socket_pair();
        auto sender =
            strix::TcpDistributedControlByteStream::
                adopt_connected(
                    fds[0],
                    16);
        auto receiver =
            strix::TcpDistributedControlByteStream::
                adopt_connected(
                    fds[1],
                    16);

        const std::vector<std::byte>
            too_large(
                17,
                std::byte{0});
        bool send_rejected = false;
        try {
            sender.send(
                too_large);
        } catch (const std::invalid_argument&) {
            send_rejected = true;
        }
        expect(
            send_rejected,
            "TCP control stream rejects oversized outbound frame");

        const std::byte oversized_prefix[4]{
            std::byte{17},
            std::byte{0},
            std::byte{0},
            std::byte{0},
        };
        if (::send(
                sender.native_handle(),
                oversized_prefix,
                sizeof(oversized_prefix),
                0) !=
            static_cast<ssize_t>(
                sizeof(oversized_prefix))) {
            throw std::runtime_error(
                "raw prefix send failed");
        }
        bool receive_rejected = false;
        try {
            static_cast<void>(
                receiver.receive());
        } catch (const std::invalid_argument&) {
            receive_rejected = true;
        }
        expect(
            receive_rejected,
            "TCP control stream rejects oversized inbound length before allocation");
    }

    if (failures == 0) {
        std::cout
            << "distributed TCP control stream tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
