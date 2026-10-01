#include "strix/direct_transport_frames.hpp"

#include <iostream>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
    strix::transport::DirectFrameGeometry geometry;
    geometry.frame_bytes = 4096;
    geometry.ring_frames = 4095;
    geometry.dedicated_terminal_frame = true;

    const auto exact =
        strix::transport::plan_direct_message_frames(
            64ULL * 1024ULL,
            geometry,
            0);
    expect(exact.payload_frames == 16 &&
               exact.terminal_frames == 1 &&
               exact.total_frames == 17 &&
               exact.terminal_frame == 16 &&
               exact.next_frame == 17 &&
               !exact.wraps,
           "exact 64 KiB payload reserves sixteen payload frames plus terminal frame");

    const auto wrapped =
        strix::transport::plan_direct_message_frames(
            64ULL * 1024ULL,
            geometry,
            4090);
    expect(wrapped.wraps &&
               wrapped.terminal_frame == 11 &&
               wrapped.next_frame == 12,
           "message advances modulo physical frame ring when crossing wrap");

    const auto small =
        strix::transport::plan_direct_message_frames(
            32,
            geometry,
            100);
    expect(small.payload_frames == 1 &&
               small.terminal_frames == 1 &&
               small.total_frames == 2 &&
               small.terminal_frame == 101,
           "small payload still owns a separate terminal commit frame");

    bool too_large = false;
    try {
        static_cast<void>(
            strix::transport::plan_direct_message_frames(
                static_cast<std::uint64_t>(
                    geometry.frame_bytes) *
                    geometry.ring_frames,
                geometry,
                0));
    } catch (const std::invalid_argument&) {
        too_large = true;
    }
    expect(too_large,
           "payload consuming whole ring is rejected because terminal frame needs capacity");

    bool zero_rejected = false;
    try {
        static_cast<void>(
            strix::transport::plan_direct_message_frames(
                0,
                geometry,
                0));
    } catch (const std::invalid_argument&) {
        zero_rejected = true;
    }
    expect(zero_rejected,
           "zero-length direct message is rejected");

    if (failures == 0) {
        std::cout << "direct transport frame tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
