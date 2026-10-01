#pragma once

#include <cstdint>

namespace strix::transport {

struct DirectFrameGeometry {
    std::uint32_t frame_bytes = 4096;
    std::uint32_t ring_frames = 0;
    bool dedicated_terminal_frame = true;
};

struct DirectMessageFramePlan {
    std::uint64_t payload_bytes = 0;
    std::uint32_t payload_frames = 0;
    std::uint32_t terminal_frames = 0;
    std::uint32_t total_frames = 0;
    std::uint32_t start_frame = 0;
    std::uint32_t terminal_frame = 0;
    std::uint32_t next_frame = 0;
    bool wraps = false;
};

[[nodiscard]] DirectMessageFramePlan
plan_direct_message_frames(
    std::uint64_t payload_bytes,
    const DirectFrameGeometry& geometry,
    std::uint32_t start_frame);

}  // namespace strix::transport
