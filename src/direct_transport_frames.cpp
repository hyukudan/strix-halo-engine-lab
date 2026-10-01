#include "strix/direct_transport_frames.hpp"

#include <limits>
#include <stdexcept>

namespace strix::transport {
namespace {

std::uint64_t ceil_div(
    std::uint64_t numerator,
    std::uint64_t denominator) {
    if (denominator == 0) {
        throw std::invalid_argument(
            "direct transport frame divisor must be non-zero");
    }
    return numerator / denominator +
           (numerator % denominator != 0 ? 1ULL : 0ULL);
}

}  // namespace

DirectMessageFramePlan
plan_direct_message_frames(
    std::uint64_t payload_bytes,
    const DirectFrameGeometry& geometry,
    std::uint32_t start_frame) {
    if (payload_bytes == 0 ||
        geometry.frame_bytes == 0 ||
        geometry.ring_frames < 2 ||
        start_frame >= geometry.ring_frames) {
        throw std::invalid_argument(
            "invalid direct transport frame geometry");
    }

    const auto payload_frames_u64 =
        ceil_div(
            payload_bytes,
            geometry.frame_bytes);
    const auto terminal_frames_u64 =
        geometry.dedicated_terminal_frame
            ? 1ULL
            : 0ULL;
    const auto total_frames_u64 =
        payload_frames_u64 +
        terminal_frames_u64;

    if (payload_frames_u64 >
            std::numeric_limits<std::uint32_t>::max() ||
        total_frames_u64 >
            geometry.ring_frames) {
        throw std::invalid_argument(
            "direct message does not fit one ring traversal");
    }

    DirectMessageFramePlan plan;
    plan.payload_bytes = payload_bytes;
    plan.payload_frames =
        static_cast<std::uint32_t>(
            payload_frames_u64);
    plan.terminal_frames =
        static_cast<std::uint32_t>(
            terminal_frames_u64);
    plan.total_frames =
        static_cast<std::uint32_t>(
            total_frames_u64);
    plan.start_frame = start_frame;

    const auto terminal_offset =
        geometry.dedicated_terminal_frame
            ? plan.payload_frames
            : plan.payload_frames - 1U;
    plan.terminal_frame =
        static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(
                 start_frame) +
             terminal_offset) %
            geometry.ring_frames);

    const auto end_linear =
        static_cast<std::uint64_t>(
            start_frame) +
        plan.total_frames;
    plan.wraps =
        end_linear >
        geometry.ring_frames;
    plan.next_frame =
        static_cast<std::uint32_t>(
            end_linear %
            geometry.ring_frames);

    return plan;
}

}  // namespace strix::transport
