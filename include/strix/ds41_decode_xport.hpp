#pragma once

// STRIX_DS41_DECODE_XPORT=communicator|channel (also 0|1): which implementation carries the pair decode exchanges of
// BatchForward::run_step. `channel` leases the communicator's transport to the shared device-waited xport::Channel for
// the step: producers publish their TX slot on the GPU, consumers wait on the RX flag with a stream wait packet, and the
// host keeps enqueuing instead of spinning in finish() for each of the 80 o_b / MoE half-sum exchanges. Same kernels'
// arithmetic, same canonical order (rank 0's half first): bitwise. Default communicator until its ABAB receipt.
//
// One rule for every entry point: the pair defaults resolver (CLI and serve, --print-effective-config) and the engine
// both call decode_xport_choice(); there is no second reading of the variable.

#include "strix/lever.hpp"

#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::ds41 {

inline constexpr lever::Decl decode_xport_lever{
    "ds41-decode-xport", "STRIX_DS41_DECODE_XPORT", lever::Exactness::bitwise, false, {}, {},
    "LINKCOSTO S1: pending ABAB receipt (1 and 8 chats, 8K and 32K context)"};

// The value as written ("communicator" / "channel" / "0" / "1" / unset) -> the lever choice. Throws on anything else.
[[nodiscard]] inline lever::Choice decode_xport_choice(const char* value) {
    const std::string_view text = value != nullptr ? std::string_view(value) : std::string_view{};
    const char* normal = value;
    if (text == "communicator") normal = "0";
    else if (text == "channel") normal = "1";
    else if (!text.empty() && text != "0" && text != "1")
        throw std::invalid_argument(std::string(decode_xport_lever.env) + "=" + value + ": expected communicator or channel");
    return lever::select(decode_xport_lever, lever::Inputs{}, normal);
}

[[nodiscard]] inline const char* decode_xport_name(bool channel) noexcept { return channel ? "channel" : "communicator"; }

} // namespace strix::ds41
