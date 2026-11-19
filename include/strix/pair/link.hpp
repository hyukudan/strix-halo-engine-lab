#pragma once
// Pair-route link setup shared by every engine's two-node route (Qwen, Gemma 4). The command line (strip_flags:
// --rank/--peer/--transport, or a node topology) is in pair/link_flags.hpp; open_link() opens the ordered byte
// transport the pair pipe runs on from the exported <PREFIX>_RANK / _PEER / _TRANSPORT.
#include "strix/dist/open_transport.hpp"
#include "strix/pair/link_flags.hpp"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix::pair {
// transport: usb4auto (default; the striped USB4 host path), usb4gpu (the same control path plus the
// device exchange, for a link that moves seam pieces device to device: xport/pair_device_link.hpp) or
// tcp (rank 0 listens on port, rank 1 connects to peer), opened by the single shared opener
// (strix/dist/open_transport.hpp). Both ranks block here until the peer appears: the first common
// point of a pair run.
inline dist::OpenedTransport open_device_link(unsigned rank, std::string_view prefix, std::uint16_t port, bool device = true) {
    const std::string base(prefix);
    const char* transport_env = std::getenv((base + "_TRANSPORT").c_str());
    std::string spec = transport_env ? transport_env : "usb4auto";
    dist::OpenTransportOptions options;
    options.usb.rank = rank;
    if (spec == "tcp") {
        const char* peer = std::getenv((base + "_PEER").c_str());
        if (rank == 1U && !peer)
            throw std::invalid_argument("pair prefill TCP rank 1 requires --peer HOST (port " + std::to_string(port) + ")");
        spec = rank == 0U ? "tcp-listen:" + std::to_string(port) : "tcp-connect:" + std::string(peer) + ":" + std::to_string(port);
        options.usb.timeout = std::chrono::milliseconds(120000);
    } else if (device && spec == "usb4gpu") {
        options.spellings = {"usb4auto", "usb4gpu", "usb4-auto", false, false, false};
    } else if (spec != "usb4auto")
        throw std::invalid_argument(device ? "pair prefill transport must be usb4auto, usb4gpu or tcp"
                                           : "pair prefill transport must be usb4auto or tcp");
    return dist::open_transport(spec, options);
}
// The host path only (usb4auto or tcp).
inline std::unique_ptr<dist::Transport> open_link(unsigned rank, std::string_view prefix, std::uint16_t port) {
    return open_device_link(rank, prefix, port, false).host;
}
} // namespace strix::pair
