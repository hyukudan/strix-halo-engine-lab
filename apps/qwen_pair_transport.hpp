#pragma once
#include "strix/qwen_prefill_pair.hpp"
#include "../src/xport/pair_device_link.hpp"
#include <cstdlib>

namespace strix::qwen_pp {
// Qwen's pair link: the shared pair::DeviceLink (control on the host transport, seam pieces one way on the device
// route with device CRC) opened from this engine's transport setting.
class PairTransport final : public pair::DeviceLink {
public:
    explicit PairTransport(unsigned rank) : pair::DeviceLink(rank, 0x51575050ULL /* "QWPP" */, open(rank), "qwen-pair") {}
    // A link the deployment handshake already opened (it exchanged measured
    // facts and plan IDs on it): keep using it, host path only.
    explicit PairTransport(std::unique_ptr<dist::Transport> link) : pair::DeviceLink(std::move(link), "qwen-pair") {}
private:
    static dist::OpenedTransport open(unsigned rank) {
        dist::OpenTransportOptions options;
        options.usb.rank = rank;
        options.spellings = {"usb4auto", "usb4gpu", "usb4-auto", false, false, false};
        const char* value = std::getenv("STRIX_QWEN_PF_PIPE_TRANSPORT");
        std::string spec = value ? value : "usb4gpu";
        if (spec == "tcp") {
            const char* peer = std::getenv("STRIX_QWEN_PF_PIPE_PEER");
            if (rank == 1U && !peer) throw std::invalid_argument("pair prefill TCP rank 1 requires --peer HOST (port 29872)");
            spec = rank == 0U ? "tcp-listen:29872" : "tcp-connect:" + std::string(peer) + ":29872";
            options.spellings.tcp = true;
            options.usb.timeout = std::chrono::milliseconds(120000);
        }
        return dist::open_transport(spec, options);
    }
};
} // namespace strix::qwen_pp
