#include "strix/dist/link_layout.hpp"
#include "strix/dist/transport_spec.hpp"
#include <iostream>

namespace {
void require(bool ok) { if (!ok) throw std::runtime_error("transport parser contract"); }
void rejects(const std::string& text, strix::dist::TransportSpellings spellings) {
    bool caught = false;
    try { (void)strix::dist::parse_transport(text, spellings); }
    catch (const std::invalid_argument&) { caught = true; }
    require(caught);
}
}
int main() try {
    using namespace strix::dist;
    require(resolve_device_route({}) == "table");
    require(device_route_identity({}) == device_route_identity("table"));
    require(device_route_identity("auto") != device_route_identity("table"));
    require(device_route_identity("65536:host,inf:gpu:2") != device_route_identity("65536:host,inf:gpu:1"));
    // One dialect for every pair binary: both spellings of each USB4 kind, TCP endpoints and device lists.
    const auto dialect = pair_transport;
    for (const char* host : {"usb4auto", "usb4-auto"})
        require(parse_transport(host, dialect).kind == TransportKind::usb4_auto);
    for (const char* gpu : {"usb4gpu", "usb4-gpu"})
        require(parse_transport(gpu, dialect).kind == TransportKind::usb4_gpu);
    // Negotiation names the route in the caller's canonical spelling, whichever spelling was requested.
    for (const char* gpu : {"usb4gpu", "usb4-gpu"}) {
        require(negotiated_transport(gpu, false) == "usb4auto");
        require(negotiated_transport(gpu, true) == "usb4gpu");
    }
    require(negotiated_transport("usb4-auto:ignored", false) == "usb4auto");
    // MiniMax's route names.
    require(negotiated_transport("usb4-gpu", false, "usb4-auto", "usb4gpu") == "usb4-auto");
    require(negotiated_transport("usb4auto", true, "usb4-auto", "usb4gpu") == "usb4-auto");
    require(negotiated_transport("usb4-gpu", true, "usb4-auto", "usb4gpu") == "usb4gpu");
    require(negotiated_transport("tcp-listen:29872", false) == "tcp-listen:29872");
    const auto listen = parse_transport("tcp-listen:29872", dialect);
    require(listen.kind == TransportKind::tcp_listen && listen.port == 29872);
    const auto connect = parse_transport("tcp-connect:peer:29872", dialect);
    require(connect.kind == TransportKind::tcp_connect && connect.host == "peer" && connect.port == 29872);
    rejects("TCP-listen:29872", dialect);
    rejects("tcp-listen:", dialect);
    rejects("tcp-connect:peer:bad", dialect);
    rejects("usb4_gpu_auto", dialect); // manifest mapping remains an app concern
    // The former kind/rest split, including ignored USB4 suffixes and the stoul/narrowing TCP behavior.
    require(parse_transport("usb4-auto:ignored", dialect).kind == TransportKind::usb4_auto);
    require(parse_transport("tcp-listen:65537suffix", dialect).port == 1);
    const auto devices = parse_transport("usb4:a,,b,", dialect);
    require(devices.kind == TransportKind::usb4_host && devices.devices == std::vector<std::string>({"a", "", "b"}));
    // A restricted dialect (no TCP, no device list, no suffix) still rejects what it does not admit.
    const TransportSpellings strict{"usb4auto", "usb4gpu", "usb4-auto", false, false, false};
    require(parse_transport("usb4-auto", strict).kind == TransportKind::usb4_auto);
    require(parse_transport("usb4-gpu", strict).kind == TransportKind::usb4_gpu);
    rejects("tcp-listen:29872", strict);
    rejects("usb4:a", strict);
    rejects("usb4auto:ignored", strict);
    rejects("usb4gpu:ignored", strict);
    // Link layout keys: the measured pair layouts are named by cables x host streams per cable x link rate.
    struct Stream { std::string xdomain; double link_gbps; };
    const std::vector<Stream> layout1{{"a", 40.0}, {"b", 40.0}, {"a", 40.0}, {"b", 40.0}};  // data0+data1 host, 2 cables
    const std::vector<Stream> layout2bq{{"a", 40.0}, {"b", 40.0}};                          // data0 host, data1 gpu_direct
    require(classify_link(layout1, false).key() == "usb4-host-2x2x40");
    require(classify_link(layout2bq, false).key() == "usb4-host-2x1x40");
    require(classify_link(layout2bq, true).key() == "usb4-host-2x1x40+direct");
    require(classify_link(std::vector<Stream>{{"a", 40.0}, {"a", 40.0}, {"b", 40.0}}, false).key() == "unknown");  // uneven
    require(classify_link(std::vector<Stream>{{"a", 40.0}, {"b", 20.0}}, false).key() == "unknown");               // mixed rate
    require(classify_link(std::vector<Stream>{{"", 40.0}, {"b", 40.0}}, false).key() == "unknown");                // no XDomain
    require(classify_link(std::vector<Stream>{}, false).key() == "unknown");
    constexpr std::string_view measured[] = {"usb4-host-2x2x40", "usb4-host-2x1x40"};
    require(measured_layout(classify_link(layout1, false), measured));
    require(measured_layout(classify_link(layout2bq, false), measured));
    require(!measured_layout(classify_link(layout2bq, true), measured));
    require(!measured_layout(classify_link(std::vector<Stream>{{"a", 40.0}}, false), measured));
    std::cout << "PASS: one pair dialect, TCP endpoints, USB4 devices, aliases, rejection contracts and link layout keys\n";
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
