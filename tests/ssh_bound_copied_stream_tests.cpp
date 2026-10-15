#include "strix/ssh_bound_copied_stream.hpp"
#include "strix/copied_stream_transport.hpp"
#include <algorithm>
#include <cstring>
#include <exception>
#include <future>
#include <iostream>
#include <netinet/in.h>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

#if defined(STRIX_ENABLE_SSH_BOUND_BOOTSTRAP)
#include <openssl/evp.h>
#endif

namespace {
using namespace strix;
using namespace strix::transport;
int failures = 0;
void expect(bool condition, const char* message) { if (!condition) { ++failures; std::cerr << "FAIL: " << message << '\n'; } }
template<class Function> bool rejected(Function&& function) { try { function(); } catch (const std::exception&) { return true; } return false; }
template<class Function> bool protocol_rejected(Function&& function, std::string_view expected_reason = {}) {
    try { function(); } catch (const std::exception& error) {
        const std::string message = error.what();
        if (!expected_reason.empty()) return message.find(expected_reason) != std::string::npos;
        return message.find("contract mismatch") != std::string::npos ||
            message.find("authentication failed") != std::string::npos ||
            message.find("transcript mismatch") != std::string::npos;
    }
    return false;
}
struct Fd {
    int value;
    explicit Fd(int fd) : value(fd) {}
    ~Fd() { if (value >= 0) ::close(value); }
    int release() { const auto fd = value; value = -1; return fd; }
};
std::array<int, 2> sockets() {
    Fd listener(::socket(AF_INET, SOCK_STREAM, 0));
    sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (listener.value < 0 || ::bind(listener.value, reinterpret_cast<sockaddr*>(&address), sizeof(address)) || ::listen(listener.value, 1))
        throw std::runtime_error("bootstrap loopback fixture listen failed");
    socklen_t length = sizeof(address);
    if (::getsockname(listener.value, reinterpret_cast<sockaddr*>(&address), &length)) throw std::runtime_error("bootstrap fixture address failed");
    Fd client(::socket(AF_INET, SOCK_STREAM, 0));
    if (client.value < 0 || ::connect(client.value, reinterpret_cast<sockaddr*>(&address), sizeof(address))) throw std::runtime_error("bootstrap fixture connect failed");
    Fd server(::accept(listener.value, nullptr, nullptr));
    if (server.value < 0) throw std::runtime_error("bootstrap fixture accept failed");
    return {client.release(), server.release()};
}
constexpr std::uint64_t application_limit = 65536, authenticated_overhead = 172;
TcpCopiedPayloadByteStream tcp(int fd) { return TcpCopiedPayloadByteStream::adopt_connected_loopback(fd,
    {application_limit + authenticated_overhead, 4096, std::chrono::milliseconds(1000)}); }
SshBootstrapNonce key() { SshBootstrapNonce bytes{}; bytes.fill(std::byte{0x42}); return bytes; }

#if defined(STRIX_ENABLE_SSH_BOUND_BOOTSTRAP)
SshCopiedSessionBinding binding() {
    SshCopiedSessionBinding value;
    value.session_nonce = generate_ssh_bootstrap_nonce(); value.route_identity = generate_ssh_bootstrap_nonce();
    for (auto& nonce : value.node_instances) nonce = generate_ssh_bootstrap_nonce();
    for (auto& nonce : value.ssh_host_key_pins) nonce = generate_ssh_bootstrap_nonce();
    for (auto& nonce : value.channel_nonces) nonce = generate_ssh_bootstrap_nonce();
    value.physical_link_id = "bootstrap-fixture-route"; value.model_content_identity.bytes[0] = 1;
    value.execution_plan_identity = 17; value.placement_plan_identity = 31;
    value.physical_epoch = 51; value.connection_epoch = 9; value.request_id = 11; value.generation = 2;
    value.maximum_application_frame_bytes = application_limit; value.maximum_payload_bytes = 20480; value.dtype_id = 7;
    return value;
}
using Pair = std::pair<SshBoundCopiedByteStream, SshBoundCopiedByteStream>;
Pair pair(const SshCopiedSessionBinding& left_binding, const SshCopiedSessionBinding& right_binding,
          SshBootstrapNonce left_key = key(), SshBootstrapNonce right_key = key(),
          SshCopiedChannel left_channel = SshCopiedChannel::payload, SshCopiedChannel right_channel = SshCopiedChannel::payload,
          std::uint32_t right_node = 1) {
    const auto fd = sockets();
    auto left_tcp = tcp(fd[0]); auto right_tcp = tcp(fd[1]);
    std::promise<SshBoundCopiedByteStream> promise; auto future = promise.get_future();
    std::jthread initiating([&] {
        try { promise.set_value(SshBoundCopiedByteStream::bootstrap(std::move(left_tcp), left_binding, left_key, 0, left_channel)); }
        catch (...) { promise.set_exception(std::current_exception()); }
    });
    auto right = SshBoundCopiedByteStream::bootstrap(std::move(right_tcp), right_binding, right_key, right_node, right_channel);
    auto left = future.get(); initiating.join(); return {std::move(left), std::move(right)};
}
void honest_tests() {
    const auto expected = binding();
    auto both = pair(expected, expected);
    const std::vector<std::byte> first{std::byte{0}, std::byte{255}, std::byte{3}};
    both.first.send(first); expect(both.second.receive() == first, "authenticated payload preserves exact bytes");
    const std::vector<std::byte> second(4096, std::byte{0xab});
    both.second.send(second); expect(both.first.receive() == second, "independent opposite-direction HKDF key authenticates replies");
    auto moved = std::move(both.first);
    expect(!both.first.valid() && moved.valid(), "authenticated channel capability is move-owned");
    expect(rejected([&] { (void)both.first.binding(); }), "moved capability exposes no binding authority");
    expect(rejected([&] { moved.send({}); }) && moved.failed(), "empty application send poisons authenticated channel");
    expect(rejected([&] { moved.send(first); }), "failed channel cannot restart or downgrade");
    expect(rejected([&] { (void)moved.binding(); }), "poisoned capability exposes no binding authority");
    both.second.poison(); expect(!both.second.valid() && rejected([&] { (void)both.second.receive(); }), "explicit cancel is one-shot and closes owned channel");
    auto control = pair(expected, expected, key(), key(), SshCopiedChannel::control, SshCopiedChannel::control);
    control.first.send(first); expect(control.second.receive() == first, "control channel is independently authenticated");
    CopiedStreamConnectionState state({expected.physical_link_id, expected.physical_epoch, 0, 1, 0});
    expect(rejected([&] { CopiedStreamTransportBackend wrong(std::move(control.first), state, {20480, application_limit, 16}); }),
           "control capability cannot be used as copied payload authority");
    expect(control.first.valid(), "rejected payload constructor retains caller's control capability");
}
void mismatch_tests() {
    const auto expected = binding();
    auto wrong_key = key(); wrong_key[0] ^= std::byte{1};
    expect(protocol_rejected([&] { (void)pair(expected, expected, key(), wrong_key); }, "authentication failed"), "bootstrap requires matching secret possession");
    expect(protocol_rejected([&] { (void)pair(expected, expected, key(), key(), SshCopiedChannel::control, SshCopiedChannel::payload); }, "contract mismatch"),
           "cross-channel bootstrap is rejected");
    expect(protocol_rejected([&] { (void)pair(expected, expected, key(), key(), SshCopiedChannel::payload, SshCopiedChannel::payload, 0); }, "contract mismatch"),
           "role reflection cannot authenticate two initiators");
    for (unsigned field = 0; field < 19; ++field) {
        auto foreign = expected;
        if (field == 0) foreign.session_nonce[0] ^= std::byte{1};
        else if (field == 1) ++foreign.request_id;
        else if (field == 2) ++foreign.generation;
        else if (field == 3) ++foreign.connection_epoch;
        else if (field == 4) ++foreign.dtype_id;
        else if (field == 5) ++foreign.placement_plan_identity;
        else if (field == 6) foreign.model_content_identity.bytes[0] ^= 2U;
        else if (field == 7) foreign.channel_nonces[0][0] ^= std::byte{1};
        else if (field == 8) foreign.channel_nonces[1][0] ^= std::byte{1};
        else if (field == 9) foreign.route_identity[0] ^= std::byte{1};
        else if (field == 10) foreign.physical_link_id += "-foreign";
        else if (field == 11) foreign.node_instances[0][0] ^= std::byte{1};
        else if (field == 12) foreign.node_instances[1][0] ^= std::byte{1};
        else if (field == 13) foreign.ssh_host_key_pins[0][0] ^= std::byte{1};
        else if (field == 14) foreign.ssh_host_key_pins[1][0] ^= std::byte{1};
        else if (field == 15) ++foreign.execution_plan_identity;
        else if (field == 16) ++foreign.physical_epoch;
        else if (field == 17) --foreign.maximum_payload_bytes;
        else ++foreign.link_index;
        expect(protocol_rejected([&] { (void)pair(expected, foreign); }, "contract mismatch"), "foreign expected session facts fail bootstrap for a protocol reason");
    }
}
void bootstrap_tag_tamper_test(unsigned attacked_phase) {
    const auto expected = binding(); const auto left = sockets(), right = sockets();
    auto left_tcp = tcp(left[0]), relay_left = tcp(left[1]);
    auto relay_right = tcp(right[0]), right_tcp = tcp(right[1]);
    auto initiating = std::async(std::launch::async, [&] {
        return SshBoundCopiedByteStream::bootstrap(std::move(left_tcp), expected, key(), 0, SshCopiedChannel::payload);
    });
    auto responding = std::async(std::launch::async, [&] {
        return SshBoundCopiedByteStream::bootstrap(std::move(right_tcp), expected, key(), 1, SshCopiedChannel::payload);
    });
    for (unsigned phase = 1; phase <= attacked_phase; ++phase) {
        auto& from = phase % 2U ? relay_left : relay_right;
        auto& to = phase % 2U ? relay_right : relay_left;
        auto frame = from.receive(); if (phase == attacked_phase) frame.back() ^= std::byte{1};
        to.send(frame);
    }
    expect(protocol_rejected([&] { (void)initiating.get(); }, "authentication failed"), "bootstrap reply/final-confirm tag tamper rejects for authentication reason");
    relay_left.poison(); relay_right.poison();
    try { auto responder = responding.get(); responder.poison(); } catch (const std::exception&) {
        // Phase2 attack prevents phase3; closing the malicious relay releases the responder.
    }
}
void unavailable_provider_test() {
    const auto expected = binding(); const auto fd = sockets();
    auto left = tcp(fd[0]); Fd other(fd[1]);
    if (EVP_set_default_properties(nullptr, "provider=nonexistent-bootstrap-test-provider") != 1)
        throw std::runtime_error("provider-negative fixture setup failed");
    const bool unavailable_rejected = protocol_rejected([&] {
        (void)SshBoundCopiedByteStream::bootstrap(std::move(left), expected, key(), 0, SshCopiedChannel::payload);
    }, "SHA256 unavailable");
    if (EVP_set_default_properties(nullptr, "") != 1) throw std::runtime_error("provider-negative fixture restore failed");
    expect(unavailable_rejected && !left.valid(), "unavailable crypto provider closes bootstrap instead of plaintext fallback");
}
enum class Fault { tamper, replay, reorder, reflect, foreign_frame, empty_frame, oversized_frame };
std::vector<std::byte> relay_fault(Fault fault, const SshCopiedSessionBinding& expected,
                                 std::vector<std::byte> old_frame = {}, SshCopiedChannel channel = SshCopiedChannel::payload) {
    const auto left = sockets(), right = sockets();
    auto left_tcp = tcp(left[0]), relay_left = tcp(left[1]);
    auto relay_right = tcp(right[0]), right_tcp = tcp(right[1]);
    std::vector<std::byte> captured;
    std::exception_ptr relay_error;
    std::jthread relay([&] {
        try {
            for (unsigned phase = 0; phase < 4; ++phase) {
                auto& from = phase % 2U == 0 ? relay_left : relay_right;
                auto& to = phase % 2U == 0 ? relay_right : relay_left;
                to.send(from.receive());
            }
            auto frame = relay_left.receive(); captured = frame;
            if (fault == Fault::tamper) { frame.back() ^= std::byte{1}; relay_right.send(frame); }
            else if (fault == Fault::replay) { relay_right.send(frame); relay_right.send(frame); }
            else if (fault == Fault::reorder) { const auto later = relay_left.receive(); relay_right.send(later); relay_right.send(frame); }
            else if (fault == Fault::reflect) relay_left.send(frame);
            else if (fault == Fault::foreign_frame) relay_right.send(old_frame);
            else {
                // Malicious relay owns this endpoint, and deliberately bypasses
                // its framing API. The authenticated receiver's FD is not exposed.
                const auto length = fault == Fault::empty_frame ? 0U : static_cast<std::uint32_t>(application_limit + authenticated_overhead + 1U);
                std::array<std::byte, 4> prefix{};
                for (unsigned shift = 0; shift < 32; shift += 8) prefix[shift / 8U] = static_cast<std::byte>((length >> shift) & 255U);
                if (::send(right[0], prefix.data(), prefix.size(), 0) != 4) throw std::runtime_error("malicious framing prefix send failed");
            }
        } catch (...) { relay_error = std::current_exception(); }
    });
    std::promise<SshBoundCopiedByteStream> promise; auto future = promise.get_future();
    std::jthread initiating([&] {
        try { promise.set_value(SshBoundCopiedByteStream::bootstrap(std::move(left_tcp), expected, key(), 0, channel)); }
        catch (...) { promise.set_exception(std::current_exception()); }
    });
    auto receiving = SshBoundCopiedByteStream::bootstrap(std::move(right_tcp), expected, key(), 1, channel);
    auto sending = future.get(); initiating.join();
    const std::vector<std::byte> payload(31, std::byte{0x91}); sending.send(payload);
    if (fault == Fault::reorder) sending.send(std::vector<std::byte>(23, std::byte{0x72}));
    if (fault == Fault::replay) expect(receiving.receive() == payload, "first authentic relay frame is accepted before replay rejection");
    auto& target = fault == Fault::reflect ? sending : receiving;
    const bool rejected_for_reason = fault == Fault::empty_frame || fault == Fault::oversized_frame
        ? rejected([&] { (void)target.receive(); }) : protocol_rejected([&] { (void)target.receive(); });
    expect(rejected_for_reason && target.failed(),
           "tamper/replay/reorder/reflection/foreign transcript fails before exposing data");
    expect(rejected([&] { target.send(payload); }), "authenticated protocol failure cannot resume");
    relay.join();
    // A rejected receiver can close while the malicious relay is writing its
    // second frame, which is the expected peer failure path.
    if (relay_error && fault != Fault::reorder && fault != Fault::replay) std::rethrow_exception(relay_error);
    return captured;
}
void authenticated_copied_test(unsigned foreign_field) {
    const auto expected = binding(); auto both = pair(expected, expected);
    CopiedStreamConnectionState state0({expected.physical_link_id, expected.physical_epoch, 0, 1, 0});
    CopiedStreamConnectionState state1({expected.physical_link_id, expected.physical_epoch, 1, 0, 0});
    const CopiedStreamTransportPolicy policy{20480, application_limit, 16};
    CopiedStreamTransportBackend sender(std::move(both.first), state0, policy), receiver(std::move(both.second), state1, policy);
    DistributedPlacementPlan plan; plan.plan_identity = expected.placement_plan_identity; plan.executable = true; plan.local_node = 0;
    DistributedExecutionBoundary boundary{91, 1, 2, 0, 1, 20480, {}}; plan.boundaries = {boundary};
    DistributedBoundaryTransfer transfer; transfer.flow = classify_distributed_boundary_flow(plan, boundary);
    transfer.descriptor = make_distributed_boundary_descriptor(plan, boundary, expected.connection_epoch, expected.request_id,
        expected.generation, 1, expected.dtype_id);
    transfer.link_plan.kind = PlanKind::single_link; transfer.link_plan.slices = {{0, 0, 20480}};
    AllocationGenerationSource generations;
    auto source = std::make_shared<CopiedStreamHostBuffer>(20480, generations);
    auto destination = std::make_shared<CopiedStreamHostBuffer>(20480, generations);
    for (std::size_t index = 0; index < source->bytes().size(); ++index) source->writable_bytes()[index] = static_cast<std::byte>((index * 17U) % 251U);
    const auto tx = sender.prepare_send(transfer, {{source, source->view()}, make_copied_stream_synchronous_proof(
        {transfer.descriptor, source->view(), CopiedStreamCompletionScope::producer_released})});
    const auto rx = receiver.prepare_receive(transfer, {destination, destination->view()});
    std::exception_ptr send_error; bool retired = false;
    std::jthread worker([&] { try {
        sender.submit(tx);
        for (unsigned poll = 0; poll < 4; ++poll) {
            const auto completion = sender.poll();
            if (completion && completion->kind == CompletionKind::remote_consumer_complete) { retired = true; break; }
        }
    } catch (...) { send_error = std::current_exception(); } });
    receiver.submit(rx); const auto complete = receiver.poll();
    expect(complete && complete->kind == CompletionKind::receive_complete, "MAC-protected copied envelope reaches validated RX");
    const auto view = receiver.receive_buffer(rx);
    expect(std::memcmp(reinterpret_cast<void*>(view.address), source->bytes().data(), 20480) == 0, "authenticated copied payload remains bit exact");
    receiver.finish_consumer(rx, make_copied_stream_synchronous_proof(
        {transfer.descriptor, view, CopiedStreamCompletionScope::consumer_complete}));
    const auto consumer = receiver.poll(); expect(consumer && consumer->kind == CompletionKind::local_buffer_reusable, "authenticated consumer proof retires RX");
    worker.join(); if (send_error) std::rethrow_exception(send_error);
    expect(retired && !sender.retains_buffer(), "authenticated per-frame acknowledgements retire TX");
    auto foreign = transfer;
    if (foreign_field == 0) ++foreign.descriptor.request_id;
    else if (foreign_field == 1) ++foreign.descriptor.generation;
    else if (foreign_field == 2) ++foreign.descriptor.connection_epoch;
    else if (foreign_field == 3) ++foreign.descriptor.dtype_id;
    else if (foreign_field == 4) ++foreign.descriptor.model_plan_id;
    else ++foreign.descriptor.byte_length;
    bool binding_rejection = false;
    try { (void)sender.prepare_send(foreign, {{source, source->view()}, make_copied_stream_synchronous_proof(
        {foreign.descriptor, source->view(), CopiedStreamCompletionScope::producer_released})}); }
    catch (const std::invalid_argument& error) {
        binding_rejection = std::string(error.what()).find("authenticated bootstrap binding") != std::string::npos;
    }
    expect(binding_rejection && sender.failed(), "verified capability rejects foreign logical facts/bytes for an admission reason");
}
#endif
} // namespace
int main() try {
#if defined(STRIX_ENABLE_SSH_BOUND_BOOTSTRAP)
    expect(ssh_bootstrap_crypto_compiled(), "crypto build capability is explicit");
    honest_tests(); mismatch_tests();
    bootstrap_tag_tamper_test(2); bootstrap_tag_tamper_test(4);
    const auto expected = binding();
    for (const auto fault : {Fault::tamper, Fault::replay, Fault::reorder, Fault::reflect, Fault::empty_frame, Fault::oversized_frame}) relay_fault(fault, expected);
    const auto old_frame = relay_fault(Fault::tamper, expected);
    relay_fault(Fault::foreign_frame, expected, old_frame); // Fresh handshake, same PSK/binding: old transcript still rejected.
    relay_fault(Fault::foreign_frame, expected, old_frame, SshCopiedChannel::control);
    for (unsigned field = 0; field < 6; ++field) authenticated_copied_test(field);
    unavailable_provider_test();
#else
    expect(!ssh_bootstrap_crypto_compiled(), "disabled crypto exposes no authenticated capability");
    expect(rejected([] { (void)generate_ssh_bootstrap_nonce(); }), "unavailable crypto never substitutes noncryptographic randomness");
    const auto fd = sockets(); auto left = tcp(fd[0]); Fd other(fd[1]);
    expect(rejected([&] { (void)SshBoundCopiedByteStream::bootstrap(std::move(left), {}, key(), 0, SshCopiedChannel::payload); }),
           "crypto-disabled bootstrap fails rather than downgrading to raw TCP");
#endif
    if (!failures) std::cout << "SSH-bound copied stream tests passed\n";
    return failures ? 1 : 0;
} catch (const std::exception& error) { std::cerr << "SSH-bound tests: " << error.what() << '\n'; return 2; }
