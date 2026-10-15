#pragma once

#include "strix/tcp_copied_payload_stream.hpp"
#include "strix/model_runtime_binding.hpp"
#include <array>
#include <memory>
#include <string>

namespace strix {

using SshBootstrapNonce = std::array<std::byte, 32>;
enum class SshCopiedChannel : std::uint32_t { control = 1, payload = 2 };

// Trusted bootstrap owner's expected facts, independently reconciled through
// pinned SSH. This library binds these facts cryptographically; it cannot
// verify an SSH process, host-key pin or loaded model merely from their bytes.
struct SshCopiedSessionBinding {
    SshBootstrapNonce session_nonce{}, route_identity{};
    std::string physical_link_id;
    std::array<SshBootstrapNonce, 2> node_instances{}, ssh_host_key_pins{}, channel_nonces{};
    ModelContentIdentity model_content_identity;
    std::uint64_t execution_plan_identity = 0, placement_plan_identity = 0;
    std::uint64_t physical_epoch = 0, connection_epoch = 0, request_id = 0, generation = 0;
    std::uint64_t maximum_application_frame_bytes = 0, maximum_payload_bytes = 0;
    std::uint32_t wire_codec_version = 2, dtype_id = 0, link_index = 0;
};

[[nodiscard]] bool ssh_bootstrap_crypto_compiled() noexcept;
// OpenSSL CSPRNG only. Caller owns provisioning and wiping these ephemeral
// bytes; never place them in argv, logs or plaintext credential files.
[[nodiscard]] SshBootstrapNonce generate_ssh_bootstrap_nonce();

// Experimental PSK-possession/channel binding, NOT SSH provisioning, mTLS,
// model admission or distributed inference qualification. Bootstrap is four
// role/direction-separated messages. Every application frame then uses a
// fresh-transcript HKDF key, strict direction/channel sequence and HMAC-SHA256.
// Move-only capability: owns its bounded TCP socket, exposes no bare socket,
// and permanently poisons/closes on cancellation or any framing/MAC failure.
// A single serialized progress owner must call send, receive and poison.
// Cross-thread cancellation is unsupported; a blocked call ends at its deadline.
class SshBoundCopiedByteStream final : public DistributedControlByteStream {
public:
    // local_node must be 0 (initiator) or 1 (responder). The 32-byte secret
    // comes from the trusted pinned-SSH session owner, never from loopback.
    [[nodiscard]] static SshBoundCopiedByteStream bootstrap(
        TcpCopiedPayloadByteStream&&, const SshCopiedSessionBinding&,
        std::span<const std::byte> ephemeral_secret, std::uint32_t local_node, SshCopiedChannel);
    SshBoundCopiedByteStream(const SshBoundCopiedByteStream&) = delete;
    SshBoundCopiedByteStream& operator=(const SshBoundCopiedByteStream&) = delete;
    SshBoundCopiedByteStream(SshBoundCopiedByteStream&&) noexcept;
    SshBoundCopiedByteStream& operator=(SshBoundCopiedByteStream&&) noexcept;
    ~SshBoundCopiedByteStream() override;
    void send(std::span<const std::byte>) override;
    [[nodiscard]] std::vector<std::byte> receive() override;
    void poison() noexcept;
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] bool failed() const noexcept;
    [[nodiscard]] SshCopiedChannel channel() const noexcept;
    [[nodiscard]] std::uint32_t local_node() const noexcept;
    [[nodiscard]] std::uint64_t maximum_frame_bytes() const noexcept;
    [[nodiscard]] const SshCopiedSessionBinding& binding() const;
private:
    struct Impl;
    explicit SshBoundCopiedByteStream(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
};

} // namespace strix
