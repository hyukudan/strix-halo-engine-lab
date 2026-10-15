#include "strix/ssh_bound_copied_stream.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

#if defined(STRIX_ENABLE_SSH_BOUND_BOOTSTRAP)
#include <openssl/core_names.h>
#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <openssl/kdf.h>
#include <openssl/params.h>
#include <openssl/rand.h>
#endif

namespace strix {
namespace {
#if defined(STRIX_ENABLE_SSH_BOUND_BOOTSTRAP)
constexpr std::size_t header_bytes = 140, tag_bytes = 32, envelope_bytes = header_bytes + tag_bytes;
using Bytes = std::vector<std::byte>;
using Key = std::array<unsigned char, 32>;
constexpr std::array<std::byte, 8> magic{std::byte{'S'}, std::byte{'H'}, std::byte{'C'}, std::byte{'P'},
    std::byte{'0'}, std::byte{'0'}, std::byte{'0'}, std::byte{'1'}};
bool nonzero(const SshBootstrapNonce& value) { return std::any_of(value.begin(), value.end(), [](std::byte b) { return b != std::byte{}; }); }
void append(Bytes& output, std::span<const std::byte> bytes) { output.insert(output.end(), bytes.begin(), bytes.end()); }
void number(Bytes& output, std::uint64_t value, unsigned width) {
    for (unsigned shift = 0; shift < width; shift += 8) output.push_back(static_cast<std::byte>((value >> shift) & 255U));
}
std::uint64_t number_at(std::span<const std::byte> bytes, std::size_t offset, unsigned width) {
    std::uint64_t value = 0;
    for (unsigned shift = 0; shift < width; shift += 8)
        value |= static_cast<std::uint64_t>(std::to_integer<unsigned char>(bytes[offset + shift / 8U])) << shift;
    return value;
}
Key digest(std::span<const std::byte> bytes) {
    Key result{}; std::size_t size = 0;
    if (EVP_Q_digest(nullptr, "SHA256", nullptr, bytes.data(), bytes.size(), result.data(), &size) != 1 || size != result.size())
        throw std::runtime_error("SSH bootstrap SHA256 unavailable");
    return result;
}
Key mac(const Key& key, std::span<const std::byte> bytes) {
    std::unique_ptr<EVP_MAC, decltype(&EVP_MAC_free)> algorithm(EVP_MAC_fetch(nullptr, "HMAC", nullptr), EVP_MAC_free);
    std::unique_ptr<EVP_MAC_CTX, decltype(&EVP_MAC_CTX_free)> context(algorithm ? EVP_MAC_CTX_new(algorithm.get()) : nullptr, EVP_MAC_CTX_free);
    char sha256[] = "SHA256";
    OSSL_PARAM parameters[]{OSSL_PARAM_construct_utf8_string(OSSL_MAC_PARAM_DIGEST, sha256, 0), OSSL_PARAM_construct_end()};
    Key result{}; std::size_t size = 0;
    if (!context || EVP_MAC_init(context.get(), key.data(), key.size(), parameters) != 1 ||
        EVP_MAC_update(context.get(), reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size()) != 1 ||
        EVP_MAC_final(context.get(), result.data(), &size, result.size()) != 1 || size != result.size())
        throw std::runtime_error("SSH bootstrap HMAC unavailable");
    return result;
}
void derive(Key& output, const Key& secret, const Key& binding_digest, const SshBootstrapNonce& initiator,
            const SshBootstrapNonce& responder, SshCopiedChannel channel, std::uint32_t sender) {
    std::unique_ptr<EVP_KDF, decltype(&EVP_KDF_free)> algorithm(EVP_KDF_fetch(nullptr, "HKDF", nullptr), EVP_KDF_free);
    std::unique_ptr<EVP_KDF_CTX, decltype(&EVP_KDF_CTX_free)> context(algorithm ? EVP_KDF_CTX_new(algorithm.get()) : nullptr, EVP_KDF_CTX_free);
    Bytes info(magic.begin(), magic.end()); number(info, 5, 32); number(info, static_cast<std::uint32_t>(channel), 32);
    number(info, sender, 32); number(info, 1U - sender, 32); append(info, initiator); append(info, responder);
    char sha256[] = "SHA256";
    char mode[] = "EXTRACT_AND_EXPAND";
    OSSL_PARAM parameters[]{OSSL_PARAM_construct_utf8_string(OSSL_KDF_PARAM_DIGEST, sha256, 0),
        OSSL_PARAM_construct_utf8_string(OSSL_KDF_PARAM_MODE, mode, 0),
        OSSL_PARAM_construct_octet_string(OSSL_KDF_PARAM_KEY, const_cast<unsigned char*>(secret.data()), secret.size()),
        OSSL_PARAM_construct_octet_string(OSSL_KDF_PARAM_SALT, const_cast<unsigned char*>(binding_digest.data()), binding_digest.size()),
        OSSL_PARAM_construct_octet_string(OSSL_KDF_PARAM_INFO, info.data(), info.size()), OSSL_PARAM_construct_end()};
    if (!context || EVP_KDF_derive(context.get(), output.data(), output.size(), parameters) != 1)
        throw std::runtime_error("SSH bootstrap HKDF unavailable");
}
Bytes canonical(const SshCopiedSessionBinding& binding) {
    if (!nonzero(binding.session_nonce) || !nonzero(binding.route_identity) || !binding.model_content_identity.valid() ||
        binding.physical_link_id.empty() || binding.physical_link_id.size() > 256 ||
        !nonzero(binding.node_instances[0]) || !nonzero(binding.node_instances[1]) || binding.node_instances[0] == binding.node_instances[1] ||
        !nonzero(binding.ssh_host_key_pins[0]) || !nonzero(binding.ssh_host_key_pins[1]) ||
        !nonzero(binding.channel_nonces[0]) || !nonzero(binding.channel_nonces[1]) || binding.channel_nonces[0] == binding.channel_nonces[1] ||
        !binding.execution_plan_identity || !binding.placement_plan_identity || !binding.physical_epoch || !binding.connection_epoch ||
        !binding.request_id || !binding.generation || binding.wire_codec_version != 2 || !binding.dtype_id || binding.link_index > 1 ||
        !binding.maximum_payload_bytes || binding.maximum_application_frame_bytes < 512 ||
        binding.maximum_payload_bytes > binding.maximum_application_frame_bytes - 512 ||
        binding.maximum_application_frame_bytes > 64ULL * 1024ULL * 1024ULL - envelope_bytes)
        throw std::invalid_argument("SSH bootstrap expected session binding is invalid");
    Bytes bytes(magic.begin(), magic.end()); append(bytes, binding.session_nonce); append(bytes, binding.route_identity);
    number(bytes, binding.physical_link_id.size(), 32);
    append(bytes, {reinterpret_cast<const std::byte*>(binding.physical_link_id.data()), binding.physical_link_id.size()});
    for (const auto& instance : binding.node_instances) append(bytes, instance);
    for (const auto& pin : binding.ssh_host_key_pins) append(bytes, pin);
    for (const auto& nonce : binding.channel_nonces) append(bytes, nonce);
    for (const auto byte : binding.model_content_identity.bytes) bytes.push_back(static_cast<std::byte>(byte));
    for (const auto value : {binding.execution_plan_identity, binding.placement_plan_identity, binding.physical_epoch,
                            binding.connection_epoch, binding.request_id, binding.generation,
                            binding.maximum_application_frame_bytes, binding.maximum_payload_bytes}) number(bytes, value, 64);
    number(bytes, binding.wire_codec_version, 32); number(bytes, binding.dtype_id, 32); number(bytes, binding.link_index, 32);
    return bytes;
}
#endif
} // namespace

bool ssh_bootstrap_crypto_compiled() noexcept {
#if defined(STRIX_ENABLE_SSH_BOUND_BOOTSTRAP)
    return true;
#else
    return false;
#endif
}
SshBootstrapNonce generate_ssh_bootstrap_nonce() {
#if defined(STRIX_ENABLE_SSH_BOUND_BOOTSTRAP)
    SshBootstrapNonce result{};
    if (RAND_bytes(reinterpret_cast<unsigned char*>(result.data()), static_cast<int>(result.size())) != 1 || !nonzero(result))
        throw std::runtime_error("SSH bootstrap secure nonce unavailable");
    return result;
#else
    throw std::runtime_error("SSH bootstrap crypto not built; no plaintext fallback");
#endif
}
struct SshBoundCopiedByteStream::Impl {
    SshCopiedSessionBinding binding;
    SshCopiedChannel channel = SshCopiedChannel::payload;
    bool verified = false, failed = false;
    std::uint32_t local = 0;
#if defined(STRIX_ENABLE_SSH_BOUND_BOOTSTRAP)
    TcpCopiedPayloadByteStream socket;
    Key binding_digest{}, secret{}, send_key{}, receive_key{};
    SshBootstrapNonce initiator{}, responder{};
    std::uint64_t next_send = 1, next_receive = 1;
    Impl(TcpCopiedPayloadByteStream&& stream, const SshCopiedSessionBinding& expected, std::span<const std::byte> key,
         std::uint32_t owner, SshCopiedChannel kind) : binding(expected), channel(kind), local(owner), socket(std::move(stream)) {
        binding_digest = digest(canonical(binding));
        std::memcpy(secret.data(), key.data(), secret.size());
    }
    ~Impl() { clear_keys(); }
    void clear_keys() noexcept {
        OPENSSL_cleanse(secret.data(), secret.size()); OPENSSL_cleanse(send_key.data(), send_key.size());
        OPENSSL_cleanse(receive_key.data(), receive_key.size());
    }
    void poison() noexcept { failed = true; verified = false; socket.poison(); clear_keys(); }
    void healthy() const { if (failed || !verified || !socket.valid()) throw std::logic_error("SSH-bound stream is unavailable or poisoned"); }
    Bytes encode(std::uint32_t phase, std::uint32_t sender, std::uint64_t sequence, std::span<const std::byte> payload, const Key& key) {
        Bytes frame(magic.begin(), magic.end()); number(frame, 1, 32); number(frame, phase, 32);
        number(frame, static_cast<std::uint32_t>(channel), 32); number(frame, sender, 32); number(frame, 1U - sender, 32);
        number(frame, sequence, 64); number(frame, payload.size(), 64);
        append(frame, {reinterpret_cast<const std::byte*>(binding_digest.data()), binding_digest.size()});
        append(frame, initiator); append(frame, responder); append(frame, payload);
        const auto tag = mac(key, frame); append(frame, {reinterpret_cast<const std::byte*>(tag.data()), tag.size()});
        return frame;
    }
    std::span<const std::byte> verify(const Bytes& frame, std::uint32_t phase, std::uint32_t sender,
                                    std::uint64_t sequence, const Key& key, bool initial = false) {
        if (frame.size() < envelope_bytes || !std::equal(magic.begin(), magic.end(), frame.begin()) ||
            number_at(frame, 8, 32) != 1 || number_at(frame, 12, 32) != phase ||
            number_at(frame, 16, 32) != static_cast<std::uint32_t>(channel) || number_at(frame, 20, 32) != sender ||
            number_at(frame, 24, 32) != 1U - sender || number_at(frame, 28, 64) != sequence ||
            number_at(frame, 36, 64) > binding.maximum_application_frame_bytes ||
            number_at(frame, 36, 64) != frame.size() - envelope_bytes ||
            std::memcmp(frame.data() + 44, binding_digest.data(), binding_digest.size()) != 0 ||
            (phase != 5 && frame.size() != envelope_bytes) || (phase == 5 && frame.size() == envelope_bytes))
            throw std::runtime_error("SSH-bound frame contract mismatch");
        const auto tag = mac(key, {frame.data(), frame.size() - tag_bytes});
        if (CRYPTO_memcmp(tag.data(), frame.data() + frame.size() - tag_bytes, tag.size()) != 0)
            throw std::runtime_error("SSH-bound frame authentication failed");
        if (!initial && (std::memcmp(frame.data() + 76, initiator.data(), initiator.size()) != 0 ||
                         std::memcmp(frame.data() + 108, responder.data(), responder.size()) != 0))
            throw std::runtime_error("SSH-bound frame transcript mismatch");
        return {frame.data() + header_bytes, frame.size() - envelope_bytes};
    }
    void bootstrap() {
        if (local == 0) {
            initiator = generate_ssh_bootstrap_nonce();
            socket.send(encode(1, 0, 1, {}, secret));
            const auto reply = socket.receive(); verify(reply, 2, 1, 2, secret, true);
            if (std::memcmp(reply.data() + 76, initiator.data(), initiator.size()) != 0)
                throw std::runtime_error("SSH bootstrap initiator nonce mismatch");
            std::memcpy(responder.data(), reply.data() + 108, responder.size());
            if (!nonzero(responder) || responder == initiator) throw std::runtime_error("SSH bootstrap responder nonce invalid");
            socket.send(encode(3, 0, 3, {}, secret));
            const auto ready = socket.receive(); verify(ready, 4, 1, 4, secret);
        } else {
            const auto hello = socket.receive(); verify(hello, 1, 0, 1, secret, true);
            std::memcpy(initiator.data(), hello.data() + 76, initiator.size());
            if (!nonzero(initiator) || std::any_of(hello.begin() + 108, hello.begin() + 140, [](std::byte byte) { return byte != std::byte{}; }))
                throw std::runtime_error("SSH bootstrap hello nonce invalid");
            responder = generate_ssh_bootstrap_nonce();
            if (responder == initiator) throw std::runtime_error("SSH bootstrap nonce collision");
            socket.send(encode(2, 1, 2, {}, secret));
            const auto confirm = socket.receive(); verify(confirm, 3, 0, 3, secret);
            socket.send(encode(4, 1, 4, {}, secret));
        }
        derive(send_key, secret, binding_digest, initiator, responder, channel, local);
        derive(receive_key, secret, binding_digest, initiator, responder, channel, 1U - local);
        OPENSSL_cleanse(secret.data(), secret.size());
        verified = true;
    }
#else
    void poison() noexcept { failed = true; verified = false; }
#endif
};
SshBoundCopiedByteStream::SshBoundCopiedByteStream(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SshBoundCopiedByteStream SshBoundCopiedByteStream::bootstrap(TcpCopiedPayloadByteStream&& socket,
    const SshCopiedSessionBinding& binding, std::span<const std::byte> secret, std::uint32_t local_node, SshCopiedChannel channel) {
#if defined(STRIX_ENABLE_SSH_BOUND_BOOTSTRAP)
    if (secret.size() != 32 || std::none_of(secret.begin(), secret.end(), [](std::byte byte) { return byte != std::byte{}; }) ||
        local_node > 1 || (channel != SshCopiedChannel::control && channel != SshCopiedChannel::payload) || !socket.valid())
        throw std::invalid_argument("SSH bootstrap requires trusted ephemeral key, owner and channel");
    static_cast<void>(canonical(binding));
    if (socket.maximum_frame_bytes() != binding.maximum_application_frame_bytes + envelope_bytes)
        throw std::invalid_argument("SSH bootstrap socket ceiling does not match admitted frame budget");
    auto impl = std::make_unique<Impl>(std::move(socket), binding, secret, local_node, channel);
    try { impl->bootstrap(); } catch (...) { impl->poison(); throw; }
    return SshBoundCopiedByteStream(std::move(impl));
#else
    static_cast<void>(socket); static_cast<void>(binding); static_cast<void>(secret); static_cast<void>(local_node); static_cast<void>(channel);
    throw std::runtime_error("SSH bootstrap crypto not built; no plaintext fallback");
#endif
}
SshBoundCopiedByteStream::SshBoundCopiedByteStream(SshBoundCopiedByteStream&&) noexcept = default;
SshBoundCopiedByteStream& SshBoundCopiedByteStream::operator=(SshBoundCopiedByteStream&& other) noexcept {
    if (this != &other) { poison(); impl_ = std::move(other.impl_); }
    return *this;
}
SshBoundCopiedByteStream::~SshBoundCopiedByteStream() { poison(); }
void SshBoundCopiedByteStream::poison() noexcept { if (impl_) impl_->poison(); }
bool SshBoundCopiedByteStream::valid() const noexcept { return impl_ && impl_->verified && !impl_->failed; }
bool SshBoundCopiedByteStream::failed() const noexcept { return !impl_ || impl_->failed; }
SshCopiedChannel SshBoundCopiedByteStream::channel() const noexcept { return impl_ ? impl_->channel : SshCopiedChannel::payload; }
std::uint32_t SshBoundCopiedByteStream::local_node() const noexcept { return impl_ ? impl_->local : 0; }
std::uint64_t SshBoundCopiedByteStream::maximum_frame_bytes() const noexcept { return impl_ ? impl_->binding.maximum_application_frame_bytes : 0; }
const SshCopiedSessionBinding& SshBoundCopiedByteStream::binding() const {
    if (!valid()) throw std::logic_error("SSH-bound session binding is unavailable");
    return impl_->binding;
}
void SshBoundCopiedByteStream::send(std::span<const std::byte> frame) {
#if defined(STRIX_ENABLE_SSH_BOUND_BOOTSTRAP)
    if (!impl_) throw std::logic_error("SSH-bound stream is moved from");
    try {
        impl_->healthy();
        if (frame.empty() || frame.size() > maximum_frame_bytes() || impl_->next_send == std::numeric_limits<std::uint64_t>::max())
            throw std::invalid_argument("SSH-bound frame size or sequence invalid");
        impl_->socket.send(impl_->encode(5, impl_->local, impl_->next_send, frame, impl_->send_key));
        ++impl_->next_send;
    } catch (...) { poison(); throw; }
#else
    static_cast<void>(frame); throw std::runtime_error("SSH bootstrap crypto not built; no plaintext fallback");
#endif
}
std::vector<std::byte> SshBoundCopiedByteStream::receive() {
#if defined(STRIX_ENABLE_SSH_BOUND_BOOTSTRAP)
    if (!impl_) throw std::logic_error("SSH-bound stream is moved from");
    try {
        impl_->healthy();
        if (impl_->next_receive == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("SSH-bound receive sequence exhausted");
        const auto frame = impl_->socket.receive();
        const auto payload = impl_->verify(frame, 5, 1U - impl_->local, impl_->next_receive, impl_->receive_key);
        ++impl_->next_receive;
        return {payload.begin(), payload.end()};
    } catch (...) { poison(); throw; }
#else
    throw std::runtime_error("SSH bootstrap crypto not built; no plaintext fallback");
#endif
}
} // namespace strix
