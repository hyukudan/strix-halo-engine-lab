#pragma once
#include "strix/qwen_prefill_pair.hpp"
#include "strix/pair/state_staging.hpp"
#include <limits>
#include <future>

namespace strix::qwen_pp {
inline bool decode_enabled() {
    const char* value = std::getenv("STRIX_QWEN_PF_PIPE_DECODE");
    return value && std::string_view(value) == "1";
}
// pair handoff v1: ordered, bounded pieces. All header words are compared before any
// payload is installed. ACK follows CRC validation AND the consumer's fence.
struct StatePart { std::uint32_t layer; std::uint32_t component; std::uint64_t bytes; };
// MTP-only extension. Plain pair handoff v1 framing and component order stay unchanged.
// Header: magic, version, predictor frontier, terminal input token, tap bytes,
// KV/raw-index row strides, numerical profile and relocatable drafter binding.
using MtpStateDescriptor = std::array<std::uint64_t, 9>;
inline constexpr std::uint64_t mtp_state_magic = 0x3150544d335050ULL;
inline constexpr std::uint32_t mtp_state_layer = 49U;
inline std::array<StatePart, 4> mtp_state_parts(std::uint64_t position,
    const MtpStateDescriptor& descriptor) {
    if (descriptor[0] != mtp_state_magic || descriptor[1] != 1U || !position ||
        descriptor[2] != position - 1U || descriptor[3] > std::numeric_limits<std::uint32_t>::max() ||
        !descriptor[4] || !descriptor[5] || !descriptor[6])
        throw std::invalid_argument("pair handoff MTP descriptor mismatch");
    const auto cache_bytes = [&](std::uint64_t stride) {
        if (descriptor[2] > std::numeric_limits<std::uint64_t>::max() / stride)
            throw std::overflow_error("pair handoff MTP cache size overflow");
        return descriptor[2] * stride;
    };
    return {{{mtp_state_layer, 0U, sizeof(MtpStateDescriptor)},
        {mtp_state_layer, 1U, descriptor[4]},
        {mtp_state_layer, 2U, cache_bytes(descriptor[5])},
        {mtp_state_layer, 3U, cache_bytes(descriptor[6])}}};
}
inline std::uint64_t mtp_state_identity(std::uint64_t identity, const MtpStateDescriptor& descriptor) {
    for (const auto word : descriptor)
        for (unsigned shift = 0; shift < 64U; shift += 8U)
            identity = (identity ^ ((word >> shift) & 255U)) * 1099511628211ULL;
    return identity;
}

using StateHeader = std::array<std::uint64_t, 12>;
inline StateHeader state_header(const StatePart& part, std::uint64_t identity,
    std::uint64_t position, std::uint32_t cut, std::uint64_t sequence,
    std::uint64_t offset, std::uint64_t bytes) {
    return {0x315453335050ULL, 1U, identity, position, cut, sequence,
        part.layer, part.component, part.bytes, offset, bytes, 0U};
}
template<class Send, class Receive, class Copy>
std::uint64_t handoff(bool producer, std::span<const StatePart> parts,
    std::uint64_t identity, std::uint64_t position, std::uint32_t cut,
    Send send, Receive receive, Copy copy) {
    // Synthetic layer 48 carries only terminal logits. The shared input PLE
    // history belongs to stage A. Check before transport or device copies.
    for (const auto& part : parts)
        if (part.layer == 48U && part.component != 0xffffffffU)
            throw std::invalid_argument("pair handoff shared input state is owned by stage A");
    std::vector<std::byte> staging(seam_piece_bytes);
    std::uint64_t total = 0, sequence = 0;
    for (std::size_t part_index = 0; part_index < parts.size(); ++part_index) {
        const auto& part = parts[part_index];
        for (std::uint64_t offset = 0; offset < part.bytes;) {
            const auto size = std::min<std::uint64_t>(seam_piece_bytes, part.bytes - offset);
            auto payload = std::span<std::byte>(staging).first(static_cast<std::size_t>(size));
            auto header = state_header(part, identity, position, cut, sequence, offset, size);
            if (producer) {
                copy(true, part_index, offset, payload);
                SeamCrc crc; crc.update(payload); header.back() = crc.value();
                send(std::as_bytes(std::span(header))); send(payload);
                StateHeader ack{}; receive(std::as_writable_bytes(std::span(ack)));
                if (ack != header) throw std::runtime_error("pair handoff state ACK mismatch");
            } else {
                StateHeader incoming{}; receive(std::as_writable_bytes(std::span(incoming)));
                const auto expected_crc = incoming.back(); incoming.back() = 0U;
                if (incoming != header) throw std::runtime_error("pair handoff state framing mismatch");
                receive(payload);
                SeamCrc crc; crc.update(payload);
                if (expected_crc != crc.value()) throw std::runtime_error("pair handoff state CRC mismatch");
                copy(false, part_index, offset, payload);
                header.back() = expected_crc; send(std::as_bytes(std::span(header)));
            }
            if (total > std::numeric_limits<std::uint64_t>::max() - size)
                throw std::overflow_error("pair handoff byte count overflow");
            total += size; offset += size; ++sequence;
        }
    }
    // Completion receipt detects missing trailing components, including empty ones.
    StateHeader finish{0x315453335050ULL, 1U, identity, position, cut, sequence,
        parts.size(), total, 0U, 0U, 0U, 0U};
    StateHeader peer{};
    if (producer) { send(std::as_bytes(std::span(finish))); receive(std::as_writable_bytes(std::span(peer))); }
    else { receive(std::as_writable_bytes(std::span(peer))); }
    if (peer != finish) throw std::runtime_error("pair handoff completion mismatch");
    if (!producer) send(std::as_bytes(std::span(finish)));
    return total;
}

using pair::StateStaging;
std::unique_ptr<StateStaging> make_hip_state_staging();

template<class Send, class Receive, class Copy, class Checksum, class Abort>
std::uint64_t handoff_overlap(bool producer, std::span<const StatePart> parts,
    std::uint64_t identity, std::uint64_t position, std::uint32_t cut,
    StateStaging& staging, Send send, Receive receive, Copy copy, Checksum checksum, Abort abort) {
    struct Piece { std::size_t part; std::uint64_t offset; StateHeader header; };
    std::vector<Piece> pieces;
    std::uint64_t total = 0;
    for (std::size_t index = 0; index < parts.size(); ++index) {
        const auto& part = parts[index];
        if (part.layer == 48U && part.component != 0xffffffffU)
            throw std::invalid_argument("pair handoff shared input state is owned by stage A");
        for (std::uint64_t offset = 0; offset < part.bytes;) {
            const auto size = std::min<std::uint64_t>(seam_piece_bytes, part.bytes - offset);
            auto header = state_header(part, identity, position, cut, pieces.size(), offset, size);
            header[1] = 2U; // v2: completion ACK after all authenticated H2D copies
            pieces.push_back({index, offset, header});
            if (total > std::numeric_limits<std::uint64_t>::max() - size)
                throw std::overflow_error("pair handoff byte count overflow");
            total += size; offset += size;
        }
    }
    const auto payload = [&](std::size_t index) {
        const auto size = static_cast<std::size_t>(pieces[index].header[10]);
        auto slot = staging.buffer(index % 2U);
        if (slot.size() < size) throw std::invalid_argument("pair handoff staging slot too small");
        return slot.first(size);
    };
    std::future<void> copying;
    try {
        if (producer && !pieces.empty()) copy(true, pieces[0].part, pieces[0].offset, payload(0));
        for (std::size_t index = 0; index < pieces.size(); ++index) {
            auto header = pieces[index].header;
            auto data = payload(index);
            if (producer) {
                if (copying.valid()) copying.get();
                if (index + 1U < pieces.size()) {
                    copying = std::async(std::launch::async, [&, next = index + 1U] {
                        copy(true, pieces[next].part, pieces[next].offset, payload(next));
                    });
                }
                header.back() = checksum(std::span<const std::byte>(data));
                send(std::as_bytes(std::span(header))); send(data);
            } else {
                StateHeader incoming{};
                receive(std::as_writable_bytes(std::span(incoming)));
                const auto crc = incoming.back(); incoming.back() = 0U;
                if (incoming != header) throw std::runtime_error("pair handoff state framing mismatch");
                receive(data);
                if (crc != checksum(std::span<const std::byte>(data)))
                    throw std::runtime_error("pair handoff state CRC mismatch");
                if (copying.valid()) copying.get();
                copying = std::async(std::launch::async, [&, index, data] {
                    copy(false, pieces[index].part, pieces[index].offset, data);
                });
            }
        }
        if (copying.valid()) copying.get();
        StateHeader finish{0x315453335050ULL, 2U, identity, position, cut, pieces.size(),
            parts.size(), total, 0U, 0U, 0U, 0U};
        StateHeader peer{};
        if (producer) { send(std::as_bytes(std::span(finish))); receive(std::as_writable_bytes(std::span(peer))); }
        else receive(std::as_writable_bytes(std::span(peer)));
        if (peer != finish) throw std::runtime_error("pair handoff completion mismatch");
        if (!producer) send(std::as_bytes(std::span(finish)));
    } catch (...) {
        abort();
        if (copying.valid()) copying.wait();
        throw;
    }
    return total;
}
} // namespace strix::qwen_pp
