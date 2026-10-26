#pragma once
// The DeepSeek V4.1 attention-state file (strix/ds41-attention-state-v1, serialize_attention_state) cut into the pieces
// a streamed hand-off sends (streaming protocol): the file is
//   head   "DS41ATST" | u32 version (1) | u32 position | u32 next_token | u32 layers          (24 bytes)
//   record one per layer: u8 flags | u32 ratio | ring | kv | idx_keys | carry | topk         (each vector: u64 count + data)
//   tail   u64 token count | tokens | u64 digest | u64 checksum (FNV-1a of every byte before it)
// A layer's record is final once the prefill finished that layer, so it can leave during the prefill; the head and the
// tail (next_token, tokens, digest, checksum) go last. The receiver concatenates head | records | tail into the very
// bytes serialize_attention_state produces, so the existing import path (deserialize: checksum + digest checks) runs
// unchanged and the state is byte-identical by construction.
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

#include "strix/deepseek41_attention_state.hpp"

namespace strix::deepseek41::pieces {

inline constexpr std::size_t kHeadBytes = 24;

namespace detail {
template <class T> void put(std::vector<std::uint8_t>& out, const T& v) {
    const auto* b = reinterpret_cast<const std::uint8_t*>(&v);
    out.insert(out.end(), b, b + sizeof(T));
}
template <class T> void put_vec(std::vector<std::uint8_t>& out, const std::vector<T>& v) {
    put<std::uint64_t>(out, v.size());
    const auto* b = reinterpret_cast<const std::uint8_t*>(v.data());
    out.insert(out.end(), b, b + v.size() * sizeof(T));
}
inline std::uint64_t get64(std::span<const std::uint8_t> b, std::size_t at) {
    if (at + 8 > b.size()) throw std::runtime_error("deepseek41 state pieces: truncated state bytes");
    std::uint64_t v;
    std::memcpy(&v, b.data() + at, 8);
    return v;
}
}  // namespace detail

// One layer's record, byte for byte as serialize_attention_state writes it.
inline std::vector<std::uint8_t> layer_record(const AttentionLayerState& L) {
    std::vector<std::uint8_t> out;
    out.reserve(1 + 4 + 5 * 8 + L.ring.size() + L.kv.size() + L.idx_keys.size() + L.carry.size() * 4 + L.topk.size() * 4);
    const std::uint8_t flags = (L.owns_kv ? 1U : 0U) | (L.owns_index ? 2U : 0U) | (L.carry_valid ? 4U : 0U) | (L.topk_known ? 8U : 0U);
    detail::put(out, flags);
    detail::put(out, L.ratio);
    detail::put_vec(out, L.ring);
    detail::put_vec(out, L.kv);
    detail::put_vec(out, L.idx_keys);
    detail::put_vec(out, L.carry);
    detail::put_vec(out, L.topk);
    return out;
}

// Offsets of the pieces inside a serialized state (sizes from the u64 counts; the caller's deserialize checks the rest).
struct Layout {
    std::span<const std::uint8_t> head, tail;
    std::vector<std::span<const std::uint8_t>> records;
};
inline Layout split(std::span<const std::uint8_t> bytes) {
    if (bytes.size() < kHeadBytes || std::memcmp(bytes.data(), "DS41ATST", 8) != 0) throw std::runtime_error("deepseek41 state pieces: not a state");
    std::uint32_t layers;
    std::memcpy(&layers, bytes.data() + 20, 4);
    if (layers > 64) throw std::runtime_error("deepseek41 state pieces: corrupt layer count");
    Layout out;
    out.head = bytes.subspan(0, kHeadBytes);
    std::size_t at = kHeadBytes;
    static constexpr std::size_t elem[5] = {1, 1, 1, 4, 4};  // ring, kv, idx_keys (u8), carry (f32), topk (u32)
    for (std::uint32_t l = 0; l < layers; ++l) {
        const std::size_t start = at;
        at += 1 + 4;
        for (const std::size_t e : elem) {
            const std::uint64_t n = detail::get64(bytes, at);
            if (n > (std::uint64_t{1} << 36) || at + 8 + n * e > bytes.size()) throw std::runtime_error("deepseek41 state pieces: corrupt record size");
            at += 8 + n * e;
        }
        out.records.push_back(bytes.subspan(start, at - start));
    }
    out.tail = bytes.subspan(at);
    return out;
}

// head | records | tail
inline std::vector<std::uint8_t> assemble(std::span<const std::uint8_t> head, const std::vector<std::vector<std::uint8_t>>& records,
                                          std::span<const std::uint8_t> tail) {
    std::size_t n = head.size() + tail.size();
    for (const auto& r : records) n += r.size();
    std::vector<std::uint8_t> out;
    out.reserve(n);
    out.insert(out.end(), head.begin(), head.end());
    for (const auto& r : records) out.insert(out.end(), r.begin(), r.end());
    out.insert(out.end(), tail.begin(), tail.end());
    return out;
}

}  // namespace strix::deepseek41::pieces
