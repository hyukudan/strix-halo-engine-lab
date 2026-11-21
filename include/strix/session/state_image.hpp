#pragma once
// Device state image: backend-neutral host codec for the cross-position state of a forward (its caches, position and
// token history), so a long teacher-forced run can stop at a ticket boundary and continue in a new process with the
// same bits. The adapter owns what the regions are and copies them to / from its device; this file owns the framing,
// the layout key check and the checksums once (host test: tests/session_state_image_tests.cpp). Files are published
// with session/checkpoint_io.hpp (atomic_write / read).
//
// Format strix-state-image/v1 (little-endian): "STXSTIM1", u32 version 1, u32 0,
//   u64 layout bytes, layout (the adapter's key: model, rank range, context, ring rows ... ; decode refuses another key),
//   u64 position, u64 token count, u32 tokens[count],
//   u64 region count, per region: u64 name bytes, name, u64 bytes, u64 fnv64(bytes), bytes,
//   u64 fnv64 of every byte before it (the trailer).
// A decode that does not match in every field throws std::runtime_error naming the first defect; there is no partial
// image.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix::session::state_image {

inline constexpr char kMagic[8] = {'S', 'T', 'X', 'S', 'T', 'I', 'M', '1'};
inline constexpr std::uint32_t kVersion = 1;

struct Region {
    std::string name;
    std::vector<std::byte> bytes;
};

struct Image {
    std::string layout;                  // adapter key; decode() refuses an image of another layout
    std::uint64_t position = 0;          // next position the forward computes
    std::vector<std::uint32_t> tokens;   // token history (an adapter may require tokens.size() == position)
    std::vector<Region> regions;         // device buffers in the adapter's order
};

inline std::uint64_t fnv64(std::span<const std::byte> bytes, std::uint64_t h = 0xcbf29ce484222325ULL) noexcept {
    for (const std::byte b : bytes) {
        h ^= static_cast<std::uint64_t>(std::to_integer<unsigned>(b));
        h *= 0x100000001b3ULL;
    }
    return h;
}

namespace detail {
inline void put_raw(std::vector<std::byte>& out, const void* p, std::size_t n) {
    const auto* b = static_cast<const std::byte*>(p);
    out.insert(out.end(), b, b + n);
}
inline void put_u64(std::vector<std::byte>& out, std::uint64_t v) {
    std::byte b[8];
    for (unsigned i = 0; i < 8U; ++i) b[i] = static_cast<std::byte>((v >> (8U * i)) & 0xFFU);
    put_raw(out, b, 8);
}
inline void put_u32(std::vector<std::byte>& out, std::uint32_t v) {
    std::byte b[4];
    for (unsigned i = 0; i < 4U; ++i) b[i] = static_cast<std::byte>((v >> (8U * i)) & 0xFFU);
    put_raw(out, b, 4);
}

class Reader {
public:
    explicit Reader(std::span<const std::byte> s) : s_(s) {}
    std::span<const std::byte> take(std::uint64_t n, const char* what) {
        if (n > s_.size() - at_) throw std::runtime_error(std::string("state image: truncated at ") + what);
        const auto out = s_.subspan(at_, static_cast<std::size_t>(n));
        at_ += static_cast<std::size_t>(n);
        return out;
    }
    std::uint64_t u64(const char* what) {
        const auto b = take(8, what);
        std::uint64_t v = 0;
        for (unsigned i = 0; i < 8U; ++i) v |= static_cast<std::uint64_t>(std::to_integer<unsigned>(b[i])) << (8U * i);
        return v;
    }
    std::uint32_t u32(const char* what) {
        const auto b = take(4, what);
        std::uint32_t v = 0;
        for (unsigned i = 0; i < 4U; ++i) v |= static_cast<std::uint32_t>(std::to_integer<unsigned>(b[i])) << (8U * i);
        return v;
    }
    [[nodiscard]] std::size_t at() const noexcept { return at_; }
    [[nodiscard]] std::size_t left() const noexcept { return s_.size() - at_; }

private:
    std::span<const std::byte> s_;
    std::size_t at_ = 0;
};
}  // namespace detail

inline std::vector<std::byte> encode(const Image& image) {
    std::size_t total = 8 + 8 + 8 + image.layout.size() + 16 + 4 * image.tokens.size() + 8 + 8;
    for (const auto& r : image.regions) total += 24 + r.name.size() + r.bytes.size();
    std::vector<std::byte> out;
    out.reserve(total);
    detail::put_raw(out, kMagic, 8);
    detail::put_u32(out, kVersion);
    detail::put_u32(out, 0);
    detail::put_u64(out, image.layout.size());
    detail::put_raw(out, image.layout.data(), image.layout.size());
    detail::put_u64(out, image.position);
    detail::put_u64(out, image.tokens.size());
    for (const std::uint32_t t : image.tokens) detail::put_u32(out, t);
    detail::put_u64(out, image.regions.size());
    for (const auto& r : image.regions) {
        detail::put_u64(out, r.name.size());
        detail::put_raw(out, r.name.data(), r.name.size());
        detail::put_u64(out, r.bytes.size());
        detail::put_u64(out, fnv64(r.bytes));
        detail::put_raw(out, r.bytes.data(), r.bytes.size());
    }
    detail::put_u64(out, fnv64(out));
    return out;
}

// expected_layout: the live forward's key; an image of any other layout is refused (rings sized for other rows, another
// context or rank range would be read at the wrong offsets).
inline Image decode(std::span<const std::byte> data, std::string_view expected_layout) {
    if (data.size() < 8 + 8 + 8) throw std::runtime_error("state image: shorter than its header");
    const auto body = data.first(data.size() - 8);
    detail::Reader tail(data.subspan(data.size() - 8));
    if (tail.u64("trailer") != fnv64(body)) throw std::runtime_error("state image: checksum mismatch (corrupt or truncated file)");
    detail::Reader r(body);
    if (std::memcmp(r.take(8, "magic").data(), kMagic, 8) != 0) throw std::runtime_error("state image: not a strix-state-image");
    if (const auto v = r.u32("version"); v != kVersion)
        throw std::runtime_error("state image: version " + std::to_string(v) + ", this build reads " + std::to_string(kVersion));
    (void)r.u32("reserved");
    Image image;
    const auto lb = r.take(r.u64("layout size"), "layout");
    image.layout.assign(reinterpret_cast<const char*>(lb.data()), lb.size());
    if (image.layout != expected_layout)
        throw std::runtime_error("state image: layout '" + image.layout + "' is not this forward's '" + std::string(expected_layout) + "'");
    image.position = r.u64("position");
    const auto count = r.u64("token count");
    if (count > r.left() / 4U) throw std::runtime_error("state image: token count past the end");
    image.tokens.resize(static_cast<std::size_t>(count));
    for (auto& t : image.tokens) t = r.u32("token");
    const auto regions = r.u64("region count");
    if (regions > r.left() / 24U) throw std::runtime_error("state image: region count past the end");
    image.regions.resize(static_cast<std::size_t>(regions));
    for (auto& reg : image.regions) {
        const auto nb = r.take(r.u64("region name size"), "region name");
        reg.name.assign(reinterpret_cast<const char*>(nb.data()), nb.size());
        const auto n = r.u64("region size");
        const auto sum = r.u64("region checksum");
        const auto bytes = r.take(n, "region bytes");
        if (fnv64(bytes) != sum) throw std::runtime_error("state image: region " + reg.name + " checksum mismatch");
        reg.bytes.assign(bytes.begin(), bytes.end());
    }
    if (r.left() != 0) throw std::runtime_error("state image: trailing bytes after the last region");
    return image;
}

}  // namespace strix::session::state_image
