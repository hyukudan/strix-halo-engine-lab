#pragma once
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>
namespace strix::h3::io {
// Minimal PNG writer (RGB8, zlib stream of stored deflate blocks: no compression library needed).
inline std::uint32_t crc32_png(const unsigned char* p, std::size_t n, std::uint32_t c = 0xffffffffU) {
    static std::uint32_t tab[256];
    static bool init = false;
    if (!init) {
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t v = i;
            for (int k = 0; k < 8; ++k) v = (v & 1U) ? 0xedb88320U ^ (v >> 1U) : v >> 1U;
            tab[i] = v;
        }
        init = true;
    }
    for (std::size_t i = 0; i < n; ++i) c = tab[(c ^ p[i]) & 0xffU] ^ (c >> 8U);
    return c;
}
inline void write_png(const std::filesystem::path& path, const std::uint8_t* rgb, int w, int h) {
    std::vector<unsigned char> raw;
    raw.reserve(static_cast<std::size_t>(h) * (static_cast<std::size_t>(w) * 3 + 1));
    for (int y = 0; y < h; ++y) {
        raw.push_back(0);
        raw.insert(raw.end(), rgb + static_cast<std::size_t>(y) * w * 3, rgb + static_cast<std::size_t>(y + 1) * w * 3);
    }
    std::vector<unsigned char> z = {0x78, 0x01};
    std::uint32_t a1 = 1, a2 = 0;
    for (unsigned char c : raw) {
        a1 = (a1 + c) % 65521U;
        a2 = (a2 + a1) % 65521U;
    }
    for (std::size_t off = 0; off < raw.size() || off == 0; off += 65535) {
        const std::size_t n = std::min<std::size_t>(65535, raw.size() - off);
        z.push_back(off + n >= raw.size() ? 1 : 0);
        z.push_back(static_cast<unsigned char>(n & 0xffU));
        z.push_back(static_cast<unsigned char>(n >> 8U));
        z.push_back(static_cast<unsigned char>(~n & 0xffU));
        z.push_back(static_cast<unsigned char>((~n >> 8U) & 0xffU));
        z.insert(z.end(), raw.begin() + static_cast<std::ptrdiff_t>(off), raw.begin() + static_cast<std::ptrdiff_t>(off + n));
        if (raw.empty()) break;
    }
    const std::uint32_t ad = (a2 << 16U) | a1;
    for (int k = 3; k >= 0; --k) z.push_back(static_cast<unsigned char>(ad >> (8 * k)));
    std::ofstream f(path, std::ios::binary);
    auto be32 = [&](std::uint32_t v) {
        const unsigned char b[4] = {static_cast<unsigned char>(v >> 24U), static_cast<unsigned char>(v >> 16U),
            static_cast<unsigned char>(v >> 8U), static_cast<unsigned char>(v)};
        f.write(reinterpret_cast<const char*>(b), 4);
    };
    auto chunk = [&](const char* type, const std::vector<unsigned char>& d) {
        be32(static_cast<std::uint32_t>(d.size()));
        std::vector<unsigned char> td(type, type + 4);
        td.insert(td.end(), d.begin(), d.end());
        f.write(reinterpret_cast<const char*>(td.data()), static_cast<std::streamsize>(td.size()));
        be32(crc32_png(td.data(), td.size()) ^ 0xffffffffU);
    };
    f.write("\x89PNG\r\n\x1a\n", 8);
    std::vector<unsigned char> ihdr(13, 0);
    for (int k = 0; k < 4; ++k) {
        ihdr[static_cast<std::size_t>(k)] = static_cast<unsigned char>(static_cast<std::uint32_t>(w) >> (8 * (3 - k)));
        ihdr[static_cast<std::size_t>(4 + k)] = static_cast<unsigned char>(static_cast<std::uint32_t>(h) >> (8 * (3 - k)));
    }
    ihdr[8] = 8;
    ihdr[9] = 2;
    chunk("IHDR", ihdr);
    chunk("IDAT", z);
    chunk("IEND", {});
    f.flush();
    if (!f) throw std::runtime_error("write " + path.string());
}

}
