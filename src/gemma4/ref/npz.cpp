#include "npz.hpp"
#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <regex>
#include <set>

namespace strix::gemma4 {
namespace {
constexpr std::size_t limit = 512U * 1024U * 1024U; // long teacher fixtures reach ~232 MB
std::uint32_t u32(const std::vector<std::uint8_t>& b, std::size_t p, unsigned n = 4) {
    if (p > b.size() || n > b.size() - p) throw std::runtime_error("truncated NPZ integer");
    std::uint32_t x = 0;
    for (unsigned i = 0; i < n; ++i) x |= std::uint32_t(b[p + i]) << (8U * i);
    return x;
}
struct Bits {
    const std::vector<std::uint8_t>& data;
    std::size_t pos = 0;
    unsigned bit = 0;
    unsigned take(unsigned n) {
        unsigned x = 0;
        for (unsigned i = 0; i < n; ++i) {
            if (pos >= data.size()) throw std::runtime_error("truncated DEFLATE stream");
            x |= ((data[pos] >> bit) & 1U) << i;
            if (++bit == 8) { bit = 0; ++pos; }
        }
        return x;
    }
    void align() { if (bit) { bit = 0; ++pos; } }
};
struct Huffman {
    std::array<unsigned, 16> count{};
    std::vector<unsigned> symbols;
    explicit Huffman(const std::vector<unsigned>& lengths) {
        for (const auto n : lengths) { if (n > 15) throw std::runtime_error("invalid DEFLATE code length"); ++count[n]; }
        int available = 1;
        for (unsigned n = 1; n < 16; ++n) { available = 2 * available - int(count[n]); if (available < 0) throw std::runtime_error("oversubscribed DEFLATE tree"); }
        for (unsigned n = 1; n < 16; ++n)
            for (unsigned s = 0; s < lengths.size(); ++s) if (lengths[s] == n) symbols.push_back(s);
    }
    unsigned decode(Bits& in) const {
        unsigned code = 0, first = 0, index = 0;
        for (unsigned n = 1; n <= 15; ++n) {
            code |= in.take(1);
            if (code >= first && code - first < count[n]) return symbols[index + code - first];
            index += count[n]; first = (first + count[n]) << 1U; code <<= 1U;
        }
        throw std::runtime_error("invalid DEFLATE Huffman code");
    }
};
std::vector<std::uint8_t> inflate(const std::vector<std::uint8_t>& packed, std::size_t expected) {
    Bits in{packed}; std::vector<std::uint8_t> out; out.reserve(expected);
    auto emit = [&](unsigned v) { if (out.size() >= expected) throw std::runtime_error("DEFLATE output overflow"); out.push_back(std::uint8_t(v)); };
    bool final = false;
    while (!final) {
        final = in.take(1) != 0; const auto type = in.take(2);
        if (type == 0) {
            in.align(); const auto n = in.take(16), inverse = in.take(16);
            if ((n ^ inverse) != 65535) throw std::runtime_error("invalid stored DEFLATE block");
            for (unsigned i = 0; i < n; ++i) emit(in.take(8));
            continue;
        }
        std::vector<unsigned> literal, distance;
        if (type == 1) {
            for (unsigned i = 0; i < 288; ++i) literal.push_back(i < 144 ? 8U : (i < 256 ? 9U : (i < 280 ? 7U : 8U)));
            distance.assign(32, 5);
        } else if (type == 2) {
            const auto nl = in.take(5) + 257, nd = in.take(5) + 1, nc = in.take(4) + 4;
            if (nl > 286) throw std::runtime_error("invalid DEFLATE literal count");
            constexpr unsigned order[] = {16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15};
            std::vector<unsigned> codes(19), lengths;
            for (unsigned i = 0; i < nc; ++i) codes[order[i]] = in.take(3);
            const Huffman tree(codes);
            while (lengths.size() < nl + nd) {
                const auto s = tree.decode(in);
                if (s < 16) lengths.push_back(s);
                else {
                    if (s == 16 && lengths.empty()) throw std::runtime_error("DEFLATE repeat without predecessor");
                    const auto repeats = s == 16 ? 3 + in.take(2) : (s == 17 ? 3 + in.take(3) : 11 + in.take(7));
                    const auto value = s == 16 ? lengths.back() : 0;
                    if (repeats > nl + nd - lengths.size()) throw std::runtime_error("DEFLATE code-length overflow");
                    lengths.insert(lengths.end(), repeats, value);
                }
            }
            literal.assign(lengths.begin(), lengths.begin() + nl); distance.assign(lengths.begin() + nl, lengths.end());
        } else throw std::runtime_error("reserved DEFLATE block");
        if (literal.size() <= 256 || literal[256] == 0) throw std::runtime_error("missing DEFLATE end code");
        const Huffman lit(literal), dist(distance);
        constexpr unsigned lb[] = {3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258};
        constexpr unsigned le[] = {0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0};
        constexpr unsigned db[] = {1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577};
        constexpr unsigned de[] = {0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};
        for (;;) {
            const auto s = lit.decode(in);
            if (s < 256) emit(s);
            else if (s == 256) break;
            else {
                if (s > 285) throw std::runtime_error("invalid DEFLATE length symbol");
                const auto length = lb[s-257] + in.take(le[s-257]), ds = dist.decode(in);
                if (ds >= 30) throw std::runtime_error("invalid DEFLATE distance symbol");
                const auto back = db[ds] + in.take(de[ds]);
                if (back > out.size()) throw std::runtime_error("DEFLATE distance before output");
                for (unsigned i = 0; i < length; ++i) emit(out[out.size() - back]);
            }
        }
    }
    if (out.size() != expected) throw std::runtime_error("DEFLATE output size mismatch");
    return out;
}
std::uint32_t crc32(const std::vector<std::uint8_t>& data) {
    std::uint32_t crc = 0xffffffffU;
    for (auto b : data) {
        crc ^= b;
        for (unsigned i = 0; i < 8; ++i) crc = (crc >> 1U) ^ (0xedb88320U & (0U - (crc & 1U)));
    }
    return ~crc;
}
NpyArray npy(const std::vector<std::uint8_t>& data) {
    if (data.size() < 10 || std::string(reinterpret_cast<const char*>(data.data()), 6) != "\x93NUMPY" || data[7] != 0 || (data[6] != 1 && data[6] != 2)) throw std::runtime_error("invalid NPY header");
    const auto prefix = data[6] == 1 ? 10U : 12U, hlen = u32(data, 8, data[6] == 1 ? 2 : 4);
    if (prefix > data.size() || hlen > data.size() - prefix) throw std::runtime_error("truncated NPY header");
    const std::string header(reinterpret_cast<const char*>(data.data() + prefix), hlen);
    std::smatch m; NpyArray out;
    if (!std::regex_search(header, m, std::regex("'descr'\\s*:\\s*'([<|][ifu][1248])'"))) throw std::runtime_error("unsupported NPY dtype");
    out.dtype = m[1];
    if (!std::regex_search(header, std::regex("'fortran_order'\\s*:\\s*False"))) throw std::runtime_error("NPY must be row-major");
    if (!std::regex_search(header, m, std::regex("'shape'\\s*:\\s*\\(([0-9, ]*)\\)"))) throw std::runtime_error("invalid NPY shape");
    const auto shape = m[1].str();
    std::size_t count = 1;
    const std::regex digits("[0-9]+");
    for (auto it = std::sregex_iterator(shape.begin(), shape.end(), digits); it != std::sregex_iterator(); ++it) {
        const auto n = std::stoull(it->str());
        if (n > limit || (n && count > limit / n)) throw std::runtime_error("NPY shape overflow");
        count *= std::size_t(n); out.shape.push_back(std::size_t(n));
    }
    const auto item = std::size_t(out.dtype.back() - '0');
    if (count > limit / item || count * item != data.size() - prefix - hlen) throw std::runtime_error("NPY payload size mismatch");
    out.bytes.assign(data.begin() + prefix + hlen, data.end());
    return out;
}
}
std::map<std::string, NpyArray> read_npz(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() < 22 || f.tellg() > std::streamoff(limit)) throw std::runtime_error("NPZ missing or too large");
    std::vector<std::uint8_t> data(std::size_t(f.tellg())); f.seekg(0); f.read(reinterpret_cast<char*>(data.data()), std::streamsize(data.size()));
    std::size_t end = data.size() - 22;
    while (u32(data, end) != 0x06054b50U || u32(data, end + 20, 2) != data.size() - end - 22) {
        if (end == 0 || data.size() - end > 65557) throw std::runtime_error("NPZ central directory missing");
        --end;
    }
    if (u32(data, end+4, 2) || u32(data, end+6, 2) || u32(data, end+8, 2) != u32(data, end+10, 2)) throw std::runtime_error("multipart NPZ refused");
    std::size_t cursor = u32(data, end+16), total = 0;
    const auto directory_end = cursor + u32(data, end+12);
    if (directory_end != end) throw std::runtime_error("invalid NPZ directory range");
    std::map<std::string, NpyArray> result;
    for (std::uint32_t member = 0; member < u32(data, end+10, 2); ++member) {
        if (u32(data, cursor) != 0x02014b50U) throw std::runtime_error("invalid NPZ directory entry");
        const auto flags = u32(data, cursor+8, 2), method = u32(data, cursor+10, 2), crc = u32(data, cursor+16);
        const auto packed = u32(data, cursor+20), size = u32(data, cursor+24);
        const auto nl = u32(data, cursor+28, 2), extra = u32(data, cursor+30, 2), comment = u32(data, cursor+32, 2);
        const auto local = u32(data, cursor+42);
        if ((flags & ~0x808U) || (method != 0 && method != 8) || size > limit - total || cursor + 46ULL + nl + extra + comment > directory_end) throw std::runtime_error("unsupported NPZ member");
        const std::string name(reinterpret_cast<const char*>(data.data() + cursor+46), nl);
        if (!name.ends_with(".npy") || name.find_first_of("/\\") != std::string::npos) throw std::runtime_error("invalid NPZ member name");
        if (u32(data, local) != 0x04034b50U || u32(data, local+8,2) != method || u32(data,local+6,2) != flags) throw std::runtime_error("NPZ local header mismatch");
        const auto local_name = u32(data,local+26,2);
        const auto begin = std::size_t(local) + 30 + local_name + u32(data, local+28,2);
        if (begin > data.size() || packed > data.size() - begin || begin + packed > u32(data, end+16)) throw std::runtime_error("NPZ payload range invalid");
        if (local_name != nl || std::string(reinterpret_cast<const char*>(data.data()+local+30),local_name) != name) throw std::runtime_error("NPZ local name mismatch");
        std::vector<std::uint8_t> bytes(data.begin() + std::ptrdiff_t(begin), data.begin() + std::ptrdiff_t(begin + packed));
        auto unpacked = method == 8 ? inflate(bytes, size) : std::move(bytes);
        if (unpacked.size() != size || crc32(unpacked) != crc) throw std::runtime_error("NPZ CRC/size mismatch");
        if (!result.emplace(name.substr(0, name.size()-4), npy(unpacked)).second) throw std::runtime_error("duplicate NPZ member");
        total += size; cursor += 46ULL + nl + extra + comment;
    }
    if (cursor != directory_end) throw std::runtime_error("NPZ directory extent mismatch");
    return result;
}
} // namespace strix::gemma4
