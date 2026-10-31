#include "features.hpp"
#include <bit>
#include <cstring>
#include <fstream>
#include <limits>

namespace strix::gemma4::features {
namespace {
using Bytes = std::vector<std::uint8_t>;
void u16(Bytes& b, std::uint16_t n) { b.push_back(std::uint8_t(n)); b.push_back(std::uint8_t(n >> 8)); }
void u32(Bytes& b, std::uint32_t n) { u16(b,std::uint16_t(n)); u16(b,std::uint16_t(n >> 16)); }
void text(Bytes& b, const std::string& s) { b.insert(b.end(),s.begin(),s.end()); }
std::uint32_t crc(const Bytes& b) {
    std::uint32_t c = UINT32_MAX;
    for (auto x : b) { c ^= x; for (int i = 0; i < 8; ++i) c = (c >> 1) ^ (0xedb88320U & (0U-(c & 1U))); }
    return ~c;
}
Bytes npy(std::span<const float> values, std::size_t rows, std::size_t width) {
    if (rows * width != values.size()) throw std::runtime_error("invalid feature array shape");
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': ("+std::to_string(rows)+", "+std::to_string(width)+"), }";
    header.append((64-(10+header.size()+1)%64)%64,' '); header += '\n';
    Bytes out{0x93,'N','U','M','P','Y',1,0}; u16(out,std::uint16_t(header.size())); text(out,header);
    for (const float x : values) u32(out,std::bit_cast<std::uint32_t>(x));
    return out;
}
}
void write_npz(const std::filesystem::path& path, const Result& result) {
    if (std::filesystem::exists(path)) throw std::invalid_argument("feature archive already exists");
    struct Entry { std::string name; std::uint32_t checksum, size, offset; };
    Bytes zip; std::vector<Entry> entries;
    const auto add = [&](std::string name, const Bytes& data) {
        if (data.size() > UINT32_MAX || zip.size()+data.size()+name.size()+30 > UINT32_MAX || entries.size() >= UINT16_MAX)
            throw std::runtime_error("feature archive exceeds ZIP32 limits");
        Entry e{name,crc(data),std::uint32_t(data.size()),std::uint32_t(zip.size())};
        u32(zip,0x04034b50); u16(zip,20); u16(zip,0); u16(zip,0); u16(zip,0); u16(zip,0);
        u32(zip,e.checksum); u32(zip,e.size); u32(zip,e.size); u16(zip,std::uint16_t(name.size())); u16(zip,0);
        text(zip,name); zip.insert(zip.end(),data.begin(),data.end()); entries.push_back(e);
    };
    for (std::size_t b = 0; b < result.branches.size(); ++b) {
        const auto& branch = result.branches[b]; const auto name = "branch_"+std::to_string(b);
        add(name+"_read.npy",npy(branch.read.values,1,branch.read.values.size()));
        std::vector<float> options;
        for (const auto& row : branch.options) options.insert(options.end(),row.values.begin(),row.values.end());
        add(name+"_options.npy",npy(options,branch.options.size(),branch.read.values.size()));
        add(name+"_code_logits.npy",npy(branch.code_logits,1,branch.code_logits.size()));
    }
    const auto start = zip.size();
    for (const auto& e : entries) {
        u32(zip,0x02014b50); u16(zip,20); u16(zip,20); u16(zip,0); u16(zip,0); u16(zip,0); u16(zip,0);
        u32(zip,e.checksum); u32(zip,e.size); u32(zip,e.size); u16(zip,std::uint16_t(e.name.size()));
        u16(zip,0); u16(zip,0); u16(zip,0); u16(zip,0); u32(zip,0); u32(zip,e.offset); text(zip,e.name);
    }
    const auto size = zip.size()-start;
    if (zip.size() > UINT32_MAX) throw std::runtime_error("feature archive exceeds ZIP32 limits");
    u32(zip,0x06054b50); u16(zip,0); u16(zip,0); u16(zip,std::uint16_t(entries.size())); u16(zip,std::uint16_t(entries.size()));
    u32(zip,std::uint32_t(size)); u32(zip,std::uint32_t(start)); u16(zip,0);
    const auto temporary = std::filesystem::path(path.string()+".tmp");
    {
        std::ofstream f(temporary,std::ios::binary|std::ios::trunc);
        f.write(reinterpret_cast<const char*>(zip.data()),std::streamsize(zip.size())); f.close();
        if (!f) throw std::runtime_error("cannot write feature archive");
    }
    std::filesystem::rename(temporary,path);
}
} // namespace strix::gemma4::features
