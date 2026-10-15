#include "strix/hc_repacked_down_pack.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/model_content_hash.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strix::hc_repacked_down;
using Bytes = std::vector<std::uint8_t>;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Function>
void rejects(Function&& function) {
    bool rejected = false;
    try { function(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "expected invalid_argument");
}
template<class Function>
void overflows(Function&& function) {
    bool rejected = false;
    try { function(); } catch (const std::overflow_error&) { rejected = true; }
    require(rejected, "expected overflow_error");
}

// Decode complete LE words without production index helpers or transfer logic.
std::uint32_t word(const Bytes& bytes, std::size_t offset) {
    return std::uint32_t{bytes.at(offset)} |
        (std::uint32_t{bytes.at(offset + 1)} << 8) |
        (std::uint32_t{bytes.at(offset + 2)} << 16) |
        (std::uint32_t{bytes.at(offset + 3)} << 24);
}
std::uint8_t decoded_code(const Geometry& g, const Bytes& plane,
                          std::uint32_t output, std::uint32_t feature) {
    const std::size_t slab = feature / 16;
    const std::size_t lane_word = feature % 16 / 4;
    const auto packed = word(plane, (slab * g.rank_pad + output) * 16 + lane_word * 4);
    return static_cast<std::uint8_t>(packed >> (8 * (feature % 4)));
}
std::uint16_t decoded_scale(const Geometry& g, const Bytes& plane,
                            std::uint32_t output, std::uint32_t block) {
    const auto packed = word(plane, static_cast<std::size_t>(g.codes_bytes) +
        (std::size_t{block / 2} * g.rank_pad + output) * 4);
    return static_cast<std::uint16_t>(packed >> (16 * (block % 2)));
}

void shape(std::uint32_t width, std::uint32_t rank, std::mt19937& random) {
    const auto g = geometry(width, rank);
    require(g.blocks == width / 32 && g.rank_pad == (rank + 31) / 32 * 32,
            "geometry dimensions");
    require(g.codes_bytes == std::uint64_t{width} * g.rank_pad &&
            g.scales_bytes == std::uint64_t{width / 16} * g.rank_pad &&
            g.scales_offset == g.codes_bytes && g.scales_offset % 1024 == 0 &&
            g.plane_bytes % 256 == 0 &&
            g.padding_bytes == g.plane_bytes - g.original_bytes,
            "geometry byte counts");
    constexpr std::array<std::uint16_t, 12> scales{
        0x0000,0x8000,0x0001,0x03ff,0x7bff,0xfbff,
        0x7c00,0xfc00,0x7c01,0x7e00,0xfe55,0xffff};
    Bytes original(static_cast<std::size_t>(g.original_bytes));
    for (auto& byte : original) byte = static_cast<std::uint8_t>(random());
    // Every admitted shape contains all 256 code bytes. The full scale-pattern
    // sweep below also covers the smallest shape, which has only eight blocks.
    for (std::uint32_t o = 0; o < rank; ++o) {
        for (std::uint32_t b = 0; b < g.blocks; ++b) {
            const auto offset = (std::size_t{o} * g.blocks + b) * 34;
            const auto s = scales[(std::size_t{o} * g.blocks + b) % scales.size()];
            original[offset] = static_cast<std::uint8_t>(s);
            original[offset + 1] = static_cast<std::uint8_t>(s >> 8);
            for (std::uint32_t k = 0; k < 32; ++k) {
                // The first row traverses all code values, including signed extremes.
                if (o == 0) original[offset + 2 + k] = static_cast<std::uint8_t>(b * 32 + k);
            }
        }
    }
    Bytes plane(static_cast<std::size_t>(g.plane_bytes), 0xa5);
    pack(g, original, plane);
    require(padding_zero(g, plane), "pack left nonzero padding");
    Bytes recovered(original.size(), 0xcd);
    unpack(g, plane, recovered);
    require(original == recovered && sha256_hex(original) == sha256_hex(recovered),
            "bytewise/SHA roundtrip");
    for (std::uint32_t o = 0; o < rank; ++o) {
        for (std::uint32_t f = 0; f < width; ++f) {
            const auto offset = (std::size_t{o} * g.blocks + f / 32) * 34;
            require(decoded_code(g, plane, o, f) == original[offset + 2 + f % 32],
                    "independent code decode");
        }
        for (std::uint32_t b = 0; b < g.blocks; ++b) {
            const auto offset = (std::size_t{o} * g.blocks + b) * 34;
            const auto expected = std::uint16_t{original[offset]} |
                (std::uint16_t{original[offset + 1]} << 8);
            require(decoded_scale(g, plane, o, b) == expected, "independent raw scale decode");
        }
    }
    // Independently enumerate padding by physical planes, including any tail.
    Bytes padding;
    for (std::uint32_t slab = 0; slab < width / 16; ++slab) {
        for (std::uint32_t o = rank; o < g.rank_pad; ++o) {
            const auto offset = (std::size_t{slab} * g.rank_pad + o) * 16;
            padding.insert(padding.end(), plane.begin() + static_cast<std::ptrdiff_t>(offset),
                           plane.begin() + static_cast<std::ptrdiff_t>(offset + 16));
        }
    }
    for (std::uint32_t pair = 0; pair < g.blocks / 2; ++pair) {
        for (std::uint32_t o = rank; o < g.rank_pad; ++o) {
            const auto offset = static_cast<std::size_t>(g.codes_bytes) +
                (std::size_t{pair} * g.rank_pad + o) * 4;
            padding.insert(padding.end(), plane.begin() + static_cast<std::ptrdiff_t>(offset),
                           plane.begin() + static_cast<std::ptrdiff_t>(offset + 4));
        }
    }
    padding.insert(padding.end(), plane.begin() + static_cast<std::ptrdiff_t>(g.codes_bytes + g.scales_bytes), plane.end());
    require(padding.size() == g.padding_bytes &&
            sha256_hex(padding) == sha256_hex(Bytes(padding.size(), 0)), "padding SHA");
    const auto packed_sha = sha256_hex(plane);
    for (const auto offset : {std::size_t{0}, static_cast<std::size_t>(g.scales_offset)}) {
        plane[offset] ^= 1;
        unpack(g, plane, recovered);
        require(recovered != original && sha256_hex(plane) != packed_sha, "payload mutation undetected");
        plane[offset] ^= 1;
    }
    if (rank != g.rank_pad) {
        for (const auto offset : {std::size_t{rank} * 16,
                                 static_cast<std::size_t>(g.scales_offset) + std::size_t{rank} * 4}) {
            plane[offset] = 1;
            require(!padding_zero(g, plane) && sha256_hex(plane) != packed_sha,
                    "padding mutation undetected");
            unpack(g, plane, recovered);
            require(recovered == original, "padding must not change logical payload");
            plane[offset] = 0;
        }
    }
    // Separate random byte corpus, including arbitrary half payload bits.
    for (auto& byte : original) byte = static_cast<std::uint8_t>(random());
    pack(g, original, plane);
    unpack(g, plane, recovered);
    require(original == recovered && sha256_hex(original) == sha256_hex(recovered) &&
            padding_zero(g, plane), "random corpus roundtrip");
    if (width == 256 && rank == 1) {
        for (const auto scale : scales) {
            for (std::uint32_t b = 0; b < g.blocks; ++b) {
                original[std::size_t{b} * 34] = static_cast<std::uint8_t>(scale);
                original[std::size_t{b} * 34 + 1] = static_cast<std::uint8_t>(scale >> 8);
            }
            pack(g, original, plane);
            for (std::uint32_t b = 0; b < g.blocks; ++b) {
                require(decoded_scale(g, plane, 0, b) == scale, "adversarial half payload");
            }
            unpack(g, plane, recovered);
            require(original == recovered, "adversarial half roundtrip");
        }
    }
}

void validation() {
    for (const auto width : {0U,255U,257U,10241U,10496U}) rejects([&]{ (void)geometry(width, 32); });
    for (const auto rank : {0U,513U,std::numeric_limits<std::uint32_t>::max()}) rejects([&]{ (void)geometry(256, rank); });
    rejects([]{ (void)code_unit_index(0,2,0,32); });
    rejects([]{ (void)code_unit_index(0,0,32,32); });
    rejects([]{ (void)scale_word_index(0,0,33); });
    overflows([]{ (void)code_unit_index(std::numeric_limits<std::uint64_t>::max(),0,0,32); });
    overflows([]{ (void)scale_word_index(std::numeric_limits<std::uint64_t>::max(),0,512); });
    const auto g = geometry(256, 1);
    Bytes original(static_cast<std::size_t>(g.original_bytes), 0x5a);
    Bytes plane(static_cast<std::size_t>(g.plane_bytes), 0xa5);
    const auto old_original = original, old_plane = plane;
    rejects([&]{ pack(g, std::span(original).first(original.size()-1), plane); });
    rejects([&]{ pack(g, original, std::span(plane).first(plane.size()-1)); });
    rejects([&]{ unpack(g, std::span(plane).first(plane.size()-1), original); });
    rejects([&]{ unpack(g, plane, std::span(original).first(original.size()-1)); });
    rejects([&]{ (void)padding_zero(g, std::span(plane).first(plane.size()-1)); });
    Bytes large_original(original.size() + 1, 0x5a), large_plane(plane.size() + 1, 0xa5);
    rejects([&]{ pack(g, large_original, plane); });
    rejects([&]{ pack(g, original, large_plane); });
    rejects([&]{ unpack(g, large_plane, original); });
    rejects([&]{ unpack(g, plane, large_original); });
    rejects([&]{ (void)padding_zero(g, large_plane); });
    require(original == old_original && plane == old_plane, "size rejection wrote destination");
    // Corrupt every derived geometry field independently; none may be trusted.
    for (unsigned field = 0; field < 8; ++field) {
        auto bad = g;
        switch (field) {
        case 0: ++bad.blocks; break;
        case 1: ++bad.rank_pad; break;
        case 2: ++bad.codes_bytes; break;
        case 3: ++bad.scales_bytes; break;
        case 4: ++bad.scales_offset; break;
        case 5: ++bad.plane_bytes; break;
        case 6: ++bad.padding_bytes; break;
        default: ++bad.original_bytes; break;
        }
        rejects([&]{ pack(bad, original, plane); });
        rejects([&]{ unpack(bad, plane, original); });
        rejects([&]{ (void)padding_zero(bad, plane); });
    }
    require(original == old_original && plane == old_plane, "geometry rejection wrote destination");
    Bytes overlap(plane.size() + original.size(), 0x6b);
    const auto before = overlap;
    rejects([&]{ pack(g, std::span(overlap).first(original.size()), std::span(overlap).first(plane.size())); });
    rejects([&]{ unpack(g, std::span(overlap).first(plane.size()), std::span(overlap).subspan(1, original.size())); });
    require(overlap == before, "overlap rejection wrote destination");
    // Deliberately unaligned host spans, with canaries on both sides. Host
    // packing has no GPU-allocation alignment precondition.
    Bytes guarded_source(original.size() + 2, 0x71), guarded_plane(plane.size() + 2, 0x92);
    Bytes guarded_result(original.size() + 2, 0x83);
    std::copy(original.begin(), original.end(), guarded_source.begin() + 1);
    const auto source_snapshot = guarded_source;
    auto source_view = std::span(guarded_source).subspan(1, original.size());
    auto plane_view = std::span(guarded_plane).subspan(1, plane.size());
    auto result_view = std::span(guarded_result).subspan(1, original.size());
    pack(g, source_view, plane_view);
    const auto plane_snapshot = guarded_plane;
    unpack(g, plane_view, result_view);
    require(guarded_source == source_snapshot && guarded_plane == plane_snapshot &&
            guarded_plane.front() == 0x92 && guarded_plane.back() == 0x92 &&
            guarded_result.front() == 0x83 && guarded_result.back() == 0x83 &&
            std::equal(result_view.begin(), result_view.end(), original.begin()),
            "guarded spans or immutable inputs changed");
    require(sha256_hex(Bytes{}) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "SHA utility binding");
}

static_assert(geometry(10240,320).plane_bytes == 3481600);
static_assert(geometry(10240,320).codes_bytes == 3276800);
static_assert(geometry(10240,320).scales_bytes == 204800);
static_assert(geometry(10240,320).padding_bytes == 0);
static_assert(code_unit_index(3,1,2,32) == 226);
static_assert(scale_word_index(3,2,32) == 34);

struct RealOptions {
    std::filesystem::path model, output;
    std::string model_sha;
};

RealOptions real_options(int argc, char** argv) {
    RealOptions options;
    std::set<std::string> seen;
    for (int at = 2; at < argc; at += 2) {
        require(at + 1 < argc, "missing real-down option value");
        const std::string key = argv[at], value = argv[at + 1];
        require(seen.insert(key).second, "duplicate real-down option");
        if (key == "--model") options.model = value;
        else if (key == "--model-sha256") options.model_sha = value;
        else if (key == "--output") options.output = value;
        else require(false, "unknown real-down option");
    }
    require(!options.model.empty() && !options.output.empty() && options.model_sha.size() == 64 &&
        std::all_of(options.model_sha.begin(), options.model_sha.end(), [](char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        }), "need model, lowercase model SHA256 and fresh output");
    require(std::filesystem::is_regular_file(options.model) && !std::filesystem::is_symlink(options.model),
            "model must be a regular nonsymlink file");
    require(!std::filesystem::exists(options.output) && !std::filesystem::is_symlink(options.output),
            "real-down ledger output occupied");
    return options;
}

std::string file_sha(const std::filesystem::path& path) {
    return strix::model_content_identity_hex(strix::sha256_file_content(path));
}

// CPU-only admission gate. No HIP headers, runtime calls or device allocation.
// The canonical 97-name set is independent of the later captured-input corpus.
void qualify_real_downs(const RealOptions& options) {
    const auto size_before = std::filesystem::file_size(options.model);
    const auto time_before = std::filesystem::last_write_time(options.model);
    require(file_sha(options.model) == options.model_sha, "whole-model SHA256 mismatch before CPU qualification");
    const auto info = strix::gguf::read_file(options.model);
    require(info.file_size == size_before, "GGUF parser file extent mismatch");
    std::ifstream model(options.model, std::ios::binary);
    require(model.good(), "open GGUF tensor windows");
    const auto g = geometry(10240, 320);
    std::ostringstream ledger;
    ledger << "{\"schema\":\"strix-hc-real-down-pack-v1\",\"layout_id\":\"" << kLayoutId
           << "\",\"model_sha256\":\"" << options.model_sha << "\",\"model_bytes\":" << size_before
           << ",\"cpu_only\":true,\"tensor_count\":97,\"tensors\":[";
    std::uint64_t total = 0;
    std::set<std::pair<std::uint64_t,std::uint64_t>> windows;
    for (unsigned index = 0; index < 97; ++index) {
        const std::string name = index == 96 ? "output_hc_down.weight" :
            "blk." + std::to_string(index / 2) + (index % 2 == 0 ? ".hc_attn_down.weight" : ".hc_ffn_down.weight");
        const strix::gguf::TensorInfo* tensor = nullptr;
        for (const auto& candidate : info.tensors) {
            if (candidate.name == name) {
                require(tensor == nullptr, "duplicate real-down GGUF name");
                tensor = &candidate;
            }
        }
        require(tensor != nullptr && tensor->ggml_type == 8 &&
                tensor->dimensions == std::vector<std::uint64_t>{10240,320} &&
                tensor->byte_size == g.original_bytes, "real-down tensor type/shape/size mismatch");
        const auto offset = tensor->absolute_offset;
        require(offset <= size_before && tensor->byte_size <= size_before - offset &&
                offset <= static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()),
                "real-down tensor window outside source");
        const auto end = offset + tensor->byte_size;
        for (const auto& prior : windows) {
            require(end <= prior.first || offset >= prior.second, "overlapping real-down tensor windows");
        }
        windows.emplace(offset,end);
        Bytes original(static_cast<std::size_t>(g.original_bytes));
        model.seekg(static_cast<std::streamoff>(offset));
        model.read(reinterpret_cast<char*>(original.data()), static_cast<std::streamsize>(original.size()));
        require(model.good(), "truncated real-down window read");
        Bytes plane(static_cast<std::size_t>(g.plane_bytes),0xa5), recovered(original.size(),0xcd);
        const auto source_sha = sha256_hex(original);
        pack(g,original,plane);
        unpack(g,plane,recovered);
        require(original == recovered && sha256_hex(recovered) == source_sha && padding_zero(g,plane),
                "real-down pack roundtrip/padding failed");
        for (std::uint32_t o = 0; o < g.rank; ++o) {
            for (std::uint32_t f = 0; f < g.width; ++f) {
                const auto at = (std::size_t{o} * g.blocks + f / 32) * 34;
                require(decoded_code(g,plane,o,f) == original[at + 2 + f % 32], "real-down independent code decode");
            }
            for (std::uint32_t b = 0; b < g.blocks; ++b) {
                const auto at = (std::size_t{o} * g.blocks + b) * 34;
                const auto expected = std::uint16_t{original[at]} | (std::uint16_t{original[at+1]} << 8);
                require(decoded_scale(g,plane,o,b) == expected, "real-down independent scale decode");
            }
        }
        // Canonical real shape has no padded rows or tail. Synthetic shape
        // tests separately extract/hash nonempty padding across the full grid.
        require(g.padding_bytes == 0, "unexpected real-down padding geometry");
        if (index != 0) ledger << ',';
        ledger << "{\"name\":\"" << name << "\",\"index\":" << index
               << ",\"file_offset\":" << offset << ",\"size\":" << original.size()
               << ",\"width\":" << g.width << ",\"rank\":" << g.rank << ",\"rank_pad\":" << g.rank_pad
               << ",\"source_sha256\":\"" << source_sha << "\",\"roundtrip_sha256\":\"" << sha256_hex(recovered)
               << "\",\"packed_sha256\":\"" << sha256_hex(plane) << "\",\"padding_bytes\":0,\"padding_sha256\":\""
               << sha256_hex(Bytes{}) << "\",\"independent_decode_exact\":true}";
        total += g.original_bytes;
    }
    require(total == 337715200ULL && windows.size() == 97, "real-down total corpus byte/count mismatch");
    require(std::filesystem::file_size(options.model) == size_before &&
            std::filesystem::last_write_time(options.model) == time_before &&
            file_sha(options.model) == options.model_sha, "model changed during CPU qualification");
    ledger << "],\"original_bytes_total\":" << total
           << ",\"model_before_after_exact\":true,\"passed\":true}\n";
    require(!std::filesystem::exists(options.output) && !std::filesystem::is_symlink(options.output),
            "real-down output became occupied");
    std::ofstream output(options.output,std::ios::binary);
    require(output.good(), "create real-down CPU ledger");
    output << ledger.str();
    output.close();
    require(output.good(), "write/close real-down CPU ledger");
    std::cout << "HC real-down CPU qualification: 97 tensors, 337715200 bytes passed\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 1) {
            require(std::string(argv[1]) == "--qualify-real-downs", "unknown CPU test mode");
            qualify_real_downs(real_options(argc,argv));
            return 0;
        }
        validation();
        std::mt19937 random(0x165);
        for (const auto width : {256U,512U,2048U,10240U}) {
            for (const auto rank : {1U,31U,32U,33U,316U,318U,320U,512U}) shape(width, rank, random);
        }
        std::cout << "HC exact repack CPU: 32 shapes and contract checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "HC exact repack CPU failed: " << error.what() << '\n';
        return 1;
    }
}
