#include "strix/weight_archive.hpp"
#include "strix/ggml_dequant.hpp"
#include <bit>
#include <cmath>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strix::weights {
MappedFile::MappedFile(const std::filesystem::path& path) {
#ifdef _WIN32
    const auto f = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open model asset");
    LARGE_INTEGER length{};
    if (!GetFileSizeEx(f, &length) || length.QuadPart <= 0 || std::uint64_t(length.QuadPart) > SIZE_MAX) {
        CloseHandle(f); throw std::runtime_error("invalid model asset size");
    }
    const auto m = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!m) { CloseHandle(f); throw std::runtime_error("cannot map model asset"); }
    data_ = static_cast<const std::uint8_t*>(MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0));
    if (!data_) { CloseHandle(m); CloseHandle(f); throw std::runtime_error("cannot view model asset"); }
    file_ = f; mapping_ = m; size_ = std::size_t(length.QuadPart);
#else
    fd_ = open(path.c_str(), O_RDONLY);
    struct stat st{};
    if (fd_ < 0 || fstat(fd_, &st) != 0 || st.st_size <= 0 || std::uint64_t(st.st_size) > SIZE_MAX) {
        if (fd_ >= 0) close(fd_);
        throw std::runtime_error("cannot open model asset");
    }
    size_ = std::size_t(st.st_size);
    const auto p = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (p == MAP_FAILED) { close(fd_); throw std::runtime_error("cannot map model asset"); }
    data_ = static_cast<const std::uint8_t*>(p);
#endif
}
MappedFile::~MappedFile() {
#ifdef _WIN32
    if (data_) UnmapViewOfFile(data_);
    if (mapping_) CloseHandle(mapping_);
    if (file_) CloseHandle(file_);
#else
    if (data_) munmap(const_cast<std::uint8_t*>(data_), size_);
    if (fd_ >= 0) close(fd_);
#endif
}
serve::Json read_json(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open JSON asset");
    f.seekg(0, std::ios::end);
    const auto size = f.tellg();
    if (size < 0 || size > 64 * 1024 * 1024) throw std::runtime_error("JSON asset too large");
    f.seekg(0);
    std::string text(static_cast<std::size_t>(size), ' ' );
    if (!f.read(text.data(), static_cast<std::streamsize>(text.size()))) throw std::runtime_error("short JSON read");
    if (text.size() > 64U * 1024U * 1024U) throw std::runtime_error("JSON asset too large");
    return serve::parse_json_unique(text);
}
namespace {
const serve::Json& require(const serve::Json& value, const char* key) {
    const auto* p = value.find(key);
    if (!p) throw std::runtime_error(std::string("missing asset field: ") + key);
    return *p;
}
std::uint64_t integer(const serve::Json& value) {
    const auto n = value.as_int();
    if (n < 0) throw std::runtime_error("negative asset extent");
    return std::uint64_t(n);
}
}
SafeFile read_safetensors(const std::filesystem::path& path) {

    SafeFile out;
    out.storage = std::make_shared<MappedFile>(path);
    const auto* data = out.storage->data();
    const auto size = out.storage->size();
    if (size < 8) throw std::runtime_error("truncated safetensors prefix");
    std::uint64_t header = 0;
    for (unsigned i = 0; i < 8; ++i) header |= std::uint64_t(data[i]) << (8U * i);
    if (header > 64ULL * 1024 * 1024 || header > size - 8) throw std::runtime_error("invalid safetensors header size");
    const auto root = serve::parse_json_unique(std::string_view(reinterpret_cast<const char*>(data + 8), std::size_t(header)));
    const auto begin = 8 + header;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    const std::map<std::string, std::size_t> dtypes = {{"BF16",2},{"F16",2},{"F32",4},{"F64",8},
        {"I8",1},{"U8",1},{"BOOL",1},{"I16",2},{"U16",2},{"I32",4},{"U32",4},{"I64",8},{"U64",8},{"F8_E4M3",1},{"F8_E4M3FN",1},{"F8_E5M2",1}};
    for (const auto& [name, desc] : root.as_object()) {
        if (name == "__metadata__") {
            for (const auto& [key, value] : desc.as_object()) { (void)key; (void)value.as_string(); }
            continue;
        }
        SafeTensor t; t.name = name; t.dtype = require(desc, "dtype").as_string();
        if (desc.as_object().size() != 3) throw std::runtime_error("unsupported safetensors descriptor fields/strides");
        const auto dt = dtypes.find(t.dtype);
        if (dt == dtypes.end()) throw std::runtime_error("unsupported safetensors dtype: " + t.dtype);
        std::uint64_t count = 1;
        for (const auto& n : require(desc, "shape").as_array()) {
            const auto dim = integer(n);
            if (dim != 0 && count > UINT64_MAX / dim) throw std::runtime_error("safetensors shape overflow");
            count *= dim; t.shape.push_back(dim);
        }
        if (t.shape.size() > 8 || count > UINT64_MAX / dt->second) throw std::runtime_error("safetensors extent overflow");
        const auto& offsets = require(desc, "data_offsets").as_array();
        if (offsets.size() != 2) throw std::runtime_error("safetensors offsets need two elements");
        const auto b = integer(offsets[0]), e = integer(offsets[1]);
        if (e < b || e > size - begin || e - b != count * dt->second) throw std::runtime_error("invalid safetensors range: " + name);
        t.offset = begin + b; t.bytes = e - b;
        if (e != b) ranges.emplace_back(b, e);
        out.tensors.push_back(std::move(t));
    }
    std::sort(ranges.begin(), ranges.end());
    std::uint64_t end = 0;
    for (const auto& [b, e] : ranges) {
        if (b != end) throw std::runtime_error("safetensors overlap or unclaimed hole");
        end = e;
    }
    if (end != size - begin) throw std::runtime_error("unclaimed safetensors payload");
    return out;
}
std::span<const std::byte> Tensor::bytes() const {
    if (!storage || offset > storage->size() || size > storage->size() - offset)
        throw std::runtime_error("weight view outside storage");
    return {reinterpret_cast<const std::byte*>(storage->data() + offset), static_cast<std::size_t>(size)};
}
Archive load_gguf(const std::filesystem::path& path, const gguf::FileInfo& info) {
    auto storage = std::make_shared<MappedFile>(path);
    if (storage->size() != info.file_size) throw std::runtime_error("GGUF changed during import");
    Archive out;
    for (const auto& t : info.tensors) {
        Tensor view{storage, {t.dimensions.rbegin(), t.dimensions.rend()}, "GGUF",
                    t.ggml_type, t.absolute_offset, t.byte_size};
        (void)view.bytes();
        if (!out.emplace(t.name, std::move(view)).second) throw std::runtime_error("duplicate GGUF tensor");
    }
    return out;
}
Archive load_safetensors(const std::filesystem::path& path) {
    Archive out;
    const auto append = [&out](const std::filesystem::path& file) {
        const auto safe = read_safetensors(file);
        for (const auto& t : safe.tensors)
            if (!out.emplace(t.name, Tensor{safe.storage, t.shape, t.dtype, 0, t.offset, t.bytes}).second)
                throw std::runtime_error("duplicate tensor across shards: " + t.name);
    };
    if (!std::filesystem::is_directory(path)) { append(path); return out; }
    const auto index_path = path / "model.safetensors.index.json";
    if (!std::filesystem::exists(index_path)) { append(path / "model.safetensors"); return out; }
    const auto index = read_json(index_path);
    const auto& map = require(index, "weight_map").as_object();
    if (map.empty()) throw std::runtime_error("empty safetensors index");
    std::set<std::string> shards;
    std::map<std::string, std::string> owners;
    for (const auto& [name, file] : map) {
        (void)name;
        const auto& s = file.as_string();
        if (s.empty() || s == "." || s == ".." || s.find_first_of("/\\:") != std::string::npos ||
            !s.ends_with(".safetensors")) throw std::runtime_error("index shard must be a local safetensors basename");
        if (shards.insert(s).second) {
            append(path / s);
            // Record each shard's names once, without reopening it per tensor.
            for (const auto& [tensor_name, tensor] : out) {
                (void)tensor;
                if (!owners.contains(tensor_name)) owners.emplace(tensor_name, s);
            }
        }
        const auto it = out.find(name);
        if (it == out.end()) throw std::runtime_error("indexed tensor missing from its shard: " + name);
        // The owner must be the declared shard, even if a different shard supplied the same name earlier.
        if (owners.at(name) != s)
            throw std::runtime_error("index tensor assigned to the wrong shard: " + name);
    }
    if (out.size() != map.size()) throw std::runtime_error("unindexed tensors in safetensors shards");
    return out;
}
std::vector<float> decode_row(const Tensor& t, std::uint64_t row) {
    if (t.shape.empty() || t.shape.size() > 2) throw std::runtime_error("weight row needs rank 1 or 2");
    const auto columns = t.shape.back(), rows = t.shape.size() == 1 ? 1 : t.shape.front();
    if (!columns || row >= rows || columns > SIZE_MAX / sizeof(float)) throw std::out_of_range("weight row out of range");
    std::vector<float> out(static_cast<std::size_t>(columns));
    const auto data = t.bytes();
    if (t.dtype == "GGUF") {
        const auto stride = ggml_dequant::row_bytes(t.ggml_type, static_cast<std::size_t>(columns));
        if (rows > data.size() / stride) throw std::runtime_error("short GGUF row payload");
        ggml_dequant::dequantize_row(t.ggml_type, data.subspan(static_cast<std::size_t>(row) * stride, stride), out);
    } else {
        const std::size_t width = t.dtype == "F32" ? 4 : (t.dtype == "F16" || t.dtype == "BF16") ? 2 : 0;
        if (!width || columns > SIZE_MAX / width || rows > data.size() / (columns * width))
            throw std::runtime_error("unsupported or short scalar weight payload");
        for (std::size_t i = 0; i < out.size(); ++i) {
            const auto* p = data.data() + (row * columns + i) * width;
            if (width == 4) { std::memcpy(&out[i], p, 4); }
            else { std::uint16_t bits; std::memcpy(&bits, p, 2);
                out[i] = t.dtype == "F16" ? ggml_dequant::fp16(bits) : ggml_dequant::bf16(bits); }
        }
    }
    return out;
}
namespace {
float e4m3(std::uint8_t bits) {
    const auto exp = (bits >> 3) & 15, mantissa = bits & 7;
    if ((bits & 127) == 127) return std::numeric_limits<float>::quiet_NaN();
    const float value = exp == 0 ? std::ldexp(static_cast<float>(mantissa), -9) :
        std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F, exp - 7);
    return bits & 128 ? -value : value;
}
float global_value(const Tensor& t) {
    float value;
    if (t.dtype != "F32" || t.size != 4) throw std::runtime_error("NVFP4 global scale must be an F32 scalar");
    std::memcpy(&value, t.bytes().data(), 4);
    if (!std::isfinite(value) || value <= 0) throw std::runtime_error("NVFP4 global scale must be positive and finite");
    return value;
}
}
Nvfp4 nvfp4(const Archive& archive, const std::string& base, std::uint64_t rows, std::uint64_t columns) {
    const auto get = [&archive](const std::string& name) -> const Tensor& {
        const auto it = archive.find(name);
        if (it == archive.end()) throw std::runtime_error("missing NVFP4 tensor: " + name);
        return it->second;
    };
    Nvfp4 out{get(base + ".weight"), get(base + ".weight_scale"), get(base + ".weight_scale_2"), rows, columns};
    if (!rows || !columns || columns % 16 || rows > UINT64_MAX / columns || out.packed.dtype != "U8" ||
        out.packed.shape != std::vector<std::uint64_t>{rows, columns / 2} ||
        out.block_scales.shape != std::vector<std::uint64_t>{rows, columns / 16} ||
        (out.block_scales.dtype != "F8_E4M3" && out.block_scales.dtype != "F8_E4M3FN") ||
        out.packed.size != rows * (columns / 2) || out.block_scales.size != rows * (columns / 16) ||
        !(out.global_scale.shape.empty() || out.global_scale.shape == std::vector<std::uint64_t>{1}))
        throw std::runtime_error("unsupported NVFP4 layout: " + base);
    (void)out.packed.bytes();
    (void)global_value(out.global_scale);
    for (auto b : out.block_scales.bytes()) {
        const float s = e4m3(std::to_integer<std::uint8_t>(b));
        if (!std::isfinite(s) || s < 0) throw std::runtime_error("NVFP4 block scale must be nonnegative and finite");
    }
    return out;
}
std::vector<float> Nvfp4::decode_row(std::uint64_t row) const {
    if (row >= rows) throw std::out_of_range("NVFP4 row out of range");
    constexpr float magnitudes[8] = {0, 0.5F, 1, 1.5F, 2, 3, 4, 6};
    std::vector<float> out(static_cast<std::size_t>(columns));
    const auto* codes = packed.bytes().data() + row * (columns / 2);
    const auto* scales = block_scales.bytes().data() + row * (columns / 16);
    const float global = global_value(global_scale);
    for (std::size_t i = 0; i < out.size(); ++i) {
        const auto code = (std::to_integer<unsigned>(codes[i / 2]) >> (4 * (i % 2))) & 15;
        const float value = code & 8 ? -magnitudes[code & 7] : magnitudes[code & 7];
        out[i] = (value * e4m3(std::to_integer<std::uint8_t>(scales[i / 16]))) * global;
    }
    return out;
}
} // namespace strix::weights
