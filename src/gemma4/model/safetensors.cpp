#include "model.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strix::gemma4 {
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
    std::string text((std::istreambuf_iterator<char>(f)), {});
    if (text.size() > 64U * 1024U * 1024U) throw std::runtime_error("JSON asset too large");
    return serve::parse_json_unique(text);
}
namespace model_detail {
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
    using namespace model_detail;
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
        {"I8",1},{"U8",1},{"BOOL",1},{"I16",2},{"U16",2},{"I32",4},{"U32",4},{"I64",8},{"U64",8},{"F8_E4M3",1},{"F8_E5M2",1}};
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
} // namespace strix::gemma4
