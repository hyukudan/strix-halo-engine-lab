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
    SafeFile out;
    std::shared_ptr<const weight::Source> source;
    auto status = weight::open_source(path.string(), source);
    if (status) status = weight::index_safetensors(source, out.index);
    if (!status) throw std::runtime_error(status.detail);
    out.storage = std::make_shared<MappedFile>(path);
    for (const auto& raw : out.index.tensors) {
        SafeTensor t; t.name = raw.name; t.dtype = raw.wire_dtype;
        t.shape.assign(raw.shape.dims.begin(), raw.shape.dims.begin() + raw.shape.rank);
        t.offset = raw.payload.range.offset; t.bytes = raw.payload.range.bytes;
        out.tensors.push_back(std::move(t));
    }
    return out;
}

} // namespace strix::gemma4
