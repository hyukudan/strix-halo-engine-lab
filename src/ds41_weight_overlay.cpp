#include "strix/file_snapshot.hpp"
#include "strix/ds41_weight_overlay.hpp"

#include "strix/gguf_reader.hpp"

#include <cstdio>
#include <stdexcept>
#include <unordered_map>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strix::ds41 {

WeightOverlay::WeightOverlay(const std::filesystem::path& path, const ModelManifest& base) : path_(path) {
    const auto info = gguf::read_file(path);
    std::unordered_map<std::string, const ManifestTensor*> model;
    for (const auto& t : base.tensors) model.emplace(t.name, &t);
    for (const auto& t : info.tensors) {
        const auto it = model.find(t.name);
        if (it == model.end()) throw std::invalid_argument("ds41 weight overlay: " + t.name + " is not a model tensor");
        const auto& m = *it->second;
        if (m.dimensions != t.dimensions || m.ggml_type != t.ggml_type || m.byte_size != t.byte_size)
            throw std::invalid_argument("ds41 weight overlay: " + t.name + " differs from the model in shape, type or size");
        if (!entries_.emplace(t.name, Entry{t.absolute_offset, t.byte_size}).second)
            throw std::invalid_argument("ds41 weight overlay: duplicate tensor " + t.name);
        bytes_ += t.byte_size;
    }
    if (entries_.empty()) throw std::invalid_argument("ds41 weight overlay: no tensors in " + path.string());
    if (const auto* id = info.find_metadata("ds41.requant.identity"); id != nullptr && id->string_value)
        identity_ = *id->string_value;
    else
        throw std::invalid_argument("ds41 weight overlay: missing ds41.requant.identity");
#if defined(_WIN32)
    // read-only view of the whole file; the view keeps the mapping alive once both handles are closed
    const HANDLE file = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                      FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) throw std::runtime_error("ds41 weight overlay: cannot open " + path.string());
    LARGE_INTEGER file_bytes{};
    const HANDLE mapping = ::GetFileSizeEx(file, &file_bytes) && file_bytes.QuadPart > 0
                               ? ::CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr)
                               : nullptr;
    void* p = mapping != nullptr ? ::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0) : nullptr;
    if (mapping != nullptr) ::CloseHandle(mapping);
    ::CloseHandle(file);
    if (p == nullptr) throw std::runtime_error("ds41 weight overlay: cannot map " + path.string());
    size_ = static_cast<std::uint64_t>(file_bytes.QuadPart);
#else
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) throw std::runtime_error("ds41 weight overlay: cannot open " + path.string());
    strix::LoaderCapture::record(path, fd_);
    struct stat st {};
    if (::fstat(fd_, &st) != 0) throw std::runtime_error("ds41 weight overlay: cannot stat " + path.string());
    size_ = static_cast<std::uint64_t>(st.st_size);
    void* p = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (p == MAP_FAILED) throw std::runtime_error("ds41 weight overlay: mmap failed");
#endif
    base_ = static_cast<const std::uint8_t*>(p);
    fnv_ = 14695981039346656037ULL;
    for (const auto& [name, e] : entries_) {  // name order: the identity does not depend on the file layout
        for (const char ch : name) fnv_ = (fnv_ ^ static_cast<std::uint8_t>(ch)) * 1099511628211ULL;
        for (const auto b : bytes(e)) fnv_ = (fnv_ ^ b) * 1099511628211ULL;
    }
}

WeightOverlay::~WeightOverlay() {
#if defined(_WIN32)
    if (base_ != nullptr) ::UnmapViewOfFile(base_);
#else
    if (base_ != nullptr) ::munmap(const_cast<std::uint8_t*>(base_), size_);
    if (fd_ >= 0) ::close(fd_);
#endif
}

const WeightOverlay::Entry* WeightOverlay::find(const std::string& name) const {
    const auto it = entries_.find(name);
    return it == entries_.end() ? nullptr : &it->second;
}

std::span<const std::uint8_t> WeightOverlay::bytes(const Entry& e) const {
    if (e.absolute_offset > size_ || e.byte_size > size_ - e.absolute_offset)
        throw std::out_of_range("ds41 weight overlay: tensor outside the file");
    return {base_ + e.absolute_offset, static_cast<std::size_t>(e.byte_size)};
}

std::string WeightOverlay::describe() const {
    char line[160];
    std::snprintf(line, sizeof line, " tensors %zu bytes %llu content_fnv %016llx", entries_.size(),
                  static_cast<unsigned long long>(bytes_), static_cast<unsigned long long>(fnv_));
    return "overlay " + path_.string() + " identity " + identity_ + line;
}

}  // namespace strix::ds41
