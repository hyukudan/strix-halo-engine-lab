#include "strix/ds41_weight_overlay.hpp"

#include "strix/gguf_reader.hpp"

#include <cstdio>
#include <stdexcept>
#include <unordered_map>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strix::ds41 {

WeightOverlay::WeightOverlay(const std::filesystem::path& path, const ModelManifest& base) : path_(path) {
#if defined(_WIN32)
    throw std::runtime_error("ds41 weight overlay: file mapping is implemented for Linux only");
#else
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
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) throw std::runtime_error("ds41 weight overlay: cannot open " + path.string());
    struct stat st {};
    if (::fstat(fd_, &st) != 0) throw std::runtime_error("ds41 weight overlay: cannot stat " + path.string());
    size_ = static_cast<std::uint64_t>(st.st_size);
    void* p = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (p == MAP_FAILED) throw std::runtime_error("ds41 weight overlay: mmap failed");
    base_ = static_cast<const std::uint8_t*>(p);
    fnv_ = 14695981039346656037ULL;
    for (const auto& [name, e] : entries_) {  // name order: the identity does not depend on the file layout
        for (const char ch : name) fnv_ = (fnv_ ^ static_cast<std::uint8_t>(ch)) * 1099511628211ULL;
        for (const auto b : bytes(e)) fnv_ = (fnv_ ^ b) * 1099511628211ULL;
    }
#endif
}

WeightOverlay::~WeightOverlay() {
#if !defined(_WIN32)
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
