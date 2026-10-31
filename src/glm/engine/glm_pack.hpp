#pragma once
// GLM engine: safetensors pack index + O_DIRECT reads through one bounce buffer (copied from the MiniMax engine's
// mm_common.hpp / mm_pack.cpp, model-agnostic): no file page is ever handed to HIP.
#include "glm_json.hpp"

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace strix::glm::engine::detail {

// ---------------------------------------------------------------- minimal JSON (safetensors headers, index)
// ---------------------------------------------------------------- safetensors pack, O_DIRECT reads
struct TensorInfo {
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::uint32_t file = 0;
    std::uint64_t offset = 0;  // absolute byte offset in the file
    std::uint64_t bytes = 0;
};
class PackReader {
public:
    explicit PackReader(std::string dir);
    ~PackReader();
    PackReader(const PackReader&) = delete;
    PackReader& operator=(const PackReader&) = delete;
    const TensorInfo& info(const std::string& name) const;
    // Bytes [offset, offset + n) of the tensor into dst (any host memory, typically the pinned staging buffer).
    void read(const TensorInfo& t, std::uint64_t offset, std::uint64_t n, void* dst);
    // Plain tensors (BF16 / F16 / F32) converted to FP32, as the gates' Pack::floats.
    std::vector<float> floats(const std::string& name);
    double gib_read() const { return static_cast<double>(bytes_read_) / 1073741824.0; }
    std::uint64_t bytes_read() const { return bytes_read_; }
    // Shards opened O_DIRECT vs buffered (buffered reads drop their pages with posix_fadvise DONTNEED after each read).
    std::size_t direct_files() const;
    std::size_t files() const { return fds_.size(); }
    static std::uint64_t bounce_bytes();

private:
    // [pos, pos + n) of shard `file`: O_DIRECT through the bounce, or buffered (no readahead) + DONTNEED.
    void read_at(std::uint32_t file, std::uint64_t pos, std::uint64_t n, void* dst);
    void open_shards(const std::vector<std::string>& files);
    std::string dir_;
    std::vector<int> fds_;
    std::vector<int> direct_;  // 1 if opened with O_DIRECT
    std::map<std::string, TensorInfo> tensors_;
    std::uint8_t* bounce_ = nullptr;
    std::uint64_t bytes_read_ = 0;
};

}  // namespace strix::glm::engine::detail
