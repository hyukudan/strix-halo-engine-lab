// Host access to raw tensor bytes of the production Qwen GGUF for the CUDA Qwen prefill tests.
// Default path: Qwen3.8-Flash-Next-UD-Q4_K_XL-merged.gguf in the working directory (override with STRIX_QWEN_GGUF).
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "strix/gguf_reader.hpp"

namespace strix::qwen_gguf {

class Model {
public:
    explicit Model(std::string path = {}) {
        if (path.empty()) {
            const char* e = std::getenv("STRIX_QWEN_GGUF");
            path = e && *e ? e : "Qwen3.8-Flash-Next-UD-Q4_K_XL-merged.gguf";
        }
        path_ = path;
        info_ = gguf::read_file(path_);
    }
    [[nodiscard]] const gguf::FileInfo& info() const noexcept { return info_; }
    [[nodiscard]] const gguf::TensorInfo& tensor(const std::string& name) const {
        for (const auto& t : info_.tensors) if (t.name == name) return t;
        throw std::runtime_error("tensor not found: " + name);
    }
    [[nodiscard]] std::vector<std::uint8_t> bytes(const std::string& name) const {
        const auto& t = tensor(name);
        std::vector<std::uint8_t> v(static_cast<std::size_t>(t.byte_size));
        std::FILE* f = std::fopen(path_.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open " + path_);
        _fseeki64(f, static_cast<long long>(t.absolute_offset), SEEK_SET);
        const bool ok = std::fread(v.data(), 1, v.size(), f) == v.size();
        std::fclose(f);
        if (!ok) throw std::runtime_error("short read " + name);
        return v;
    }

private:
    std::string path_;
    gguf::FileInfo info_;
};

}  // namespace strix::qwen_gguf
