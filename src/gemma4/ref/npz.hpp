#pragma once
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::gemma4 {
struct NpyArray {
    std::string dtype;
    std::vector<std::size_t> shape;
    std::vector<std::uint8_t> bytes;
    template<class T> std::vector<T> values(const std::string& expected) const {
        if (dtype != expected || bytes.size() % sizeof(T)) throw std::runtime_error("NPY dtype mismatch");
        std::vector<T> out(bytes.size() / sizeof(T));
        std::memcpy(out.data(), bytes.data(), bytes.size());
        return out;
    }
};
// Bounded ZIP/NPY reader for fixture files; stored and raw-DEFLATE members, CRC checked.
[[nodiscard]] std::map<std::string, NpyArray> read_npz(const std::filesystem::path& path);
} // namespace strix::gemma4
