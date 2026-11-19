#pragma once
// Test support for the wfmt GGUF tests: a minimal GGUF v3 writer (metadata + tensor directory, zero data), the parser
// call with test limits, and a failure counter. Header-only, one copy per test executable.
#include "strix/gguf_reader.hpp"
#include "strix/weight/tensor_map.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace gguf = strix::gguf;
namespace tm = strix::weight::tensor_map;

int failures = 0;
void check(bool condition, const char* what) {
    if (!condition) {
        ++failures;
        std::printf("FAIL %s\n", what);
    }
}

// Minimal GGUF v3 builder (little-endian), enough for metadata scalars/arrays and a tensor directory with zero data.
struct Builder {
    std::string kv;
    std::uint64_t kv_count = 0;
    struct Tensor { std::string name; std::vector<std::uint64_t> dims; std::uint32_t type; std::uint64_t bytes; };
    std::vector<Tensor> tensors;

    template <class T> static void put(std::string& out, T value) {
        char bytes[sizeof(T)];
        std::memcpy(bytes, &value, sizeof(T));
        out.append(bytes, sizeof(T));
    }
    static void put_string(std::string& out, const std::string& text) {
        put<std::uint64_t>(out, text.size());
        out += text;
    }
    void key(const std::string& name, std::uint32_t type) {
        put_string(kv, name);
        put<std::uint32_t>(kv, type);
        ++kv_count;
    }
    void u32(const std::string& name, std::uint32_t value) { key(name, 4); put(kv, value); }
    void str(const std::string& name, const std::string& value) { key(name, 8); put_string(kv, value); }
    void i32_array(const std::string& name, const std::vector<std::int32_t>& values) {
        key(name, 9);
        put<std::uint32_t>(kv, 5);
        put<std::uint64_t>(kv, values.size());
        for (const auto value : values) put(kv, value);
    }
    void u64_array(const std::string& name, const std::vector<std::uint64_t>& values) {
        key(name, 9);
        put<std::uint32_t>(kv, 10);
        put<std::uint64_t>(kv, values.size());
        for (const auto value : values) put(kv, value);
    }
    void bool_array(const std::string& name, const std::vector<bool>& values) {
        key(name, 9);
        put<std::uint32_t>(kv, 7);
        put<std::uint64_t>(kv, values.size());
        for (const bool value : values) put<std::uint8_t>(kv, value ? 1 : 0);
    }
    void tensor(const std::string& name, std::vector<std::uint64_t> dims, std::uint32_t type, std::uint64_t bytes) {
        tensors.push_back({name, std::move(dims), type, bytes});
    }
    std::string build() const {
        std::string out = "GGUF";
        put<std::uint32_t>(out, 3);
        put<std::uint64_t>(out, tensors.size());
        put<std::uint64_t>(out, kv_count);
        out += kv;
        std::uint64_t offset = 0;
        for (const auto& t : tensors) {
            put_string(out, t.name);
            put<std::uint32_t>(out, static_cast<std::uint32_t>(t.dims.size()));
            for (const auto d : t.dims) put(out, d);
            put(out, t.type);
            put(out, offset);
            offset += (t.bytes + 31) / 32 * 32;
        }
        out.append((32 - out.size() % 32) % 32, '\0');
        out.append(offset, '\0');
        return out;
    }
};

gguf::FileInfo parse(const std::string& bytes) {
    std::istringstream input(bytes, std::ios::binary);
    gguf::Limits limits;
    limits.max_retained_array_elements = 4096;
    return gguf::read(input, bytes.size(), limits);
}

}  // namespace
