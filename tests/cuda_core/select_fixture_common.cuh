#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace k5c_fixture {

inline void check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "k5c %s: %s\n", operation, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

template <class T>
class Device {
public:
    explicit Device(const std::vector<T>& values) : size_(values.size()) {
        check(cudaMalloc(&data, size_ * sizeof(T)), "allocation");
        check(cudaMemcpy(data, values.data(), size_ * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    Device(std::size_t size, T initial) : Device(std::vector<T>(size, initial)) {}
    ~Device() { (void)cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    std::vector<T> read() const {
        std::vector<T> values(size_);
        check(cudaMemcpy(values.data(), data, size_ * sizeof(T), cudaMemcpyDeviceToHost), "download");
        return values;
    }

    T* data = nullptr;
private:
    std::size_t size_;
};

template <class T>
void compare(const char* name, const std::vector<T>& expected, const std::vector<T>& actual) {
    if (expected.size() == actual.size() &&
        std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(T)) == 0) return;
    std::size_t i = 0;
    while (i < expected.size() && i < actual.size() && std::memcmp(&expected[i], &actual[i], sizeof(T)) == 0) ++i;
    std::fprintf(stderr, "k5c %s FAIL at element %zu\n", name, i);
    std::exit(EXIT_FAILURE);
}

template <class T>
void compare(const char* name, const Device<T>& expected, const Device<T>& actual) {
    compare(name, expected.read(), actual.read());
}

inline std::vector<float> values(std::size_t n, unsigned seed) {
    std::vector<float> result(n);
    for (auto& value : result) {
        seed = seed * 1664525U + 1013904223U;
        value = static_cast<float>(static_cast<int>(seed >> 16U) - 32768) / 65536.0F;
    }
    return result;
}

constexpr unsigned poison = 0xdeadbeefU;
constexpr float score_poison = -991.0F;

} // namespace k5c_fixture
