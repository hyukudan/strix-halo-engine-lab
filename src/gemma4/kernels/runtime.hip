#include <hip/hip_runtime.h>
#include "ops.hpp"
#include <cstring>
#include <stdexcept>
#include <string>

namespace strix::gemma4::gpu {
namespace {
void checked(hipError_t e) {
    if (e != hipSuccess)
        throw std::runtime_error(std::string("gemma4 HIP: ") + hipGetErrorString(e));
}
} // namespace
Buffer::Buffer(std::size_t bytes) {
    resize(bytes);
}
Buffer::~Buffer() {
    if (data_)
        (void)hipFree(data_);
}
Buffer::Buffer(Buffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}
Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        if (data_)
            (void)hipFree(data_);
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
    }
    return *this;
}
void Buffer::resize(std::size_t bytes) {
    if (bytes == size_)
        return;
    void* next = nullptr;
    if (bytes)
        checked(hipMalloc(&next, bytes));
    if (data_)
        (void)hipFree(data_);
    data_ = next;
    size_ = bytes;
}
void Buffer::upload(const void* src, std::size_t bytes, std::size_t offset) {
    if (offset > size_ || bytes > size_ - offset || (bytes && !src))
        throw std::runtime_error("device upload bounds");
    if (bytes)
        checked(hipMemcpy(static_cast<std::uint8_t*>(data_) + offset, src, bytes, hipMemcpyHostToDevice));
}
void Buffer::download(void* dst, std::size_t bytes, std::size_t offset) const {
    if (offset > size_ || bytes > size_ - offset || (bytes && !dst))
        throw std::runtime_error("device download bounds");
    if (bytes)
        checked(hipMemcpy(dst, static_cast<const std::uint8_t*>(data_) + offset, bytes, hipMemcpyDeviceToHost));
}
void Buffer::zero() {
    if (size_)
        checked(hipMemset(data_, 0, size_));
}
void synchronize(Queue q) {
    checked(hipStreamSynchronize(static_cast<hipStream_t>(q)));
}
void require_device() {
    int device = 0;
    checked(hipGetDevice(&device));
    hipDeviceProp_t p{};
    checked(hipGetDeviceProperties(&p, device));
    if (std::string(p.gcnArchName).find("gfx1151") != 0 || p.warpSize != 32)
        throw std::runtime_error("g4-v1 device implementation requires gfx1151 wave32");
}
} // namespace strix::gemma4::gpu
