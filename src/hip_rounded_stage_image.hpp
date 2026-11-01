#pragma once

#include "strix/compact_stage_images.hpp"

namespace strix {

// A view of a BF16-rounded image. Only the storage changes; all arithmetic and
// status checks still see precisely the incumbent FP32 bits.
struct RoundedStageImage {
    void* data = nullptr;
    bool compact = false;
    __host__ __device__ RoundedStageImage() = default;
    __host__ __device__ RoundedStageImage(float* p, bool c = false) : data(p), compact(c) {}
    struct Reference {
        void* data;
        bool compact;
        std::uint64_t index;
        __device__ __forceinline__ operator float() const {
            return compact ? __uint_as_float(static_cast<unsigned>(static_cast<const std::uint16_t*>(data)[index]) << 16U)
                           : static_cast<const float*>(data)[index];
        }
        __device__ __forceinline__ void operator=(float value) const {
            if (compact) static_cast<std::uint16_t*>(data)[index] = static_cast<std::uint16_t>(__float_as_uint(value) >> 16U);
            else static_cast<float*>(data)[index] = value;
        }
    };
    __device__ __forceinline__ Reference operator[](std::uint64_t i) const { return {data, compact, i}; }
    __device__ __forceinline__ RoundedStageImage operator+(std::uint64_t i) const {
        RoundedStageImage result = *this;
        result.data = static_cast<unsigned char*>(data) + i * (compact ? 2U : 4U);
        return result;
    }
};

} // namespace strix
