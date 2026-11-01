#pragma once
#include "strix/known_prefill_limits.hpp"
#include "strix/prefill_stage_reservation.hpp"
#include <hip/hip_runtime.h>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>

namespace strix {
inline int hip_prefill_device() {
    int device = -1;
    if (hipGetDevice(&device) != hipSuccess)
        throw std::runtime_error("prefill pools: cannot query owning device");
    return device;
}
// One process scheduler owns submission AND completion of these shared scratch
// images. Setup precedes every launch, on the window's device, before the corresponding
// compiled owner executes. Subsequent owners may reserve a larger T at setup on the same
// device; setup drains old readers before replacing scratch. Mutexes protect host metadata; they do not permit concurrent
// GPU consumers of the same scratch. Execution never grows scratch.
class HipPrefillPoolOwner {
    mutable std::recursive_mutex mutex_;
    std::uint32_t rows_ = 0;
    int device_ = -1;
    bool ready_ = false, reserving_ = false;
public:
    template<class Reserve> void setup(std::uint32_t rows, Reserve reserve) {
        const std::lock_guard lock(mutex_);
        if (!rows || rows > known_prefill_max_rows)
            throw std::invalid_argument("prefill pools: invalid configured maximum rows");
        const auto capacity = known_prefill_pool_rows(rows);
        const auto device = hip_prefill_device();
        if (rows_ && device != device_)
            throw std::runtime_error("prefill pools: setup cannot change device");
        if (ready_ && capacity <= rows_) return;
        // Tail owners share these pools. Keep the high-water reservation's
        // extent, including every stride/offset derived from it, for all callers.
        const auto previous_rows = rows_;
        rows_ = capacity; device_ = device;
        reserving_ = true;
        // A failed reserve can leave individual pools grown, including HC overflow
        // counters initialized at the new capacity's offset. Restoring rows_ only
        // restores owner metadata; setup failure aborts execution, so these partially
        // grown pools must not be consumed after the exception.
        try { reserve(); } catch (...) { rows_ = previous_rows; reserving_ = false; throw; }
        reserving_ = false; ready_ = true;
    }
    std::uint32_t rows(std::uint32_t call_rows = 0) const {
        const std::lock_guard lock(mutex_);
        if ((!ready_ && !reserving_) || call_rows > rows_ || hip_prefill_device() != device_)
            throw std::runtime_error("prefill pools: call exceeds setup capacity or uses another device");
        return rows_;
    }
};
class HipPrefillPool {
    mutable std::mutex mutex_;
    void* data_ = nullptr;
    std::uint64_t bytes_ = 0, generation_ = 0;
    int device_ = -1;
public:
    // Only setup may allocate. Execution always checks the existing reservation.
    void* get(std::uint64_t need, const char* label, bool setup = false, bool zero = false) {
        const std::lock_guard lock(mutex_);
        const auto device = hip_prefill_device();
        if (data_ && (device != device_ || (need > bytes_ && !setup)))
            throw std::runtime_error(std::string(label) + ": immutable reservation/device exceeded");
        if (!data_ || need > bytes_) {
            if (!setup) throw std::runtime_error(std::string(label) + ": pool was not reserved at setup");
            const auto check = [&](hipError_t status) {
                if (status != hipSuccess) throw std::runtime_error(std::string(label) + ": " + hipGetErrorString(status));
            };
            void* allocation = nullptr;
            check(hipMalloc(&allocation, need));
            if (zero) {
                const auto status = hipMemset(allocation, 0, need);
                if (status != hipSuccess) { (void)hipFree(allocation); check(status); }
            }
            if (data_) {
                // Scratch pointers are obtained per launch. Drain earlier consumers
                // before setup replaces the high-water reservation.
                const auto status = hipDeviceSynchronize();
                if (status != hipSuccess) { (void)hipFree(allocation); check(status); }
                check(hipFree(data_));
            }
            data_ = allocation; bytes_ = need; device_ = device; ++generation_;
        }
        return data_;
    }
    std::uint64_t generation() const {
        const std::lock_guard lock(mutex_);
        return generation_;
    }
};
} // namespace strix
