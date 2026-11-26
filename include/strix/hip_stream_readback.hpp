#pragma once

#include "strix/env_flags.hpp"
#include "strix/hip_check.hpp"
#include "strix/session/stream_readback.hpp"

namespace strix::device {

// Online schedulers can use observed reply times in their next step's policy.
[[nodiscard]] inline bool hip_pinned_replies_enabled() {
    return envcfg::flag("STRIX_HIP_PINNED_REPLIES", false, EnvFlagMatch::exact);
}

struct HipReadbackBackend {
    using Stream = hipStream_t;

    static void* try_pin(std::size_t bytes) noexcept {
        void* storage = nullptr;
        if (hipHostMalloc(&storage, bytes, hipHostMallocDefault) == hipSuccess) return storage;
        (void)hipGetLastError();
        return nullptr;
    }

    static void unpin(void* storage) noexcept { (void)hipHostFree(storage); }

    static void copy_to_host(void* dst, const void* src, std::size_t bytes, Stream stream, const char* what) {
        hip_check(hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, stream), what);
    }

    static void wait(Stream stream, const char* what) { hip_check(hipStreamSynchronize(stream), what); }
    static bool drain(Stream stream) noexcept { return hipStreamSynchronize(stream) == hipSuccess; }
};

template<class Reply>
using HipStreamReadback = session::StreamReadback<Reply, HipReadbackBackend>;

}  // namespace strix::device
