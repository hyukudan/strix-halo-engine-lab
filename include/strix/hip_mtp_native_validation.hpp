#pragma once
#include "strix/hip_mtp_resources.hpp"
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <initializer_list>
#include <limits>
#include <stdexcept>
namespace strix::mtp_native {
inline std::uint64_t sum(std::uint64_t a,std::uint64_t b) {
    if(b>std::numeric_limits<std::uint64_t>::max()-a)
        throw std::overflow_error("MTP native extent sum overflow");
    return a+b;
}
inline std::uint64_t product(std::uint64_t a,std::uint64_t b) {
    if (b && a>std::numeric_limits<std::uint64_t>::max()/b)
        throw std::overflow_error("MTP native extent overflow");
    return a*b;
}
inline void context(const MtpNativeContext& c) {
    if (!c.resources || !c.model || !c.plan || !c.tensors ||
        c.model!=&c.resources->model() || c.plan!=&c.resources->plan() ||
        c.tensors!=&c.resources->tensors())
        throw std::invalid_argument("MTP native owner context mismatch");
    c.resources->validate_stage();
    if (c.position>=c.resources->context_capacity())
        throw std::out_of_range("MTP native position exceeds retained capacity");
}
inline MtpDeviceView buffer(const MtpNativeContext& c,HipMtpBuffer id,
                           std::uint64_t bytes,std::uint64_t alignment=16) {
    auto v=c.resources->view(id);
    if (!bytes || !v.address || !v.allocation_generation || v.bytes<bytes ||
        v.address%alignment || v.bytes>std::numeric_limits<std::uintptr_t>::max()-v.address)
        throw std::invalid_argument("MTP native buffer extent/alignment/generation invalid");
    return v;
}
inline void disjoint(std::initializer_list<MtpDeviceView> views) {
    for(auto a=views.begin();a!=views.end();++a)
        for(auto b=a+1;b!=views.end();++b)
            if(a->address<b->address+b->bytes && b->address<a->address+a->bytes)
                throw std::invalid_argument("MTP native buffers overlap");
}
inline void tensor(const MtpTensorView& t,std::initializer_list<std::uint64_t> shape,
                   std::uint64_t bytes) {
    if (!t.manifest || !t.runtime || !t.runtime->address ||
        !t.runtime->allocation_generation || !runtime_tensor_gpu_visible(t.runtime->storage) ||
        t.runtime->bytes<bytes || t.manifest->byte_size<bytes ||
        t.runtime->bytes>std::numeric_limits<std::uintptr_t>::max()-t.runtime->address ||
        t.manifest->dimensions.size()!=shape.size() ||
        !std::equal(t.manifest->dimensions.begin(),t.manifest->dimensions.end(),shape.begin()))
        throw std::invalid_argument("MTP native tensor shape/storage/extent invalid");
}
inline void stream(const MtpNativeContext& c,void* stream) {
    if(!stream || stream!=c.resources->stream())
        throw std::invalid_argument("MTP native launch stream differs from retained owner");
    hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;
    if(hipStreamIsCapturing(reinterpret_cast<hipStream_t>(stream),&capture)!=hipSuccess)
        throw std::runtime_error("MTP native stream capture query failed");
    if(capture!=hipStreamCaptureStatusNone)
        throw std::invalid_argument("MTP native entry points require eager noncapture stream");
}
} // namespace strix::mtp_native
