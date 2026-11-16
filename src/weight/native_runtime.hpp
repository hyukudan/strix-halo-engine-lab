#pragma once
#include <strix/weight/runtime.hpp>
#include <strix/weight/source.hpp>
#include <algorithm>
#include <array>
#include <mutex>
#include <vector>
namespace strix::weight::detail {
using namespace strix::weight;
struct Pending { void* data; std::vector<void*> events; std::uint32_t device; bool failed; };
template<class Api> struct NativeRuntime {
inline static std::mutex retired_mutex;
inline static std::vector<Pending> retired;
static Status check(typename Api::Error error) { return error==Api::Success ? Status{Error::ok,{}} : Status{Error::device_failure,Api::GetErrorString(error)}; }
static void collect() {
    std::lock_guard lock(retired_mutex);
    for (auto it=retired.begin(); it!=retired.end();) {
        Api::SetDevice(static_cast<int>(it->device));
        bool done=!it->failed;
        for (auto event:it->events) done=done && Api::EventQuery(static_cast<typename Api::Event>(event))==Api::Success;
        if (!done) { ++it; continue; }
        for (auto event:it->events) Api::EventDestroy(static_cast<typename Api::Event>(event));
        Api::Free(it->data); it=retired.erase(it);
    }
}

static strix::weight::Status bind(void* stream, std::uint32_t device, ExecutionContext& context, Hardware& hardware) {
    if (device>static_cast<std::uint32_t>(INT32_MAX)) return {Error::device_failure,"device index overflow"};
    auto status=check(Api::SetDevice(static_cast<int>(device))); if (!status) return status;
    typename Api::Properties properties{}; status=check(Api::GetDeviceProperties(&properties,static_cast<int>(device))); if (!status) return status;
    std::size_t free_bytes=0,total_bytes=0; status=check(Api::MemGetInfo(&free_bytes,&total_bytes)); if (!status) return status;
    Hardware h{}; h.backend=Api::backend; h.arch=Api::arch(properties);
    h.compiler_id=current_compiler_id(); h.device_class=h.arch; h.free_bytes=free_bytes;
    h.warp_or_wave=static_cast<std::uint32_t>(properties.warpSize);
    h.shared_bytes_per_sm=static_cast<std::uint32_t>(properties.sharedMemPerMultiprocessor);
    // Calibrated rates and image budget are supplied by deployment/receipts, never guessed here.
    auto callbacks=std::make_shared<RuntimeCallbacks>();
    callbacks->allocate=[device](std::uint64_t bytes,void*& data) -> Status {
        collect(); auto s=check(Api::SetDevice(static_cast<int>(device))); if (!s) return s;
        if (bytes>SIZE_MAX) return {Error::memory_budget,"allocation size overflow"};
        return check(Api::Malloc(&data,static_cast<std::size_t>(bytes ? bytes : 1)));
    };
    callbacks->free=[device](void* data) { Api::SetDevice(static_cast<int>(device)); Api::Free(data); };
    callbacks->record_event=[device](ExecutionContext& c,void*& value) -> Status {
        auto s=check(Api::SetDevice(static_cast<int>(device))); if (!s) return s;
        typename Api::Event event{}; s=check(Api::EventCreateWithFlags(&event,Api::EventDisableTiming)); if (!s) return s;
        s=check(Api::EventRecord(event,static_cast<typename Api::Stream>(c.stream)));
        if (!s) { Api::EventDestroy(event); return s; } value=event; return s;
    };
    callbacks->wait_event=[device](ExecutionContext& c,void* event) -> Status {
        auto s=check(Api::SetDevice(static_cast<int>(device))); if (!s) return s;
        return check(Api::StreamWaitEvent(static_cast<typename Api::Stream>(c.stream),static_cast<typename Api::Event>(event),0));
    };
    callbacks->event_complete=[device](void* event) { Api::SetDevice(static_cast<int>(device)); return Api::EventQuery(static_cast<typename Api::Event>(event))==Api::Success; };
    callbacks->destroy_event=[device](void* event) { Api::SetDevice(static_cast<int>(device)); Api::EventDestroy(static_cast<typename Api::Event>(event)); };
    callbacks->convert=[device](const WeightView& w,const Plan& p,void* data,ExecutionContext&) -> Status {
        if (p.selected.strategy!=Strategy::native || p.selected.resources.image_bytes!=w.payload.range.bytes)
            return {Error::unsupported_arithmetic,"image conversion source not registered"};
        auto s=check(Api::SetDevice(static_cast<int>(device))); if (!s) return s;
        std::array<Byte,65536> chunk{};
        for (std::uint64_t at=0; at<w.payload.range.bytes;) {
            auto bytes=static_cast<std::size_t>(std::min<std::uint64_t>(chunk.size(),w.payload.range.bytes-at));
            s=w.payload.source->read({w.payload.range.offset+at,bytes},{chunk.data(),bytes}); if (!s) return s;
            s=check(Api::Memcpy(static_cast<Byte*>(data)+at,chunk.data(),bytes,Api::MemcpyHostToDevice)); if (!s) return s;
            at+=bytes;
        }
        return {Error::ok,{}};
    };
    callbacks->retire=[device](void* data,std::vector<void*> events,ExecutionContext c) {
        Api::SetDevice(static_cast<int>(device)); typename Api::Event fence{};
        bool failed=Api::EventCreateWithFlags(&fence,Api::EventDisableTiming)!=Api::Success;
        if (!failed) {
            if (Api::EventRecord(fence,static_cast<typename Api::Stream>(c.stream))!=Api::Success) { Api::EventDestroy(fence); failed=true; }
            else events.push_back(fence);
        }
        { std::lock_guard lock(retired_mutex); retired.push_back({data,std::move(events),device,failed}); }
        collect();
    };
    context={Api::backend,stream,device,std::move(callbacks)}; hardware=std::move(h);
    return {Error::ok,{}};
}
};

}
