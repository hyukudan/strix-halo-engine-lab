#include "backend.hpp"

#include "../arch/xe2/caps.hpp"

#if defined(STRIX_HAVE_SYCL) && STRIX_HAVE_SYCL
#include "strix/device/intel_enumerator.hpp"

#include <sycl/sycl.hpp>
#endif

namespace strix::intel {

const char* status_name(Status status) noexcept {
    switch (status) {
        case Status::ok: return "ok";
        case Status::not_built: return "not_built";
        case Status::unsupported: return "unsupported";
        case Status::invalid: return "invalid";
        default: return "device_error";
    }
}

#if defined(STRIX_HAVE_SYCL) && STRIX_HAVE_SYCL
// Not compiled in CI yet (no oneAPI toolchain on the build hosts): written against SYCL 2020 + the Intel extensions.
namespace {
sycl::queue* q(Queue queue) { return static_cast<sycl::queue*>(queue.impl); }
} // namespace

bool sycl_built() noexcept { return true; }

Status open_queue(int ordinal, Queue* queue, Capabilities* caps) {
    if (!queue) return Status::invalid;
    try {
        int seen = 0;
        for (const auto& d : sycl::device::get_devices(sycl::info::device_type::gpu)) {
            if (d.get_backend() != sycl::backend::ext_oneapi_level_zero || seen++ != ordinal) continue;
            queue->impl = new sycl::queue(d, sycl::property::queue::in_order{});
            if (caps) {
                if (d.has(sycl::aspect::ext_intel_device_id))
                    caps->arch = device::intel_arch_from_device_id(d.get_info<sycl::ext::intel::info::device::device_id>());
                caps->compute_units = d.get_info<sycl::info::device::max_compute_units>();
                caps->global_bytes = d.get_info<sycl::info::device::global_mem_size>();
                caps->local_bytes = d.get_info<sycl::info::device::local_mem_size>();
                caps->integrated = device::intel_arch_integrated(caps->arch);
                caps->subgroup = arch::xe2::kSubgroup;
                caps->xmx = arch::xe2::selects(caps->arch);
            }
            return Status::ok;
        }
        return Status::unsupported;
    } catch (const sycl::exception&) {
        return Status::device_error;
    }
}

void close_queue(Queue& queue) noexcept {
    delete q(queue);
    queue.impl = nullptr;
}

Status allocate(Queue queue, MemoryKind kind, std::size_t bytes, void** out) {
    if (!q(queue) || !out) return Status::invalid;
    switch (kind) {
        case MemoryKind::device: *out = sycl::malloc_device(bytes, *q(queue)); break;
        case MemoryKind::shared: *out = sycl::malloc_shared(bytes, *q(queue)); break;
        default: *out = sycl::malloc_host(bytes, *q(queue)); break;
    }
    return *out ? Status::ok : Status::device_error;
}

Status release(Queue queue, void* pointer) {
    if (!q(queue)) return Status::invalid;
    sycl::free(pointer, *q(queue));
    return Status::ok;
}

Status copy(Queue queue, void* dst, const void* src, std::size_t bytes) {
    if (!q(queue)) return Status::invalid;
    q(queue)->memcpy(dst, src, bytes);
    return Status::ok;
}

Status synchronize(Queue queue) {
    if (!q(queue)) return Status::invalid;
    try {
        q(queue)->wait_and_throw();
        return Status::ok;
    } catch (const sycl::exception&) {
        return Status::device_error;
    }
}

Status enqueue_dense_q8a8(Queue queue, const dense::Q8A8View& view) {
    if (!q(queue)) return Status::invalid;
    if (!dense::supported(view)) return Status::unsupported;
    const dense::Q8A8View v = view;
    q(queue)->parallel_for(sycl::range<2>(v.rows, v.count), [=](sycl::id<2> i) {
        dense::q8a8_cell(v, static_cast<std::uint32_t>(i[0]), static_cast<std::uint32_t>(i[1]));
    });
    return Status::ok;
}
#else
// Build without SYCL: the interface exists and binds nothing, as src/device/hip_gate_none.cpp does for HIP.
bool sycl_built() noexcept { return false; }
Status open_queue(int, Queue* queue, Capabilities*) {
    if (queue) queue->impl = nullptr;
    return Status::not_built;
}
void close_queue(Queue& queue) noexcept { queue.impl = nullptr; }
Status allocate(Queue, MemoryKind, std::size_t, void** out) {
    if (out) *out = nullptr;
    return Status::not_built;
}
Status release(Queue, void*) { return Status::not_built; }
Status copy(Queue, void*, const void*, std::size_t) { return Status::not_built; }
Status synchronize(Queue) { return Status::not_built; }
Status enqueue_dense_q8a8(Queue, const dense::Q8A8View& view) {
    return dense::supported(view) ? Status::not_built : Status::unsupported;
}
#endif

} // namespace strix::intel
