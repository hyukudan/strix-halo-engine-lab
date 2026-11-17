#pragma once

// Runtime host memory as driver pages, ONE home for every HIP binary (host only, no HIP headers).
//
// By default the HIP runtime backs every host allocation it makes (its queues, kernel arguments, code objects, and
// each hipHostMalloc) with shared anonymous memory wrapped in a user-pointer buffer object. Those pages are movable:
// when page compaction or reclaim migrates one of them the driver evicts every queue of the process until the buffer
// is revalidated (tens to hundreds of milliseconds of lost GPU time). The engine's own large buffers are already
// immune (strix/pinned_host.hpp: anonymous, long-term pinned); the runtime's internal memory is not, and no
// user-space lock or pin protects shared memory from compaction.
//
// The runtime can instead take that memory from the driver (HSA_USERPTR_FOR_PAGED_MEM=0): driver-owned pages, the
// same kind that holds the model weights, which the kernel never migrates. With it no host page the device can see is
// movable, so compaction has nothing to evict, whatever the system's memory settings are. Kernel arithmetic and every
// byte read or written by the device are unchanged (bitwise): only which physical pages back the runtime's memory.
//
// One consequence is handled here and in strix/pinned_host.hpp: driver pages cannot be the target of direct file I/O
// (O_DIRECT reads and fixed I/O buffers take page references, which the kernel refuses on device mappings). A buffer
// that is both a device-transfer source and a direct-I/O target therefore must not come from the runtime allocator in
// this mode: host_pin::allocate_io() follows host_io_route() below.
//
// The runtime reads the value while it initialises, which with eager code loading happens before main(): the start
// gate inserts it before the eager-code restart so one restart carries both, and restarts the image itself when that
// restart does not happen.
//
// Policy (STRIX_HOST_DRIVER_PAGES): "1" = driver pages, "0" / unset = the runtime default (opt-in until the pair
// receipt exists). An explicitly supplied HSA_USERPTR_FOR_PAGED_MEM always wins, even empty, and is never overwritten.

#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>

#include "strix/lever.hpp"

namespace strix::device {

inline constexpr std::string_view kHostDriverPagesName = "host.driver_pages";
inline constexpr lever::Decl kHostDriverPages{kHostDriverPagesName, "STRIX_HOST_DRIVER_PAGES", lever::Exactness::bitwise,
                                              false, {}, {}, "pair receipt pending (in-window evicted_ms, 1 and 3 prefills)"};
inline constexpr const char* kRuntimePagedMemory = "HSA_USERPTR_FOR_PAGED_MEM";

// The runtime's own rule: user pages unless the value is exactly "0".
[[nodiscard]] constexpr bool runtime_host_memory_is_driver_pages(const char* value) {
    return value != nullptr && std::string_view(value) == "0";
}

// Where a buffer that is also a direct-I/O target may come from.
enum class HostIoRoute {
    shared_allocator,  // the shared host allocator's usual choice (runtime memory is valid for direct I/O)
    own_pages,         // always the engine's own anonymous pages, pinned when the system allows it
};
[[nodiscard]] constexpr HostIoRoute host_io_route(bool runtime_driver_pages) {
    return runtime_driver_pages ? HostIoRoute::own_pages : HostIoRoute::shared_allocator;
}

// Returns true when it set HSA_USERPTR_FOR_PAGED_MEM=0 (the caller makes sure a restarted image sees it).
template<class Get, class Put>
inline bool apply_runtime_host_driver_pages(Get get, Put put) {
    if (get(kRuntimePagedMemory) != nullptr) return false;
    if (!lever::select(kHostDriverPages, lever::Inputs{}, get("STRIX_HOST_DRIVER_PAGES")).on) return false;
    if (!put(kRuntimePagedMemory, "0")) throw std::runtime_error("cannot configure runtime host memory");
    return true;
}

inline int& runtime_host_memory_restart_error() {
    static int error = 0;
    return error;
}

// After the eager-code restart did not replace the image: restart for the inserted value, and own it until exec
// succeeds. A failed exec must not leave a value this already-initialised runtime did not read: the engine would
// route direct-I/O buffers for a mode the runtime is not in.
template<class Erase, class Restart>
inline int finish_runtime_host_driver_pages(bool inserted, Erase erase, Restart restart) {
    if (!inserted) return 0;
    const int error = restart();
    runtime_host_memory_restart_error() = error;
    if (!erase(kRuntimePagedMemory)) throw std::runtime_error("cannot roll back runtime host memory after failed restart");
    return error;
}

// Receipt line, empty in the runtime's default mode (the output of an unchanged configuration stays as it was).
[[nodiscard]] inline std::string runtime_host_memory_receipt(const char* runtime_value, const char* policy, int restart_error) {
    if (restart_error != 0)
        return std::string("runtime host memory: user pages (restart failed: ") + std::strerror(restart_error) + ")";
    if (!runtime_host_memory_is_driver_pages(runtime_value)) return {};
    const bool lever_on = policy != nullptr && std::string_view(policy) == "1";
    return std::string("runtime host memory: driver pages (") + (lever_on ? "STRIX_HOST_DRIVER_PAGES" : kRuntimePagedMemory) +
           (lever_on ? "=1" : "=0") + "), direct-I/O staging on engine pages";
}

} // namespace strix::device
