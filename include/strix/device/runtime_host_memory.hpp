#pragma once

// No host page the device can see is movable: ONE home for every HIP binary (host only, no HIP headers).
//
// The driver evicts every queue of a process whenever the kernel changes the CPU mapping of a host page the device can
// see: page compaction or reclaim migrating one such page costs tens to hundreds of milliseconds of lost GPU time.
// On a GPU whose device memory is host memory, loading a model that nearly fills it always runs compaction, so any
// movable GPU-visible page is hit right behind the load, i.e. in the first request. By default the HIP runtime owns
// three kinds of such pages; the engine's own large buffers are not among them (strix/pinned_host.hpp: anonymous,
// long-term pinned before they are registered).
//
//   1. Its host allocations (queues, kernel arguments, code objects, each hipHostMalloc): shared anonymous memory
//      behind a user-pointer buffer.               HSA_USERPTR_FOR_PAGED_MEM=0 takes them from the driver instead.
//   2. The context-save area of every hardware queue (about 18 MiB each, one per stream the process uses): private
//      anonymous memory registered as an always-mapped shared-virtual-memory range.
//                                                  HSA_USE_SVM=0 takes it from the driver too, and makes host
//                                                  registration a user-pointer buffer over the (pinned) engine pages.
//   3. Every pageable buffer handed to a device copy of 2 MiB or more: registered in place and, with driver pages,
//      left registered for as long as it is mapped (heap memory stays GPU-visible after the copy).
//                                                  GPU_PINNED_XFER_SIZE / GPU_PINNED_MIN_XFER_SIZE above any copy
//                                                  size send those copies through the runtime's own staging buffer.
//
// Driver-owned pages are the kind that already holds the model weights; the kernel never migrates them. With the three
// values no GPU-visible host page is movable, so compaction has nothing to evict, whatever the system's memory
// settings, the allocation order or other processes do. Kernel arithmetic and every byte read or written by the
// device are unchanged (bitwise): only which physical pages back host memory, and one extra host copy for pageable
// transfers.
//
// One consequence is handled here and in strix/pinned_host.hpp: driver pages cannot be the target of direct file I/O
// (O_DIRECT reads and fixed I/O buffers take page references, which the kernel refuses on device mappings). A buffer
// that is both a device-transfer source and a direct-I/O target therefore must not come from the runtime allocator in
// this mode: host_pin::allocate_io() follows host_io_route() below.
//
// The runtime reads the values while it initialises, which with eager code loading happens before main(): the start
// gate inserts them before the eager-code restart so one restart carries everything, and restarts the image itself
// when that restart does not happen.
//
// Policy (STRIX_HOST_DRIVER_PAGES): "1" = on, "0" = the runtime defaults, unset = ON when the default applies: every
// GPU of the driver's topology keeps its device memory in host memory (kfd_gpus_share_host_memory(), read from sysfs
// before the runtime exists) and the binary uses the shared code-load policy. Binaries with their own policy register
// model file mappings with the runtime (src/hip_model_residency.cpp), which has not been run without shared virtual
// memory: they keep the runtime defaults until their owner has a receipt, and still take the switch.
// An explicitly supplied HSA_USERPTR_FOR_PAGED_MEM always wins, even empty: nothing is inserted then. Any other value
// of the set that is already present is kept as supplied.

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

#include "strix/lever.hpp"

namespace strix::device {

inline constexpr std::string_view kHostDriverPagesName = "host.driver_pages";
inline constexpr lever::Decl kHostDriverPages{kHostDriverPagesName, "STRIX_HOST_DRIVER_PAGES", lever::Exactness::bitwise,
                                              true, {}, {},
                                              "two-node 8K prefill tickets, 1 and 3 prefills: evicted_ms 0 in the load and in every prefill window"};
inline constexpr const char* kRuntimePagedMemory = "HSA_USERPTR_FOR_PAGED_MEM";
inline constexpr const char* kRuntimeSharedVirtualMemory = "HSA_USE_SVM";
inline constexpr const char* kRuntimePinnedCopySize = "GPU_PINNED_XFER_SIZE";
inline constexpr const char* kRuntimePinnedCopyMinimum = "GPU_PINNED_MIN_XFER_SIZE";
inline constexpr const char* kNoPinnedCopies = "65536";  // MiB: above any host copy, so none is registered in place

struct RuntimeValue {
    const char* name;
    const char* value;
};
// The set, in insertion order. The first is the mode switch the engine routes direct I/O by.
inline constexpr std::array<RuntimeValue, 4> kRuntimeImmovableHostMemory{{
    {kRuntimePagedMemory, "0"},
    {kRuntimeSharedVirtualMemory, "0"},
    {kRuntimePinnedCopySize, kNoPinnedCopies},
    {kRuntimePinnedCopyMinimum, kNoPinnedCopies},
}};

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

// One node of the driver's topology, from the text of its "properties" file ("name value" lines).
enum class KfdNodeMemory {
    not_a_gpu,    // no compute units: a CPU-only node
    host_shared,  // a GPU without local memory: its device memory is host memory
    dedicated,    // a GPU with its own memory, or one that does not say
};
[[nodiscard]] inline KfdNodeMemory kfd_node_memory(std::string_view properties) {
    const auto number = [&](std::string_view key, unsigned long long& out) {
        for (std::size_t at = 0; at < properties.size();) {
            const std::size_t end = std::min(properties.find('\n', at), properties.size());
            const std::string_view line = properties.substr(at, end - at);
            at = end + 1;
            if (line.size() <= key.size() || line.substr(0, key.size()) != key || line[key.size()] != ' ') continue;
            const std::string_view text = line.substr(key.size() + 1);
            const auto parsed = std::from_chars(text.data(), text.data() + text.size(), out);
            return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
        }
        return false;
    };
    unsigned long long units = 0, local = 0;
    if (!number("simd_count", units) || units == 0) return KfdNodeMemory::not_a_gpu;
    if (!number("local_mem_size", local)) return KfdNodeMemory::dedicated;
    return local == 0 ? KfdNodeMemory::host_shared : KfdNodeMemory::dedicated;
}

// True when the topology has at least one GPU and every GPU keeps its device memory in host memory. Needs no runtime.
[[nodiscard]] inline bool kfd_gpus_share_host_memory(const std::filesystem::path& nodes = "/sys/class/kfd/kfd/topology/nodes") {
    std::error_code error;
    bool any = false;
    for (std::filesystem::directory_iterator it(nodes, error), end; !error && it != end; it.increment(error)) {
        std::ifstream input(it->path() / "properties", std::ios::binary);
        if (!input.good()) continue;
        const std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        switch (kfd_node_memory(text)) {
        case KfdNodeMemory::not_a_gpu: break;
        case KfdNodeMemory::host_shared: any = true; break;
        case KfdNodeMemory::dedicated: return false;
        }
    }
    return any;
}

// Inserts the values of the set that are absent. Returns which (bit i = kRuntimeImmovableHostMemory[i]); 0 = nothing
// changed. The caller makes sure a restarted image sees them.
template<class Get, class Put>
inline unsigned apply_runtime_host_driver_pages(Get get, Put put, bool default_applies) {
    if (get(kRuntimePagedMemory) != nullptr) return 0;
    const lever::Choice choice = lever::select(kHostDriverPages, lever::Inputs{}, get("STRIX_HOST_DRIVER_PAGES"));
    if (!(choice.origin == lever::Origin::explicit_switch ? choice.on : choice.on && default_applies)) return 0;
    unsigned inserted = 0;
    for (std::size_t i = 0; i < kRuntimeImmovableHostMemory.size(); ++i) {
        const RuntimeValue& item = kRuntimeImmovableHostMemory[i];
        if (get(item.name) != nullptr) continue;
        if (!put(item.name, item.value)) throw std::runtime_error("cannot configure runtime host memory");
        inserted |= 1U << i;
    }
    return inserted;
}

inline int& runtime_host_memory_restart_error() {
    static int error = 0;
    return error;
}

// After the eager-code restart did not replace the image: restart for the inserted values, and own them until exec
// succeeds. A failed exec must not leave values this already-initialised runtime did not read: the engine would
// route direct-I/O buffers for a mode the runtime is not in.
template<class Erase, class Restart>
inline int finish_runtime_host_driver_pages(unsigned inserted, Erase erase, Restart restart) {
    if (inserted == 0) return 0;
    const int error = restart();
    runtime_host_memory_restart_error() = error;
    for (std::size_t i = 0; i < kRuntimeImmovableHostMemory.size(); ++i)
        if (((inserted >> i) & 1U) != 0 && !erase(kRuntimeImmovableHostMemory[i].name))
            throw std::runtime_error("cannot roll back runtime host memory after failed restart");
    return error;
}

// Receipt line of the effective mode. Empty when the runtime defaults are in effect and nothing asked otherwise (the
// output of a binary the policy does not apply to stays as it was).
template<class Get>
[[nodiscard]] inline std::string runtime_host_memory_receipt(Get get, int restart_error, bool default_applies) {
    if (restart_error != 0)
        return std::string("runtime host memory: user pages (restart failed: ") + std::strerror(restart_error) + ")";
    const auto is = [&](const char* name, std::string_view want) {
        const char* value = get(name);
        return value != nullptr && std::string_view(value) == want;
    };
    const char* policy = get("STRIX_HOST_DRIVER_PAGES");
    const bool automatic = policy == nullptr || *policy == '\0';
    if (!is(kRuntimePagedMemory, "0")) {
        if (default_applies && get(kRuntimePagedMemory) == nullptr && is("STRIX_HOST_DRIVER_PAGES", "0"))
            return "runtime host memory: user pages (STRIX_HOST_DRIVER_PAGES=0)";
        return {};
    }
    const bool queues = is(kRuntimeSharedVirtualMemory, "0");
    const bool staged = is(kRuntimePinnedCopySize, kNoPinnedCopies) && is(kRuntimePinnedCopyMinimum, kNoPinnedCopies);
    const char* origin = is("STRIX_HOST_DRIVER_PAGES", "1") ? "STRIX_HOST_DRIVER_PAGES=1"
                         : automatic && default_applies && queues && staged ? "default: device memory is host memory"
                                                                            : "HSA_USERPTR_FOR_PAGED_MEM=0";
    return std::string("runtime host memory: driver pages (") + origin + "), direct-I/O staging on engine pages; queue context " +
           (queues ? "on driver pages" : "on movable pages") + "; pageable copies " + (staged ? "staged" : "registered in place");
}

}  // namespace strix::device
