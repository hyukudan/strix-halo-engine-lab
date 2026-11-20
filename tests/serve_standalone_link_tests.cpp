// Link check of the standalone serving graph (cmake/StrixServeStandalone.cmake): the API library of the
// STRIX_CUDA_QWEN_ONLY configuration, linked here without the device gate or the core library. The stored member
// pointer pulls src/serve/api.cpp into the link, so every symbol that file calls (the shared power sampler among
// them) must come from a library that graph names. The prefix directory members the RTX serve's in-flight prefix
// join binds (apps/cuda_qwen_serve_main.cu) are pinned the same way. This file calls none of them itself.
#include "strix/serve/api.hpp"
#include "strix/serve/prefix_directory.hpp"

#include <cstdint>
#include <iostream>

namespace {
using strix::serve::PrefixDirectory;
using strix::serve::PrefixStatus;
using Handle = void (strix::serve::ApiService::*)(const strix::serve::HttpRequest&, strix::serve::HttpResponseWriter&);
using Acquire = bool (PrefixDirectory::*)(std::uint64_t) noexcept;
using Release = void (PrefixDirectory::*)(std::uint64_t) noexcept;
using Status = PrefixStatus (PrefixDirectory::*)(std::uint64_t) const noexcept;
// volatile: the references stay in the object file
volatile Handle g_handle = &strix::serve::ApiService::handle;
volatile Acquire g_acquire = &PrefixDirectory::acquire;
volatile Release g_release = &PrefixDirectory::release;
volatile Status g_status = &PrefixDirectory::status;
}  // namespace

int main() {
    const Handle handle = g_handle;
    const Acquire acquire = g_acquire;
    const Release release = g_release;
    const Status status = g_status;
    if (!handle || !acquire || !release || !status) return 1;
    std::cout << "standalone serve link test passed\n";
    return 0;
}
