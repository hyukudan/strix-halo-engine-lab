#include "strix/device/hip_gate.hpp"

#include "strix/device/hip_enumerator.hpp"
#include "strix/device/process_topology.hpp"
#include "strix/device/runtime_host_memory.hpp"
#include "strix/env_flags.hpp"
#include "strix/power/sampler.hpp"
#include "strix/route_receipt.hpp"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <optional>

namespace strix::device {
namespace {
struct Process {
    std::mutex mutex;
    std::optional<Bound> bound;
};
Process& process() {
    static Process state;
    return state;
}
[[noreturn]] void fail(const std::string& why) { throw Refusal("device gate: refusing to start: " + why); }
} // namespace

const Bound& bind_hip(Request request, const std::string& devices_path, std::FILE* receipt) {
    auto& state = process();
    const std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.bound) {
        request.vendor = Vendor::amd;
        HipEnumerator cards;
        auto chosen = start_gate(cards, std::move(request), devices_path);
        if (chosen.visible.ordinal != 0)
            fail("declared device " + chosen.device.id + " (pci " + chosen.visible.pci + ") is HIP index " + std::to_string(chosen.visible.ordinal) +
                 " and threads start on index 0: set HIP_VISIBLE_DEVICES=" + std::to_string(chosen.visible.ordinal) + " before starting");
        if (const auto status = hipSetDevice(0); status != hipSuccess) fail(std::string("hipSetDevice failed: ") + hipGetErrorString(status));
        publish_bound_plans(chosen, r1_rows_tiles::Runtime::hip, 0);  // the device is ours: its launches read its plans
        if (receipt) {
            std::fprintf(receipt, "%s\n", receipt_line(chosen).c_str());
            std::fflush(receipt);
        }
        state.bound = std::move(chosen);
        power::start_process_sampler();  // shared power receipt: every binary that binds a device samples from here on
        ::strix::route::arm_exit_print();         // shared route receipt: printed once when the process exits
    } else if (const auto status = hipSetDevice(state.bound->visible.ordinal); status != hipSuccess) {
        fail(std::string("hipSetDevice failed: ") + hipGetErrorString(status));
    }
    return *state.bound;
}

void bind_hip_main(int& argc, char** argv, std::FILE* receipt, CodeLoad code_load, HostRegistration registration) {
    // Runtime host memory mode first (runtime_host_memory.hpp): the eager-code restart then carries every value in one
    // exec. Still here afterwards with values this image inserted: restart for them, or roll them back.
    // The default applies where the device memory is host memory, to binaries that register only their own pinned pages.
    // Binaries that register the model FILE mapping (HostRegistration::model_file: apps/hip_decode_token_main.cpp,
    // apps/hip_model_residency_main.cpp; src/hip_model_residency.cpp create_impl) decline it and refuse "1": with
    // HSA_USE_SVM=0 that registration is refused ("hipHostRegister(model range): invalid argument", measured 8K one node,
    // 2 of 2 runs). Their GPU-visible file pages are kept from evicting by registered_backing_lifetime.hpp's lock order.
    // Independent of the code-load policy (CodeLoad only decides who applies eager code loading).
    const bool host_pages_default = runtime_host_pages_default(registration, kfd_gpus_share_host_memory());
    unsigned host_pages_inserted = 0;
    try {
        host_pages_inserted = apply_runtime_host_driver_pages(
            [](const char* name) { return strix::envcfg::text(name); },
            [](const char* name, const char* value) { return ::setenv(name, value, 0) == 0; }, host_pages_default, registration);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        std::fflush(stderr);
        std::exit(2);
    }
    eager_code_load_or_restart(argv, code_load);  // before anything touches the device: the restart repeats the process
    if (const int error = finish_runtime_host_driver_pages(host_pages_inserted,
            [](const char* name) { return ::unsetenv(name) == 0; }, [&] { return restart_own_image(argv); }))
        std::fprintf(stderr, "runtime host memory: restart failed (%s); continuing with user pages\n", std::strerror(error));
    {  // effective config: the lever with the mode this image really runs in (the same line for every entry point)
        const char* policy = strix::envcfg::text("STRIX_HOST_DRIVER_PAGES");  // recorded again below with the effective mode
        strix::envcfg::record("STRIX_HOST_DRIVER_PAGES", policy != nullptr && *policy != '\0' ? policy : nullptr,
                              runtime_host_memory_is_driver_pages(std::getenv(kRuntimePagedMemory)) ? "1" : "0");
    }
    if (receipt && code_load == CodeLoad::shared) std::fprintf(receipt, "%s\n", code_load_receipt(code_load).c_str());
    if (receipt) {
        const std::string line = runtime_host_memory_receipt([](const char* name) { return strix::envcfg::text(name); },
                                                             runtime_host_memory_restart_error(), host_pages_default);
        if (!line.empty()) std::fprintf(receipt, "%s\n", line.c_str());
    }
    try {
        const auto cli = take_command_line(argc, argv);
        Request request;
        request.select = cli.select;
        request.profile = cli.profile;
        (void)bind_hip(std::move(request), cli.devices_path, receipt);
        // Node topology (strix-topology-v1): keep --topology/--node for the family; `topology:` beside `device:`.
        keep_topology_command_line(take_topology_command_line(argc, argv));
        if (const auto line = gate_topology_line(argc, argv); receipt && !line.empty()) std::fprintf(receipt, "%s\n", line.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        std::fflush(stderr);
        std::exit(2);
    }
}

HostWait set_host_wait(HostWait engine_choice, std::FILE* receipt) {
    const char* text = std::getenv("STRIX_HOST_WAIT");
    const auto policy = resolve_host_wait(engine_choice, text);
    if (policy != HostWait::runtime) {
        const unsigned flag = policy == HostWait::spin ? hipDeviceScheduleSpin : policy == HostWait::yield ? hipDeviceScheduleYield : hipDeviceScheduleBlockingSync;
        if (const auto status = hipSetDeviceFlags(flag); status != hipSuccess) fail(std::string("host wait: hipSetDeviceFlags failed: ") + hipGetErrorString(status));
    }
    if (receipt) {
        std::fprintf(receipt, "host wait: %s (%s)\n", host_wait_name(policy), text && *text ? "STRIX_HOST_WAIT" : "engine");
        std::fflush(receipt);
    }
    return policy;
}

const Bound* hip_bound() noexcept {
    auto& state = process();
    const std::lock_guard<std::mutex> lock(state.mutex);
    return state.bound ? &*state.bound : nullptr;
}

} // namespace strix::device
