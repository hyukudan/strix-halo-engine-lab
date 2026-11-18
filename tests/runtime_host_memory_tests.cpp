// Host test of the runtime host-memory mode (include/strix/device/runtime_host_memory.hpp): the environment step of the
// start gate, its rollback, the receipt, and the rule that a direct-I/O target never comes from the runtime allocator
// while the runtime backs its host memory with driver pages.

#include "strix/device/runtime_host_memory.hpp"

#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

namespace dev = strix::device;

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}

struct Env {
    std::map<std::string, std::string> values;
    int puts = 0;
    bool put_ok = true;
    const char* get(const char* name) const {
        const auto it = values.find(name);
        return it == values.end() ? nullptr : it->second.c_str();
    }
    bool put(const char* name, const char* value) {
        ++puts;
        if (put_ok) values[name] = value;
        return put_ok;
    }
    unsigned apply(bool default_applies = false, dev::HostRegistration registration = dev::HostRegistration::engine_pages) {
        return dev::apply_runtime_host_driver_pages([&](const char* name) { return get(name); },
                                                    [&](const char* name, const char* value) { return put(name, value); },
                                                    default_applies, registration);
    }
    bool driver_pages() const { return dev::runtime_host_memory_is_driver_pages(get(dev::kRuntimePagedMemory)); }
    // every value of the set is present with the policy's value
    bool whole_set() const {
        for (const auto& item : dev::kRuntimeImmovableHostMemory)
            if (get(item.name) == nullptr || std::string_view(get(item.name)) != item.value) return false;
        return true;
    }
    std::string receipt(bool default_applies, int restart_error = 0) const {
        return dev::runtime_host_memory_receipt([&](const char* name) { return get(name); }, restart_error, default_applies);
    }
};
constexpr unsigned kAll = (1U << dev::kRuntimeImmovableHostMemory.size()) - 1U;

static_assert(!dev::runtime_host_memory_is_driver_pages(nullptr));
static_assert(!dev::runtime_host_memory_is_driver_pages(""));
static_assert(dev::runtime_host_memory_is_driver_pages("0"));
static_assert(!dev::runtime_host_memory_is_driver_pages("1"));
static_assert(!dev::runtime_host_memory_is_driver_pages("00"));
static_assert(!dev::runtime_host_memory_is_driver_pages(" 0"));
static_assert(dev::host_io_route(true) == dev::HostIoRoute::own_pages);
static_assert(dev::host_io_route(false) == dev::HostIoRoute::shared_allocator);
static_assert(dev::kHostDriverPages.default_on, "default on where the default applies (pair receipt: no eviction in any window)");
static_assert(dev::kHostDriverPages.exactness == strix::lever::Exactness::bitwise);
static_assert(std::string_view(dev::kRuntimeImmovableHostMemory[0].name) == dev::kRuntimePagedMemory,
              "the first value of the set is the mode switch the engine routes direct I/O by");

void environment_step() {
    {
        Env env;  // nothing set, a device the default does not apply to: the runtime default stays, nothing is written
        require(env.apply(false) == 0 && env.puts == 0 && !env.driver_pages(), "no default where it does not apply");
    }
    {
        Env env;  // nothing set, device memory is host memory: the whole set, once
        require(env.apply(true) == kAll && env.puts == 4 && env.driver_pages() && env.whole_set(), "default inserts the whole set");
        require(env.apply(true) == 0 && env.puts == 4, "the restarted image finds the values and does not insert again");
    }
    {
        Env env;  // the switch decides on any device
        env.values["STRIX_HOST_DRIVER_PAGES"] = "1";
        require(env.apply(false) == kAll && env.puts == 4 && env.whole_set(), "lever on inserts the whole set");
        require(env.apply(false) == 0 && env.puts == 4, "the restarted image finds the values and does not insert again");
    }
    for (const char* explicit_value : {"1", "", "0"}) {
        for (const bool default_applies : {false, true}) {
            Env env;
            env.values["STRIX_HOST_DRIVER_PAGES"] = "1";
            env.values[dev::kRuntimePagedMemory] = explicit_value;
            require(env.apply(default_applies) == 0 && env.puts == 0, "an explicit runtime mode always wins: nothing is inserted");
            require(env.driver_pages() == (std::string_view(explicit_value) == "0"), "mode follows the explicit value");
        }
    }
    {
        Env env;  // a supplied secondary value is kept, the rest of the set is inserted
        env.values[dev::kRuntimeSharedVirtualMemory] = "1";
        const unsigned inserted = env.apply(true);
        require(inserted == (kAll & ~2U) && env.puts == 3 && env.driver_pages(), "absent values only");
        require(std::string_view(env.get(dev::kRuntimeSharedVirtualMemory)) == "1" && !env.whole_set(), "the supplied value stays");
    }
    for (const bool default_applies : {false, true}) {
        Env env;
        env.values["STRIX_HOST_DRIVER_PAGES"] = "0";
        require(env.apply(default_applies) == 0 && env.puts == 0, "off switch");
    }
    {
        Env env;
        env.values["STRIX_HOST_DRIVER_PAGES"] = "2";
        bool refused = false;
        try { (void)env.apply(); } catch (const std::invalid_argument&) { refused = true; }
        require(refused && env.puts == 0, "invalid switch value is refused before anything is written");
    }
    {
        Env env;
        env.values["STRIX_HOST_DRIVER_PAGES"] = "1";
        env.put_ok = false;
        bool failed = false;
        try { (void)env.apply(); } catch (const std::runtime_error&) { failed = true; }
        require(failed && !env.driver_pages(), "a failed insert is an error, not a silent user-pages run");
    }
}

// The declared host registration (F3/F8): a binary that registers the model file mapping never takes the default, and an
// explicit "1" is refused with a reason before anything is inserted; "0" and unset stay accepted.
void registration_step() {
    using dev::HostRegistration;
    static_assert(dev::runtime_host_pages_default(HostRegistration::engine_pages, true));
    static_assert(!dev::runtime_host_pages_default(HostRegistration::engine_pages, false));
    static_assert(!dev::runtime_host_pages_default(HostRegistration::model_file, true));
    static_assert(!dev::runtime_host_pages_default(HostRegistration::model_file, false));
    {
        Env env;  // unset policy on a model-file binary: nothing, even where the device memory is host memory
        require(env.apply(dev::runtime_host_pages_default(HostRegistration::model_file, true), HostRegistration::model_file) == 0 &&
                    env.puts == 0, "model-file binary must decline the default");
    }
    {
        Env env;
        env.values["STRIX_HOST_DRIVER_PAGES"] = "1";
        bool refused = false;
        try {
            (void)env.apply(false, HostRegistration::model_file);
        } catch (const std::invalid_argument& e) {
            refused = std::string_view(e.what()).find("STRIX_HOST_DRIVER_PAGES=1 refused") != std::string_view::npos;
        }
        require(refused && env.puts == 0, "explicit 1 on a model-file binary must be refused with a reason, nothing inserted");
    }
    {
        Env env;
        env.values["STRIX_HOST_DRIVER_PAGES"] = "0";
        require(env.apply(false, HostRegistration::model_file) == 0 && env.puts == 0, "0 stays accepted on a model-file binary");
    }
    {
        Env env;  // an explicit runtime value always wins: no refusal, nothing inserted
        env.values["STRIX_HOST_DRIVER_PAGES"] = "1";
        env.values[dev::kRuntimePagedMemory] = "1";
        require(env.apply(false, HostRegistration::model_file) == 0 && env.puts == 0, "explicit runtime value wins before the refusal");
    }
    {
        Env env;  // engine-pages binaries keep the switch
        env.values["STRIX_HOST_DRIVER_PAGES"] = "1";
        require(env.apply(false, HostRegistration::engine_pages) == kAll && env.whole_set(), "explicit 1 still applies to engine-pages binaries");
    }
}

void restart_step() {
    int restarts = 0, erased = 0;
    const auto erase_ok = [&](const char*) { ++erased; return true; };
    require(dev::finish_runtime_host_driver_pages(0U, erase_ok, [&] { ++restarts; return 0; }) == 0 && restarts == 0 && erased == 0,
            "nothing inserted: no restart, nothing erased");
    // exec returned: the already-initialised runtime did not read the values, so the engine must not route for them
    Env env;
    env.values[dev::kRuntimeSharedVirtualMemory] = "1";  // supplied by the operator: not ours to erase
    const unsigned inserted = env.apply(true);
    require(inserted == (kAll & ~2U) && env.driver_pages(), "inserted");
    const int error = dev::finish_runtime_host_driver_pages(
        inserted, [&](const char* name) { ++erased; return env.values.erase(name) == 1; }, [&] { ++restarts; return EACCES; });
    require(error == EACCES && restarts == 1 && erased == 3, "failed restart reports the exec error and rolls back what it inserted");
    require(!env.driver_pages() && dev::host_io_route(env.driver_pages()) == dev::HostIoRoute::shared_allocator,
            "after rollback the engine routes for the runtime default");
    require(env.values.size() == 1 && std::string_view(env.get(dev::kRuntimeSharedVirtualMemory)) == "1",
            "rollback leaves exactly the operator's value");
    require(dev::runtime_host_memory_restart_error() == EACCES, "restart error kept for the receipt");
    bool failed = false;
    try {
        (void)dev::finish_runtime_host_driver_pages(1U, [](const char*) { return false; }, [] { return EACCES; });
    } catch (const std::runtime_error&) { failed = true; }
    require(failed, "a rollback that cannot be done is an error");
}

void receipt_lines() {
    const std::string tail = ", direct-I/O staging on engine pages; queue context on driver pages; pageable copies staged";
    {
        Env env;
        require(env.receipt(false).empty() && env.receipt(true).empty(), "runtime defaults, nothing asked: no line");
        env.values["STRIX_HOST_DRIVER_PAGES"] = "0";
        require(env.receipt(false).empty(), "off switch where the default does not apply: no line");
        require(env.receipt(true) == "runtime host memory: user pages (STRIX_HOST_DRIVER_PAGES=0)", "off switch receipt");
    }
    {
        Env env;
        require(env.apply(true) == kAll, "default");
        require(env.receipt(true) == "runtime host memory: driver pages (default: device memory is host memory)" + tail, "default receipt");
    }
    {
        Env env;
        env.values["STRIX_HOST_DRIVER_PAGES"] = "1";
        require(env.apply(false) == kAll, "lever");
        require(env.receipt(false) == "runtime host memory: driver pages (STRIX_HOST_DRIVER_PAGES=1)" + tail, "lever receipt");
    }
    {
        Env env;  // the operator's own mode: the line says what is and is not covered
        env.values[dev::kRuntimePagedMemory] = "0";
        require(env.receipt(true) == "runtime host memory: driver pages (HSA_USERPTR_FOR_PAGED_MEM=0), direct-I/O staging on engine pages; "
                                     "queue context on movable pages; pageable copies registered in place",
                "explicit runtime value receipt");
        env.values[dev::kRuntimePagedMemory] = "1";
        env.values["STRIX_HOST_DRIVER_PAGES"] = "1";
        require(env.receipt(true).empty(), "explicit user pages prints nothing");
    }
    {
        Env env;
        env.values["STRIX_HOST_DRIVER_PAGES"] = "1";
        require(env.receipt(true, EACCES).starts_with("runtime host memory: user pages (restart failed: "), "restart failure receipt");
    }
}

// The device class the default applies to, from the driver's topology text (no runtime, no device).
void topology_probe() {
    using M = dev::KfdNodeMemory;
    require(dev::kfd_node_memory("cpu_cores_count 16\nsimd_count 0\nlocal_mem_size 0\n") == M::not_a_gpu, "CPU node");
    require(dev::kfd_node_memory("simd_count 80\nmem_banks_count 1\nlocal_mem_size 0\nfw_version 34\n") == M::host_shared, "host-shared GPU");
    require(dev::kfd_node_memory("simd_count 384\nlocal_mem_size 25753026560\n") == M::dedicated, "GPU with its own memory");
    require(dev::kfd_node_memory("simd_count 80\n") == M::dedicated, "a GPU that does not say is not claimed");
    require(dev::kfd_node_memory("max_simd_count 80\nlocal_mem_size 0\n") == M::not_a_gpu, "key must start the line");
    require(dev::kfd_node_memory("simd_count 80\nlocal_mem_size 0x0\n") == M::dedicated, "unparsed value is not claimed");
    require(dev::kfd_node_memory("") == M::not_a_gpu, "empty text");

    const std::filesystem::path root = std::filesystem::temp_directory_path() / "strix-runtime-host-memory-tests";
    std::filesystem::remove_all(root);
    const auto node = [&](const char* name, const char* text) {
        std::filesystem::create_directories(root / name);
        std::ofstream(root / name / "properties", std::ios::binary) << text;
    };
    require(!dev::kfd_gpus_share_host_memory(root / "missing"), "no topology: the default does not apply");
    node("0", "cpu_cores_count 16\nsimd_count 0\nlocal_mem_size 0\n");
    require(!dev::kfd_gpus_share_host_memory(root), "no GPU: the default does not apply");
    node("1", "simd_count 80\nlocal_mem_size 0\n");
    require(dev::kfd_gpus_share_host_memory(root), "one host-shared GPU");
    node("2", "simd_count 384\nlocal_mem_size 25753026560\n");
    require(!dev::kfd_gpus_share_host_memory(root), "a GPU with its own memory beside it: not claimed");
    std::filesystem::remove_all(root);
}

std::string read(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "cannot read " + path.generic_string());
    return std::string((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
}

// The device side cannot run here: pin the source of the rule instead. Every buffer an O_DIRECT descriptor reads into
// on a HIP path takes allocate_io, and allocate_io's own-pages branch never reaches the runtime allocator.
void direct_io_sources(const std::filesystem::path& root) {
    const std::string pin = read(root / "include/strix/pinned_host.hpp");
    const auto begin = pin.find("inline hipError_t allocate_io(");
    const auto end = pin.find("// hipHostFree for memory from allocate() / allocate_io()");
    require(begin != std::string::npos && end != std::string::npos && begin < end, "allocate_io not found");
    const std::string body = pin.substr(begin, end - begin);
    for (const char* contract : {
             "if (device::host_io_route(runtime_driver_pages()) == device::HostIoRoute::shared_allocator) return allocate(host, bytes, flags);",
             "if (p == MAP_FAILED) return hipErrorOutOfMemory;", "hipHostRegister(p, size, reg)", "if (rec.ring < 0) ++r.unpinned;"})
        require(body.find(contract) != std::string::npos, std::string("allocate_io contract changed: ") + contract);
    require(body.find("hipHostMalloc(") == std::string::npos, "allocate_io own-pages branch must not use the runtime allocator");
    require(pin.find("device::runtime_host_memory_is_driver_pages(strix::envcfg::text(device::kRuntimePagedMemory))") != std::string::npos,
            "mode must be read from the runtime's own variable");
    const struct { const char* file; const char* call; } sites[] = {
        {"src/core/hip_weight_uploader.hpp", "strix::host_pin::allocate_io(&p, static_cast<std::size_t>(bytes), hipHostMallocDefault)"},
        {"src/glm/load/glm_peerload.hip", "strix::host_pin::allocate_io(reinterpret_cast<void**>(&s), kChunk + 4096, hipHostMallocDefault)"},
        {"src/hip_ds41_prepared_load.hpp", "strix::host_pin::allocate_io(reinterpret_cast<void**>(&pinned_), staging_slots * chunk_bytes + 4096"},
        {"src/hip_ds41_expert_backend.hip", "buffer_ = Pinned<std::uint8_t>(chunk + 2U * align, true);"},
        {"src/hip_ds41_expert_backend.hip", "direct_io ? strix::host_pin::allocate_io(reinterpret_cast<void**>(&p), bytes, hipHostMallocDefault)"},
        {"src/device/hip_gate.cpp", "host_pages_inserted = apply_runtime_host_driver_pages("},
        {"src/device/hip_gate.cpp", "finish_runtime_host_driver_pages(host_pages_inserted,"},
    };
    for (const auto& site : sites)
        require(read(root / site.file).find(site.call) != std::string::npos, std::string("direct-I/O site changed: ") + site.file);
    const std::string gate = read(root / "src/device/hip_gate.cpp");
    require(gate.find("apply_runtime_host_driver_pages(") < gate.find("eager_code_load_or_restart(argv, code_load);") &&
                gate.find("eager_code_load_or_restart(argv, code_load);") < gate.find("finish_runtime_host_driver_pages("),
            "the runtime value must be inserted before the eager-code restart and settled after it");
    // One resolver for every entry point (command line and serving alike): the default is decided in the start gate
    // only, from the topology probe and the code-load policy, and the effective mode is recorded there.
    require(gate.find("const bool host_pages_default = runtime_host_pages_default(registration, kfd_gpus_share_host_memory());") != std::string::npos,
            "the default must come from the topology probe and the binary's declared host registration");
    require(gate.find("host_pages_default, registration);") != std::string::npos, "the declaration must reach the resolver (refusal of 1)");
    // Every main that registers the model file mapping declares it (F3/F8): the default is declined and "1" refused there.
    for (const char* file : {"apps/hip_decode_token_main.cpp", "apps/hip_model_residency_main.cpp"})
        require(read(root / file).find("strix::device::HostRegistration::model_file") != std::string::npos,
                std::string("model-file registration not declared at the start gate: ") + file);
    require(gate.find("strix::envcfg::record(\"STRIX_HOST_DRIVER_PAGES\"") != std::string::npos, "effective mode must be recorded");
    for (const char* file : {"include/strix/pinned_host.hpp", "src/core/hip_weight_uploader.hpp", "apps/minimax_rank_main.cpp"})
        require(read(root / file).find("apply_runtime_host_driver_pages(") == std::string::npos,
                std::string("a second resolver of the runtime host memory mode: ") + file);
}

}  // namespace

int main() {
    try {
        environment_step();
        registration_step();
        restart_step();
        receipt_lines();
        topology_probe();
        direct_io_sources(std::filesystem::path(STRIX_RUNTIME_HOST_MEMORY_SOURCE_DIR));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "runtime host memory tests: FAILED: %s\n", e.what());
        return 1;
    }
    std::printf("runtime host memory tests: ok\n");
    return 0;
}
