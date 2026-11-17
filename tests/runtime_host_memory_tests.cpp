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
    bool apply() {
        return dev::apply_runtime_host_driver_pages([&](const char* name) { return get(name); },
                                                    [&](const char* name, const char* value) { return put(name, value); });
    }
    bool driver_pages() const { return dev::runtime_host_memory_is_driver_pages(get(dev::kRuntimePagedMemory)); }
};

static_assert(!dev::runtime_host_memory_is_driver_pages(nullptr));
static_assert(!dev::runtime_host_memory_is_driver_pages(""));
static_assert(dev::runtime_host_memory_is_driver_pages("0"));
static_assert(!dev::runtime_host_memory_is_driver_pages("1"));
static_assert(!dev::runtime_host_memory_is_driver_pages("00"));
static_assert(!dev::runtime_host_memory_is_driver_pages(" 0"));
static_assert(dev::host_io_route(true) == dev::HostIoRoute::own_pages);
static_assert(dev::host_io_route(false) == dev::HostIoRoute::shared_allocator);
static_assert(!dev::kHostDriverPages.default_on, "opt-in until the pair receipt exists");
static_assert(dev::kHostDriverPages.exactness == strix::lever::Exactness::bitwise);

void environment_step() {
    {
        Env env;  // nothing set: the runtime default stays, nothing is written
        require(!env.apply() && env.puts == 0 && !env.driver_pages(), "default must not touch the runtime mode");
    }
    {
        Env env;
        env.values["STRIX_HOST_DRIVER_PAGES"] = "1";
        require(env.apply() && env.puts == 1 && env.driver_pages(), "lever on inserts the runtime value");
        require(!env.apply() && env.puts == 1, "the restarted image finds the value and does not insert again");
    }
    for (const char* explicit_value : {"1", "", "0"}) {
        Env env;
        env.values["STRIX_HOST_DRIVER_PAGES"] = "1";
        env.values[dev::kRuntimePagedMemory] = explicit_value;
        require(!env.apply() && env.puts == 0, "an explicit runtime value always wins");
        require(env.driver_pages() == (std::string_view(explicit_value) == "0"), "mode follows the explicit value");
    }
    {
        Env env;
        env.values["STRIX_HOST_DRIVER_PAGES"] = "0";
        require(!env.apply() && env.puts == 0, "lever off");
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

void restart_step() {
    int restarts = 0, erased = 0;
    const auto erase_ok = [&](const char* name) { erased += std::string_view(name) == dev::kRuntimePagedMemory; return true; };
    require(dev::finish_runtime_host_driver_pages(false, erase_ok, [&] { ++restarts; return 0; }) == 0 && restarts == 0 && erased == 0,
            "nothing inserted: no restart, nothing erased");
    // exec returned: the already-initialised runtime did not read the value, so the engine must not route for it
    Env env;
    env.values["STRIX_HOST_DRIVER_PAGES"] = "1";
    require(env.apply() && env.driver_pages(), "inserted");
    const int error = dev::finish_runtime_host_driver_pages(
        true, [&](const char* name) { ++erased; return env.values.erase(name) == 1; }, [&] { ++restarts; return EACCES; });
    require(error == EACCES && restarts == 1 && erased == 1, "failed restart reports the exec error and rolls back");
    require(!env.driver_pages() && dev::host_io_route(env.driver_pages()) == dev::HostIoRoute::shared_allocator,
            "after rollback the engine routes for the runtime default");
    require(dev::runtime_host_memory_restart_error() == EACCES, "restart error kept for the receipt");
    bool failed = false;
    try {
        (void)dev::finish_runtime_host_driver_pages(true, [](const char*) { return false; }, [] { return EACCES; });
    } catch (const std::runtime_error&) { failed = true; }
    require(failed, "a rollback that cannot be done is an error");
}

void receipt_lines() {
    require(dev::runtime_host_memory_receipt(nullptr, nullptr, 0).empty(), "default mode prints nothing");
    require(dev::runtime_host_memory_receipt("1", "1", 0).empty(), "explicit user pages prints nothing");
    require(dev::runtime_host_memory_receipt("0", "1", 0) ==
                "runtime host memory: driver pages (STRIX_HOST_DRIVER_PAGES=1), direct-I/O staging on engine pages",
            "lever receipt");
    require(dev::runtime_host_memory_receipt("0", nullptr, 0) ==
                "runtime host memory: driver pages (HSA_USERPTR_FOR_PAGED_MEM=0), direct-I/O staging on engine pages",
            "explicit runtime value receipt");
    require(dev::runtime_host_memory_receipt(nullptr, "1", EACCES).starts_with("runtime host memory: user pages (restart failed: "),
            "restart failure receipt");
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
}

}  // namespace

int main() {
    try {
        environment_step();
        restart_step();
        receipt_lines();
        direct_io_sources(std::filesystem::path(STRIX_RUNTIME_HOST_MEMORY_SOURCE_DIR));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "runtime host memory tests: FAILED: %s\n", e.what());
        return 1;
    }
    std::printf("runtime host memory tests: ok\n");
    return 0;
}
