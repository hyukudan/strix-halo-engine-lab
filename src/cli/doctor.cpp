#include "detail.hpp"
#include "cpu/core/isa.hpp"
#include "strix/deploy/device.hpp"
#include "strix/device/intel_enumerator.hpp"
#include "strix/linux_uma_observation.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <iostream>
#include <stdexcept>
#include <system_error>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/utsname.h>
#include <unistd.h>
#endif

namespace strix::cli {
namespace {
using serve::Json;
namespace fs = std::filesystem;
constexpr std::array<std::string_view, 3> backends{"hip", "cuda", "intel"};
constexpr std::array<std::string_view, 6> components{"core", "tools", "docs", "hip", "cuda", "intel"};

std::string_view status_name(Status status) {
    switch (status) {
    case Status::ok: return "ok";
    case Status::warn: return "warn";
    case Status::fail: return "fail";
    case Status::skip: return "skip";
    }
    return "skip";
}

class NativeProbes final : public DoctorProbes {
public:
    explicit NativeProbes(fs::path data) : support_(std::move(data) / "support") {}

    Platform platform() const override {
#if defined(_WIN32)
        return Platform::windows_host;
#elif defined(__linux__)
        return Platform::linux_host;
#elif defined(__APPLE__)
        return Platform::macos_host;
#else
        return Platform::other;
#endif
    }

    Check os() const override {
#if defined(_WIN32)
        return {"os", Status::ok, "Windows"};
#elif defined(__linux__)
        utsname info{};
        if (uname(&info) == 0)
            return {"os", Status::ok, std::string(info.sysname) + " " + info.release + " " + info.machine};
        return {"os", Status::warn, "Linux kernel identity unavailable"};
#else
        return {"os", Status::skip, "host OS probe unavailable"};
#endif
    }

    Check cpu() const override {
        const auto flags = cpu::caps();
        const auto model = cpu::cpu_model();
        const auto yes_no = [](bool value) { return value ? "yes" : "no"; };
        return {"cpu", model.empty() ? Status::warn : Status::ok,
                (model.empty() ? "model unavailable" : model) + std::string("; usable ISA: AVX2=") +
                yes_no(flags.avx2) + ", AVX-512F=" + yes_no(flags.avx512f) +
                "; OS state: AVX=" + yes_no(flags.os_avx) + ", AVX-512=" + yes_no(flags.os_avx512) +
                "; hardware AMX-INT8=" + yes_no(flags.amx_int8)};
    }

    std::vector<std::string> architectures(std::string_view backend) const override {
        if (backend == "cpu") {
#if defined(__x86_64__) || defined(_M_X64)
            return {"x86_64"};
#elif defined(__aarch64__) || defined(_M_ARM64)
            return {"aarch64"};
#else
            return {};
#endif
        }
        std::vector<std::string> archs;
        if (backend == "hip") {
            for (const auto& device : deploy::probe_devices())
                if (!device.gfx_target.empty()) archs.push_back(device.gfx_target);
        } else if (backend == "cuda" || backend == "intel") {
            const auto devices = backend == "cuda" ? device::query_nvidia_smi_with_arch() : device::query_xpu_smi();
            if (devices) {
                for (const auto& device : *devices) {
                    if (device.arch.empty()) continue;
                    // Intel observations include a device suffix. The matrix uses the architecture generation.
                    archs.push_back(backend == "intel" ? device.arch.substr(0, device.arch.find('-')) : device.arch);
                }
            }
        }
        return archs;
    }

    Check ram() const override {
#if defined(_WIN32)
        MEMORYSTATUSEX memory{};
        memory.dwLength = static_cast<DWORD>(sizeof(memory));
        if (GlobalMemoryStatusEx(&memory))
            return memory_check(memory.ullTotalPhys, memory.ullAvailPhys);
#elif defined(__linux__)
        const auto observation = observe_linux_uma_memory();
        return memory_check(observation.memory.mem_total_bytes, observation.memory.mem_available_bytes);
#endif
        return {"ram", Status::warn, "total and available bytes unavailable"};
    }

    Check runtime(std::string_view backend) const override {
        const auto name = "runtime-" + std::string(backend);
#if defined(_WIN32)
        const wchar_t* library = backend == "hip" ? L"amdhip64.dll" :
            backend == "cuda" ? L"nvcuda.dll" : L"ze_loader.dll";
        const auto module = LoadLibraryExW(library, nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (module) FreeLibrary(module);
        const bool loaded = module != nullptr;
#elif defined(__linux__)
        const char* library = backend == "hip" ? "libamdhip64.so" :
            backend == "cuda" ? "libcuda.so.1" : "libze_loader.so.1";
        auto* module = dlopen(library, RTLD_NOW | RTLD_LOCAL);
        if (!module && backend != "cuda") {
            library = backend == "hip" ? "libamdhip64.so.6" : "libze_loader.so";
            module = dlopen(library, RTLD_NOW | RTLD_LOCAL);
        }
        if (!module && backend == "hip") {
            library = "libamdhip64.so.7";
            module = dlopen(library, RTLD_NOW | RTLD_LOCAL);
        }
        if (module) dlclose(module);
        const bool loaded = module != nullptr;
#else
        return {name, Status::skip, "dynamic loader probe unavailable"};
#endif
#if defined(_WIN32) || defined(__linux__)
        return {name, loaded ? Status::ok : Status::warn,
                fs::path(library).string() + (loaded ? " loads; device initialization untested" : " could not be loaded")};
#endif
    }

    Check node(std::string_view name) const override {
        const auto label = std::string(name);
#if defined(__linux__)
        if (name == "/dev/kfd") {
            const auto error = open_node(label.c_str());
            return {label, error == 0 ? Status::ok : (error == ENOENT ? Status::skip : Status::fail),
                    error == 0 ? "read/write open succeeded" : std::generic_category().message(error)};
        }
        std::error_code ec;
        unsigned found = 0, accessible = 0;
        for (fs::directory_iterator it("/dev/dri", ec), end; !ec && it != end; it.increment(ec)) {
            if (!it->path().filename().string().starts_with("renderD")) continue;
            ++found;
            if (open_node(it->path().c_str()) == 0) ++accessible;
        }
        if (ec && ec != std::errc::no_such_file_or_directory) return {label, Status::fail, ec.message()};
        if (!found) return {label, Status::skip, "no render nodes"};
        const auto status = accessible == found ? Status::ok : (accessible ? Status::warn : Status::fail);
        return {label, status, std::to_string(accessible) + "/" + std::to_string(found) + " render nodes open read/write"};
#else
        return {label, Status::skip, "Linux device node"};
#endif
    }

    Check component(std::string_view component_name) const override {
        const auto name = std::string(component_name);
        for (const auto& filename : {name + "-archs.txt", "build-" + name + ".txt", "executables-" + name + ".txt"}) {
            const auto [receipt, status] = read_linux_uma_text(support_ / filename);
            if (status == LinuxUmaReadStatus::read && !receipt.empty())
                return {"component-" + name, Status::ok, filename + "\n" + receipt};
            if (status != LinuxUmaReadStatus::unavailable)
                return {"component-" + name, Status::warn, filename + " is empty or unreadable"};
        }
        return {"component-" + name, Status::skip, "installed support receipt unavailable"};
    }

private:
    static Check memory_check(std::optional<std::uint64_t> total, std::optional<std::uint64_t> available) {
        const auto format = [](auto value) { return value ? std::to_string(*value) : "unavailable"; };
        return {"ram", total && available ? Status::ok : Status::warn,
                "total_bytes=" + format(total) + " available_bytes=" + format(available)};
    }
#if defined(__linux__)
    static int open_node(const char* path) {
        const int fd = open(path, O_RDWR | O_CLOEXEC | O_NONBLOCK);
        if (fd < 0) return errno;
        close(fd);
        return 0;
    }
#endif
    fs::path support_;
};
} // namespace

std::string doctor(const DoctorProbes& probes, const support::Matrix& matrix, bool json) {
    auto kfd = probes.node("/dev/kfd");
    auto dri = probes.node("/dev/dri");
    const bool hip_nodes_open = kfd.status == Status::ok && dri.status == Status::ok;
    std::vector<Check> checks{probes.os(), probes.cpu(), probes.ram(), std::move(kfd), std::move(dri)};
    std::string candidates;
    bool recommended = false;
    const auto platform = probes.platform();
    const auto os = platform_name(platform);
    const auto append = [&](const std::string& detail) {
        if (!candidates.empty()) candidates += '\n';
        candidates += detail;
    };
    const auto match_rows = [&](std::string_view backend) {
        auto archs = probes.architectures(backend);
        if (archs.empty()) {
            append(std::string(backend) + ": architecture unavailable");
            return;
        }
        std::ranges::sort(archs);
        archs.erase(std::ranges::unique(archs).begin(), archs.end());
        for (const auto& row : matrix.rows) {
            if (row.os != os || row.backend != backend || std::ranges::find(archs, row.arch) == archs.end())
                continue;
            auto detail = support_row(row);
            if (row.state >= support::State::packaged) {
                if (backend == "hip" && platform == Platform::linux_host && !hip_nodes_open) {
                    detail += "; package recommendation requires /dev/kfd and /dev/dri access";
                } else {
                    detail += "; recommended package component " + row.component;
                    recommended = true;
                }
            }
            if (!row.note.empty()) detail += "; " + row.note;
            append(detail);
        }
        for (const auto& arch : archs) {
            if (std::ranges::none_of(matrix.rows, [&](const auto& row) {
                return row.os == os && row.backend == backend && row.arch == arch;
            })) append(std::string(backend) + '/' + arch + '/' + std::string(os) + ": no support matrix row");
        }
    };
    match_rows("cpu");
    for (const auto backend : backends) {
        auto runtime = probes.runtime(backend);
        if (runtime.status == Status::ok) match_rows(backend);
        checks.push_back(std::move(runtime));
    }
    for (const auto component : components) checks.push_back(probes.component(component));
    if (!recommended) append("strix-tools and strix-core for host tools; no backend package recommended");
    checks.push_back({"recommendation", recommended ? Status::ok : Status::warn, std::move(candidates)});
    Json rows = Json::array();
    std::string output;
    for (const auto& check : checks) {
        if (json) rows.push(Json::Object{{"name", check.name}, {"status", status_name(check.status)}, {"detail", check.detail}});
        else output += check.name + " [" + std::string(status_name(check.status)) + "] " + check.detail + '\n';
    }
    return json ? Json(Json::Object{{"checks", rows}}).dump() + '\n' : output;
}

fs::path data_directory(const fs::path& bin) {
    const auto build_data = bin / "share/strix";
    if (fs::exists(build_data / "models/registry.toml")) return build_data;
    const auto installed = (bin / STRIX_CLI_DATA_FROM_BIN).lexically_normal();
    if (fs::exists(installed / "models/registry.toml")) return installed;
    const auto multi_config = bin.parent_path() / "share/strix";
    return fs::exists(multi_config / "models/registry.toml") ? multi_config : installed;
}

int launch_native(const Request& req, const model_registry::Entry& entry,
                  const model_description::ModelDescription& description, const fs::path& data, std::ostream& out) {
    const auto bin = executable_directory();
    const auto support_data = fs::exists(data / "support/matrix.toml") ? data : data_directory(bin);
    return launch(req, {entry}, &description, bin, support_data, NativeProbes(support_data), executable_exists, out);
}

int main(const std::vector<std::string>& args) {
    try {
        const auto request = parse(args);
        const auto bin = executable_directory();
        const auto data = data_directory(bin);
        return dispatch(request, bin, data, NativeProbes(data), executable_exists, std::cout);
    } catch (const std::exception& error) {
        std::cerr << "strix: " << error.what() << '\n';
        return 2;
    }
}
} // namespace strix::cli
