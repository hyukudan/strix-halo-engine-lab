#include "strix/device/gate.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

#ifdef _WIN32
#define STRIX_POPEN _popen
#define STRIX_PCLOSE _pclose
#define STRIX_NULL_DEVICE "NUL"
#else
#define STRIX_NULL_DEVICE "/dev/null"
#define STRIX_POPEN popen
#define STRIX_PCLOSE pclose
#endif

namespace strix::device {
namespace {
std::string trim(std::string_view text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) text.remove_prefix(1);
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t' || text.back() == '\r')) text.remove_suffix(1);
    return std::string(text);
}
std::vector<Visible> parse_listing(std::string_view csv, bool with_compute_capability) {
    std::vector<Visible> out;
    while (!csv.empty()) {
        const auto end = csv.find('\n');
        auto line = csv.substr(0, end);
        csv.remove_prefix(end == std::string_view::npos ? csv.size() : end + 1);
        if (trim(line).empty()) continue;
        // Keep the free-text name last so it can contain commas.
        const int name_index = with_compute_capability ? 5 : 4;
        std::string field[6];
        for (int i = 0; i <= name_index; ++i) {
            const auto comma = i < name_index ? line.find(',') : std::string_view::npos;
            if (i < name_index && comma == std::string_view::npos)
                throw std::invalid_argument("nvidia-smi: unexpected line " + std::string(line));
            field[i] = trim(line.substr(0, comma));
            if (i < name_index) line.remove_prefix(comma + 1);
        }
        Visible v;
        v.vendor = Vendor::nvidia;
        try {
            std::size_t used = 0;
            v.ordinal = std::stoi(field[0], &used);
            if (used != field[0].size() || v.ordinal < 0) throw std::invalid_argument("index");
            v.uuid = normalise_uuid(field[1]);
            v.pci = normalise_pci(field[2]);
            v.memory_bytes = std::stoull(field[3], &used) << 20U;
            if (used != field[3].size()) throw std::invalid_argument("memory");
        } catch (const std::exception& e) {
            throw std::invalid_argument("nvidia-smi: unexpected line (" + std::string(e.what()) + ")");
        }
        if (with_compute_capability) {
            const auto& capability = field[4];
            const auto dot = capability.find('.');
            if (dot != std::string::npos && dot > 0 && dot + 2 == capability.size() &&
                capability.substr(0, dot).find_first_not_of("0123456789") == std::string::npos &&
                capability.back() >= '0' && capability.back() <= '9')
                v.arch = capability.substr(0, dot) + capability.back();
        }
        v.name = field[name_index];
        out.push_back(std::move(v));
    }
    return out;
}

std::optional<std::vector<Visible>> query_nvidia(const char* command, bool with_compute_capability) {
    std::FILE* pipe = STRIX_POPEN(command, "r");
    if (!pipe) return std::nullopt;
    std::string text;
    char buffer[512];
    while (std::fgets(buffer, sizeof buffer, pipe)) text += buffer;
    if (STRIX_PCLOSE(pipe) != 0) return std::nullopt;
    try {
        auto devices = parse_listing(text, with_compute_capability);
        if (devices.empty()) return std::nullopt;
        return devices;
    } catch (const std::invalid_argument&) {
        return std::nullopt;
    }
}
} // namespace

std::vector<Visible> parse_nvidia_smi(std::string_view csv) {
    return parse_listing(csv, false);
}

std::vector<Visible> parse_nvidia_smi_with_arch(std::string_view csv) {
    return parse_listing(csv, true);
}

std::optional<std::vector<Visible>> query_nvidia_smi() {
    return query_nvidia("nvidia-smi --query-gpu=index,uuid,pci.bus_id,memory.total,name"
        " --format=csv,noheader,nounits 2>" STRIX_NULL_DEVICE, false);
}

std::optional<std::vector<Visible>> query_nvidia_smi_with_arch() {
    auto devices = query_nvidia("nvidia-smi --query-gpu=index,uuid,pci.bus_id,memory.total,compute_cap,name"
        " --format=csv,noheader,nounits 2>" STRIX_NULL_DEVICE, true);
    if (devices) return devices;
    return query_nvidia_smi();
}

void set_process_environment(const char* name, const std::string& value) {
#ifdef _WIN32
    if (_putenv_s(name, value.c_str()) != 0) throw Refusal(std::string("device gate: refusing to start: cannot set ") + name);
#else
    if (setenv(name, value.c_str(), 1) != 0) throw Refusal(std::string("device gate: refusing to start: cannot set ") + name);
#endif
}

} // namespace strix::device
