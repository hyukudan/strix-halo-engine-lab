#include "strix/device/profiles.hpp"
#include "strix/device/link_class.hpp"
#include "strix/device/profiles_data.hpp"
#include "strix/hip/arch.hpp"
#include "strix/model_description.hpp"

#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace strix::device {
namespace {

using model_description::Value;
using Table = Value::Table;

template <typename T>
const T& field(const Table& table, const char* key) {
    const auto it = table.find(key);
    if (it != table.end())
        if (const auto* value = std::get_if<T>(&it->second.data)) return *value;
    throw std::invalid_argument(std::string("device profiles: invalid or missing ") + key);
}

template <typename T>
T unsigned_field(const Table& table, const char* key) {
    const auto value = field<std::int64_t>(table, key);
    if (!std::in_range<T>(value)) throw std::invalid_argument(std::string("device profiles: out of range ") + key);
    return static_cast<T>(value);
}

double real_field(const Table& table, const char* key) {
    const auto it = table.find(key);
    if (it != table.end())
        if (const auto* value = std::get_if<std::int64_t>(&it->second.data)) return static_cast<double>(*value);
    return field<double>(table, key);
}

Profile parse_profile(const Table& row) {
    const auto& pcie = field<Table>(row, "pcie");
    const auto& peaks = field<Table>(row, "peaks");
    Profile profile{
        field<std::string>(row, "name"), field<std::string>(row, "display"),
        field<std::string>(row, "vendor"), field<std::string>(row, "arch"),
        unsigned_field<std::uint64_t>(row, "memory_bytes"), real_field(row, "memory_gbytes_ps"),
        {real_field(peaks, "fp32"), real_field(peaks, "fp16"), real_field(peaks, "bf16"),
         real_field(peaks, "fp8"), real_field(peaks, "fp4"), real_field(peaks, "int8")},
        unsigned_field<std::uint32_t>(row, "units"), field<std::string>(row, "unit_kind"),
        unsigned_field<std::uint32_t>(pcie, "gen"), unsigned_field<std::uint32_t>(pcie, "width"),
        field<std::string>(row, "host_link"), field<bool>(row, "unified"), field<std::string>(row, "measured"),
        row.contains("note") ? std::string_view(field<std::string>(row, "note")) : std::string_view{}};
    if (row.contains("lowbit_rows")) {
        const auto& lowbit = field<Table>(row, "lowbit_rows");
        profile.lowbit_rows = {unsigned_field<std::uint32_t>(lowbit, "small_rows"),
                               unsigned_field<std::uint32_t>(lowbit, "wide_panels")};
    }
    if (row.contains("legacy_hip_receipt_product"))
        profile.legacy_hip_receipt_product = field<std::string>(row, "legacy_hip_receipt_product");
    if (row.contains("unmeasured")) profile.unmeasured = field<bool>(row, "unmeasured");
    if (profile.unmeasured && (!profile.measured.empty() || profile.lowbit_rows.small_rows ||
                               profile.memory_bytes || profile.memory_gbytes_ps || profile.peaks.fp32 ||
                               profile.peaks.fp16 || profile.peaks.bf16 || profile.peaks.fp8 ||
                               profile.peaks.fp4 || profile.peaks.int8 || !profile.legacy_hip_receipt_product.empty()))
        throw std::invalid_argument("device profiles: an unmeasured profile supplies hardware identity only");
    return profile;
}

std::vector<Profile> parse_catalogue(const Table& root) {
    if (field<std::string>(root, "schema") != "strix-device-profiles-v1")
        throw std::invalid_argument("device profiles: expected strix-device-profiles-v1");
    const auto& rows = field<Value::Array>(root, "profiles");
    std::vector<Profile> profiles;
    profiles.reserve(rows.size());
    for (const auto& row : rows) profiles.push_back(parse_profile(std::get<Table>(row.data)));
    return profiles;
}

std::vector<Interconnect> parse_interconnects(const Table& root) {
    std::vector<Interconnect> links;
    for (const auto& entry : field<Value::Array>(root, "interconnects")) {
        const auto& row = std::get<Table>(entry.data);
        const auto& name = field<std::string>(row, "name");
        const auto* link_class = find_link_class(name);
        if (!link_class) throw std::invalid_argument("device profiles: unknown interconnect " + name);
        const auto note = row.contains("note") ? std::string_view(field<std::string>(row, "note")) : std::string_view{};
        links.push_back({link_class->name, link_class->gbytes_ps, link_class->rtt_us, note});
    }
    return links;
}

const Table& catalogue_data() {
    static const auto data = model_description::parse(detail::profiles_toml, model_description::TableSyntax::root_array_tables);
    return data.fields;
}

std::string num(double v) {
    std::ostringstream os;
    os << std::setprecision(10) << v;
    return os.str();
}

std::string quoted(std::string_view s) {
    std::string out = "\"";
    for (const char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out + "\"";
}

} // namespace

const std::vector<Profile>& catalogue() {
    // Profile views refer to strings in this process-lifetime TOML parse.
    static const auto profiles = parse_catalogue(catalogue_data());
    return profiles;
}

const Profile* find_profile(std::string_view name) noexcept {
    for (const auto& p : catalogue())
        if (p.name == name) return &p;
    return nullptr;
}

const Profile& describe_hip_profile(std::string_view arch, unsigned compute_units, bool integrated) {
    arch = hip::arch::base_name(arch);
    const Profile* match = nullptr;
    if (!arch.empty()) {
        for (const auto& profile : catalogue()) {
            if (profile.vendor != "amd" || profile.arch != arch || profile.unified != integrated ||
                (profile.units != 0U && profile.units != compute_units)) continue;
            if (match != nullptr) {
                match = nullptr;
                break;
            }
            match = &profile;
        }
    }
    if (match != nullptr) return *match;
    const auto* generic = find_profile("generic-hip");
    if (generic == nullptr || !generic->unmeasured)
        throw std::invalid_argument("device profiles: missing the unmeasured generic-hip fallback");
    return *generic;
}

bool hip_receipt_profile_matches(std::string_view name, std::string_view cost_identity) {
    const auto* profile = find_profile(name);
    if (profile == nullptr || profile->vendor != "amd" || profile->legacy_hip_receipt_product.empty()) return false;
    const auto arch_end = cost_identity.find('|');
    if (arch_end == std::string_view::npos ||
        hip::arch::base_name(cost_identity.substr(0, arch_end)) != profile->arch) return false;
    cost_identity.remove_prefix(arch_end + 1);
    const auto product_end = cost_identity.find('|');
    return product_end != std::string_view::npos &&
           cost_identity.substr(0, product_end) == profile->legacy_hip_receipt_product;
}

std::string profile_names() {
    std::string out;
    for (const auto& p : catalogue()) out += (out.empty() ? "" : ", ") + std::string(p.name);
    return out;
}

const std::vector<Interconnect>& interconnects() {
    static const auto links = parse_interconnects(catalogue_data());
    return links;
}

const Interconnect* find_interconnect(std::string_view name) noexcept {
    for (const auto& i : interconnects())
        if (i.name == name) return &i;
    return nullptr;
}

std::string catalogue_toml() {
    std::ostringstream os;
    // Preserve the CLI export bytes. The input TOML has its own editable-file header.
    os << "# Generated from src/device/profiles.cpp (strix::device::catalogue_toml); do not edit by hand.\n"
          "# Peaks are dense TFLOPS / TOPS; 0 = no tensor path or not stated. `measured` lists the fields read on our\n"
          "# own devices; everything else is the vendor's public specification.\n"
          "schema = \"strix-device-profiles-v1\"\n";
    for (const auto& p : catalogue()) {
        os << "\n[profiles." << p.name << "]\n"
           << "display = " << quoted(p.display) << "\n"
           << "vendor = " << quoted(p.vendor) << "\n"
           << "arch = " << quoted(p.arch) << "\n"
           << "memory_bytes = " << p.memory_bytes << "\n"
           << "memory_gbytes_ps = " << num(p.memory_gbytes_ps) << "\n"
           << "peaks = { fp32 = " << num(p.peaks.fp32) << ", fp16 = " << num(p.peaks.fp16) << ", bf16 = " << num(p.peaks.bf16)
           << ", fp8 = " << num(p.peaks.fp8) << ", fp4 = " << num(p.peaks.fp4) << ", int8 = " << num(p.peaks.int8) << " }\n"
           << "units = " << p.units << "\n"
           << "unit_kind = " << quoted(p.unit_kind) << "\n"
           << "pcie = { gen = " << p.pcie_gen << ", width = " << p.pcie_width << " }\n"
           << "host_link = " << quoted(p.host_link) << "\n"
           << "unified = " << (p.unified ? "true" : "false") << "\n"
           << "measured = " << quoted(p.measured) << "\n";
        if (p.lowbit_rows.small_rows != 0U)
            os << "lowbit_rows = { small_rows = " << p.lowbit_rows.small_rows
               << ", wide_panels = " << p.lowbit_rows.wide_panels << " }\n";
        if (!p.note.empty()) os << "note = " << quoted(p.note) << "\n";
    }
    os << "\n# Interconnect classes for `count = N` shapes: nominal one-direction GB/s and round trip (us).\n";
    for (const auto& i : interconnects())
        os << "[interconnects." << quoted(i.name) << "]\ngbytes_ps = " << num(i.gbytes_ps) << "\nrtt_us = " << num(i.rtt_us)
           << "\n";
    return os.str();
}

} // namespace strix::device
