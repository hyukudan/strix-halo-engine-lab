#include "strix/device/profiles.hpp"
#include "strix/device/link_class.hpp"
#include "strix/device/profiles_data.hpp"
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

// A view of the one link-class table (link_class.hpp): names, order and notes for `count = N` shapes; the rates are
// the class rates (no second copy of a number).
Interconnect from_class(std::string_view name, std::string_view note) {
    const auto* c = find_link_class(name);
    return {c->name, c->gbytes_ps, c->rtt_us, note};
}
const std::vector<Interconnect> kInterconnects = {
    from_class("pcie5x16", "peer copies through the host bridge, no NVLink"),
    from_class("pcie4x16", "peer copies through the host bridge"),
    from_class("pcie4x4", ""),
    from_class("oculink4x4", ""),
    from_class("pcie3x4", ""),
    from_class("usb4", "measured between our two nodes"),
    from_class("100gbe", ""),
    from_class("25gbe", ""),
    from_class("10gbe", ""),
    from_class("2.5gbe", ""),
    from_class("1gbe", ""),
};

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
    static const auto data = model_description::parse(detail::profiles_toml, model_description::TableSyntax::root_array_tables);
    static const auto profiles = parse_catalogue(data.fields);
    return profiles;
}

const Profile* find_profile(std::string_view name) noexcept {
    for (const auto& p : catalogue())
        if (p.name == name) return &p;
    return nullptr;
}

std::string profile_names() {
    std::string out;
    for (const auto& p : catalogue()) out += (out.empty() ? "" : ", ") + std::string(p.name);
    return out;
}

const std::vector<Interconnect>& interconnects() { return kInterconnects; }

const Interconnect* find_interconnect(std::string_view name) noexcept {
    for (const auto& i : kInterconnects)
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
    for (const auto& i : kInterconnects)
        os << "[interconnects." << quoted(i.name) << "]\ngbytes_ps = " << num(i.gbytes_ps) << "\nrtt_us = " << num(i.rtt_us)
           << "\n";
    return os.str();
}

} // namespace strix::device
