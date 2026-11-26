// Device catalogue (strix/device/profiles.hpp), declared budgets (budget.hpp, gate.hpp) and the shared memory plan
// (strix/deploy/memory_plan.hpp). Host only.
#include "strix/deploy/device.hpp"
#include "strix/deploy/memory_plan.hpp"
#include "strix/device/process_budget.hpp"
#include "strix/device/budget.hpp"
#include "strix/device/declaration.hpp"
#include "strix/device/gate.hpp"
#include "strix/device/link_class.hpp"
#include "strix/device/profiles.hpp"
#include "strix/model_description.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <utility>

namespace {

int failures = 0;
void require(bool ok, const char* message) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
}
template <class F> bool throws(F&& f, const char* needle = nullptr) {
    try {
        f();
    } catch (const std::exception& e) {
        if (needle && std::string(e.what()).find(needle) == std::string::npos) {
            std::fprintf(stderr, "  (message was: %s)\n", e.what());
            return false;
        }
        return true;
    }
    return false;
}

using namespace strix;
constexpr std::uint64_t GiB = 1073741824ULL;
const char* kPro = "GPU-11111111-2222-3333-4444-555555555555";
constexpr std::uint64_t kProBytes = 102641958912ULL;

device::Visible pro_card() {
    device::Visible v;
    v.ordinal = 0;
    v.vendor = device::Vendor::nvidia;
    v.name = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition";
    v.uuid = device::normalise_uuid(kPro);
    v.arch = "sm_120";
    v.memory_bytes = kProBytes;
    return v;
}

std::string decl(const std::string& body) {
    return "schema = \"strix-devices-v1\"\n" + body;
}

void catalogue() {
    std::set<std::string_view> names;
    for (const auto& p : device::catalogue()) require(names.insert(p.name).second, "catalogue names are unique");
    for (const char* n : {"rtx-pro-6000-blackwell-96", "rtx-5090-32", "rtx-5080-16", "rtx-4090-24", "strix-halo-128",
                          "dgx-spark-gb10-128", "rx-9070-xt-16", "custom"})
        require(device::find_profile(n) != nullptr, n);
    require(device::hip_receipt_profile_matches("strix-halo-128", "gfx1151:xnack-|AMD Radeon 8060S Graphics|fixture"),
            "legacy receipt accepts the recorded architecture and product");
    require(!device::hip_receipt_profile_matches("strix-halo-128", "gfx1152|AMD Radeon 8060S Graphics|fixture") &&
                !device::hip_receipt_profile_matches("strix-halo-128", "gfx1151|another product|AMD Radeon 8060S Graphics|fixture") &&
                !device::hip_receipt_profile_matches("strix-halo-32cu", "gfx1151|AMD Radeon 8060S Graphics|fixture"),
            "the receipt does not attach to another architecture, field or profile");
    const auto* p5090 = device::find_profile("rtx-5090-32");
    require(p5090->units == 170 && p5090->arch == "sm_120" && p5090->memory_bytes == 32607ULL * 1048576ULL, "5090 entry");
    require(device::find_profile("rtx-pro-6000-blackwell-96")->memory_bytes == kProBytes, "PRO memory is the measured total");
    for (const char* n : {"strix-halo-32cu", "gorgon-gfx1152", "gorgon-gfx1153", "amd-gfx942", "amd-gfx950",
                          "intel-arc-xe2", "apple-silicon", "generic-hip"}) {
        const auto* profile = device::find_profile(n);
        require(profile && profile->unmeasured, n);
        require(profile->memory_bytes == 0 && profile->memory_gbytes_ps == 0 && profile->lowbit_rows.small_rows == 0,
                "unmeasured catalogue row cannot carry a memory cap, bandwidth or schedule");
    }
    for (const auto& p : device::catalogue())
        if (p.name != "custom" && !p.unmeasured)
            require(p.memory_bytes > 0 && p.memory_gbytes_ps > 0 && p.units > 0, "every sized profile retains its facts");
    require(device::find_interconnect("pcie5x16") && !device::find_interconnect("nvlink"), "interconnect classes");
    // The exported TOML keeps the named v1 tables, with all source values in catalogue order.
    using model_description::Value;
    const auto source = model_description::load(std::string(STRIX_SOURCE_ROOT) + "/data/devices/profiles.toml",
                                                model_description::TableSyntax::root_array_tables);
    const auto output_text = device::catalogue_toml();
    // Legacy CLI exports quote interconnect keys. The shared model parser accepts bare keys.
    const auto link_section = output_text.find("\n# Interconnect classes");
    require(link_section != std::string::npos, "the export includes its interconnect section");
    const auto output = model_description::parse(std::string_view(output_text).substr(0, link_section));
    const auto& ordered = std::get<Value::Array>(source.fields.at("profiles").data);
    const auto& exported = std::get<Value::Table>(output.fields.at("profiles").data);
    require(source.fields.at("schema") == output.fields.at("schema"), "the export uses the catalogue schema");
    require(ordered.size() == device::catalogue().size() && ordered.size() == exported.size(),
            "the catalogue and the export cover every profile");
    for (std::size_t i = 0; i < ordered.size(); ++i) {
        auto expected = std::get<Value::Table>(ordered[i].data);
        const auto key = std::get<std::string>(expected.at("name").data);
        expected.erase("name");
        // Internal admission metadata does not extend the v1 CLI export schema.
        expected.erase("unmeasured");
        expected.erase("legacy_hip_receipt_product");
        require(device::catalogue().at(i).name == key, "catalogue order follows the data");
        require(std::get<Value::Table>(exported.at(key).data) == expected, "exported profile matches its data");
    }
    const auto& link_data = std::get<Value::Array>(source.fields.at("interconnects").data);
    require(link_data.size() == device::interconnects().size(), "interconnect catalogue covers the data");
    for (std::size_t i = 0; i < link_data.size(); ++i) {
        const auto& link = device::interconnects().at(i);
        const auto& expected = std::get<Value::Table>(link_data[i].data);
        require(link.name == std::get<std::string>(expected.at("name").data), "interconnect order follows the data");
        if (expected.contains("note"))
            require(link.note == std::get<std::string>(expected.at("note").data), "interconnect note follows the data");
        const auto* shared = device::find_link_class(link.name);
        require(shared && shared->gbytes_ps == link.gbytes_ps && shared->rtt_us == link.rtt_us, "rates follow the shared link classes");
        const auto header = "[interconnects.\"" + std::string(link.name) + "\"]\n";
        require(output_text.find(header, link_section) != std::string::npos, "the export covers every catalogue interconnect");
    }
}

void declaration() {
    const auto apple = device::parse_declaration(decl("[devices.gpu]\nprofile = \"apple-silicon\"\n"));
    require(apple.devices.at(0).vendor == device::Vendor::apple && apple.devices.at(0).planned,
            "Apple profile supplies a vendor for a planning entry");
    const auto d = device::parse_declaration(decl("[devices.main]\nuuid = \"" + std::string(kPro) +
                                                  "\"\nprofile = \"rtx-5090-32\"\nroles = [\"replica\"]\n"));
    require(d.devices.size() == 1 && d.devices[0].vendor == device::Vendor::nvidia, "vendor comes from the profile");
    require(d.devices[0].profile == "rtx-5090-32" && !d.devices[0].planned, "a profile with an identity binds");
    {
        // Optional measured low-bit route switch points of one declared GPU (core LowBitRowPlan): both keys > 0.
        const std::string gpu = "[devices.a]\nvendor = \"amd\"\npci = \"0000:c5:00.0\"\nprofile = \"strix-halo-128\"\n";
        const auto with = device::parse_declaration(decl(gpu + "lowbit_rows = { small_rows = 64, wide_panels = 32 }\n"));
        const auto without = device::parse_declaration(decl(gpu));
        require(with.devices.at(0).lowbit_rows.small_rows == 64U && with.devices.at(0).lowbit_rows.wide_panels == 32U &&
                    without.devices.at(0).lowbit_rows.small_rows == 0U && without.devices.at(0).lowbit_rows.wide_panels == 0U,
                "lowbit_rows: declared switch points, absent = none");
        require(throws([&] { (void)device::parse_declaration(decl(gpu + "lowbit_rows = { small_rows = 64 }\n")); }, "required"),
                "lowbit_rows: both keys required");
        require(throws([&] { (void)device::parse_declaration(decl(gpu + "lowbit_rows = { small_rows = 0, wide_panels = 4 }\n")); }, "1 to 65535"),
                "lowbit_rows: zero refused");
        require(throws([&] { (void)device::parse_declaration(decl(gpu + "lowbit_rows = { small_rows = 4, wide_panels = 4, x = 1 }\n")); }, "unknown key"),
                "lowbit_rows: unknown key refused");
        require(throws([] { (void)device::parse_declaration(decl("[devices.host]\nkind = \"cpu\"\nlowbit_rows = { small_rows = 4, wide_panels = 4 }\n")); }, "only a GPU"),
                "lowbit_rows: GPU entries only");
    }
    require(throws([] { (void)device::parse_declaration(decl("[devices.a]\nuuid = \"GPU-11111111-2222-3333-4444-555555555555\"\nprofile = \"rtx-9999\"\n")); }, "unknown profile"),
            "unknown profile refused with the known list");
    require(throws([] { (void)device::parse_declaration(decl("[devices.a]\nvendor = \"amd\"\nuuid = \"GPU-11111111-2222-3333-4444-555555555555\"\nprofile = \"rtx-5090-32\"\n")); }, "contradicts"),
            "vendor contradicting the profile refused");
    require(throws([] { (void)device::parse_declaration(decl("[devices.a]\nvendor = \"nvidia\"\nprofile = \"custom\"\n")); }, "memory_gib"),
            "custom needs memory_gib");
    const auto shape = device::parse_declaration(decl("[devices.pair]\nprofile = \"rtx-5090-32\"\ncount = 2\ninterconnect = \"pcie5x16\"\n"));
    require(shape.devices[0].planned && shape.devices[0].count == 2 && shape.devices[0].interconnect == "pcie5x16", "2x 5090 shape");
    require(throws([] { (void)device::parse_declaration(decl("[devices.p]\nprofile = \"rtx-5090-32\"\ncount = 2\n")); }, "interconnect"),
            "count > 1 needs an interconnect");
    require(throws([] { (void)device::parse_declaration(decl("[devices.p]\nprofile = \"rtx-5090-32\"\ncount = 2\ninterconnect = \"nvlink\"\n")); }, "NVLink"),
            "nvlink is not a catalogue interconnect");
    require(throws([] { (void)device::parse_declaration(decl("[devices.p]\nuuid = \"GPU-11111111-2222-3333-4444-555555555555\"\nprofile = \"rtx-5090-32\"\ncount = 2\ninterconnect = \"pcie5x16\"\n")); }, "no identity"),
            "a shape takes no identity");
    require(throws([] { (void)device::parse_declaration(decl("[devices.p]\nvendor = \"nvidia\"\n")); }, "needs an identity"),
            "no identity and no profile is still refused");
    require(throws([] { (void)device::parse_declaration(decl("[devices.host]\nkind = \"cpu\"\nprofile = \"rtx-5090-32\"\n")); }, "only a GPU"),
            "a cpu unit takes no profile");
    const auto host = device::parse_declaration(decl("[devices.host]\nkind = \"cpu\"\nmemory_gib = 64\n"));
    require(host.devices[0].budget_bytes == 64 * GiB, "cpu unit memory_gib = host budget");
}

void budget() {
    device::Descriptor g;
    g.id = "main";
    g.profile = "rtx-5090-32";
    auto b = device::resolve_budget(g, nullptr, kProBytes);
    require(b.device_bytes == 32607ULL * 1048576ULL && b.device_source == "profile rtx-5090-32", "profile gives the budget");
    require(b.emulated() && b.units == 170 && b.bandwidth_gbytes_ps == 1792, "below the card = emulation");
    g.budget_bytes = 30 * GiB;
    require(device::resolve_budget(g, nullptr, kProBytes).device_bytes == 30 * GiB, "memory_gib wins over the profile");
    device::BudgetOverrides o;
    o.vram_total_gib = 24.0;
    require(device::resolve_budget(g, nullptr, kProBytes, o).device_bytes == 24 * GiB, "the route's flag wins");
    o.vram_total_gib = -1.0;
    require(!device::resolve_budget(g, nullptr, kProBytes, o).device_capped(), "--vram-total-gib none removes the cap");
    device::Descriptor big;
    big.id = "x";
    big.profile = "strix-halo-128";
    require(throws([&] { (void)device::resolve_budget(big, nullptr, 32 * GiB); }, "cannot exceed"), "a profile cannot add memory");
    device::Descriptor none;
    none.id = "plain";
    require(!device::resolve_budget(none, nullptr, kProBytes).device_capped(), "no profile = no cap (today's path)");
    device::Descriptor cpu;
    cpu.id = "host";
    cpu.kind = device::Kind::cpu;
    cpu.budget_bytes = 64 * GiB;
    require(device::resolve_budget(g, &cpu, kProBytes).host_bytes == 64 * GiB, "host budget from the cpu unit");
    require(device::parse_total_gib("--x", "none") < 0 && device::parse_total_gib("--x", "32") == 32.0, "flag form");
    require(throws([] { (void)device::parse_total_gib("--x", "0"); }), "zero refused");
}

void gate() {
    const std::string path_free = decl("[devices.main]\nvendor = \"nvidia\"\nuuid = \"" + std::string(kPro) + "\"\nroles = [\"replica\"]\n");
    const auto plain = device::parse_declaration(path_free);
    device::Request r;
    r.vendor = device::Vendor::nvidia;
    const auto b0 = device::resolve(&plain, {pro_card()}, r);
    require(!b0.budget.device_capped() && device::receipt_json(b0).find("memory_budget") == std::string::npos,
            "no profile: receipt bytes unchanged");
    r.profile = "rtx-5090-32";
    const auto b1 = device::resolve(&plain, {pro_card()}, r);
    require(b1.budget.emulated() && b1.device.profile == "rtx-5090-32", "--device-profile emulates on the bound card");
    require(device::receipt_json(b1).find("strix-memory-budget-v1") != std::string::npos, "receipt carries the budget");
    require(device::receipt_line(b1).find("profile=rtx-5090-32") != std::string::npos, "receipt line names the profile");
    r.profile = "rtx-9999";
    require(throws([&] { (void)device::resolve(&plain, {pro_card()}, r); }, "unknown device profile"), "unknown --device-profile");
    // A declared profile above the card refuses at the gate.
    const auto too_big = device::parse_declaration(decl("[devices.main]\nuuid = \"" + std::string(kPro) +
                                                        "\"\nprofile = \"dgx-spark-gb10-128\"\nvendor = \"nvidia\"\nroles = [\"replica\"]\n"));
    r.profile.clear();
    require(throws([&] { (void)device::resolve(&too_big, {pro_card()}, r); }, "cannot exceed"), "profile above the card refused");
    // A planned shape is never bound.
    const auto mixed = device::parse_declaration(decl("[devices.main]\nvendor = \"nvidia\"\nuuid = \"" + std::string(kPro) +
                                                      "\"\nroles = [\"replica\"]\n[devices.pair]\nprofile = \"rtx-5090-32\"\ncount = 2\n"
                                                      "interconnect = \"pcie5x16\"\nroles = [\"replica\"]\n"));
    r.select = "pair";
    require(throws([&] { (void)device::resolve(&mixed, {pro_card()}, r); }, "planned shape"), "planned entry refused at bind");
    r.select.clear();
    require(device::resolve(&mixed, {pro_card()}, r).device.id == "main", "the bindable card still binds beside a shape");
    // ONE RESOLVER: the server / CLI bound path and the planning tool resolve the same request to the same budget.
    device::BudgetOverrides o;
    o.ram_total_gib = 48.0;
    const auto via_gate = device::route_budget(b1, o);
    const auto via_tool = device::resolve_budget(b1.device, nullptr, kProBytes, o);
    require(via_gate.json() == via_tool.json(), "gate path and planning path give the same budget");
    // start_gate records the budget for the planners that never see the binding (deploy::probe_device).
    const std::string file = "device_profiles_tests_devices.toml";
    {
        std::ofstream out(file, std::ios::binary);
        out << decl("[devices.main]\nuuid = \"" + std::string(kPro) + "\"\nprofile = \"rtx-5080-16\"\nroles = [\"replica\"]\n");
    }
    device::FakeEnumerator cards(device::Vendor::nvidia, {pro_card()});
    const auto via_start = device::start_gate(cards, {}, file);
    std::remove(file.c_str());
    require(via_start.budget.device_bytes == 16303ULL * 1048576ULL &&
                device::process_device_budget_bytes().load() == via_start.budget.device_bytes,
            "start_gate records the process budget");
    deploy::DeviceFacts f;
    f.mem_total_bytes = 96 * GiB;
    f.mem_free_bytes = 90 * GiB;
    deploy::apply_budget(f, 32 * GiB);
    require(f.mem_total_bytes == 32 * GiB && f.mem_free_bytes == 26 * GiB, "apply_budget: free = budget - in use");
    deploy::DeviceFacts g = {};
    g.mem_total_bytes = 16 * GiB;
    g.mem_free_bytes = 15 * GiB;
    deploy::apply_budget(g, 32 * GiB);
    require(g.mem_total_bytes == 16 * GiB && g.mem_free_bytes == 15 * GiB, "apply_budget: a budget above the device changes nothing");
}

void lowbit_profile() {
    const auto* profile = device::find_profile("strix-halo-128");
    require(profile && profile->lowbit_rows.small_rows == 64U && profile->lowbit_rows.wide_panels == 64U,
            "Strix Halo catalogue declares both measured switch points");
    for (const auto& entry : device::catalogue()) {
        if (entry.name != "strix-halo-128")
            require(entry.lowbit_rows.small_rows == 0U && entry.lowbit_rows.wide_panels == 0U,
                    "other profiles do not inherit Strix Halo switch points");
    }
    device::Visible card;
    card.vendor = device::Vendor::amd;
    card.arch = "gfx1151";
    card.pci = "0000:c5:00.0";
    card.memory_bytes = 128 * GiB;
    card.integrated = true;
    device::Request request;
    request.vendor = device::Vendor::amd;
    const auto device_entry = "[devices.main]\npci = \"0000:c5:00.0\"\nroles = [\"decode\"]\n";
    const auto declared = device::parse_declaration(decl(std::string(device_entry) + "profile = \"strix-halo-128\"\n"));
    const auto overridden = device::parse_declaration(decl(std::string(device_entry) + "profile = \"strix-halo-128\"\n"
        "lowbit_rows = { small_rows = 32, wide_panels = 8 }\n"));
    const auto implicit = device::resolve(nullptr, {card}, request).device.lowbit_rows;
    require(implicit.small_rows == 0U && !implicit.from_profile, "implicit binding without a profile has no plan");
    const auto inherited = device::resolve(&declared, {card}, request).device.lowbit_rows;
    require(inherited.small_rows == 64U && inherited.wide_panels == 64U && inherited.from_profile,
            "bound declaration inherits the selected catalogue plan");
    const auto explicit_rows = device::resolve(&overridden, {card}, request).device.lowbit_rows;
    require(explicit_rows.small_rows == 32U && explicit_rows.wide_panels == 8U && !explicit_rows.from_profile,
            "entry switch points override catalogue data");
    request.profile = "strix-halo-128";
    require(device::resolve(nullptr, {card}, request).device.lowbit_rows.from_profile,
            "profile selection also works on an implicit binding");
    request.profile = "rx-9070-xt-16";
    require(device::resolve(&declared, {card}, request).device.lowbit_rows.small_rows == 0U &&
                device::resolve(&overridden, {card}, request).device.lowbit_rows.small_rows == 32U,
            "profile replacement replaces the catalogue plan and keeps explicit entry switch points");
    card.arch = "gfx1201";
    require(device::resolve(&declared, {card}, {}).device.lowbit_rows.small_rows == 0U,
            "emulating a profile on another architecture does not import its schedule");
}

deploy::RouteNeeds moe_route() {
    deploy::RouteNeeds n;
    n.route = "moe";
    n.device_fixed = 4 * GiB;
    n.weights_resident = 10 * GiB;
    n.tier_units = 1000;
    n.tier_unit_bytes = 64ULL << 20; // 62.5 GiB of experts
    n.min_device_units = 16;
    n.host_tier = n.file_tier = true;
    n.host_fixed = 8 * GiB;
    n.session_per_position = 64 << 10;
    n.positions = 32768;
    n.min_positions = 2560;
    n.slots = 4;
    n.margin_bytes = 1 * GiB;
    return n;
}

void plan() {
    device::MemoryBudget b;
    b.device_bytes = 32 * GiB;
    b.host_bytes = 40 * GiB;
    b.device_source = "test";
    auto n = moe_route();
    const auto p = deploy::plan_memory(b, n);
    // 4 + 10 + 1 + 4 x 2 GiB sessions = 23 GiB, 9 GiB -> 144 units in VRAM; host 32 GiB -> 512 units; 344 to the file.
    require(!p.refused && p.slots == 4 && p.positions == 32768, "sessions kept");
    require(p.units_device == 144 && p.units_host == 512 && p.units_file == 344, "VRAM / pinned / file split");
    b.device_bytes = 18 * GiB;
    const auto q = deploy::plan_memory(b, n);
    require(!q.refused && q.positions == 32768 && q.slots == 1, "slots drop before positions");
    b.device_bytes = 17 * GiB;
    const auto r = deploy::plan_memory(b, n);
    require(!r.refused && r.slots == 1 && r.positions < 32768 && r.positions >= 2560, "then positions halve");
    b.device_bytes = 15 * GiB;
    const auto s = deploy::plan_memory(b, n);
    require(s.refused && s.min_device_bytes > 15 * GiB && s.refusal.find("needs at least") != std::string::npos, "refusal names the minimum");
    // A route without tiers (all weights resident): refuses below the weights and says the minimum.
    deploy::RouteNeeds dense;
    dense.route = "dense";
    dense.weights_resident = 82 * GiB;
    dense.device_fixed = 2 * GiB;
    dense.session_per_position = 31616;
    dense.session_fixed = 233ULL << 20;
    dense.positions = 34816;
    dense.slots = 3;
    b.device_bytes = 32 * GiB;
    const auto d = deploy::plan_memory(b, dense);
    require(d.refused && d.min_device_bytes > 84 * GiB, "dense route refuses on 32 GiB with the minimum");
    b.device_bytes = 96 * GiB;
    require(!deploy::plan_memory(b, dense).refused, "and fits 96 GiB");
    // Needs file round trip.
    const auto parsed = deploy::parse_route_needs("schema = \"strix-route-needs-v1\"\nroute = \"x\"\nweights_resident = 1024\nhost_tier = true\ndecode_efficiency = 0.5\n");
    require(parsed.route == "x" && parsed.weights_resident == 1024 && parsed.host_tier && parsed.decode_efficiency == 0.5, "needs parse");
    require(throws([] { (void)deploy::parse_route_needs("schema = \"strix-route-needs-v1\"\nroute = \"x\"\nbogus = 1\n"); }, "unknown key"),
            "needs unknown key");
    // 2x RTX 5090 over PCIe 5: each card holds half, the prediction counts one hop.
    device::Descriptor pair;
    pair.id = "pair";
    pair.profile = "rtx-5090-32";
    pair.count = 2;
    pair.interconnect = "pcie5x16";
    auto m = moe_route();
    m.bytes_per_token = 11ULL * 1000000000ULL;
    m.tiered_bytes_per_token = 6ULL * 1000000000ULL;
    m.hop_bytes_per_token = 16384;
    m.decode_efficiency = 0.5;
    const auto two = deploy::plan_shape(pair, nullptr, m);
    require(!two.per_device.refused && two.per_device.units_device > 0, "shape fits per device");
    require(two.predicted_decode_tok_s > 0 && two.hop_ms > 0, "shape prediction");
    pair.count = 1;
    const auto one = deploy::plan_shape(pair, nullptr, m);
    require(one.per_device.units_device < two.per_device.units_device * 2, "two cards hold more experts than one");
}

} // namespace

int main() {
    for (const auto& [name, run] : {std::pair<const char*, void (*)()>{"catalogue", catalogue}, {"declaration", declaration},
                                    {"budget", budget}, {"gate", gate}, {"lowbit_profile", lowbit_profile}, {"plan", plan}}) {
        try {
            run();
        } catch (const std::exception& e) {
            ++failures;
            std::fprintf(stderr, "FAIL: %s threw: %s\n", name, e.what());
        }
    }
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("device profiles tests: OK\n");
    return 0;
}
