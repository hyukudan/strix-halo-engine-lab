// Topology bindings (QUEUE-topology items 2, 3, 6): pair flag strip vs declaration, the gate's `topology:` line and the
// effective-config field, and the one link-rate source. Invariant under test: no declaration (legacy flags) resolves
// exactly today's values; a declared two-node lab topology resolves the same rank/transport/rates.
#include "strix/device/link_class.hpp"
#include "strix/device/process_topology.hpp"
#include "strix/device/profiles.hpp"
#include "strix/handoff_policy.hpp"
#include "strix/pair/layercut.hpp"
#include "strix/pair/link_flags.hpp"
#include "strix/pair/state_owners.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _MSC_VER
#include <crtdbg.h>
#endif

namespace {
using namespace strix::device;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
template <class F> void rejects(F f, const char* needle, const std::string& message) {
    try {
        f();
    } catch (const std::exception& e) {
        if (std::string(e.what()).find(needle) == std::string::npos)
            throw std::runtime_error(message + ": wrong error: " + e.what());
        return;
    }
    throw std::runtime_error(message + ": accepted");
}
bool has(const std::string& text, const std::string& needle) { return text.find(needle) != std::string::npos; }
std::string env(const std::string& key) {
    const char* v = std::getenv(key.c_str());
    return v ? v : "<unset>";
}
void unset(const char* key) {
#ifdef _WIN32
    (void)_putenv_s(key, "");
#else
    (void)unsetenv(key);
#endif
}
void fresh() {
    reset_process_topology_for_tests();
    unset("STRIX_TOPOLOGY");
    unset("STRIX_NODE");
}
std::string write_file(const std::string& name, const std::string& text) {
    std::ofstream(name, std::ios::binary) << text;
    return name;
}
std::string node(const char* id, int rank, const char* host, const char* profile = "strix-halo-128", int gib = 124) {
    return std::string("[nodes.") + id + "]\nrank = " + std::to_string(rank) + "\nhost = \"" + host + "\"\nprofile = \"" + profile +
           "\"\nmemory_gib = " + std::to_string(gib) + "\n";
}
std::string link(const char* name, const char* a, const char* b, const char* cls, const char* extra = "") {
    return std::string("[links.") + name + "]\nbetween = [\"" + a + "\", \"" + b + "\"]\nclass = \"" + cls + "\"\n" + extra;
}
const std::string kHead = "schema = \"strix-topology-v1\"\nname = \"t\"\n";

struct Argv {
    std::vector<std::string> text;
    std::vector<char*> ptr;
    explicit Argv(std::vector<std::string> args) : text(std::move(args)) {
        for (auto& s : text) ptr.push_back(s.data());
        ptr.push_back(nullptr);
    }
    int argc() const { return static_cast<int>(text.size()); }
};
// Runs strip_flags; returns "RANK|PEER|TRANSPORT|remaining argv".
std::string strip(const std::string& prefix, std::vector<std::string> args, bool require_rank = true) {
    for (const char* k : {"_RANK", "_PEER", "_TRANSPORT"}) unset((prefix + k).c_str());
    Argv a(std::move(args));
    int argc = a.argc();
    char** argv = a.ptr.data();
    const auto kept = strix::pair::strip_flags(argc, argv, prefix, require_rank);
    std::string rest;
    for (int i = 1; i < argc; ++i) rest += std::string(rest.empty() ? "" : " ") + argv[i];
    require(argv[argc] == nullptr, "argv terminated");
    return env(prefix + "_RANK") + "|" + env(prefix + "_PEER") + "|" + env(prefix + "_TRANSPORT") + "|" + rest;
}

// Item 6: one class table, calibration rows reproduce every former constant bit for bit.
void link_rates() {
    static_assert(find_link_class("usb4")->gbytes_ps * 1000.0 == 4580.0, "qwen_chain_plan pair_mbps default");
    static_assert(link_rate_mbytes_ps(LinkUse::relay_host_path, "usb4") == 4400.0, "handoff_policy / state_owners relay");
    static_assert(link_rate_mbytes_ps(LinkUse::host_egress, "1gbe") * 1e6 == 125e6, "state_owners host egress");
    static_assert(link_rate_mbytes_ps(LinkUse::relay_host_path, "usb4") * 1e6 == 4.4e9, "state_owners relay");
    require(strix::pair::state::Links{}.host_bytes_per_s == 125e6 && strix::pair::state::Links{}.relay_bytes_per_s == 4.4e9,
            "state_owners Links defaults unchanged");
    const auto* usb4 = strix::handoff::find_link("usb4");
    require(usb4 && usb4->goodput_MBps == 4400.0 && usb4->per_stream_MBps == 4400.0 && usb4->per_message_s == 60e-6, "handoff_policy usb4 row unchanged");
    const std::vector<std::pair<const char*, double>> layercut{{"1gbe", 118}, {"2.5gbe", 290}, {"10gbe", 1180}, {"usb4", 4600}};
    for (const auto& [name, mbps] : layercut) require(strix::pair::layercut::link_class(name) == mbps, std::string("layercut ") + name);
    require(strix::pair::layercut::link_class("100gbe") == 11500.0, "layercut: a new class is a declaration, not a code change");
    rejects([] { (void)strix::pair::layercut::link_class("nvlink"); }, "expected 1gbe|", "layercut unknown class");
    // interconnects() is a view: same names, order, numbers as the former private table.
    const std::vector<std::tuple<const char*, double, double>> former{
        {"pcie5x16", 50.0, 5.0}, {"pcie4x16", 25.0, 5.0}, {"pcie4x4", 7.0, 5.0}, {"oculink4x4", 7.0, 5.0}, {"pcie3x4", 3.2, 5.0},
        {"usb4", 4.58, 30.0}, {"100gbe", 11.5, 20.0}, {"25gbe", 2.8, 60.0}, {"10gbe", 1.1, 100.0}, {"2.5gbe", 0.29, 150.0},
        {"1gbe", 0.118, 200.0}};
    const auto& ic = interconnects();
    require(ic.size() == former.size(), "interconnect count");
    for (std::size_t i = 0; i < former.size(); ++i)
        require(ic[i].name == std::get<0>(former[i]) && ic[i].gbytes_ps == std::get<1>(former[i]) && ic[i].rtt_us == std::get<2>(former[i]),
                "interconnect row " + std::to_string(i));
    require(ic[5].note == "measured between our two nodes" && ic[0].note == "peer copies through the host bridge, no NVLink", "notes");
    // From the declaration: legacy / lab nominal usb4 = today's numbers; another class or a measurement flips.
    require(link_rate_mbytes_ps(LinkUse::relay_host_path, nullptr) == 4400.0, "no declaration = legacy");
    const auto legacy_tcp = legacy_pair_topology(1, "tcp");
    require(link_rate_mbytes_ps(LinkUse::relay_host_path, local_pair_link(legacy_tcp)) == 4400.0, "legacy tcp shape = legacy");
    const auto lab = parse_topology(kHead + node("a", 0, "node-a") + node("b", 1, "node-b") + link("ab", "a", "b", "usb4"));
    require(link_rate_mbytes_ps(LinkUse::relay_host_path, local_pair_link(lab)) == 4400.0, "lab declaration = legacy relay");
    require(link_rate_mbytes_ps(LinkUse::layer_cut_handoff, local_pair_link(lab)) == 4600.0, "lab declaration = legacy layercut");
    require(link_rate_mbytes_ps(LinkUse::host_egress, local_pair_link(lab)) == 125.0, "host egress is not the pair link");
    const auto eth = parse_topology(kHead + node("a", 0, "node-a") + node("b", 1, "node-b") + link("ab", "a", "b", "10gbe"));
    require(link_rate_mbytes_ps(LinkUse::relay_host_path, local_pair_link(eth)) == 1100.0, "10gbe relay = class nominal");
    require(link_rate_mbytes_ps(LinkUse::layer_cut_handoff, local_pair_link(eth)) == 1180.0, "10gbe layercut = its calibration");
    const auto measured = parse_topology(kHead + node("a", 0, "node-a") + node("b", 1, "node-b") +
                                         link("ab", "a", "b", "25gbe", "gbps_measured = 21.6\n"));
    const auto* ml = local_pair_link(measured);
    require(ml && ml->measured && link_rate_mbytes_ps(LinkUse::relay_host_path, ml) == ml->gbytes_ps * 1000.0, "measured wins");
}

// Item 2: legacy flags give today's exports; a declaration fills rank / peer / transport.
void pair_flags() {
    fresh();
    for (const char* rank : {"0", "1"})
        for (const char* t : {"usb4auto", "usb4gpu", "tcp"}) {
            const auto got = strip("TP_LEG", {"bin", "--model", "m", "--rank", rank, "--transport", t, "--peer", "h", "-x"});
            require(got == std::string(rank) + "|h|" + t + "|--model m -x", "legacy " + std::string(rank) + " " + t + ": " + got);
        }
    require(strip("TP_LEG", {"bin", "--rank", "1"}) == "1|<unset>|<unset>|", "legacy rank only");
    rejects([] { (void)strip("TP_LEG", {"bin", "--rank", "2"}); }, "pair prefill requires one --rank 0|1", "rank 2 without a file");
    rejects([] { (void)strip("TP_LEG", {"bin", "--peer"}); }, "pair prefill pair flag requires a value", "missing value");
    rejects([] { (void)strip("TP_LEG", {"bin", "-x"}); }, "pair prefill requires --rank 0|1", "missing rank");
    require(strip("TP_LEG", {"bin", "-x"}, false) == "<unset>|<unset>|<unset>|-x", "deploy path: rank optional");
    require(resolved_process_topology() == nullptr, "legacy strip resolves no topology");
    rejects([] { (void)strip("TP_LEG", {"bin", "--node", "a"}); }, "no topology declared", "--node without a file");
    fresh();

    const auto usb4 = write_file("tb_two_usb4.toml", kHead + node("a", 0, "node-a") + node("b", 1, "node-b") + link("ab", "a", "b", "usb4"));
    require(strip("TP_T1", {"bin", "--topology", usb4, "--node", "b", "-x"}) == "1|node-a|<unset>|-x", "declared usb4: family default transport");
    fresh();
    // Legacy --rank with the lab declaration resolves the same rank; the peer comes from the file.
    require(strip("TP_T2", {"bin", "--topology", usb4, "--rank", "0"}) == "0|node-b|<unset>|", "declared + legacy rank");
    fresh();
    rejects([&] { (void)strip("TP_T3", {"bin", "--topology", usb4, "--node", "b", "--rank", "0"}); }, "is rank 1 but --rank 0", "contradiction");
    fresh();
    const auto eth = write_file("tb_two_10gbe_unequal.toml", kHead + node("big", 0, "h-big", "custom", 240) + node("small", 1, "h-small", "strix-halo-128", 96) +
                                                                 link("e", "small", "big", "10gbe"));
    require(strip("TP_T4", {"bin", "--topology", eth, "--node", "small"}) == "1|h-big|tcp|", "declared 10gbe, unequal nodes");
    fresh();
    require(strip("TP_T5", {"bin", "--topology", eth, "--node", "big", "--transport", "usb4auto"}) == "0|h-small|usb4auto|", "flag wins");
    fresh();
    // The device gate took --topology/--node first (kept + exported): the strip still resolves them.
    keep_topology_command_line({eth, "small"});
    require(env("STRIX_TOPOLOGY") == eth && env("STRIX_NODE") == "small", "kept flags exported for children");
    require(strip("TP_T6", {"bin", "-x"}) == "1|h-big|tcp|-x", "kept by the gate");
    fresh();
    const auto pcie = write_file("tb_two_pcie.toml", kHead + node("a", 0, "node-a") + node("b", 1, "node-b") + link("p", "a", "b", "pcie4x4"));
    rejects([&] { (void)strip("TP_T7", {"bin", "--topology", pcie, "--node", "a"}); }, "no pair transport between nodes", "pcie link");
    fresh();
    const auto one = write_file("tb_one.toml", kHead + node("solo", 0, "h0"));
    rejects([&] { (void)strip("TP_T8", {"bin", "--topology", one}); }, "the pair route has two ranks", "N=1");
    fresh();
    const auto three = write_file("tb_three.toml", kHead + node("n0", 0, "h0") + node("n1", 1, "h1") + node("n2", 2, "h2") +
                                                       link("a", "n0", "n1", "usb4") + link("b", "n1", "n2", "10gbe"));
    rejects([&] { (void)strip("TP_T9", {"bin", "--topology", three, "--node", "n2"}); }, "declares 3", "N=3");
    fresh();
    const auto four = write_file("tb_four.toml", kHead + node("n0", 0, "h0") + node("n1", 1, "h1") + node("n2", 2, "h2") + node("n3", 3, "h3") +
                                                     link("a", "n0", "n1", "usb4") + link("b", "n1", "n2", "usb4") + link("c", "n2", "n3", "usb4") +
                                                     link("d", "n3", "n0", "25gbe"));
    rejects([&] { (void)strip("TP_T10", {"bin", "--topology", four, "--node", "n1"}); }, "declares 4", "N=4");
    fresh();
    rejects([&] { (void)strip("TP_T11", {"bin", "--topology", four, "--node", "zz"}); }, "not a declared node", "unknown node");
    fresh();
}

// Item 3: the gate line and the effective-config field; CLI and serve resolve the same topology for the same argv.
void gate_and_config() {
    fresh();
    {
        Argv a({"bin", "--model", "m"});
        require(gate_topology_line(a.argc(), a.ptr.data()).empty() && resolved_process_topology() == nullptr, "legacy single: no line");
        require(with_topology_field("{\"a\":1}") == "{\"a\":1}", "unresolved: config unchanged");
    }
    {
        Argv a({"bin", "--rank", "7"});
        require(gate_topology_line(a.argc(), a.ptr.data()).empty(), "legacy rank the shape cannot describe: silent, no failure");
    }
    fresh();
    {
        Argv a({"bin", "--rank", "1", "--transport", "usb4auto"});
        const auto line = gate_topology_line(a.argc(), a.ptr.data());
        require(has(line, "topology: source=legacy") && has(line, "ranks=2") && has(line, "local=1"), "legacy pair line: " + line);
        require(has(with_topology_field("{\"a\":1}"), "{\"a\":1,\"topology\":{") && has(with_topology_field("{}"), "{\"topology\":{"),
                "config field");
        require(with_topology_field("not json") == "not json", "non-object untouched");
        rejects([] { (void)process_topology(0, ""); }, "contradicts the resolved local rank 1", "a family rank that contradicts the gate");
    }
    // CLI and serve entry points: the same argv resolves the same topology id and local rank, whichever path runs first.
    const auto file = write_file("tb_gate_three.toml", kHead + node("n0", 0, "h0") + node("n1", 1, "h1", "custom", 64) + node("n2", 2, "h2") +
                                                         link("a", "n0", "n1", "usb4") + link("b", "n1", "n2", "100gbe"));
    std::string id_gate, id_serve;
    for (int pass = 0; pass < 2; ++pass) {
        fresh();
        Argv a({"bin", "--topology", file, "--node", "n1", "--serve"});
        int argc = a.argc();
        keep_topology_command_line(take_topology_command_line(argc, a.ptr.data()));
        require(argc == 2 && std::string(a.ptr[1]) == "--serve", "gate strips --topology/--node");
        const auto& t = pass == 0 ? (void)gate_topology_line(argc, a.ptr.data()), *resolved_process_topology() : process_topology();
        require(t.size() == 3 && t.local_rank == 1, "N=3 resolved at the gate");
        (pass == 0 ? id_gate : id_serve) = topology_id(t);
        require(has(with_topology_field("{}"), "\"topology_id\":\"" + topology_id(t) + "\""), "config carries the id");
    }
    require(id_gate == id_serve, "CLI = serve topology id");
    fresh();
    {
        Argv a({"bin", "--topology", file});
        int argc = a.argc();
        keep_topology_command_line(take_topology_command_line(argc, a.ptr.data()));
        rejects([&] { (void)gate_topology_line(argc, a.ptr.data()); }, "cannot tell which node", "declared, unknown local node: gate refuses");
    }
    fresh();
    {
        Argv a({"bin", "--topology", "tb_missing_file.toml"});
        int argc = a.argc();
        keep_topology_command_line(take_topology_command_line(argc, a.ptr.data()));
        rejects([&] { (void)gate_topology_line(argc, a.ptr.data()); }, "tb_missing_file.toml", "missing file named");
    }
    fresh();
}
} // namespace

int main() {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    try {
        link_rates();
        pair_flags();
        gate_and_config();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
    std::cout << "device topology bindings tests: ok\n";
    return 0;
}
