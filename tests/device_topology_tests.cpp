// Host tests of the node topology declaration (strix/device/topology.hpp): 1, 2 equal, 2 unequal, 3 and 4 nodes,
// invalid declarations, the legacy --rank shape, local identity, discovery and the bridge to the shared link graph.
#include "strix/deploy/link.hpp"
#include "strix/device/topology.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _MSC_VER
#include <crtdbg.h>
#endif

namespace {
using namespace strix::device;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template <class F> void rejects(F f, const char* needle, const char* message) {
    try {
        f();
    } catch (const std::exception& e) {
        if (std::string(e.what()).find(needle) == std::string::npos)
            throw std::runtime_error(std::string(message) + ": wrong error: " + e.what());
        return;
    }
    throw std::runtime_error(std::string(message) + ": accepted");
}
bool has(const std::string& text, const char* needle) { return text.find(needle) != std::string::npos; }

const char* kTwoEqual = R"(schema = "strix-topology-v1"
name = "two-node-usb4"
[nodes.a]
rank = 0
host = "node-a"
profile = "strix-halo-128"
memory_gib = 124
[nodes.b]
rank = 1
host = "node-b"
profile = "strix-halo-128"
memory_gib = 124
[links.ab]
between = ["a", "b"]
class = "usb4"
)";

void single_node() {
    const auto t = parse_topology(R"(schema = "strix-topology-v1"
[nodes.solo]
rank = 0
profile = "rtx-5090-32"
)");
    require(t.size() == 1 && t.links.empty(), "one node, no link");
    require(t.nodes[0].memory_source == FactSource::profile && t.nodes[0].memory_bytes > 0, "memory from the profile");
    require(t.nodes[0].arch == "sm_120" && t.nodes[0].vendor == "nvidia", "arch from the profile");
    const auto r = resolve_topology(&t, {});
    require(r.local_known && r.local_rank == 0, "the only node is local");
    const auto implicit = resolve_topology(nullptr, {});
    require(implicit.size() == 1 && implicit.source == "single" && implicit.local_rank == 0, "no file, no rank: one node");
    require(has(topology_line(implicit), "ranks=1 local=0(local)"), "single line");
}

void two_equal() {
    const auto t = parse_topology(kTwoEqual, "two.toml");
    require(t.size() == 2 && t.uniform(), "two equal nodes are uniform");
    require(t.nodes[0].id == "a" && t.nodes[1].id == "b", "index == rank");
    require(t.nodes[1].memory_bytes == 124ULL * 1073741824ULL && t.nodes[1].memory_source == FactSource::declared, "declared memory");
    require(t.nodes[0].memory_gbytes_ps == 256 && t.nodes[0].rate_source == FactSource::profile, "profile memory rate");
    require(t.links.size() == 1 && t.links[0].a == 0 && t.links[0].b == 1, "one link");
    require(t.direct_gbytes_ps(1, 0) == 4.58 && !t.links[0].measured, "usb4 nominal class rate, either direction");
    // The same file on both nodes: same id, different local rank.
    LocalIdentity on_b;
    on_b.hostname = "node-b";
    const auto rb = resolve_topology(&t, on_b);
    LocalIdentity on_a;
    on_a.rank = 0;
    const auto ra = resolve_topology(&t, on_a);
    require(rb.local_rank == 1 && ra.local_rank == 0, "host match and legacy rank");
    require(topology_id(ra) == topology_id(rb), "one cluster, one id on every node");
    const auto line = topology_line(rb);
    require(has(line, "topology: source=two.toml name=two-node-usb4 ranks=2 local=1(b) uniform=yes"), "line head");
    require(has(line, "ab[a-b usb4 4.58GB/s nominal rtt=30us port=link-ab]"), "line link");
    require(has(topology_json(rb), "\"schema\":\"strix-topology-v1\"") && has(topology_json(rb), "\"local_rank\":1"), "json");
}

void two_unequal() {
    const auto t = parse_topology(R"(schema = "strix-topology-v1"
[nodes.big]
rank = 0
profile = "custom"
memory_gib = 240
bandwidth_gbytes_ps = 410
fp16_tflops = 120
[nodes.small]
rank = 1
profile = "strix-halo-128"
memory_gib = 96
[links.eth]
between = ["small", "big"]
class = "25gbe"
gbps_measured = 21.6
rtt_us_measured = 45
)");
    require(!t.uniform(), "unequal nodes are not uniform");
    require(t.nodes[0].id == "big" && t.nodes[0].fp16_tflops == 120 && t.nodes[0].compute_source == FactSource::declared, "declared compute");
    require(t.nodes[1].fp16_tflops == 59.4 && t.nodes[1].compute_source == FactSource::profile, "profile compute");
    require(t.links[0].measured && t.links[0].rtt_measured && t.direct_gbytes_ps(0, 1) == 2.7, "measured link wins");
    require(has(topology_line(resolve_topology(&t, LocalIdentity{"small", -1, "", ""})), "uniform=no"), "line says unequal");
}

void three_line() {
    const auto t = parse_topology(R"(schema = "strix-topology-v1"
[nodes.n0]
rank = 0
profile = "strix-halo-128"
[nodes.n1]
rank = 1
profile = "strix-halo-128"
[nodes.n2]
rank = 2
profile = "dgx-spark-gb10-128"
[links.l01]
between = ["n0", "n1"]
class = "usb4"
[links.l12]
between = ["n1", "n2"]
class = "10gbe"
)");
    require(t.size() == 3 && t.links_between(0, 2).empty() && t.direct_gbytes_ps(0, 2) == 0, "no direct 0-2 link");
    // Through the shared link graph, rank 0 reaches rank 2 over two hops at the slower rate.
    const auto d = to_declaration(t);
    require(d.devices.size() == 3 && d.links.size() == 2 && d.devices[2].id == "n2" && d.devices[2].planned, "declaration bridge");
    const auto g = strix::deploy::link_graph(d);
    const auto p = strix::deploy::best_path(g, "n0", "n2");
    require(p.path && p.path->edges.size() == 2 && p.path->gbytes_ps == 1.1, "two hops, bottleneck 10gbe");
    LocalIdentity none;
    rejects([&] { (void)resolve_topology(&t, none); }, "cannot tell which node", "three nodes need an identity");
}

void four_ring_dual() {
    const auto t = parse_topology(R"(schema = "strix-topology-v1"
[nodes.a]
rank = 0
profile = "strix-halo-128"
[nodes.b]
rank = 1
profile = "strix-halo-128"
[nodes.c]
rank = 2
profile = "strix-halo-128"
[nodes.d]
rank = 3
profile = "strix-halo-128"
[links.ab0]
between = ["a", "b"]
class = "usb4"
port = "p0"
[links.ab1]
between = ["b", "a"]
class = "usb4"
port = "p1"
[links.bc]
between = ["b", "c"]
class = "usb4"
[links.cd]
between = ["c", "d"]
class = "usb4"
[links.da]
between = ["d", "a"]
class = "usb4"
)");
    require(t.size() == 4 && t.uniform(), "four equal nodes");
    require(t.links_between(0, 1).size() == 2 && t.direct_gbytes_ps(0, 1) == 2 * 4.58, "dual link adds on distinct ports");
    LocalIdentity id;
    id.node = "c";
    require(resolve_topology(&t, id).local_rank == 2, "--node picks the rank");
    id.rank = 1;
    rejects([&] { (void)resolve_topology(&t, id); }, "is rank 2 but --rank 1", "--node and --rank disagree");
}

void invalid() {
    const std::string head = "schema = \"strix-topology-v1\"\n";
    const std::string two = head + "[nodes.a]\nrank = 0\nprofile = \"strix-halo-128\"\n[nodes.b]\nrank = 1\nprofile = \"strix-halo-128\"\n";
    rejects([&] { (void)parse_topology("schema = \"strix-devices-v1\"\n[nodes.a]\nrank = 0\n"); }, "expected strix-topology-v1", "schema");
    rejects([&] { (void)parse_topology(head); }, "nodes: required", "no node");
    // table() takes the key as a view: a non-table value still names the literal key in the error.
    rejects([&] { (void)parse_topology(head + "nodes = 3\n"); }, "nodes: expected a table", "nodes not a table");
    rejects([&] { (void)parse_topology(head + "links = 1\n[nodes.a]\nrank = 0\n"); }, "links: expected a table", "links not a table");
    rejects([&] { (void)parse_topology(head + "[nodes.a]\nrank = 0\ncolour = 1\n"); }, "nodes.a.colour: unknown key", "unknown key");
    rejects([&] { (void)parse_topology(head + "[nodes.a]\nprofile = \"strix-halo-128\"\n"); }, "nodes.a.rank: required", "rank");
    rejects([&] { (void)parse_topology(head + "[nodes.a]\nrank = 1\n"); }, "ranks are 0..0", "rank gap");
    rejects([&] { (void)parse_topology(head + "[nodes.a]\nrank = 0\n[nodes.b]\nrank = 0\n"); }, "declared twice", "rank repeat");
    // memory_gib is range-checked before llround: 1e30 GiB is not undefined behaviour, 1e-12 GiB is not 0 bytes ("no limit").
    const std::string node = head + "[nodes.a]\nrank = 0\nprofile = \"strix-halo-128\"\nmemory_gib = ";
    rejects([&] { (void)parse_topology(node + "1e30\n"); }, "nodes.a.memory_gib: out of range", "memory_gib overflow");
    rejects([&] { (void)parse_topology(node + "1e-12\n"); }, "nodes.a.memory_gib: out of range", "memory_gib rounds to 0 bytes");
    require(parse_topology(node + "0.5\n").nodes[0].memory_bytes == 536870912ULL, "fractional memory_gib still parses");
    rejects([&] { (void)parse_topology(head + "[nodes.a]\nrank = 0\nprofile = \"tpu-9\"\n"); }, "unknown profile tpu-9", "profile");
    rejects([&] { (void)parse_topology(head + "[nodes.a]\nrank = 0\nprofile = \"custom\"\n"); }, "required with profile custom", "custom");
    rejects([&] { (void)parse_topology(head + "[nodes.a]\nrank = 0\nmemory_gib = 0\n"); }, "expected a positive number", "memory");
    rejects([&] { (void)parse_topology(head + "[nodes.a]\nrank = 0\nroles = [\"host\"]\n"); }, "unknown GPU role host", "role");
    rejects([&] { (void)parse_topology(two); }, "nodes.b: no link reaches", "disconnected");
    rejects([&] { (void)parse_topology(two + "[links.x]\nbetween = [\"a\", \"a\"]\nclass = \"usb4\"\n"); }, "two different nodes", "self link");
    rejects([&] { (void)parse_topology(two + "[links.x]\nbetween = [\"a\", \"z\"]\nclass = \"usb4\"\n"); }, "unknown node z", "unknown end");
    rejects([&] { (void)parse_topology(two + "[links.x]\nbetween = [\"a\"]\nclass = \"usb4\"\n"); }, "expected two node ids", "one end");
    rejects([&] { (void)parse_topology(two + "[links.x]\nbetween = [\"a\", \"b\"]\nclass = \"usb7\"\n"); }, "unknown link class usb7", "class");
    rejects([&] { (void)parse_topology(two + "[links.x]\nbetween = [\"a\", \"b\"]\n"); }, "links.x.class: required", "no class");
    rejects([&] { (void)parse_topology(two + "[links.x]\nbetween = [\"a\", \"b\"]\nclass = \"usb4\"\n[links.y]\nbetween = [\"b\", \"a\"]\nclass = \"usb4\"\nport = \"link-x\"\n"); },
            "same port link-x", "parallel link on one port");
    const auto t = parse_topology(kTwoEqual);
    rejects([&] { (void)resolve_topology(&t, LocalIdentity{"zz", -1, "", ""}); }, "--node zz: not a declared node", "unknown node id");
    rejects([&] { (void)resolve_topology(&t, LocalIdentity{"", 2, "", ""}); }, "declares ranks 0..1", "rank beyond file");
    rejects([&] { (void)resolve_topology(nullptr, LocalIdentity{"a", -1, "", ""}); }, "no topology declared", "--node without a file");
    auto same_host = parse_topology(kTwoEqual);
    same_host.nodes[1].host = "node-a";
    rejects([&] { (void)resolve_topology(&same_host, LocalIdentity{"", -1, "", "node-a"}); }, "declared by 2 nodes", "ambiguous host");
}

void legacy() {
    // Today's pair routes: --rank R plus a transport spelling, two equal ranks, one USB4 link.
    for (const char* spelling : {"usb4auto", "usb4-auto", "usb4gpu", "usb4-gpu", "usb4:0,1"}) {
        const auto t = resolve_topology(nullptr, LocalIdentity{"", 1, spelling, ""});
        require(t.source == "legacy" && t.size() == 2 && t.local_rank == 1 && t.uniform(), "legacy pair shape");
        require(t.links.size() == 1 && t.links[0].link_class == "usb4" && t.direct_gbytes_ps(0, 1) == 4.58, "usb4 spellings");
    }
    const auto tcp = legacy_pair_topology(0, "tcp-listen:0.0.0.0:7000");
    require(tcp.local_rank == 0 && tcp.links[0].link_class.empty() && tcp.direct_gbytes_ps(0, 1) == 0, "tcp: unknown rate");
    require(topology_id(legacy_pair_topology(0, "usb4auto")) == topology_id(legacy_pair_topology(1, "usb4-gpu")), "same id both ranks");
    require(has(topology_line(tcp), "source=legacy name=pair ranks=2 local=0(rank0)"), "legacy line");
    rejects([] { (void)legacy_pair_topology(2, "usb4auto"); }, "ranks 0 and 1", "legacy rank 2");
}

void discovery() {
    auto t = resolve_topology(nullptr, LocalIdentity{"", 0, "usb4auto", ""});
    observe(t, 0, ObservedNode{"amd", "gfx1151", 120ULL * 1073741824ULL, 20});
    require(t.nodes[0].arch == "gfx1151" && t.nodes[0].memory_source == FactSource::observed && t.nodes[0].units == 20, "fills legacy facts");
    require(!t.uniform(), "one observed node, one unknown");
    auto d = parse_topology(kTwoEqual);
    rejects([&] { observe(d, 0, ObservedNode{"amd", "gfx1201", 0, 0}); }, "declared arch gfx1151", "arch mismatch");
    rejects([&] { observe(d, 1, ObservedNode{"", "", 100ULL * 1073741824ULL, 0}); }, "declares 124GiB but the node reports 100GiB", "memory above");
    observe(d, 1, ObservedNode{"", "gfx1151", 125ULL * 1073741824ULL, 0});
    require(d.nodes[1].memory_source == FactSource::declared && d.nodes[1].memory_bytes == 124ULL * 1073741824ULL, "declared stays");
    rejects([&] { observe(d, 5, {}); }, "rank 5 is not declared", "observe rank");
}

void command_line() {
    std::vector<std::string> words{"bin", "--rank", "1", "--topology", "c.toml", "--transport", "usb4-auto", "--node", "b"};
    std::vector<char*> argv;
    for (auto& w : words) argv.push_back(w.data());
    int argc = static_cast<int>(argv.size());
    const auto cli = take_topology_command_line(argc, argv.data());
    require(cli.path == "c.toml" && cli.node == "b", "flags taken");
    require(argc == 5 && std::string(argv[1]) == "--rank" && std::string(argv[4]) == "usb4-auto", "other flags kept in order");
    std::vector<std::string> bad{"bin", "--node"};
    std::vector<char*> bad_argv{bad[0].data(), bad[1].data()};
    int bad_argc = 2;
    rejects([&] { (void)take_topology_command_line(bad_argc, bad_argv.data()); }, "--node: expected a value", "flag without value");
}

} // namespace

int main() {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    try {
        single_node();
        two_equal();
        two_unequal();
        three_line();
        four_ring_dual();
        invalid();
        legacy();
        discovery();
        command_line();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
    std::cout << "device topology tests: ok\n";
    return 0;
}
