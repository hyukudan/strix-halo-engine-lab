// Host tests of the shared route planner (deploy/route.hpp) with a synthetic inventory: one discrete GPU host and
// a two-node pair reached over the network. Rates are the published receipts quoted in the design notes; models
// are stand-ins with the stated seam and state sizes, not engine code.
#include "strix/deploy/route.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <crtdbg.h>
#endif

namespace {
using namespace strix::deploy;
constexpr std::uint64_t GiB = 1ULL << 30;
void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F f, const char* what) {
    bool rejected = false;
    try { (void)f(); } catch (const std::exception&) { rejected = true; }
    require(rejected, std::string("accepted invalid input: ") + what);
}
std::string replace(std::string text, const std::string& from, const std::string& to) {
    const auto at = text.find(from);
    if (at == std::string::npos) throw std::runtime_error("fixture line not found: " + from);
    return text.replace(at, from.size(), to);
}

// The one line that states the link to the pair.
const std::string kLink1g = R"(attach = { kind = "remote", link = "1gbe" })";

std::string declaration(const std::string& link_line) {
    return R"(schema = "strix-devices-v1"
machine = "synthetic"
[devices.rtx]
vendor = "nvidia"
uuid = "GPU-00000000-0000-0000-0000-000000000001"
roles = ["prefill"]
attach = { kind = "pcie", pcie_gen = 5, pcie_width = 16, h2d_gbytes_ps = 25.0 }
[devices.halo-a]
vendor = "amd"
pci = "0000:c3:00.0"
roles = ["pair", "decode", "prefill"]
)" + link_line + R"(
[devices.halo-b]
vendor = "amd"
pci = "0001:c3:00.0"
roles = ["pair", "decode", "prefill"]
)" + link_line + R"(
[links.pair]
between = ["halo-a", "halo-b"]
class = "usb4"
)";
}

DeviceFacts facts(std::uint64_t total, std::uint64_t free_bytes) {
    DeviceFacts f;
    f.mem_total_bytes = total; f.mem_free_bytes = free_bytes; f.source = "synthetic";
    return f;
}
Inventory inventory(const std::string& link_line) {
    Inventory inv;
    inv.declaration = strix::device::parse_declaration(declaration(link_line));
    inv.facts["rtx"] = facts(96 * GiB, 90 * GiB);
    inv.facts["halo-a"] = facts(117 * GiB, 110 * GiB);
    inv.facts["halo-b"] = facts(117 * GiB, 110 * GiB);
    return inv;
}

class Footprints final : public FootprintProvider {
public:
    Footprints(std::string model, std::uint64_t weights) : weights_(weights), model_(std::move(model)) {}
    std::string model() const override { return model_; }
    std::vector<std::string> topologies() const override { return {"single", "tp2", "pipeline"}; }
    std::vector<Format> formats(const std::string&) const override { return {{"q", "q", 0, 0, 0}}; }
    Footprint footprint(const Spec&, const Candidate& c, std::uint32_t) const override {
        Footprint f;
        f.weights = c.ranks > 1 && c.topology == "tp2" ? weights_ / 2 : weights_;
        f.kv_per_token = 4096; f.workspace = 2 * GiB;
        return f;
    }
    std::uint64_t weights_;
private:
    std::string model_;
};

// Stand-in model: L layers, constant seam bytes, state = a*rows + b split evenly over layers.
class Model final : public RouteProvider {
public:
    Model(std::string name, std::uint32_t layers, std::uint64_t weights, std::uint64_t seam, double state_a, double state_b,
          std::vector<Protocol> protocols)
        : protocols_(std::move(protocols)), fp_(std::move(name), weights), layers_(layers), seam_(seam), a_(state_a), b_(state_b) {}
    const FootprintProvider& footprints() const override { return fp_; }
    std::uint32_t layers() const override { return layers_; }
    std::optional<std::uint64_t> seam_bytes(std::uint32_t, std::uint64_t) const override { return seam_; }
    std::uint64_t state_bytes(std::uint64_t rows, std::uint32_t begin, std::uint32_t end) const override {
        return static_cast<std::uint64_t>((a_ * static_cast<double>(rows) + b_) * (end - begin) / layers_);
    }
    std::vector<Protocol> protocols() const override { return protocols_; }
    std::optional<Footprint> range_footprint(const Spec& s, const Candidate& c, std::uint32_t rank, std::uint32_t begin,
                                             std::uint32_t end) const override {
        auto f = fp_.footprint(s, c, rank);
        f.weights = f.weights * (end - begin) / layers_;
        return f;
    }
    std::vector<Protocol> protocols_;
private:
    Footprints fp_;
    std::uint32_t layers_;
    std::uint64_t seam_;
    double a_, b_;
};

// DS-like: RTX 2K 4.66 s / 32K 11.65 s; pair 2K 1.7 s / 32K 19.65 s (FAST); state ~894 B/row + 2.9 MB; seam 85.5 KB.
const char* kDsCosts = R"(schema = "strix-stage-costs-v1"
model = "ds-like"
[costs.rtx-2k]
unit = ["rtx"]
identity = "ds-cuda"
context = 2048
total_ms = 4660.0
spread = 0.02
[costs.rtx-32k]
unit = ["rtx"]
identity = "ds-cuda"
context = 32768
total_ms = 11650.0
spread = 0.02
[costs.pair-2k]
unit = ["halo-a", "halo-b"]
topology = "tp2"
identity = "ds-hip"
context = 2048
total_ms = 1700.0
spread = 0.03
[costs.pair-32k]
unit = ["halo-a", "halo-b"]
topology = "tp2"
identity = "ds-hip"
context = 32768
total_ms = 19650.0
spread = 0.03
)";
Model ds_model() {
    return Model("ds-like", 40, 80 * GiB, 85500, 894.0, 2.9e6, {
        {"local", "none", "ds-pair-v1", {"ds-hip"}, {}, "pair-gate"},
        {"local", "none", "ds-cuda-v1", {"ds-cuda"}, {}, "cuda-gate"},
        {"handoff", "handoff-sho1-v2/bp-v1", "ds-cuda-prefill-hip-v1", {"ds-cuda", "ds-hip"}, {}, "synthetic-gate", 300.0, 0.0, 0},
    });
}

// Qwen-like: RTX 32K 7.65 s, one node 17.7 s, pair pipeline 11.04 s; seam 25,616 B/row; state 29,952 B/row + 116.4 MB.
std::string qwen_costs() {
    std::string uniform = "[";
    for (int i = 0; i < 48; ++i) uniform += std::string(i ? ", " : "") + "1.0";
    uniform += "]";
    return R"(schema = "strix-stage-costs-v1"
model = "qwen-like"
[costs.rtx]
unit = ["rtx"]
identity = "q-cuda"
context = 32768
total_ms = 7650.0
spread = 0.03
layers = )" + uniform + R"(
[costs.node-a]
unit = ["halo-a"]
identity = "q-hip"
context = 32768
total_ms = 17700.0
spread = 0.03
[costs.node-b]
unit = ["halo-b"]
identity = "q-hip"
context = 32768
total_ms = 17700.0
spread = 0.03
[costs.pair]
unit = ["halo-a", "halo-b"]
topology = "pipeline"
identity = "q-hip-pair"
context = 32768
total_ms = 11040.0
spread = 0.03
)";
}
Model qwen_model(const std::string& layers_gate) {
    return Model("qwen-like", 48, 40 * GiB, 25616, 29952.0, 116.4e6, {
        {"local", "none", "q-pair-v1", {"q-hip-pair"}, {}, "pair-gate"},
        {"handoff", "handoff-sho1-v2/bp-v1", "q-cuda-hip-handoff-v1", {"q-cuda", "q-hip"}, {}, "handoff-gate", 200.0, 0.0, 0},
        {"layers", "pipe-crc-ack-v1", "q-cuda-hip-layer-v1", {"q-cuda", "q-hip", "q-hip"}, {2048, 4096, 8192}, layers_gate, 200.0, 0.0, 0},
    });
}

RouteRequest request(const std::string& model, std::uint64_t prompt, std::vector<std::string> decode) {
    RouteRequest r;
    r.spec.model = model;
    r.spec.context = 32768;
    r.spec.slots = 1;
    r.spec.memory.margin_gib = 4;
    r.prompt = prompt;
    r.decode = std::move(decode);
    return r;
}
const RouteOption& option(const RoutePlan& p, const std::string& prefix) {
    for (const auto& o : p.options)
        if (o.label.rfind(prefix, 0) == 0) return o;
    throw std::runtime_error("no option " + prefix + " in plan: " + p.to_json().substr(0, 400));
}
std::string chosen(const RoutePlan& p) { return p.selected() ? p.selected()->label : "<none: " + p.reason + ">"; }

void declaration_tests() {
    const auto d = strix::device::parse_declaration(declaration(kLink1g));
    require(d.links.size() == 1 && d.links[0].link == "usb4" && d.links[0].port == "link-pair", "peer link parsed");
    const auto dev = [&](const std::string& id) -> const strix::device::Descriptor& {
        for (const auto& x : d.devices) if (x.id == id) return x;
        throw std::runtime_error("no device " + id);
    };
    require(dev("halo-a").attach.link == "1gbe" && dev("halo-a").attach.port == "attach-remote", "remote attach defaults to shared port");
    require(dev("rtx").attach.port == "attach-rtx", "pcie attach is its own resource");
    rejects([] { return strix::device::parse_declaration(declaration(R"(attach = { kind = "remote", link = "3gbe" })")); }, "unknown class");
    rejects([] { return strix::device::parse_declaration(replace(declaration(kLink1g), R"(["halo-a", "halo-b"])", R"(["halo-a", "nope"])")); },
            "link to an undeclared device");
    const auto g = link_graph(d);
    const auto p = best_path(g, "rtx", "halo-b");
    require(p.path && p.path->edges.size() == 2 && std::abs(p.path->gbytes_ps - 0.118) < 1e-9, "rtx->halo-b via host NIC at 1 GbE");
    const auto unknown = link_graph(strix::device::parse_declaration(declaration(R"(attach = { kind = "remote" })")));
    const auto none = best_path(unknown, "rtx", "halo-a");
    require(!none.path && none.reason.find("no measured rate and no link class") != std::string::npos, "unknown link is not traversed");
    rejects([] { return parse_stage_costs("schema = \"strix-stage-costs-v1\"\nmodel = \"m\"\n[costs.x]\nunit = [\"a\", \"b\"]\n"
                                          "identity = \"i\"\ncontext = 1\ntotal_ms = 1.0\n"); }, "group without topology");
}

void ds_tests() {
    const auto model = ds_model();
    const auto costs = parse_stage_costs(kDsCosts);
    const auto inv = inventory(kLink1g);
    // 32K: hand-off beats the pair (prediction in the design: ~12.2 s against 19.4-19.9 s FAST).
    auto p = plan_route(request("ds-like", 32768, {"halo-a", "halo-b"}), inv, model, costs);
    const auto h = option(p, "handoff:rtx->halo-a+halo-b");
    require(chosen(p) == h.label, "ds 32K chooses the hand-off, got " + chosen(p));
    require(std::abs(h.predicted_ms - 12223) < 20, "ds 32K hand-off ~12.2 s, got " + std::to_string(h.predicted_ms));
    require(h.bytes == static_cast<std::uint64_t>(894.0 * 32768 + 2.9e6), "state bytes reported");
    require(h.transfers.size() == 1 && h.transfers[0].rate == "nominal" && h.transfers[0].edges.size() == 2, "nominal 1 GbE path via host");
    require(option(p, "local:rtx").reason.find("decode is placed on") == 0, "rtx local refused: decode is on the pair");
    // 2K: the pair's short-prompt cost wins (no extension of the long-prompt gain to short prompts).
    p = plan_route(request("ds-like", 2048, {"halo-a", "halo-b"}), inv, model, costs);
    require(chosen(p).rfind("local:halo-a+halo-b", 0) == 0, "ds 2K chooses the pair, got " + chosen(p));
    // Throughput objective: the hand-off's producer bound (2,813 rows/s) beats the pair (1,668).
    auto t = request("ds-like", 32768, {"halo-a", "halo-b"});
    t.objective = Objective::throughput;
    p = plan_route(t, inv, model, costs);
    require(chosen(p).rfind("handoff:", 0) == 0 && std::abs(p.selected()->rows_per_s - 2812.7) < 2, "ds throughput objective");
    // Unmeasured import: predicted, never selected.
    auto m2 = model;
    m2.protocols_[2].import_ms = -1;
    p = plan_route(request("ds-like", 32768, {"halo-a", "halo-b"}), inv, m2, costs);
    require(chosen(p).rfind("local:halo-a+halo-b", 0) == 0, "unmeasured import never selected");
    require(option(p, "handoff:").predicted && !option(p, "handoff:").selectable, "unmeasured import still predicted");
    // Unknown memory and memory that does not fit, through FootprintProvider + fit_rank.
    auto inv2 = inv;
    inv2.facts.erase("rtx");
    p = plan_route(request("ds-like", 32768, {"halo-a", "halo-b"}), inv2, model, costs);
    require(option(p, "handoff:").reason.rfind("memory of device rtx is not measured", 0) == 0, "unknown memory refused");
    inv2 = inv;
    inv2.facts["rtx"] = facts(96 * GiB, 60 * GiB);
    p = plan_route(request("ds-like", 32768, {"halo-a", "halo-b"}), inv2, model, costs);
    require(option(p, "handoff:").reason.find("device rtx: ") == 0 && chosen(p).rfind("local:", 0) == 0, "rtx without room refused");
    // Receipts contrast.
    p = plan_route(request("ds-like", 32768, {"halo-a", "halo-b"}), inv, model, costs);
    const auto x = contrast(p, {{h.label, 12500}, {h.label, 20000}, {"nope", 1}});
    require(x[0].found && x[0].within && !x[0].explain, "receipt within uncertainty: predicted " + std::to_string(x[0].predicted_ms) +
            " +- " + std::to_string(x[0].uncertainty_ms) + " found " + std::to_string(x[0].found));
    require(x[1].explain && std::abs(x[1].error - (20000 / h.predicted_ms - 1)) < 1e-9, "receipt miss flagged");
    require(!x[2].found, "unknown label");
    // Pin: an unselectable option is refused with its reason, never silently replaced.
    auto pin = request("ds-like", 32768, {"halo-a", "halo-b"});
    pin.pin = option(p, "local:rtx").label;
    p = plan_route(pin, inv, model, costs);
    require(p.chosen < 0 && p.reason.find("decode is placed on") != std::string::npos, "pinned unselectable option refused");
}

void one_resolver_tests() {
    const auto model = ds_model();
    const auto costs = parse_stage_costs(kDsCosts);
    const auto cli = inventory(kLink1g);
    auto serve = cli;
    serve.load["rtx"] = {}; serve.load["halo-a"] = {2, 0, false}; // idle prefill, pair decoding two streams
    const auto r = request("ds-like", 32768, {"halo-a", "halo-b"});
    const auto a = plan_route(r, cli, model, costs), b = plan_route(r, serve, model, costs);
    require(a.hash() == b.hash() && a.to_json() == b.to_json(), "CLI and server resolve the same request identically");
    require(a.to_json().find("\"schema\":\"strix-route-v1\"") == 1 && a.to_json().find("\"route_id\":\"" + a.hash()) != std::string::npos,
            "route JSON schema and id");
    serve.load["rtx"].queued = 1; // the only explained difference: live load
    const auto c = plan_route(r, serve, model, costs);
    require(option(c, "handoff:").reason.rfind("live load: device rtx has 1 queued requests", 0) == 0 && chosen(c).rfind("local:halo", 0) == 0,
            "server load changes the decision and names it");
}

void qwen_tests() {
    const auto costs = parse_stage_costs(qwen_costs());
    const auto measured10g = R"(attach = { kind = "remote", link = "10gbe", gbps_measured = 8.8, rtt_us_measured = 100 })";
    const auto measured1g = R"(attach = { kind = "remote", link = "1gbe", gbps_measured = 0.944, rtt_us_measured = 200 })";
    // Layer split not qualified (no CUDA stage adapter yet): predicted, reported, never selected.
    auto p = plan_route(request("qwen-like", 32768, {"halo-a"}), inventory(measured10g), qwen_model(""), costs);
    const auto split = option(p, "layers:rtx[0,");
    require(split.predicted && !split.selectable && split.reason.find("not qualified") == 0, "ungated split reported only");
    require(chosen(p).rfind("handoff:rtx->halo-a", 0) == 0, "qwen 10 GbE without split chooses hand-off, got " + chosen(p));
    require(std::abs(p.selected()->predicted_ms - (7650 + 1097.9e6 / 1.1e6 + 0.105 + 200)) < 5, "qwen hand-off ~8.9-9.0 s at 10 GbE");
    // With a gate the split wins on 10 GbE and loses on 1 GbE: one declaration line decides.
    p = plan_route(request("qwen-like", 32768, {"halo-a"}), inventory(measured10g), qwen_model("synthetic-gate"), costs);
    require(chosen(p).rfind("layers:rtx[0,", 0) == 0, "qwen 10 GbE gated split chosen, got " + chosen(p));
    const auto& s = *p.selected();
    require(s.stages.size() == 3 && s.stages[0].end > 20 && s.stages[0].end < 30, "balanced first cut near 57 % of work");
    require(s.predicted_ms < 7650, "split beats RTX alone on 10 GbE");
    const auto cut10 = s.stages[0].end;
    p = plan_route(request("qwen-like", 32768, {"halo-a"}), inventory(measured1g), qwen_model("synthetic-gate"), costs);
    // At 1 GbE the RTX stage becomes link-bound: the balance moves almost all work to the nodes, and the 9.3 s raw
    // state transfer keeps the full RTX hand-off out.
    const auto s1 = option(p, "layers:rtx[0,");
    require(s1.stages[0].end < cut10 / 4 && chosen(p).rfind("handoff:", 0) != 0, "1 GbE moves the cut, got " + chosen(p));
    require(option(p, "handoff:rtx->halo-a").predicted_ms > 17000, "1 GbE hand-off pays ~9.3 s of raw state");
    require(s1.reason.find("taken from rtx") != std::string::npos, "borrowed layer work is stated");
    // Class-only link: the remote prefill role needs a measured rate (start-gate rule), so the split is refused.
    p = plan_route(request("qwen-like", 32768, {"halo-a"}), inventory(R"(attach = { kind = "remote", link = "10gbe" })"),
                   qwen_model("synthetic-gate"), costs);
    require(option(p, "layers:rtx[0,").reason.find("needs a measured rate") != std::string::npos, "class-only remote prefill refused");
    require(chosen(p).rfind("handoff:", 0) == 0, "class-only link still admits the hand-off to a decode unit");
    // No per-layer work for a stage and none to borrow: the split cannot be predicted.
    auto bare = costs;
    for (auto& e : bare.entries) e.layers.clear();
    p = plan_route(request("qwen-like", 32768, {"halo-a"}), inventory(measured10g), qwen_model("synthetic-gate"), bare);
    require(option(p, "layers:").reason.find("no per-layer work") == 0, "split without layer work refused");
}
} // namespace

int main() try {
#ifdef _WIN32
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    declaration_tests();
    ds_tests();
    one_resolver_tests();
    qwen_tests();
    std::cout << "route planner tests passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "route planner test failed: " << e.what() << '\n';
    return 1;
}
