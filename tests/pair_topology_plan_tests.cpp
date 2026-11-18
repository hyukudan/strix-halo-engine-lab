// Shared placement over N ranks from the topology declaration (topo_impl2): balance_chain (N-stage balance; the two-rank
// balance_split is its N = 2 form), the topology binding of the chain planner (chain_topology.hpp), the Qwen MoE cut
// with per-rank speed, the host mesh collectives and the facts exchange over N. Every N = 2 lab case is checked
// against a frozen copy of the code it replaced (same cut, same bits, same plan line, same wire bytes).
#include <array>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "strix/deploy/binding.hpp"
#include "strix/device/topology.hpp"
#include "strix/dist/mesh.hpp"
#include "strix/pair/balance.hpp"
#include "strix/pair/chain_topology.hpp"
#include "strix/qwen_prefill_balance.hpp"

namespace {
int failures = 0;
void require(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what.c_str()); }
}
template<class F> bool throws(F&& f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}
bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }

// ---- frozen copies of the replaced two-rank code (reference only) ----
namespace old {
using strix::pair::StageSplit;
inline StageSplit predict_split(std::span<const double> layers, double tail, std::uint32_t cut) {
    StageSplit split{cut, 0, tail};
    for (std::size_t layer = 0; layer < layers.size(); ++layer) (layer < cut ? split.a : split.b) += layers[layer];
    return split;
}
template<class Legal> StageSplit balance_split(std::span<const double> layers, double tail, Legal legal) {
    StageSplit best{};
    double score = std::numeric_limits<double>::infinity();
    for (std::uint32_t cut = 1; cut < layers.size(); ++cut) if (legal(cut)) {
        const auto candidate = predict_split(layers, tail, cut);
        const auto work = candidate.a > candidate.b ? candidate.a : candidate.b;
        if (work < score) { best = candidate; score = work; }
    }
    if (!(score < std::numeric_limits<double>::infinity())) throw std::invalid_argument("pair pipeline has no legal cut");
    return best;
}
inline StageSplit predict_split(std::span<const double> layers, std::span<const double> front, double tail,
    std::uint32_t cut, bool half) {
    auto split = predict_split(layers, tail, cut);
    if (!half) return split;
    if (cut >= layers.size() || front.size() != layers.size() || !(front[cut] <= layers[cut]))
        throw std::invalid_argument("pair pipeline half-layer cut is outside the layer costs");
    split.a += front[cut]; split.b -= front[cut]; split.half = true;
    return split;
}
template<class Legal, class LegalHalf> StageSplit balance_split(std::span<const double> layers,
    std::span<const double> front, double tail, Legal legal, LegalHalf legal_half) {
    StageSplit best{};
    double score = std::numeric_limits<double>::infinity();
    const auto offer = [&](const StageSplit& candidate) {
        const auto work = candidate.a > candidate.b ? candidate.a : candidate.b;
        if (work < score) { best = candidate; score = work; }
    };
    for (std::uint32_t cut = 1; cut < layers.size(); ++cut) {
        if (legal(cut)) offer(predict_split(layers, tail, cut));
        if (legal_half(cut)) offer(predict_split(layers, front, tail, cut, true));
    }
    if (!(score < std::numeric_limits<double>::infinity())) throw std::invalid_argument("pair pipeline has no legal cut");
    return best;
}
// The qwen35 pair app's topology glue before the shared binding (apps/hip_qwen35_pair_main.hip at f98128b72).
struct Rank { std::string name; double capacity_bytes = 0, prefill_rows_per_s = 0, read_bytes_per_ms = 0, block_fixed_ms = 0; };
struct Hop { double bytes_per_ms = 0, latency_ms = 0; };
inline void glue(const strix::device::Topology& topo, double prefill_rate, std::vector<Rank>& ranks, std::vector<Hop>& hops) {
    double max_tflops = 0.0;
    for (const auto& n : topo.nodes) max_tflops = std::max(max_tflops, n.fp16_tflops);
    for (const auto& n : topo.nodes) {
        Rank r;
        r.name = n.id;
        r.capacity_bytes = double(n.memory_bytes);
        const double base = prefill_rate > 0.0 ? prefill_rate : 1.0;
        r.prefill_rows_per_s = max_tflops > 0.0 && n.fp16_tflops > 0.0 ? base * n.fp16_tflops / max_tflops : base;
        r.read_bytes_per_ms = n.memory_gbytes_ps * 1e6;
        ranks.push_back(r);
    }
    for (std::uint32_t r = 0; r + 1U < topo.size(); ++r) {
        const auto ls = topo.links_between(r, r + 1U);
        hops.push_back({topo.direct_gbytes_ps(r, r + 1U) * 1e6, ls.front()->rtt_us / 1000.0});
    }
}
// deploy/binding.hpp exchange_pair_facts before exchange_facts: the wire record it sent.
struct Wire { std::uint64_t magic, rank, version, cu, total, free, unified; std::array<char, 64> gfx, source; };
inline Wire pair_wire(unsigned rank, const strix::deploy::DeviceFacts& local) {
    Wire mine{};
    mine.magic = 0x53544445504c3034ULL; mine.rank = rank;
    mine.version = local.gfx_target_version; mine.cu = local.cu_count;
    mine.total = local.mem_total_bytes; mine.free = local.mem_free_bytes; mine.unified = local.unified;
    std::copy(local.gfx_target.begin(), local.gfx_target.end(), mine.gfx.begin());
    std::copy(local.source.begin(), local.source.end(), mine.source.begin());
    return mine;
}
} // namespace old

// ---- 1. balance: N = 2 equal ranks bit-identical to the old two-stage search ----
void balance_two_rank_identity() {
    std::mt19937 rng(1234U);
    for (int trial = 0; trial < 4000; ++trial) {
        const auto L = static_cast<std::uint32_t>(2U + rng() % 47U);
        std::vector<double> layers(L), front(L);
        const bool integer = trial % 3 == 0;   // integer costs: many exact ties
        for (std::uint32_t l = 0; l < L; ++l) {
            layers[l] = integer ? double(1 + rng() % 4) : 0.1 + double(rng() % 100000) / 7919.0;
            front[l] = layers[l] * double(rng() % 101) / 100.0;
        }
        const double tail = integer ? double(rng() % 5) : double(rng() % 1000) / 333.0;
        std::vector<char> legal(L), legal_half(L);
        for (std::uint32_t l = 0; l < L; ++l) { legal[l] = rng() % 4 != 0; legal_half[l] = rng() % 3 == 0; }
        const auto lg = [&](std::uint32_t c) { return legal[c] != 0; };
        const auto lh = [&](std::uint32_t c) { return legal_half[c] != 0; };
        const auto cmp = [&](auto&& fresh, auto&& frozen, const char* what) {
            strix::pair::StageSplit a{}, b{};
            const bool ta = throws([&] { a = fresh(); }), tb = throws([&] { b = frozen(); });
            require(ta == tb, std::string(what) + ": same refusal");
            if (ta || tb) return;
            require(a.cut == b.cut && a.half == b.half && same_bits(a.a, b.a) && same_bits(a.b, b.b),
                    std::string(what) + ": trial " + std::to_string(trial) + " cut " + std::to_string(a.cut) + " vs " + std::to_string(b.cut));
        };
        cmp([&] { return strix::pair::balance_split(layers, tail, lg); }, [&] { return old::balance_split(layers, tail, lg); }, "whole cuts");
        cmp([&] { return strix::pair::balance_split(layers, front, tail, lg, lh); },
            [&] { return old::balance_split(layers, front, tail, lg, lh); }, "half cuts");
    }
}

// ---- 2. balance over N ranks: coverage, optimum, unequal speed ----
double brute_best(const std::vector<std::vector<double>>& costs, double tail, std::uint32_t stages) {
    const auto L = static_cast<std::uint32_t>(costs[0].size());
    double best = std::numeric_limits<double>::infinity();
    std::vector<std::uint32_t> cuts(stages + 1U, 0U);
    cuts[stages] = L;
    std::function<void(std::uint32_t)> rec = [&](std::uint32_t s) {
        if (s == stages) {
            double worst = 0.0;
            for (std::uint32_t k = 0; k < stages; ++k) {
                double w = k + 1U == stages ? tail : 0.0;
                for (auto l = cuts[k]; l < cuts[k + 1U]; ++l) w += costs[k][l];
                worst = std::max(worst, w);
            }
            best = std::min(best, worst);
            return;
        }
        for (auto c = cuts[s - 1U] + 1U; c + (stages - s) <= L; ++c) { cuts[s] = c; rec(s + 1U); }
    };
    rec(1U);
    return best;
}
void balance_n_ranks() {
    std::mt19937 rng(77U);
    for (std::uint32_t n = 1; n <= 4; ++n) {
        for (int trial = 0; trial < 60; ++trial) {
            const auto L = static_cast<std::uint32_t>(n + 1U + rng() % 14U);
            std::vector<std::vector<double>> costs(n, std::vector<double>(L));
            for (std::uint32_t l = 0; l < L; ++l) {
                const double base = 1.0 + double(rng() % 50);
                for (std::uint32_t s = 0; s < n; ++s) costs[s][l] = base * (1.0 + double(s % 2));   // odd ranks 2x slower
            }
            std::vector<std::span<const double>> spans(costs.begin(), costs.end());
            const auto r = strix::pair::balance_chain(std::span<const std::span<const double>>(spans), 3.0, [](std::uint32_t) { return true; });
            require(r.cuts.size() == n - 1U && r.work.size() == n, "N=" + std::to_string(n) + ": one cut per interior boundary");
            std::uint32_t prev = 0;
            for (const auto c : r.cuts) { require(c > prev && c < L, "N=" + std::to_string(n) + ": cuts increasing inside the model"); prev = c; }
            double worst = 0.0;
            for (const auto w : r.work) worst = std::max(worst, w);
            require(std::abs(worst - brute_best(costs, 3.0, n)) <= 1e-9 * worst, "N=" + std::to_string(n) + ": optimum equals brute force");
        }
    }
    // A 2x slower second rank gets about a third of equal layers.
    std::vector<double> fast(12, 1.0), slow(12, 2.0);
    const std::span<const double> two[2] = {fast, slow};
    const auto r = strix::pair::balance_chain(std::span<const std::span<const double>>(two), 0.0, [](std::uint32_t) { return true; });
    require(r.cuts.at(0) == 8U, "slow rank: cut 8 (8 x 1 | 4 x 2), got " + std::to_string(r.cuts.at(0)));
    require(throws([&] { (void)strix::pair::balance_chain(std::span<const std::span<const double>>(two), 0.0, [](std::uint32_t) { return false; }); }),
            "no legal cut refused");
}

// ---- 3. topology binding of the chain planner ----
std::string node(const char* id, int rank, int gib = 124, const char* extra = "") {
    return std::string("[nodes.") + id + "]\nrank = " + std::to_string(rank) + "\nhost = \"h-" + id +
           "\"\nprofile = \"strix-halo-128\"\nmemory_gib = " + std::to_string(gib) + "\n" + extra;
}
std::string link(const char* name, const char* a, const char* b, const char* cls) {
    return std::string("[links.") + name + "]\nbetween = [\"" + a + "\", \"" + b + "\"]\nclass = \"" + cls + "\"\n";
}
const std::string kHead = "schema = \"strix-topology-v1\"\nname = \"t\"\n";

namespace chain = strix::pair::chain;
chain::ModelShape model(std::uint32_t layers, double weight_gb) {
    chain::ModelShape m;
    m.layers = layers;
    m.seam_bytes_per_row = 4.0 * 5120.0;
    m.state_bytes_per_row.assign(layers, 4096.0);
    m.state_fixed_bytes.assign(layers, 1.0e6);
    m.weight_bytes.assign(layers, weight_gb * 1e9 / layers);
    m.embed_bytes = 1.0e9;
    m.head_bytes = 1.0e9;
    return m;
}
std::vector<double> shape(std::uint32_t layers) {
    std::vector<double> s(layers);
    for (std::uint32_t l = 0; l < layers; ++l) s[l] = (l + 1U) % 4U == 0U ? 16.5 : 31.0;
    return s;
}
chain::Request request(std::size_t ranks, std::uint64_t rows) {
    chain::Request q;
    q.rows = rows;
    q.blocks = {512U};
    for (std::size_t r = 0; r < ranks; ++r) q.decoders.push_back(static_cast<std::uint32_t>(r));
    q.state_rows = rows + 64U;
    q.decode_tokens = 64.0;
    q.layer_owned_state = true;
    return q;
}
struct Planned { chain::Choice choice; std::vector<chain::Device> devices; };
Planned plan(const strix::device::Topology& t, std::uint32_t layers, double weight_gb, std::uint64_t rows, double rate = 1500.0) {
    const auto ranks = chain::ranks(t, rate);
    auto m = model(layers, weight_gb);
    if (!chain::memory_declared(t)) m.weight_bytes.clear();
    Planned p;
    p.devices = chain::devices(ranks, layers, shape(layers));
    p.choice = chain::choose(p.devices, chain::links(chain::line_hops(t)), m, request(t.size(), rows));
    return p;
}
void topology_plans() {
    using strix::device::parse_topology;
    const std::uint32_t L = 64U;
    const std::uint64_t rows = 8192U;
    // Lab: 2 x 124 GiB on usb4. Byte-identical to the old app glue (weights cleared then, admission inside now).
    const auto lab = parse_topology(kHead + node("a", 0) + node("b", 1) + link("ab", "a", "b", "usb4"));
    {
        std::vector<old::Rank> oranks;
        std::vector<old::Hop> ohops;
        old::glue(lab, 1500.0, oranks, ohops);
        std::vector<chain::Device> odev;
        for (const auto& r : oranks) {
            chain::Device d;
            d.name = r.name;
            d.layer_ms_per_row = chain::scaled_costs(r.prefill_rows_per_s, L, shape(L));
            d.capacity_bytes = r.capacity_bytes;
            d.read_bytes_per_ms = r.read_bytes_per_ms;
            odev.push_back(d);
        }
        std::vector<chain::Link> olinks;
        for (const auto& h : ohops) olinks.push_back({h.bytes_per_ms, h.bytes_per_ms, h.latency_ms});
        auto om = model(L, 54.0);
        om.weight_bytes.clear();
        auto oq = request(2, rows);
        oq.layer_owned_state = false;
        const auto before = chain::choose(odev, olinks, om, oq);
        const auto after = plan(lab, L, 54.0, rows);
        const auto lb = chain::describe(before.best, odev, rows), la = chain::describe(after.choice.best, after.devices, rows);
        require(lb == la, "lab 2-node plan line identical:\n  before " + lb + "\n  after  " + la);
        require(after.choice.best.cuts == before.best.cuts && after.choice.best.decoder == before.best.decoder, "lab cuts/decoder identical");
        std::printf("lab   %s\n", la.c_str());
        const auto scale = chain::stage_cost_scale(lab);
        require(scale.size() == 2U && scale[0] == 1.0 && scale[1] == 1.0, "lab: equal ranks scale 1.0 exactly");
    }
    // N = 1: one node, the plan is that node.
    const auto one = parse_topology(kHead + node("a", 0));
    {
        const auto p = plan(one, L, 54.0, rows);
        require(p.choice.best.valid && p.choice.best.stages() == 1U, "N=1: one stage");
        std::printf("N=1   %s\n", chain::describe(p.choice.best, p.devices, rows).c_str());
    }
    // N = 3 and N = 4 on a line, model too large for fewer nodes: every node gets a contiguous range.
    for (const std::uint32_t n : {3U, 4U}) {
        std::string text = kHead;
        const char* ids[] = {"a", "b", "c", "d"};
        const char* names[] = {"ab", "bc", "cd"};
        for (std::uint32_t r = 0; r < n; ++r) text += node(ids[r], int(r), 48);
        for (std::uint32_t r = 0; r + 1U < n; ++r) text += link(names[r], ids[r], ids[r + 1U], "usb4");
        const auto t = parse_topology(text);
        const auto p = plan(t, L, 40.0 * n, rows);   // needs every node: 40 GB of layers per node at 48 GiB each
        require(p.choice.best.valid && p.choice.best.stages() == n, "N=" + std::to_string(n) + ": all ranks used, got " +
                std::to_string(p.choice.best.stages()) + " " + p.choice.best.reason);
        std::printf("N=%u   %s\n", n, chain::describe(p.choice.best, p.devices, rows).c_str());
    }
    // Unequal memory: a 32 GiB first node cannot hold half of a 120 GB model; the plan gives it fewer layers.
    {
        const auto t = parse_topology(kHead + node("small", 0, 32) + node("big", 1, 124) + link("ab", "small", "big", "usb4"));
        const auto p = plan(t, L, 140.0, rows);   // the big node alone cannot hold it
        require(p.choice.best.valid && p.choice.best.stages() == 2U, "unequal memory: two stages");
        require(p.choice.best.cuts.size() == 3U && p.choice.best.cuts[1] < L / 2U, "unequal memory: the small node gets fewer layers");
        std::printf("mem   %s\n", chain::describe(p.choice.best, p.devices, rows).c_str());
        const auto none = plan(parse_topology(kHead + node("s1", 0, 32) + node("s2", 1, 32) + link("ab", "s1", "s2", "usb4")), L, 140.0, rows);
        require(!none.choice.best.valid, "model larger than the cluster refused");
    }
    // Slow rank: one node at half the FP16 rate gets fewer layers than the fast one.
    {
        const auto t = parse_topology(kHead + node("fast", 0, 124, "fp16_tflops = 59.4\n") + node("slow", 1, 124, "fp16_tflops = 29.7\n") +
                                      link("ab", "fast", "slow", "usb4"));
        const auto scale = chain::stage_cost_scale(t);
        require(scale[0] == 1.0 && scale[1] == 2.0, "slow rank cost scale 2.0");
        const auto p = plan(t, L, 160.0, rows);   // does not fit one node: both run layers
        require(p.choice.best.valid && p.choice.best.stages() == 2U && p.choice.best.cuts[1] > L / 2U, "slow rank: the fast node takes more layers");
        std::printf("slow  %s\n", chain::describe(p.choice.best, p.devices, rows).c_str());
    }
    // Slow link: on 1 GbE a model that fits one node stays on one node.
    {
        const auto t = parse_topology(kHead + node("a", 0) + node("b", 1) + link("ab", "a", "b", "1gbe"));
        const auto p = plan(t, L, 54.0, rows, 20000.0);   // fast prefill: 1 GbE seam time exceeds the compute saved
        require(p.choice.best.valid && p.choice.best.stages() == 1U, "slow link: single node");
        std::printf("1gbe  %s\n", chain::describe(p.choice.best, p.devices, rows).c_str());
    }
    // Not a line: rank 0 - rank 2 only, refused by name.
    {
        const auto t = parse_topology(kHead + node("a", 0) + node("b", 1) + node("c", 2) + link("ac", "a", "c", "usb4") + link("bc", "b", "c", "usb4"));
        require(throws([&] { (void)chain::line_hops(t); }), "non-line topology refused");
    }
    require(chain::line_port(29650U, 0U) == 29650U && chain::line_port(29650U, 2U) == 29652U, "line port = base + lower");
}

// ---- 4. Qwen MoE cut: equal scale = today's balance; a slow second rank moves the cut up ----
void qwen_moe_cut() {
    strix::qwen_pp::LayerCosts costs;
    for (std::size_t l = 0; l < 48U; ++l) { costs.layers[l] = 10.0 + double(l % 4); costs.attention[l] = 3.0; }
    costs.has_attention = true;
    costs.lm_head = 5.0;
    strix::qwen_pp::LegalCuts legal{}, half{};
    for (std::size_t l = 1; l < 48U; ++l) { legal[l] = true; half[l] = l % 2U == 0U; }
    const auto base = strix::qwen_pp::choose_cut(costs, legal, half);
    const std::array<double, 2> equal{1.0, 1.0}, slow_b{1.0, 2.0};
    const auto same = strix::qwen_pp::choose_cut(costs, legal, half, equal);
    require(base.cut == same.cut && base.half == same.half && same_bits(base.a, same.a) && same_bits(base.b, same.b), "Qwen MoE: scale {1,1} = today");
    const auto moved = strix::qwen_pp::choose_cut(costs, legal, half, slow_b);
    require(moved.cut > base.cut, "Qwen MoE: slow rank 1 takes fewer layers (" + std::to_string(base.cut) + " -> " + std::to_string(moved.cut) + ")");
    const std::array<double, 3> three{1.0, 1.0, 2.0};
    require(throws([&] { (void)strix::qwen_pp::choose_cut(costs, legal, half, three); }), "Qwen MoE: 3 scaled ranks refused (two-party route)");
}

// ---- 5. host mesh: fake in-memory transports ----
struct Hub {
    std::mutex m;
    std::condition_variable cv;
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::deque<std::vector<std::byte>>> box;
    std::vector<std::vector<std::uint32_t>> trace;
    strix::dist::PeerExchange endpoint(std::uint32_t me) {
        return [this, me](std::uint32_t peer, std::span<const std::byte> out, std::span<std::byte> in) {
            std::unique_lock lock(m);
            trace[me].push_back(peer);
            box[{me, peer}].emplace_back(out.begin(), out.end());
            cv.notify_all();
            cv.wait(lock, [&] { return !box[{peer, me}].empty(); });
            auto msg = std::move(box[{peer, me}].front());
            box[{peer, me}].pop_front();
            if (msg.size() != in.size()) throw std::runtime_error("record size");
            std::memcpy(in.data(), msg.data(), msg.size());
        };
    }
};
void mesh_collectives() {
    std::mt19937 rng(9U);
    for (std::uint32_t n = 1; n <= 4; ++n) {
        Hub hub;
        hub.trace.resize(n);
        std::vector<std::vector<float>> parts(n, std::vector<float>(64));
        for (auto& p : parts) for (auto& v : p) v = float(int(rng() % 2000000) - 1000000) * 1.0e-3f * float(1 + rng() % 7);
        std::vector<std::vector<float>> out = parts;
        std::vector<std::vector<std::uint64_t>> gathered(n, std::vector<std::uint64_t>(n));
        std::vector<std::thread> threads;
        for (std::uint32_t r = 0; r < n; ++r)
            threads.emplace_back([&, r] {
                const std::uint64_t mine = 1000U + r;
                strix::dist::all_gather(r, n, std::as_bytes(std::span(&mine, 1)), std::as_writable_bytes(std::span(gathered[r])), hub.endpoint(r));
                strix::dist::all_reduce_ordered(r, n, out[r], hub.endpoint(r));
            });
        for (auto& t : threads) t.join();
        for (std::uint32_t r = 0; r < n; ++r) {
            for (std::uint32_t k = 0; k < n; ++k) require(gathered[r][k] == 1000U + k, "all_gather rank order");
            for (std::size_t i = 0; i < 64U; ++i) {
                float serial = parts[0][i];
                for (std::uint32_t k = 1; k < n; ++k) serial += parts[k][i];
                require(std::memcmp(&serial, &out[r][i], 4) == 0, "all_reduce_ordered = serial left fold, bit for bit");
            }
            require(hub.trace[r].size() == 2U * (n - 1U), "one exchange per peer per collective");
        }
    }
    require(throws([] { (void)strix::dist::mesh_exchange({nullptr, nullptr})(1U, {}, {}); }), "missing link refused by name");
}

// ---- 6. facts over N; the pair form sends the frozen wire bytes ----
strix::deploy::DeviceFacts facts(unsigned r) {
    strix::deploy::DeviceFacts f;
    f.gfx_target = "gfx1151"; f.source = "kfd"; f.gfx_target_version = 110501U; f.cu_count = 40U + r;
    f.mem_total_bytes = (124ULL + r) << 30; f.mem_free_bytes = 100ULL << 30; f.unified = true;
    return f;
}
void facts_exchange() {
    for (std::uint32_t n = 1; n <= 4; ++n) {
        Hub hub;
        hub.trace.resize(n);
        std::vector<std::vector<strix::deploy::DeviceFacts>> seen(n);
        std::vector<std::thread> threads;
        for (std::uint32_t r = 0; r < n; ++r)
            threads.emplace_back([&, r] { seen[r] = strix::deploy::exchange_facts(r, n, facts(r), hub.endpoint(r)); });
        for (auto& t : threads) t.join();
        for (std::uint32_t r = 0; r < n; ++r)
            for (std::uint32_t k = 0; k < n; ++k)
                require(seen[r].size() == n && seen[r][k].cu_count == 40U + k && seen[r][k].mem_total_bytes == ((124ULL + k) << 30),
                        "facts N=" + std::to_string(n) + ": every rank sees every rank");
    }
    // Pair: the bytes handed to the pair Exchange are the frozen record, and a wrong peer rank is refused.
    std::vector<std::byte> sent;
    const auto peer_wire = old::pair_wire(1U, facts(1U));
    const auto two = strix::deploy::exchange_pair_facts(0U, facts(0U), [&](std::span<const std::byte> out, std::span<std::byte> in) {
        sent.assign(out.begin(), out.end());
        std::memcpy(in.data(), &peer_wire, sizeof peer_wire);
    });
    const auto mine = old::pair_wire(0U, facts(0U));
    require(sent.size() == sizeof mine && std::memcmp(sent.data(), &mine, sizeof mine) == 0, "pair facts wire bytes unchanged");
    require(two.size() == 2U && two[1].cu_count == 41U, "pair facts decoded");
    require(throws([&] {
        (void)strix::deploy::exchange_pair_facts(0U, facts(0U), [&](std::span<const std::byte>, std::span<std::byte> in) {
            const auto wrong = old::pair_wire(0U, facts(1U));
            std::memcpy(in.data(), &wrong, sizeof wrong);
        });
    }), "pair facts: peer claiming my rank refused");
    require(throws([] { (void)strix::deploy::exchange_pair_facts(2U, facts(0U), [](auto, auto) {}); }), "pair facts: rank 2 refused");
}
} // namespace

int main() {
    try {
        balance_two_rank_identity();
        balance_n_ranks();
        topology_plans();
        qwen_moe_cut();
        mesh_collectives();
        facts_exchange();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: unexpected exception: %s\n", e.what());
        return 1;
    }
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("pair topology plan tests: PASS\n");
    return 0;
}
