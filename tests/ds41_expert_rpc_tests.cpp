// strix/ds41-expert-rpc-v1: framing, messages, the answer plan, and the exactness of the split combine (any residency,
// one or two servers, compacted rows) against the one-device combine; a loopback server (direct and relay over a TCP
// peer) with a deterministic fake backend; with STRIX_DS41_EXPERTS_GGUF (a GGUF holding a layer's routed experts, e.g.
// the fixture ds41-experts-2-20.gguf) and STRIX_DS41_LAYER, the CPU reference backend against reference_window.
#include "strix/ds41_expert_rpc.hpp"
#include "strix/ds41_expert_server.hpp"
#include "strix/ds41_formats.hpp"
#include "strix/ds41_moe.hpp"
#include "strix/gguf_reader.hpp"

#include <algorithm>
#include <atomic>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>

namespace x = strix::ds41::xrpc;

namespace {

int g_failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++g_failures;
        std::printf("FAIL %s\n", what.c_str());
    }
}
template <class F>
void expect_throw(F f, const std::string& what) {
    try {
        f();
        expect(false, what + " (no exception)");
    } catch (const std::exception&) {
    }
}
bool same_bits(const float* a, const float* b, std::size_t n) { return std::memcmp(a, b, n * 4U) == 0; }

std::uint64_t mix(std::uint64_t h) {
    h ^= h >> 33U; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33U; h *= 0xc4ceb9fe1a85ec53ULL; h ^= h >> 33U;
    return h;
}
// A deterministic "raw vector" of (layer, expert, A8 row): varied magnitudes and signs, signed zeros and subnormals,
// so that a fold in a different order or with a lost -0 would show.
float fake_value(std::uint32_t layer, std::uint32_t expert, std::uint64_t row_hash, std::uint32_t o) {
    const auto h = mix(row_hash ^ (std::uint64_t{layer} << 40U) ^ (std::uint64_t{expert} << 20U) ^ o);
    switch (h % 16U) {
    case 0: return -0.0F;
    case 1: return 1e-40F;
    default: break;
    }
    const float m = static_cast<float>(static_cast<std::int64_t>((h >> 8U) % 2000001U) - 1000000) * 1e-6F;
    return std::ldexp(m, static_cast<int>((h >> 40U) % 41U) - 20);
}
std::uint64_t row_hash(const std::int8_t* codes, const float* deltas, std::uint32_t hidden) {
    std::uint64_t h = 1469598103934665603ULL;
    for (std::uint32_t i = 0; i < hidden; ++i) h = (h ^ static_cast<std::uint8_t>(codes[i])) * 1099511628211ULL;
    for (std::uint32_t g = 0; g < hidden / 32U; ++g) {
        std::uint32_t b;
        std::memcpy(&b, deltas + g, 4);
        h = (h ^ b) * 1099511628211ULL;
    }
    return h;
}

class FakeBackend final : public x::ExpertBackend {
public:
    FakeBackend(x::Geometry g, x::Residency r) : g_(g), r_(r) {}
    [[nodiscard]] x::Residency residency() const override { return r_; }
    [[nodiscard]] std::string engine() const override { return "fake"; }
    [[nodiscard]] std::string numeric_identity() const override { return "fake-v1"; }
    void compute(const x::ExpertsRequest& req, const x::ResponsePlan& plan, std::size_t first, std::size_t count, float* out,
                 std::uint64_t& ns) override {
        std::vector<std::vector<float>> raw(req.rows * req.top_k);
        const auto raw_of = [&](std::uint32_t slot) -> const float* {
            auto& v = raw[slot];
            if (v.empty()) {
                if (!r_.serves(req.layer, req.ids[slot])) throw std::logic_error("fake backend asked for an expert it does not keep");
                const auto r = slot / req.top_k;
                const auto h = row_hash(req.codes.data() + std::size_t{r} * g_.hidden, req.deltas.data() + std::size_t{r} * g_.groups(), g_.hidden);
                v.resize(g_.hidden);
                for (std::uint32_t o = 0; o < g_.hidden; ++o) v[o] = fake_value(req.layer, req.ids[slot], h, o);
            }
            return v.data();
        };
        for (std::size_t i = 0; i < count; ++i) {
            const auto& it = plan.items[first + i];
            if (it.kind == x::PlanItem::Kind::raw) std::memcpy(out + i * g_.hidden, raw_of(it.slot), g_.vector_bytes());
            else x::fold_item(plan, first + i, req.weights, g_.hidden, raw_of, out + i * g_.hidden);
        }
        ns = 1;
    }

private:
    x::Geometry g_;
    x::Residency r_;
};

struct Window {
    std::uint32_t layer = 0, rows = 0;
    std::vector<std::uint32_t> ids;
    std::vector<float> weights, shared;
    std::vector<std::int8_t> codes;
    std::vector<float> deltas;
};
Window random_window(const x::Geometry& g, std::uint32_t rows, std::mt19937& rng) {
    Window w;
    w.layer = static_cast<std::uint32_t>(rng() % g.layers);
    w.rows = rows;
    for (std::uint32_t r = 0; r < rows; ++r) {
        std::vector<std::uint32_t> pick;
        while (pick.size() < g.top_k) {
            const auto e = static_cast<std::uint32_t>(rng() % g.experts);
            if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
        }
        w.ids.insert(w.ids.end(), pick.begin(), pick.end());
    }
    std::uniform_real_distribution<float> u(-2.0F, 2.0F);
    for (std::size_t i = 0; i < w.ids.size(); ++i) w.weights.push_back(std::abs(u(rng)) * 0.3F);
    w.shared.resize(std::size_t{rows} * g.hidden);
    for (auto& v : w.shared) v = u(rng);
    w.codes.resize(std::size_t{rows} * g.hidden);
    for (auto& c : w.codes) c = static_cast<std::int8_t>(static_cast<int>(rng() % 255U) - 127);
    w.deltas.resize(std::size_t{rows} * g.groups());
    for (auto& d : w.deltas) d = std::abs(u(rng)) * 1e-2F;
    return w;
}
std::vector<float> fake_raw(const x::Geometry& g, const Window& w) {
    std::vector<float> raw(w.ids.size() * g.hidden);
    for (std::uint32_t r = 0; r < w.rows; ++r) {
        const auto h = row_hash(w.codes.data() + std::size_t{r} * g.hidden, w.deltas.data() + std::size_t{r} * g.groups(), g.hidden);
        for (std::uint32_t k = 0; k < g.top_k; ++k)
            for (std::uint32_t o = 0; o < g.hidden; ++o)
                raw[(std::size_t{r} * g.top_k + k) * g.hidden + o] = fake_value(w.layer, w.ids[r * g.top_k + k], h, o);
    }
    return raw;
}
// the one-device combine (reference_window's)
std::vector<float> reference_y(const x::Geometry& g, const Window& w, const std::vector<float>& raw) {
    const auto routes = strix::ds41::build_window_routes(w.ids, w.rows, g.top_k, g.split);
    std::vector<float> y(std::size_t{w.rows} * g.hidden);
    for (std::uint32_t r = 0; r < w.rows; ++r)
        for (std::uint32_t o = 0; o < g.hidden; ++o) {
            float low = 0.0F, high = 0.0F;
            for (std::uint32_t k = 0; k < g.top_k; ++k) {
                const auto slot = routes.order[r * g.top_k + k];
                const float term = w.weights[slot] * raw[slot * g.hidden + o];
                if (k < routes.low_counts[r]) low = low + term;
                else high = high + term;
            }
            y[std::size_t{r} * g.hidden + o] = (w.shared[std::size_t{r} * g.hidden + o] + low) + high;
        }
    return y;
}
// the request for one server: rows with a remote slot it keeps (compacted), and the row map
x::ExpertsRequest request_for(const x::Geometry& g, const Window& w, const std::vector<std::uint32_t>& remote, const x::Residency& res,
                              std::vector<std::uint32_t>& row_map, std::uint32_t flags = 0) {
    x::ExpertsRequest q;
    q.layer = w.layer;
    q.top_k = g.top_k;
    q.flags = flags;
    row_map.clear();
    for (std::uint32_t r = 0; r < w.rows; ++r) {
        std::uint32_t mask = 0;
        for (std::uint32_t k = 0; k < g.top_k; ++k)
            if (((remote[r] >> k) & 1U) && res.serves(w.layer, w.ids[r * g.top_k + k])) mask |= 1U << k;
        if (mask == 0U) continue;
        row_map.push_back(r);
        q.ids.insert(q.ids.end(), w.ids.begin() + r * g.top_k, w.ids.begin() + (r + 1) * g.top_k);
        q.weights.insert(q.weights.end(), w.weights.begin() + r * g.top_k, w.weights.begin() + (r + 1) * g.top_k);
        q.remote.push_back(mask);
        q.codes.insert(q.codes.end(), w.codes.begin() + std::size_t{r} * g.hidden, w.codes.begin() + std::size_t{r + 1} * g.hidden);
        q.deltas.insert(q.deltas.end(), w.deltas.begin() + std::size_t{r} * g.groups(), w.deltas.begin() + std::size_t{r + 1} * g.groups());
    }
    q.rows = static_cast<std::uint32_t>(row_map.size());
    return q;
}

void test_header() {
    x::Header h;
    h.type = x::MsgType::experts;
    h.request = 77;
    h.bytes = 5000;
    h.flags = 3;
    auto b = x::encode_header(h);
    const auto d = x::decode_header(b);
    expect(d.type == x::MsgType::experts && d.request == 77 && d.bytes == 5000 && d.flags == 3, "header round trip");
    b[9] ^= 1U;
    expect_throw([&] { (void)x::decode_header(b); }, "header check");
    auto c = x::encode_header(h);
    c[0] = 0;
    expect_throw([&] { (void)x::decode_header(c); }, "header magic");
    h.bytes = x::kMaxMessageBytes + 1U;
    expect_throw([&] { (void)x::decode_header(x::encode_header(h)); }, "header size limit");
}

void test_messages() {
    x::Geometry g;
    x::Hello hello;
    hello.token[3] = 9;
    hello.model_identity = 0x1234;
    hello.client = "cuda-parts-host";
    hello.numeric_identity = "ds41-exact-v1";
    const auto h2 = x::decode_hello(x::encode(hello));
    expect(h2.token == hello.token && h2.model_identity == 0x1234 && h2.geometry == g && h2.client == hello.client, "hello round trip");
    x::HelloAck ack;
    ack.residency = {0, 192, 0xffULL};
    ack.world = 2;
    ack.engine = "hip";
    const auto a2 = x::decode_hello_ack(x::encode(ack));
    expect(a2.residency.expert_end == 192 && a2.residency.layer_mask == 0xff && a2.world == 2 && a2.engine == "hip", "hello_ack round trip");
    std::mt19937 rng(1);
    const auto w = random_window(g, 3, rng);
    std::vector<std::uint32_t> remote(3, 0x3fU), map;
    const auto q = request_for(g, w, remote, {0, 384, ~0ULL}, map);
    const auto bytes = x::encode(q, g);
    expect(bytes.size() + x::kHeaderBytes == x::request_bytes(3, 6, g), "request size formula");
    expect(x::request_bytes(1, 6, g) == 32U + 16U + 52U + 5760U, "R1 request = 5860 B");
    const auto q2 = x::decode_experts(bytes, g);
    expect(q2.ids == q.ids && q2.remote == q.remote && q2.codes == q.codes && q2.deltas == q.deltas && same_bits(q2.weights.data(), q.weights.data(), q.weights.size()),
           "experts round trip");
    auto bad = bytes;
    bad.pop_back();
    expect_throw([&] { (void)x::decode_experts(bad, g); }, "experts truncated");
    x::ResultTiming t{5, 6, 7};
    auto head = x::encode_result_head(4, 2, g.hidden, t);
    head.resize(head.size() + 2U * g.vector_bytes());
    const auto v = x::view_result(head);
    expect(v.layer == 4 && v.items == 2 && v.timing.compute_ns == 6 && head.size() + x::kHeaderBytes == x::result_bytes(2, g), "result head");
}

void test_plan_cases() {
    x::Geometry g;
    x::ExpertsRequest q;
    q.layer = 3;
    q.rows = 1;
    q.top_k = 6;
    q.ids = {5, 300, 10, 200, 190, 250};  // ascending: 5 10 190 | 200 250 300 = slots 0 2 4 | 3 5 1
    q.weights.assign(6, 0.5F);
    const x::Residency all{0, 384, ~0ULL};
    q.remote = {(1U << 2) | (1U << 4) | (1U << 5) | (1U << 1)};  // 10, 190, 250, 300 remote
    auto p = x::plan_response(q, g, all);
    expect(p.items.size() == 4 && p.items[0].kind == x::PlanItem::Kind::raw && p.items[0].slot == 2 && p.items[1].slot == 4 &&
               p.items[2].slot == 5 && p.items[3].slot == 1 && p.low[0] == 3,
           "plan: interior remote slots travel raw");
    q.remote = {(1U << 0) | (1U << 2) | (1U << 3)};  // 5, 10 (low prefix), 200 (high prefix)
    p = x::plan_response(q, g, all);
    expect(p.items.size() == 2 && p.items[0].kind == x::PlanItem::Kind::fold && p.items[0].count == 2 && p.items[1].kind == x::PlanItem::Kind::fold &&
               p.items[1].count == 1 && p.items[1].half == 1,
           "plan: leading remote slots fold");
    expect(p.entry_slots == std::vector<std::uint32_t>({0, 2, 3}), "plan: entries in ascending expert");
    q.flags = x::kNoFold;
    p = x::plan_response(q, g, all);
    expect(p.items.size() == 3 && p.items[0].kind == x::PlanItem::Kind::raw, "plan: kNoFold");
    q.flags = 0;
    q.remote = {0x3fU};
    p = x::plan_response(q, g, all);
    expect(p.items.size() == 2 && p.items[0].count == 3 && p.items[1].count == 3, "plan: all remote = two half folds");
    expect_throw([&] { (void)x::plan_response(q, g, x::Residency{0, 192, ~0ULL}); }, "plan: slot outside the residency");
    expect_throw([&] { (void)x::plan_response(q, g, x::Residency{0, 384, 0x1ULL}); }, "plan: layer outside the residency");
    q.ids = {5, 5, 10, 200, 190, 250};
    expect_throw([&] { (void)x::plan_response(q, g, all); }, "plan: repeated expert");
}

// Split combine == one-device combine, bit for bit, over random residencies, one or two servers, compacted rows.
void test_split_combine(std::mt19937& rng) {
    x::Geometry g;
    g.hidden = 256;
    int cases = 0;
    for (int it = 0; it < 400; ++it) {
        const auto rows = static_cast<std::uint32_t>(1U + rng() % 9U);
        const auto w = random_window(g, rows, rng);
        const auto raw = fake_raw(g, w);
        const auto ref = reference_y(g, w, raw);
        // residency: each expert remote with probability p (per test), whole halves sometimes
        const double p = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
        std::vector<std::uint32_t> remote(rows, 0);
        for (std::uint32_t r = 0; r < rows; ++r)
            for (std::uint32_t k = 0; k < g.top_k; ++k)
                if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < p) remote[r] |= 1U << k;
        const bool two = (it % 2) == 0;
        const std::uint32_t flags = (it % 7) == 3 ? x::kNoFold : 0U;
        std::vector<x::Residency> servers = two ? std::vector<x::Residency>{{0, g.split, ~0ULL}, {g.split, g.experts, ~0ULL}}
                                                : std::vector<x::Residency>{{0, g.experts, ~0ULL}};
        std::vector<x::ExpertsRequest> reqs(servers.size());
        std::vector<x::ResponsePlan> plans(servers.size());
        std::vector<std::vector<std::uint32_t>> maps(servers.size());
        std::vector<std::vector<float>> answers(servers.size());
        std::vector<x::Answer> ans;
        for (std::size_t s = 0; s < servers.size(); ++s) {
            reqs[s] = request_for(g, w, remote, servers[s], maps[s], flags);
            if (reqs[s].rows == 0U) continue;
            plans[s] = x::plan_response(reqs[s], g, servers[s]);
            FakeBackend be(g, servers[s]);
            answers[s].resize(plans[s].items.size() * g.hidden);
            std::uint64_t ns = 0;
            be.compute(reqs[s], plans[s], 0, plans[s].items.size(), answers[s].data(), ns);
            ans.push_back({&plans[s], answers[s].data(), maps[s]});
        }
        const auto cp = x::plan_combine(rows, g.top_k, w.ids, remote, g.split, ans);
        std::vector<float> y(std::size_t{rows} * g.hidden);
        x::combine_host(cp, ans, w.weights, w.shared, g.hidden, [&](std::uint32_t slot) { return raw.data() + std::size_t{slot} * g.hidden; }, y.data());
        if (!same_bits(y.data(), ref.data(), y.size())) {
            expect(false, "split combine differs from the one-device combine (case " + std::to_string(it) + ")");
            break;
        }
        ++cases;
    }
    std::printf("split combine: %d random windows bitwise equal to the one-device combine\n", cases);
    // a missing answer, and an answer for a local slot, are refused
    const auto w = random_window(g, 2, rng);
    std::vector<std::uint32_t> remote = {0x3U, 0x0U}, map;
    const x::Residency all{0, g.experts, ~0ULL};
    auto q = request_for(g, w, remote, all, map);
    const auto plan = x::plan_response(q, g, all);
    std::vector<float> vec(plan.items.size() * g.hidden);
    const x::Answer a{&plan, vec.data(), map};
    std::vector<std::uint32_t> more = {0x7U, 0x0U}, less = {0x1U, 0x0U};
    expect_throw([&] { (void)x::plan_combine(2, g.top_k, w.ids, more, g.split, std::span<const x::Answer>(&a, 1)); }, "combine plan: missing answer");
    expect_throw([&] { (void)x::plan_combine(2, g.top_k, w.ids, less, g.split, std::span<const x::Answer>(&a, 1)); }, "combine plan: extra answer");
}

std::string write_token(const std::string& path, char fill) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    const std::string hex(64, fill);
    std::fwrite(hex.data(), 1, hex.size(), f);
    std::fclose(f);
    return path;
}

// One window through real sockets: direct (two servers) and relay (rank 0 + rank 1 over a TCP peer).
void test_loopback(std::mt19937& rng) {
    x::Geometry g;  // the model's geometry (5120 hidden): real message sizes
    const auto token_path = write_token("ds41-xrpc-test-token.txt", 'a');
    const auto token = x::read_token_file(token_path);
    x::Hello hello;
    hello.token = token;
    hello.geometry = g;
    hello.client = "test";
    for (const bool relay : {false, true}) {
        std::vector<std::unique_ptr<FakeBackend>> backends;
        std::vector<std::thread> threads;
        std::vector<std::uint16_t> ports;
        std::unique_ptr<x::PeerLink> peer0, peer1;
        std::atomic<std::uint16_t> port_a{0}, port_b{0};
        if (relay) {
            backends.push_back(std::make_unique<FakeBackend>(g, x::Residency{0, g.split, ~0ULL}));
            backends.push_back(std::make_unique<FakeBackend>(g, x::Residency{g.split, g.experts, ~0ULL}));
            // peer link: rank 1 listens, rank 0 connects
            std::thread t1([&] { peer1 = x::tcp_peer_listen("127.0.0.1:47811", std::chrono::seconds(20)); });
            peer0 = x::tcp_peer_connect("127.0.0.1:47811", std::chrono::seconds(20));
            t1.join();
            threads.emplace_back([&] { x::follow(g, x::Residency{0, g.experts, ~0ULL}, *backends[1], *peer1); });
            x::ServerOptions o;
            o.listen = "127.0.0.1:0";
            o.token = token;
            o.geometry = g;
            o.once = true;
            o.peer_residency = {g.split, g.experts, ~0ULL};
            o.on_listen = [&](std::uint16_t p) { port_a = p; };
            threads.emplace_back([&, o] { (void)x::serve(o, *backends[0], peer0.get()); });
        } else {
            backends.push_back(std::make_unique<FakeBackend>(g, x::Residency{0, g.split, ~0ULL}));
            backends.push_back(std::make_unique<FakeBackend>(g, x::Residency{g.split, g.experts, ~0ULL}));
            for (int s = 0; s < 2; ++s) {
                x::ServerOptions o;
                o.listen = "127.0.0.1:0";
                o.token = token;
                o.geometry = g;
                o.once = true;
                o.busy_poll = s == 1;
                auto* slot = s == 0 ? &port_a : &port_b;
                o.on_listen = [slot](std::uint16_t p) { *slot = p; };
                threads.emplace_back([&, o, s] { (void)x::serve(o, *backends[s], nullptr); });
            }
        }
        while (port_a == 0 || (!relay && port_b == 0)) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        std::vector<std::unique_ptr<x::ExpertClient>> clients;
        clients.push_back(std::make_unique<x::ExpertClient>("127.0.0.1:" + std::to_string(port_a.load()), hello, std::chrono::seconds(10),
                                                            std::chrono::seconds(30), relay));
        if (!relay)
            clients.push_back(std::make_unique<x::ExpertClient>("127.0.0.1:" + std::to_string(port_b.load()), hello, std::chrono::seconds(10),
                                                                std::chrono::seconds(30), true));
        expect(clients[0]->ack().world == (relay ? 2U : 1U) && clients[0]->ack().residency.expert_end == (relay ? g.experts : g.split),
               std::string(relay ? "relay" : "direct") + ": hello ack residency");
        expect(clients[0]->ping().find("\"status\":\"ok\"") != std::string::npos, "ping");
        int windows = 0;
        // pipelined: submit three windows to every server, then collect
        std::vector<Window> ws;
        std::vector<std::vector<std::uint32_t>> remotes;
        for (const std::uint32_t rows : {1U, 3U, 8U}) {
            ws.push_back(random_window(g, rows, rng));
            std::vector<std::uint32_t> remote(rows);
            for (auto& m : remote) m = static_cast<std::uint32_t>(rng() & 0x3fU);
            remote[0] |= 0x21U;
            remotes.push_back(remote);
        }
        std::vector<std::vector<x::ExpertsRequest>> reqs(ws.size());
        std::vector<std::vector<std::vector<std::uint32_t>>> maps(ws.size());
        for (std::size_t i = 0; i < ws.size(); ++i)
            for (std::size_t c = 0; c < clients.size(); ++c) {
                maps[i].emplace_back();
                const auto res = clients[c]->ack().residency;
                reqs[i].push_back(request_for(g, ws[i], remotes[i], res, maps[i].back()));
                if (reqs[i][c].rows != 0U) (void)clients[c]->submit(reqs[i][c]);
            }
        for (std::size_t i = 0; i < ws.size(); ++i) {
            std::vector<x::ResponsePlan> plans(clients.size());
            std::vector<std::vector<std::uint8_t>> payloads(clients.size());
            std::vector<std::vector<float>> vecs(clients.size());
            std::vector<x::Answer> ans;
            for (std::size_t c = 0; c < clients.size(); ++c) {
                if (reqs[i][c].rows == 0U) continue;
                plans[c] = x::plan_response(reqs[i][c], g, clients[c]->ack().residency);
                const auto v = clients[c]->collect(payloads[c]);
                expect(v.items == plans[c].items.size() && v.layer == ws[i].layer, "answer size = plan");
                vecs[c].assign(v.vectors, v.vectors + std::size_t{v.items} * g.hidden);
                ans.push_back({&plans[c], vecs[c].data(), maps[i][c]});
            }
            const auto raw = fake_raw(g, ws[i]);
            const auto ref = reference_y(g, ws[i], raw);
            const auto cp = x::plan_combine(ws[i].rows, g.top_k, ws[i].ids, remotes[i], g.split, ans);
            std::vector<float> y(ref.size());
            x::combine_host(cp, ans, ws[i].weights, ws[i].shared, g.hidden, [&](std::uint32_t s) { return raw.data() + std::size_t{s} * g.hidden; },
                            y.data());
            expect(same_bits(y.data(), ref.data(), y.size()), std::string(relay ? "relay" : "direct") + ": loopback combine bitwise");
            ++windows;
        }
        // null compute: zeros of the planned size
        {
            std::vector<std::uint32_t> map;
            auto q = request_for(g, ws[2], remotes[2], clients[0]->ack().residency, map, x::kNullCompute);
            if (q.rows != 0U) {
                (void)clients[0]->submit(q);
                std::vector<std::uint8_t> pl;
                const auto v = clients[0]->collect(pl);
                expect(v.items == x::plan_response(q, g, clients[0]->ack().residency).items.size(), "null compute answer size");
            }
        }
        for (auto& c : clients) c->close();
        for (auto& t : threads) t.join();
        std::printf("%s: %d windows over sockets bitwise equal\n", relay ? "relay (rank 0 + rank 1 over a TCP peer)" : "direct (two servers)", windows);
    }
    // a wrong token is refused
    {
        std::atomic<std::uint16_t> port{0};
        FakeBackend be(g, {0, g.experts, ~0ULL});
        x::ServerOptions o;
        o.listen = "127.0.0.1:0";
        o.token = token;
        o.geometry = g;
        o.once = true;
        o.on_listen = [&](std::uint16_t p) { port = p; };
        std::thread t([&] { (void)x::serve(o, be, nullptr); });
        while (port == 0) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        x::Hello bad = hello;
        bad.token[0] ^= 1U;
        expect_throw([&] { x::ExpertClient c("127.0.0.1:" + std::to_string(port.load()), bad, std::chrono::seconds(5), std::chrono::seconds(5)); },
                     "wrong token refused");
        t.join();
    }
    std::remove(token_path.c_str());
}

// The CPU reference backend against reference_window on a real layer (all six slots remote, one server).
void test_host_backend(const char* path, std::uint32_t layer, std::mt19937& rng) {
    x::Geometry g;
    auto be = x::make_host_backend(path, 1ULL << layer, 0, 0, &g);
    for (const std::uint32_t rows : {1U, 2U}) {
        auto w = random_window(g, rows, rng);
        w.layer = layer;
        // A8 input from a real quantization of random values
        std::normal_distribution<float> n(0.0F, 1.0F);
        std::vector<strix::ds41::A8Activation> xs;
        for (std::uint32_t r = 0; r < rows; ++r) {
            std::vector<float> v(g.hidden);
            for (auto& f : v) f = n(rng);
            xs.push_back(strix::ds41::quantize_a8(v));
            std::copy(xs.back().codes.begin(), xs.back().codes.end(), w.codes.begin() + std::size_t{r} * g.hidden);
            std::copy(xs.back().deltas.begin(), xs.back().deltas.end(), w.deltas.begin() + std::size_t{r} * g.groups());
        }
        std::vector<std::uint32_t> remote(rows, (1U << g.top_k) - 1U), map;
        // half the rows: only some slots remote (the rest from the reference's raw, "local")
        if (rows > 1U) remote[1] = 0x2aU;
        const x::Residency all{0, g.experts, 1ULL << layer};
        const auto q = request_for(g, w, remote, all, map);
        const auto plan = x::plan_response(q, g, all);
        std::vector<float> out(plan.items.size() * g.hidden);
        std::uint64_t ns = 0;
        be->compute(q, plan, 0, plan.items.size(), out.data(), ns);
        // reference: every expert of the window
        std::ifstream in(path, std::ios::binary);
        const auto file = strix::gguf::read_file(path);
        std::vector<std::uint8_t> gate, up, down;
        for (const auto& t : file.tensors) {
            const std::string p = "blk." + std::to_string(layer) + ".";
            auto load = [&](std::vector<std::uint8_t>& dst) {
                dst.resize(t.byte_size);
                in.seekg(static_cast<std::streamoff>(t.absolute_offset));
                in.read(reinterpret_cast<char*>(dst.data()), static_cast<std::streamsize>(dst.size()));
            };
            if (t.name == p + "ffn_gate_exps.weight") load(gate);
            else if (t.name == p + "ffn_up_exps.weight") load(up);
            else if (t.name == p + "ffn_down_exps.weight") load(down);
        }
        strix::ds41::ExpertTensors et;
        et.gate = gate; et.up = up; et.down = down;
        et.hidden = g.hidden; et.inter = g.inter;
        if (const auto* m = file.find_metadata("deepseek41.swiglu_limit"); m != nullptr && m->floating_value) et.limit = static_cast<float>(*m->floating_value);
        const auto routes = strix::ds41::build_window_routes(w.ids, rows, g.top_k, g.split);
        const auto ref = strix::ds41::reference_window(et, xs, routes, w.weights, w.shared, g.top_k);
        const x::Answer a{&plan, out.data(), map};
        const auto cp = x::plan_combine(rows, g.top_k, w.ids, remote, g.split, std::span<const x::Answer>(&a, 1));
        std::vector<float> y(ref.y.size());
        x::combine_host(cp, std::span<const x::Answer>(&a, 1), w.weights, w.shared, g.hidden,
                        [&](std::uint32_t s) { return ref.raw.data() + std::size_t{s} * g.hidden; }, y.data());
        expect(same_bits(y.data(), ref.y.data(), y.size()), "host backend on real layer " + std::to_string(layer) + ": combine == reference_window");
        std::printf("host backend, real layer %u, %u rows: %zu items, %.2f s, combine %s reference_window\n", layer, rows, plan.items.size(),
                    static_cast<double>(ns) * 1e-9, same_bits(y.data(), ref.y.data(), y.size()) ? "==" : "!=");
    }
}

}  // namespace

int main() {
    try {
        std::mt19937 rng(20261020);
        test_header();
        test_messages();
        test_plan_cases();
        test_split_combine(rng);
        test_loopback(rng);
        if (const char* path = std::getenv("STRIX_DS41_EXPERTS_GGUF")) {
            const char* layer = std::getenv("STRIX_DS41_LAYER");
            test_host_backend(path, layer ? static_cast<std::uint32_t>(std::atoi(layer)) : 20U, rng);
        }
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    std::printf(g_failures ? "%d FAILURES\n" : "all passed\n", g_failures);
    return g_failures ? 1 : 0;
}
