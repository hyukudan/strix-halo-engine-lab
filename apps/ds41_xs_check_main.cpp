// strix-ds41-xs-check: exactness gate of expert servers. Sends windows of real-shaped requests (routes from a recorded
// routes file or random, random A8 inputs, random remote masks) to one or two endpoints and compares every answer item
// bit for bit with the CPU reference backend (the declared products of ds41_formats) on the same plan; also checks
// that the parts-host combine of the answers equals the one-device combine of the reference raw vectors.
//   strix-ds41-xs-check --connect HOST:PORT[,HOST:PORT] --token-file F --gguf FILE --layers 2,20 [--rows 1,2,4,8]
//       [--windows N] [--routes FILE] [--remote-p P] [--seed S]
// FILE: any GGUF holding the routed experts of the layers (the model, or a fixture subset); the server may run on
// another machine with another copy of the same weights.
#include "strix/ds41_expert_server.hpp"
#include "strix/ds41_formats.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace x = strix::ds41::xrpc;

namespace {

std::vector<std::string> split(const std::string& s, char c) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    for (std::string p; std::getline(ss, p, c);)
        if (!p.empty()) out.push_back(p);
    return out;
}

// p8-routes format: one line per token, items "layer:id,id,..."
std::vector<std::vector<std::uint32_t>> read_routes(const std::string& path, std::uint32_t layer, std::uint32_t top_k) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read routes " + path);
    std::vector<std::vector<std::uint32_t>> out;
    for (std::string line; std::getline(in, line);) {
        std::istringstream ls(line);
        for (std::string item; ls >> item;) {
            const auto colon = item.find(':');
            if (colon == std::string::npos || std::stoul(item.substr(0, colon)) != layer) continue;
            std::vector<std::uint32_t> ids;
            for (const auto& v : split(item.substr(colon + 1), ',')) ids.push_back(static_cast<std::uint32_t>(std::stoul(v)));
            if (ids.size() >= top_k) {
                ids.resize(top_k);
                out.push_back(ids);
            }
        }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::string> endpoints;
        std::string token_file, gguf, routes_path, layers = "20";
        std::vector<std::uint32_t> rows_list = {1, 2, 4, 8};
        std::uint32_t windows = 4, seed = 1;
        double remote_p = 0.5;
        for (int i = 1; i < argc; ++i) {
            const std::string s = argv[i];
            const auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument(s + " needs a value");
                return argv[++i];
            };
            if (s == "--connect") endpoints = split(next(), ',');
            else if (s == "--token-file") token_file = next();
            else if (s == "--gguf") gguf = next();
            else if (s == "--layers") layers = next();
            else if (s == "--routes") routes_path = next();
            else if (s == "--windows") windows = static_cast<std::uint32_t>(std::stoul(next()));
            else if (s == "--seed") seed = static_cast<std::uint32_t>(std::stoul(next()));
            else if (s == "--remote-p") remote_p = std::stod(next());
            else if (s == "--rows") {
                rows_list.clear();
                for (const auto& r : split(next(), ',')) rows_list.push_back(static_cast<std::uint32_t>(std::stoul(r)));
            } else throw std::invalid_argument("unknown option " + s);
        }
        if (endpoints.empty() || token_file.empty() || gguf.empty()) throw std::invalid_argument("--connect, --token-file and --gguf are required");
        const auto mask = x::parse_layer_mask(layers, 64U);
        x::Geometry g;
        auto ref = x::make_host_backend(gguf, mask, 0, 0, &g);
        x::Hello hello;
        hello.token = x::read_token_file(token_file);
        hello.geometry = g;
        hello.client = "ds41-xs-check";
        hello.numeric_identity = ref->numeric_identity();
        std::vector<std::unique_ptr<x::ExpertClient>> clients;
        for (const auto& e : endpoints) {
            clients.push_back(std::make_unique<x::ExpertClient>(e, hello, std::chrono::seconds(60), std::chrono::seconds(600)));
            const auto& a = clients.back()->ack();
            std::printf("endpoint %s: %s, experts [%u, %u), layers 0x%llx, world %u, numeric %s\n", e.c_str(), a.engine.c_str(), a.residency.expert_first,
                        a.residency.expert_end, static_cast<unsigned long long>(a.residency.layer_mask), a.world, a.numeric_identity.c_str());
        }
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0F, 1.0F);
        std::uint64_t items_total = 0, items_equal = 0, windows_total = 0, combine_equal = 0;
        double server_ms = 0.0;
        for (std::uint32_t layer = 0; layer < 64U; ++layer) {
            if (((mask >> layer) & 1U) == 0U) continue;
            const auto routes = routes_path.empty() ? std::vector<std::vector<std::uint32_t>>{} : read_routes(routes_path, layer, g.top_k);
            std::size_t next_route = 0;
            for (const auto rows : rows_list)
                for (std::uint32_t w = 0; w < windows; ++w) {
                    std::vector<std::uint32_t> ids;
                    for (std::uint32_t r = 0; r < rows; ++r) {
                        if (!routes.empty()) {
                            const auto& rr = routes[next_route++ % routes.size()];
                            ids.insert(ids.end(), rr.begin(), rr.end());
                            continue;
                        }
                        std::vector<std::uint32_t> pick;
                        while (pick.size() < g.top_k) {
                            const auto e = static_cast<std::uint32_t>(rng() % g.experts);
                            if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
                        }
                        ids.insert(ids.end(), pick.begin(), pick.end());
                    }
                    std::vector<float> weights(ids.size());
                    for (auto& v : weights) v = std::abs(normal(rng)) * 0.2F;
                    std::vector<std::int8_t> codes(std::size_t{rows} * g.hidden);
                    std::vector<float> deltas(std::size_t{rows} * g.groups());
                    for (std::uint32_t r = 0; r < rows; ++r) {
                        std::vector<float> v(g.hidden);
                        for (auto& f : v) f = normal(rng);
                        const auto a8 = strix::ds41::quantize_a8(v);
                        std::copy(a8.codes.begin(), a8.codes.end(), codes.begin() + std::size_t{r} * g.hidden);
                        std::copy(a8.deltas.begin(), a8.deltas.end(), deltas.begin() + std::size_t{r} * g.groups());
                    }
                    std::vector<std::uint32_t> remote(rows, 0);
                    for (std::uint32_t r = 0; r < rows; ++r)
                        for (std::uint32_t k = 0; k < g.top_k; ++k)
                            if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < remote_p) remote[r] |= 1U << k;
                    // one request per endpoint (rows with a remote slot it keeps)
                    std::vector<x::ExpertsRequest> reqs(clients.size());
                    std::vector<std::vector<std::uint32_t>> maps(clients.size());
                    std::vector<std::uint32_t> covered(rows, 0);
                    for (std::size_t c = 0; c < clients.size(); ++c) {
                        const auto& res = clients[c]->ack().residency;
                        auto& q = reqs[c];
                        q.layer = layer;
                        q.top_k = g.top_k;
                        for (std::uint32_t r = 0; r < rows; ++r) {
                            std::uint32_t m = 0;
                            for (std::uint32_t k = 0; k < g.top_k; ++k)
                                if (((remote[r] >> k) & 1U) && !((covered[r] >> k) & 1U) && res.serves(layer, ids[r * g.top_k + k])) m |= 1U << k;
                            if (m == 0U) continue;
                            covered[r] |= m;
                            maps[c].push_back(r);
                            q.ids.insert(q.ids.end(), ids.begin() + r * g.top_k, ids.begin() + (r + 1) * g.top_k);
                            q.weights.insert(q.weights.end(), weights.begin() + r * g.top_k, weights.begin() + (r + 1) * g.top_k);
                            q.remote.push_back(m);
                            q.codes.insert(q.codes.end(), codes.begin() + std::size_t{r} * g.hidden, codes.begin() + std::size_t{r + 1} * g.hidden);
                            q.deltas.insert(q.deltas.end(), deltas.begin() + std::size_t{r} * g.groups(), deltas.begin() + std::size_t{r + 1} * g.groups());
                            ++q.rows;
                        }
                        if (q.rows != 0U) (void)clients[c]->submit(q);
                    }
                    remote = covered;  // slots no endpoint keeps stay local
                    // reference raw vectors of every slot (for the one-device combine)
                    x::ExpertsRequest all;
                    all.layer = layer;
                    all.rows = rows;
                    all.top_k = g.top_k;
                    all.ids = ids;
                    all.weights = weights;
                    all.remote.assign(rows, (1U << g.top_k) - 1U);
                    all.codes = codes;
                    all.deltas = deltas;
                    all.flags = x::kNoFold;
                    const x::Residency every{0, g.experts, mask};
                    const auto all_plan = x::plan_response(all, g, every);
                    std::vector<float> raw_items(all_plan.items.size() * g.hidden), raw(std::size_t{rows} * g.top_k * g.hidden);
                    std::uint64_t ns = 0;
                    ref->compute(all, all_plan, 0, all_plan.items.size(), raw_items.data(), ns);
                    for (std::size_t i = 0; i < all_plan.items.size(); ++i)
                        std::memcpy(raw.data() + std::size_t{all_plan.items[i].slot} * g.hidden, raw_items.data() + i * g.hidden, g.vector_bytes());
                    // answers vs the reference items
                    std::vector<x::ResponsePlan> plans(clients.size());
                    std::vector<std::vector<float>> vecs(clients.size());
                    std::vector<x::Answer> answers;
                    for (std::size_t c = 0; c < clients.size(); ++c) {
                        if (reqs[c].rows == 0U) continue;
                        plans[c] = x::plan_response(reqs[c], g, clients[c]->ack().residency);
                        std::vector<std::uint8_t> payload;
                        const auto v = clients[c]->collect(payload);
                        server_ms += static_cast<double>(v.timing.compute_ns) * 1e-6;
                        vecs[c].assign(v.vectors, v.vectors + std::size_t{v.items} * g.hidden);
                        for (std::size_t i = 0; i < plans[c].items.size(); ++i) {
                            std::vector<float> want(g.hidden);
                            const auto& it = plans[c].items[i];
                            const auto slot_raw = [&](std::uint32_t s) {
                                return raw.data() + (std::size_t{maps[c][s / g.top_k]} * g.top_k + s % g.top_k) * g.hidden;
                            };
                            if (it.kind == x::PlanItem::Kind::raw) std::memcpy(want.data(), slot_raw(it.slot), g.vector_bytes());
                            else x::fold_item(plans[c], i, reqs[c].weights, g.hidden, slot_raw, want.data());
                            ++items_total;
                            if (std::memcmp(want.data(), vecs[c].data() + i * g.hidden, g.vector_bytes()) == 0) ++items_equal;
                            else std::printf("MISMATCH layer %u rows %u window %u endpoint %zu item %zu (%s, row %u)\n", layer, rows, w, c, i,
                                             it.kind == x::PlanItem::Kind::raw ? "raw" : "fold", it.row);
                        }
                        answers.push_back({&plans[c], vecs[c].data(), maps[c]});
                    }
                    // the combine of the answers against the one-device combine
                    std::vector<float> shared(std::size_t{rows} * g.hidden);
                    for (auto& f : shared) f = normal(rng);
                    const auto cp = x::plan_combine(rows, g.top_k, ids, remote, g.split, answers);
                    std::vector<float> y(shared.size()), y_ref(shared.size());
                    x::combine_host(cp, answers, weights, shared, g.hidden, [&](std::uint32_t s) { return raw.data() + std::size_t{s} * g.hidden; }, y.data());
                    const std::vector<std::uint32_t> none(rows, 0);
                    const auto cp_ref = x::plan_combine(rows, g.top_k, ids, none, g.split, {});
                    x::combine_host(cp_ref, {}, weights, shared, g.hidden, [&](std::uint32_t s) { return raw.data() + std::size_t{s} * g.hidden; }, y_ref.data());
                    ++windows_total;
                    if (std::memcmp(y.data(), y_ref.data(), y.size() * 4U) == 0) ++combine_equal;
                    else std::printf("COMBINE MISMATCH layer %u rows %u window %u\n", layer, rows, w);
                    std::fflush(stdout);
                }
        }
        for (auto& c : clients) c->close();
        const bool pass = items_equal == items_total && combine_equal == windows_total;
        std::printf("%s: %llu/%llu answer items bitwise equal to the CPU reference, %llu/%llu windows' combine equal to the one-device combine "
                    "(server compute %.1f ms total)\n",
                    pass ? "PASS" : "FAIL", static_cast<unsigned long long>(items_equal), static_cast<unsigned long long>(items_total),
                    static_cast<unsigned long long>(combine_equal), static_cast<unsigned long long>(windows_total), server_ms);
        return pass ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
