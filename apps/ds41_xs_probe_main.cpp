// strix-ds41-xs-probe: round trips of strix/ds41-expert-rpc-v1 requests against expert servers, with the sizes of the
// decode (R rows, m remote experts per row, FOLD or RAW answers) and pauses between requests (the device work of a
// layer), so the wire cost of a remote layer is measured as the parts-host will see it. Portable (Linux, Windows).
//   strix-ds41-xs-probe --connect HOST:PORT[,HOST:PORT] --token-file F [--rows R] [--remote M] [--raw] [--null]
//       [--count N] [--warmup W] [--pause-us A,B,...] [--layer L] [--busy-poll] [--json FILE]
// With two endpoints (direct mode), each request goes to both at once (each with the slots it keeps) and the round trip
// ends when both answers are in. --remote M: the first M slots (ascending id) of every row are remote, placed so the
// endpoint(s) keep them (with two endpoints: alternating halves).
#include "strix/ds41_expert_rpc.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace x = strix::ds41::xrpc;

namespace {

using Clock = std::chrono::steady_clock;

struct Args {
    std::vector<std::string> endpoints;
    std::string token_file, json;
    std::uint32_t rows = 1, remote = 1, layer = 20, count = 300, warmup = 30;
    std::vector<unsigned> pauses_us = {0, 1000, 12000};
    bool raw = false, null = false, busy_poll = false;
};

std::vector<std::string> split(const std::string& s, char c) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    for (std::string p; std::getline(ss, p, c);)
        if (!p.empty()) out.push_back(p);
    return out;
}

void pause_for(unsigned us) {
    if (us == 0U) return;
    const auto until = Clock::now() + std::chrono::microseconds(us);
    if (us > 3000U) std::this_thread::sleep_for(std::chrono::microseconds(us - 2000U));
    while (Clock::now() < until) {
    }
}

double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const auto i = static_cast<std::size_t>(p * static_cast<double>(v.size() - 1U) + 0.5);
    return v[std::min(i, v.size() - 1U)];
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args a;
        for (int i = 1; i < argc; ++i) {
            const std::string s = argv[i];
            const auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument(s + " needs a value");
                return argv[++i];
            };
            if (s == "--connect") a.endpoints = split(next(), ',');
            else if (s == "--token-file") a.token_file = next();
            else if (s == "--rows") a.rows = static_cast<std::uint32_t>(std::stoul(next()));
            else if (s == "--remote") a.remote = static_cast<std::uint32_t>(std::stoul(next()));
            else if (s == "--layer") a.layer = static_cast<std::uint32_t>(std::stoul(next()));
            else if (s == "--count") a.count = static_cast<std::uint32_t>(std::stoul(next()));
            else if (s == "--warmup") a.warmup = static_cast<std::uint32_t>(std::stoul(next()));
            else if (s == "--pause-us") {
                a.pauses_us.clear();
                for (const auto& p : split(next(), ',')) a.pauses_us.push_back(static_cast<unsigned>(std::stoul(p)));
            } else if (s == "--raw") a.raw = true;
            else if (s == "--null") a.null = true;
            else if (s == "--busy-poll") a.busy_poll = true;
            else if (s == "--json") a.json = next();
            else throw std::invalid_argument("unknown option " + s);
        }
        if (a.endpoints.empty() || a.token_file.empty()) throw std::invalid_argument("--connect and --token-file are required");
        x::Hello hello;
        hello.token = x::read_token_file(a.token_file);
        hello.client = "ds41-xs-probe";
        hello.numeric_identity = "none";
        // the geometry: take the server's (the probe has no model)
        std::vector<std::unique_ptr<x::ExpertClient>> clients;
        for (const auto& e : a.endpoints) {
            // first connect with the default geometry; a server with another geometry refuses with status 2
            clients.push_back(std::make_unique<x::ExpertClient>(e, hello, std::chrono::seconds(30), std::chrono::seconds(60), a.busy_poll));
            std::printf("endpoint %s: engine %s, experts [%u, %u), layer mask 0x%llx, world %u, numeric %s\n", e.c_str(),
                        clients.back()->ack().engine.c_str(), clients.back()->ack().residency.expert_first, clients.back()->ack().residency.expert_end,
                        static_cast<unsigned long long>(clients.back()->ack().residency.layer_mask), clients.back()->ack().world,
                        clients.back()->ack().numeric_identity.c_str());
        }
        const auto g = hello.geometry;
        if (a.remote > g.top_k || a.rows == 0U) throw std::invalid_argument("--remote <= top_k, --rows >= 1");
        std::mt19937 rng(7);
        std::FILE* json = a.json.empty() ? nullptr : std::fopen(a.json.c_str(), "w");
        if (json) std::fprintf(json, "{\"rows\":%u,\"remote\":%u,\"raw\":%s,\"null\":%s,\"busy_poll\":%s,\"endpoints\":%zu,\"runs\":[", a.rows, a.remote,
                               a.raw ? "true" : "false", a.null ? "true" : "false", a.busy_poll ? "true" : "false", clients.size());
        bool first_run = true;
        for (const auto pause : a.pauses_us) {
            std::vector<double> rtt, server, compute;
            std::uint64_t req_bytes = 0, ans_bytes = 0;
            std::vector<std::uint8_t> payload;
            for (std::uint32_t n = 0; n < a.warmup + a.count; ++n) {
                // one window: rows with top_k distinct experts; remote slots spread over the endpoints' ranges
                std::vector<x::ExpertsRequest> reqs(clients.size());
                for (std::size_t c = 0; c < clients.size(); ++c) {
                    auto& q = reqs[c];
                    q.layer = a.layer;
                    q.top_k = g.top_k;
                    q.flags = (a.raw ? x::kNoFold : 0U) | (a.null ? x::kNullCompute : 0U);
                }
                for (std::uint32_t r = 0; r < a.rows; ++r) {
                    // remote experts: M ids; endpoint of slot j = j % endpoints; the rest local (ids outside every range
                    // are fine: local slots are never computed by a server)
                    std::vector<std::uint32_t> ids;
                    std::vector<int> owner;
                    for (std::uint32_t j = 0; j < g.top_k; ++j) {
                        const bool rem = j < a.remote;
                        const auto& res = clients[j % clients.size()]->ack().residency;
                        for (;;) {
                            const std::uint32_t e = rem ? res.expert_first + static_cast<std::uint32_t>(rng() % (res.expert_end - res.expert_first))
                                                        : static_cast<std::uint32_t>(rng() % g.experts);
                            if (std::find(ids.begin(), ids.end(), e) == ids.end()) {
                                ids.push_back(e);
                                owner.push_back(rem ? static_cast<int>(j % clients.size()) : -1);
                                break;
                            }
                        }
                    }
                    for (std::size_t c = 0; c < clients.size(); ++c) {
                        std::uint32_t mask = 0;
                        for (std::uint32_t k = 0; k < g.top_k; ++k)
                            if (owner[k] == static_cast<int>(c)) mask |= 1U << k;
                        if (mask == 0U) continue;
                        auto& q = reqs[c];
                        q.ids.insert(q.ids.end(), ids.begin(), ids.end());
                        for (std::uint32_t k = 0; k < g.top_k; ++k) q.weights.push_back(0.1F + 0.01F * static_cast<float>(k));
                        q.remote.push_back(mask);
                        for (std::uint32_t i = 0; i < g.hidden; ++i) q.codes.push_back(static_cast<std::int8_t>(static_cast<int>(rng() % 255U) - 127));
                        for (std::uint32_t i = 0; i < g.groups(); ++i) q.deltas.push_back(1e-3F);
                        ++q.rows;
                    }
                }
                pause_for(pause);
                const auto t0 = Clock::now();
                std::uint64_t srv = 0, cmp = 0, out_bytes = 0, in_bytes = 0;
                for (std::size_t c = 0; c < clients.size(); ++c)
                    if (reqs[c].rows != 0U) {
                        (void)clients[c]->submit(reqs[c]);
                        out_bytes += x::request_bytes(reqs[c].rows, g.top_k, g);
                    }
                for (std::size_t c = 0; c < clients.size(); ++c)
                    if (reqs[c].rows != 0U) {
                        const auto v = clients[c]->collect(payload);
                        srv = std::max(srv, v.timing.server_ns);
                        cmp = std::max(cmp, v.timing.compute_ns);
                        in_bytes += x::result_bytes(v.items, g);
                    }
                const double us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
                if (n >= a.warmup) {
                    rtt.push_back(us);
                    server.push_back(static_cast<double>(srv) * 1e-3);
                    compute.push_back(static_cast<double>(cmp) * 1e-3);
                    req_bytes = out_bytes;
                    ans_bytes = in_bytes;
                }
            }
            std::printf("rows %u remote %u %s%s pause %6u us: request %llu B answer %llu B | round trip p50 %7.1f p90 %7.1f p99 %7.1f max %7.1f us | "
                        "server p50 %6.1f compute p50 %6.1f us\n",
                        a.rows, a.remote, a.raw ? "raw" : "fold", a.null ? " null" : "", pause, static_cast<unsigned long long>(req_bytes),
                        static_cast<unsigned long long>(ans_bytes), pct(rtt, 0.5), pct(rtt, 0.9), pct(rtt, 0.99), pct(rtt, 1.0), pct(server, 0.5),
                        pct(compute, 0.5));
            std::fflush(stdout);
            if (json) {
                std::fprintf(json, "%s{\"pause_us\":%u,\"request_bytes\":%llu,\"answer_bytes\":%llu,\"rtt_us\":{\"p50\":%.1f,\"p90\":%.1f,\"p99\":%.1f,\"max\":%.1f},"
                                   "\"server_us_p50\":%.1f,\"compute_us_p50\":%.1f,\"n\":%zu}",
                             first_run ? "" : ",", pause, static_cast<unsigned long long>(req_bytes), static_cast<unsigned long long>(ans_bytes), pct(rtt, 0.5),
                             pct(rtt, 0.9), pct(rtt, 0.99), pct(rtt, 1.0), pct(server, 0.5), pct(compute, 0.5), rtt.size());
                first_run = false;
            }
        }
        if (json) {
            std::fprintf(json, "]}\n");
            std::fclose(json);
        }
        for (auto& c : clients) c->close();
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
