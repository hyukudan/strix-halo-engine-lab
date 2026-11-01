#pragma once
// Opt-in decode timeline (--decode-profile): the runtime records one CUDA event on
// the compute stream after each phase it enqueues and the host time spent enqueuing
// it; after the token's synchronization the event intervals are turned into device
// milliseconds per (layer, phase). This header holds the host-only part (phase names,
// the per-token JSON and the per-arm totals) so it is tested without a device.
// Events only: no kernel, copy or ordering changes, so tokens and digests are the
// same with or without the timeline (timings include the events' small cost).
#include "strix/serve/json.hpp"
#include <array>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::glm::rtxpf::producer {
using strix::serve::Json;

enum class DecodePhase : unsigned {
    embed,         // token write + embedding
    hc_pre_attn,   // hyper-connection pre-mix + input norm
    kda,           // KDA layer: open-block checkpoint replay (state/conv copies) + the chunked KDA call
    dsa,           // DSA layer: index scoring over the context + sparse attention (one call)
    hc_post_attn,  // hyper-connection expand after attention
    hc_pre_ffn,    // pre-mix + post-attention norm
    dense_ffn,     // layers 0..2: dense FFN
    router,        // MoE router (top-8 + rates)
    export_wait,   // cache export + device-side wait for the layer's units + descriptor bind
    predict,       // predicted prefetch fork (input copy + side-stream router), when on
    routed_moe,    // routed experts (CUTLASS grouped GEMMs)
    shared_expert, // shared expert
    combine,       // routed + shared add, release
    hc_post_ffn,   // hyper-connection expand after the FFN
    head,          // final mix + norm + lm_head + argmax + result copies
    host_gap,      // device idle while the host synchronizes, checks and reads the result
    mtp,           // MTP maintenance (its own export/wait and experts) + its synchronization
    count
};
inline constexpr std::array<const char*, std::size_t(DecodePhase::count)> decode_phase_names{
    "embed", "hc_pre_attn", "kda", "dsa", "hc_post_attn", "hc_pre_ffn", "dense_ffn", "router", "export_wait",
    "predict", "routed_moe", "shared_expert", "combine", "hc_post_ffn", "head", "host_gap", "mtp"};

struct TimelineMark {
    int layer = -1;  // model layer, -1 = outside the layer loop
    DecodePhase phase = DecodePhase::embed;
    double host_ms = 0;  // host time from the previous mark to this one (enqueue cost, or a synchronization)
};

// One token. gpu_ms[i] = device time from the previous event (the token's start event
// for i = 0) to mark i's event; span_ms = first to last event.
inline Json decode_timeline_json(const std::vector<TimelineMark>& marks, const std::vector<double>& gpu_ms, double span_ms) {
    if (marks.size() != gpu_ms.size()) throw std::invalid_argument("decode timeline: one device interval per mark");
    constexpr std::size_t P = std::size_t(DecodePhase::count);
    std::array<double, P> gpu{}, host{};
    std::array<unsigned, P> n{};
    std::map<int, std::array<double, P>> layers;
    double gpu_total = 0, host_total = 0;
    for (std::size_t i = 0; i < marks.size(); ++i) {
        const auto p = std::size_t(marks[i].phase);
        if (p >= P) throw std::invalid_argument("decode timeline: phase");
        gpu[p] += gpu_ms[i];
        host[p] += marks[i].host_ms;
        ++n[p];
        gpu_total += gpu_ms[i];
        host_total += marks[i].host_ms;
        if (marks[i].layer >= 0) layers[marks[i].layer][p] += gpu_ms[i];
    }
    auto phases = Json::object();
    for (std::size_t p = 0; p < P; ++p) {
        if (!n[p]) continue;
        auto o = Json::object();
        o.set("gpu_ms", gpu[p]);
        o.set("host_ms", host[p]);
        o.set("count", std::uint64_t(n[p]));
        phases.set(decode_phase_names[p], o);
    }
    auto per_layer = Json::array();
    for (const auto& [layer, row] : layers) {
        auto o = Json::object();
        o.set("layer", layer);
        for (std::size_t p = 0; p < P; ++p)
            if (row[p] != 0) o.set(decode_phase_names[p], row[p]);
        per_layer.push(o);
    }
    auto j = Json::object();
    j.set("span_ms", span_ms);
    j.set("gpu_ms_total", gpu_total);
    j.set("host_ms_total", host_total);
    j.set("phases", phases);
    j.set("layers", per_layer);
    return j;
}

// A bench arm: per phase the device/host milliseconds summed over the steps that carry
// a timeline and their per-token means, plus the mean span; empty object = none carried one.
inline Json aggregate_decode_timelines(const Json& steps) {
    constexpr std::size_t P = std::size_t(DecodePhase::count);
    std::array<double, P> gpu{}, host{};
    double span = 0;
    std::uint64_t tokens = 0;
    for (const auto& s : steps.as_array()) {
        const Json* t = s.find("timeline");
        if (!t) continue;
        ++tokens;
        span += t->find("span_ms")->as_double();
        const Json* ph = t->find("phases");
        for (std::size_t p = 0; p < P; ++p)
            if (const Json* o = ph->find(decode_phase_names[p])) {
                gpu[p] += o->find("gpu_ms")->as_double();
                host[p] += o->find("host_ms")->as_double();
            }
    }
    auto j = Json::object();
    if (!tokens) return j;
    j.set("tokens", tokens);
    j.set("span_ms_per_token", span / double(tokens));
    auto phases = Json::object();
    for (std::size_t p = 0; p < P; ++p) {
        if (gpu[p] == 0 && host[p] == 0) continue;
        auto o = Json::object();
        o.set("gpu_ms_per_token", gpu[p] / double(tokens));
        o.set("host_ms_per_token", host[p] / double(tokens));
        o.set("gpu_ms_total", gpu[p]);
        phases.set(decode_phase_names[p], o);
    }
    j.set("phases", phases);
    return j;
}
}  // namespace strix::glm::rtxpf::producer
