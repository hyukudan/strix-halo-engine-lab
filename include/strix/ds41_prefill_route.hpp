#pragma once

// Length routing of a deepseek41 prompt prefill between the decode group itself ("local": the two-node prefill on the
// ranks that decode) and a prefill-only device ("offload": e.g. the CUDA prefill, whose strix/ds41-attention-state-v1
// state is then sent to and imported by the decode group).
//
// The offload route pays a fixed price per request (expert layers that do not fit the offload device stream over its
// host link for every prompt, plus state export + wire + import), so short prompts are faster locally and long ones
// on the offload device. The deployment plan (tensormeld/hetero-plan-v1, member "prefill_route",
// tensormeld/prefill-route-v1) carries the crossover as local_max_tokens: an effective prompt length at or below it
// prefills locally, a longer one is offloaded.
//
// Effective length = prompt tokens - tokens served by a prefix-cache hit on the decode group. A hit leaves only the
// uncached suffix to prefill locally, while the offload device would redo (or first receive) the cached prefix, so a
// hit pulls a request back to the local route. Without a hit the rule is the threshold. With a hit and the plan's
// measured curves (prefill_route.curves), the two TTFTs are compared: local(suffix) + cached_s_per_token * cached
// against offload(whole prompt); without curves the threshold applies to the suffix.
//
// Off by default: a default-constructed route never offloads (every request keeps today's local path).

#include "strix/serve/json.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strix::ds41 {

enum class PrefillSite : std::uint8_t { local, offload };

[[nodiscard]] inline const char* prefill_site_name(PrefillSite s) noexcept {
    return s == PrefillSite::offload ? "offload" : "local";
}

struct PrefillRouteDecision {
    PrefillSite site = PrefillSite::local;
    std::uint64_t effective_tokens = 0;
    std::uint64_t cached_tokens = 0;
    bool by_curves = false;                 // a prefix-cache hit decided by the measured curves (else the threshold)
    double local_s = 0.0, offload_s = 0.0;  // the compared TTFTs when by_curves
};

// End-to-end TTFT (seconds) against prompt tokens, piecewise linear, end segments extended, never below 0.
struct TtftCurve {
    std::vector<std::pair<double, double>> points;  // (tokens, seconds), ascending tokens, >= 2 points

    [[nodiscard]] bool usable() const noexcept { return points.size() >= 2; }
    [[nodiscard]] double at(double tokens) const {
        if (!usable()) throw std::logic_error("ttft curve: fewer than two points");
        std::size_t i = 1;
        while (i + 1 < points.size() && points[i].first < tokens) ++i;
        const auto& a = points[i - 1];
        const auto& b = points[i];
        return std::max(0.0, a.second + (tokens - a.first) * (b.second - a.second) / (b.first - a.first));
    }
};

struct PrefillRoute {
    static constexpr std::uint64_t never = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t local_max_tokens = never;  // effective length <= this: local; never = routing off
    std::string source = "off";              // off | configured | measured | model
    TtftCurve local_curve, offload_curve;    // optional (plan's prefill_route.curves): route prefix-cache hits
    double cached_s_per_token = 0.0;         // local cost per cached token of context on a hit

    [[nodiscard]] bool enabled() const noexcept { return local_max_tokens != never; }

    [[nodiscard]] PrefillRouteDecision decide(std::uint64_t prompt_tokens, std::uint64_t cached_tokens = 0,
                                              bool offload_up = true) const {
        if (cached_tokens > prompt_tokens) throw std::invalid_argument("prefill route: cached tokens exceed the prompt");
        PrefillRouteDecision d;
        d.cached_tokens = cached_tokens;
        d.effective_tokens = prompt_tokens - cached_tokens;
        bool offload = d.effective_tokens > local_max_tokens;
        if (cached_tokens > 0 && local_curve.usable() && offload_curve.usable()) {
            d.by_curves = true;
            d.local_s = local_curve.at(static_cast<double>(std::max<std::uint64_t>(d.effective_tokens, 1))) +
                        cached_s_per_token * static_cast<double>(cached_tokens);
            d.offload_s = offload_curve.at(static_cast<double>(prompt_tokens));
            offload = d.offload_s < d.local_s;
        }
        d.site = offload_up && enabled() && offload ? PrefillSite::offload : PrefillSite::local;
        return d;
    }

    // From a plan JSON: the plan object itself ({"prefill_route": {...}}) or the route object
    // ({"schema": "tensormeld/prefill-route-v1", "local_max_tokens": N, "source": ...}). A null route = off.
    [[nodiscard]] static PrefillRoute from_plan(const serve::Json& plan) {
        const serve::Json* r = plan.find("prefill_route");
        if (r == nullptr) r = &plan;
        PrefillRoute out;
        if (r->is_null()) return out;
        if (!r->is_object()) throw std::invalid_argument("prefill route: expected an object");
        const serve::Json* schema = r->find("schema");
        if (schema == nullptr || !schema->is_string() || schema->as_string() != "tensormeld/prefill-route-v1")
            throw std::invalid_argument("prefill route: schema must be tensormeld/prefill-route-v1");
        const serve::Json* n = r->find("local_max_tokens");
        if (n == nullptr || !n->is_number() || n->as_int() < 0)
            throw std::invalid_argument("prefill route: local_max_tokens must be a non-negative integer");
        out.local_max_tokens = static_cast<std::uint64_t>(n->as_int());
        const serve::Json* src = r->find("source");
        out.source = src != nullptr && src->is_string() ? src->as_string() : "configured";
        if (const serve::Json* c = r->find("curves"); c != nullptr && c->is_object()) {
            out.local_curve = curve(c->find("local"));
            out.offload_curve = curve(c->find("offload"));
            if (const serve::Json* k = c->find("cached_s_per_token"); k != nullptr && k->is_number())
                out.cached_s_per_token = k->as_double();
            if (out.cached_s_per_token < 0.0) throw std::invalid_argument("prefill route: negative cached_s_per_token");
        }
        return out;
    }

    [[nodiscard]] static TtftCurve curve(const serve::Json* a) {
        TtftCurve c;
        if (a == nullptr || !a->is_array()) throw std::invalid_argument("prefill route: curve must be an array");
        for (const auto& p : a->as_array()) {
            if (!p.is_array() || p.as_array().size() != 2 || !p.as_array()[0].is_number() ||
                !p.as_array()[1].is_number())
                throw std::invalid_argument("prefill route: curve point must be [tokens, seconds]");
            const double n = p.as_array()[0].as_double();
            const double t = p.as_array()[1].as_double();
            if (!(n > 0.0) || !(t >= 0.0) || (!c.points.empty() && n <= c.points.back().first))
                throw std::invalid_argument("prefill route: curve points must have ascending positive tokens");
            c.points.emplace_back(n, t);
        }
        if (!c.usable()) throw std::invalid_argument("prefill route: curve needs at least two points");
        return c;
    }

    // Command-line form: "--prefill-local-max-tokens N" (configured; overrides the plan).
    [[nodiscard]] static PrefillRoute configured(std::uint64_t local_max_tokens) {
        PrefillRoute out;
        out.local_max_tokens = local_max_tokens;
        out.source = "configured";
        return out;
    }
};

}  // namespace strix::ds41
