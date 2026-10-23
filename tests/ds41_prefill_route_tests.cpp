#include "strix/ds41_prefill_route.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}

template <class F>
bool throws(F&& f) {
    try {
        f();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

}  // namespace

int main() {
    using strix::ds41::PrefillRoute;
    using strix::ds41::PrefillSite;
    namespace serve = strix::serve;

    // off by default: everything local
    const PrefillRoute off;
    check(!off.enabled(), "default route is off");
    check(off.decide(1u << 20).site == PrefillSite::local, "off route keeps 1M tokens local");

    // threshold, prefix-cache hit, offload down
    const auto r = PrefillRoute::configured(2226);
    check(r.decide(2226).site == PrefillSite::local, "at the threshold: local");
    check(r.decide(2227).site == PrefillSite::offload, "above the threshold: offload");
    const auto hit = r.decide(8192, 7168);
    check(hit.site == PrefillSite::local && hit.effective_tokens == 1024 && hit.cached_tokens == 7168,
          "prefix-cache hit: the 1024-token suffix stays local");
    check(r.decide(8192, 1024).site == PrefillSite::offload, "short hit: the 7168-token suffix is offloaded");
    check(r.decide(32768, 0, false).site == PrefillSite::local, "offload down: local");
    check(PrefillRoute::configured(0).decide(1).site == PrefillSite::offload, "0 = always offload");
    check(throws([&] { (void)r.decide(10, 11); }), "cached > prompt rejected");

    // from a plan
    const auto plan = serve::parse_json(
        R"({"schema":"tensormeld/hetero-plan-v1","prefill_route":{"schema":"tensormeld/prefill-route-v1",)"
        R"("local":"whole:amd-pair","offload":"rtx","local_max_tokens":2226,"source":"measured"}})");
    const auto p = PrefillRoute::from_plan(plan);
    check(p.enabled() && p.local_max_tokens == 2226 && p.source == "measured", "plan route parsed");
    const auto nul = PrefillRoute::from_plan(serve::parse_json(R"({"prefill_route":null})"));
    check(!nul.enabled(), "null plan route = off");
    check(throws([] { (void)PrefillRoute::from_plan(serve::parse_json(R"({"prefill_route":{"local_max_tokens":5}})")); }),
          "schema required");
    check(throws([] {
              (void)PrefillRoute::from_plan(serve::parse_json(
                  R"({"schema":"tensormeld/prefill-route-v1","local_max_tokens":-1})"));
          }),
          "negative threshold rejected");

    // prefix-cache hits with the plan's measured curves
    const auto withc = PrefillRoute::from_plan(serve::parse_json(
        R"({"prefill_route":{"schema":"tensormeld/prefill-route-v1","local_max_tokens":2323,"source":"measured",)"
        R"("curves":{"local":[[32,0.239],[128,0.477],[512,1.336],[2048,4.192],[8000,14.046],[32000,55.446]],)"
        R"("offload":[[512,4.6],[2048,4.628],[8192,5.064],[32768,12.047]],"cached_s_per_token":6.3e-6}}})"));
    check(withc.decide(2323).site == PrefillSite::local && !withc.decide(2323).by_curves, "no hit: threshold");
    check(withc.decide(2324).site == PrefillSite::offload, "no hit: above the threshold offloads");
    const auto h1 = withc.decide(32000, 31744);
    check(h1.by_curves && h1.site == PrefillSite::local && h1.local_s < 1.2, "32000 with 31744 cached: local ~0.96 s");
    check(withc.decide(32768, 28672).site == PrefillSite::local, "4K suffix at 32K: local (offload redoes 32K)");
    check(withc.decide(32768, 16384).site == PrefillSite::offload, "16K suffix at 32K: offload");
    check(PrefillRoute::configured(2323).decide(32768, 28672).site == PrefillSite::offload,
          "without curves a hit uses the threshold on the suffix");
    check(throws([] {
              (void)PrefillRoute::from_plan(serve::parse_json(
                  R"({"schema":"tensormeld/prefill-route-v1","local_max_tokens":5,)"
                  R"("curves":{"local":[[2,1],[1,2]],"offload":[[1,1],[2,2]]}})"));
          }),
          "descending curve rejected");

    if (failures == 0) std::printf("ds41 prefill route tests: OK\n");
    else std::printf("ds41 prefill route tests: %d FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
