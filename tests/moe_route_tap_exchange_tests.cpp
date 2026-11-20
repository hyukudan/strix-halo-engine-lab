// Host test of the backend binding sequence of the FAST gate v2 route tap (include/strix/moe/route_tap_exchange.hpp),
// used by the CUDA MoE routers (include/strix/cuda/core/moe_route_tap.hpp). Invariants: an inactive tap never copies;
// record-only reads and never writes back; replay overwrites the device routes with the recorded ones; the one-ulp
// null writes back only its layer's weights (ids unchanged); a mismatching call is refused before any write.
#include "strix/moe/route_tap_exchange.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}
// "Device" = host memory; counts the copies and waits the exchange issues.
struct FakeCopier {
    int to_host_calls = 0, to_device_calls = 0, waits = 0;
    void to_host(void* host, const void* device, std::size_t bytes) { std::memcpy(host, device, bytes); ++to_host_calls; }
    void to_device(void* device, const void* host, std::size_t bytes) { std::memcpy(device, host, bytes); ++to_device_calls; }
    void wait() { ++waits; }
};
std::string temp_file(const char* name) {
    return (std::filesystem::temp_directory_path() / (std::string("strix-route-exchange-") + name)).string();
}
}  // namespace

int main() {
    using strix::moe::RouteTap;
    using strix::moe::RouteTapConfig;
    using strix::moe::route_tap_exchange;
    const auto path = temp_file("ref.rte");
    const std::vector<std::uint32_t> ref_ids = {5, 2, 9, 1, 4, 8};  // 2 rows x top-3
    const std::vector<float> ref_w = {0.5F, 0.3F, 0.2F, 0.6F, 0.25F, 0.15F};
    {  // inactive: no copy at all
        RouteTap tap(RouteTapConfig{});
        FakeCopier c;
        auto ids = ref_ids;
        auto w = ref_w;
        check(!route_tap_exchange(tap, c, 0, 2, 3, 10, ids.data(), w.data()), "inactive tap reports no change");
        check(c.to_host_calls == 0 && c.to_device_calls == 0 && c.waits == 0, "inactive tap issues no copy and no wait");
    }
    {  // record: two layers; reads, never writes back
        RouteTapConfig cfg;
        cfg.out_path = path;
        RouteTap tap(cfg);
        FakeCopier c;
        auto ids = ref_ids;
        auto w = ref_w;
        check(!route_tap_exchange(tap, c, 0, 2, 3, 10, ids.data(), w.data()), "record reports no change");
        check(!route_tap_exchange(tap, c, 1, 2, 3, 10, ids.data(), w.data()), "record reports no change (layer 1)");
        check(!route_tap_exchange(tap, c, 1, 0, 3, 10, ids.data(), w.data()), "an empty call is skipped");
        check(c.to_host_calls == 4 && c.to_device_calls == 0 && c.waits == 2, "record: 2 reads + 1 wait per call, no write");
        check(ids == ref_ids && w == ref_w, "record leaves the device routes untouched");
        check(tap.receipt().find("\"records\":2") != std::string::npos, "record receipt counts 2 calls");
    }
    {  // replay: the arm's own routes differ; the device ends with the recorded ones
        RouteTapConfig cfg;
        cfg.in_path = path;
        RouteTap tap(cfg);
        FakeCopier c;
        std::vector<std::uint32_t> ids = {5, 2, 7, 1, 4, 8};
        std::vector<float> w = {0.4F, 0.4F, 0.2F, 0.6F, 0.25F, 0.15F};
        check(route_tap_exchange(tap, c, 0, 2, 3, 10, ids.data(), w.data()), "replay reports a change");
        check(ids == ref_ids && w == ref_w, "replay writes the recorded ids and weights to the device");
        check(c.to_device_calls == 2 && c.waits == 2, "replay: one read wait + one write wait");
        // a call whose shape differs from the record is refused before anything is written
        std::vector<std::uint32_t> one = {1, 2, 3};
        std::vector<float> onew = {0.2F, 0.3F, 0.5F};
        bool refused = false;
        try {
            (void)route_tap_exchange(tap, c, 1, 1, 3, 10, one.data(), onew.data());
        } catch (const std::exception&) {
            refused = true;
        }
        check(refused, "replay refuses a call with another row count than the record");
        check(one == std::vector<std::uint32_t>{1, 2, 3} && c.to_device_calls == 2, "a refused call writes nothing");
    }
    {  // null on layer 1 only: layer 0 untouched (no write), layer 1 weights move by one ulp, ids unchanged
        RouteTapConfig cfg;
        cfg.null_on = true;
        cfg.null_layer = 1;
        cfg.null_seed = 7;
        RouteTap tap(cfg);
        FakeCopier c;
        auto ids = ref_ids;
        auto w = ref_w;
        check(!route_tap_exchange(tap, c, 0, 2, 3, 10, ids.data(), w.data()), "null: another layer is not changed");
        check(c.to_device_calls == 0 && w == ref_w, "null: no write back for another layer");
        check(route_tap_exchange(tap, c, 1, 2, 3, 10, ids.data(), w.data()), "null: its layer is changed");
        check(ids == ref_ids, "null keeps the ids");
        bool all_one_ulp = true;
        for (std::size_t i = 0; i < w.size(); ++i)
            all_one_ulp = all_one_ulp && w[i] != ref_w[i] &&
                          (w[i] == std::nextafter(ref_w[i], 1.0F) || w[i] == std::nextafter(ref_w[i], 0.0F));
        check(all_one_ulp, "null moves every weight of its layer by exactly one ulp");
    }
    // Matching geometry alone does not establish that the recorded values fit this model.
    for (const bool invalid_weight : {false, true}) {
        RouteTapConfig cfg;
        cfg.out_path = path;
        auto ids = ref_ids;
        auto weights = ref_w;
        if (invalid_weight) weights[1] = std::numeric_limits<float>::quiet_NaN();
        else ids[1] = 10;
        {
            RouteTap record(cfg);
            record.on_call(0, 2, 3, ids.data(), weights.data());
        }
        cfg.out_path.clear();
        cfg.in_path = path;
        RouteTap replay(cfg);
        FakeCopier copier;
        ids = ref_ids;
        weights = ref_w;
        bool refused = false;
        try {
            (void)route_tap_exchange(replay, copier, 0, 2, 3, 10, ids.data(), weights.data());
        } catch (const std::runtime_error&) {
            refused = true;
        }
        check(refused && copier.to_device_calls == 0 && ids == ref_ids && weights == ref_w,
              "invalid recorded values are rejected before any write to the device");
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::puts("moe_route_tap_exchange_tests: PASS");
    return 0;
}
