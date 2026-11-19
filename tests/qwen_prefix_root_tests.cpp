// HIP Qwen root mark (apps/qwen_cache_admission.hpp prefix_root_capture + prefix_capture_boundary_selected):
// 1. the root is session::root_mark on the T grid: the deepest multiple of T strictly inside the system + tools prefix
//    and at most n - 2;
// 2. on the qualified tiled plan (T 2048, tails 1024/256/64/16, schedule 12) every nonzero root is a full-T
//    continuation boundary with a certificate, so publishing it changes no plan, certificate or row (exhaustive sweep);
// 3. the selector with requested=false is the previous formula (exhaustive); a requested boundary is selected;
// 4. no-model replay: producer publishes the selected full-T boundaries, a sibling with the same prefix restores the
//    deepest one inside the prefix; restored rows with and without the root request.
#include "qwen_cache_admission.hpp"
#include "qwen_tiled_prompt_plan.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace strix::decode_cli;

namespace {
void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error("FAIL: " + what);
}

QwenTiledPlanConfig qualified_plan() {
    QwenTiledPlanConfig c;
    c.rows = 2048;
    c.block = {2048, QwenPlanOutput::terminal};
    c.schedule = 12;
    c.ladder = {1024, 256, 64, 16};
    for (const auto r : c.ladder) c.tails.push_back({r, QwenPlanOutput::terminal});
    return c;
}

bool previous_selected(std::uint64_t p, std::uint64_t covered, std::uint64_t tile, std::uint64_t cadence) {
    return tile && cadence && (p % cadence == 0 || p > covered || covered - p < tile);
}

// Deepest boundary a producer of n tokens publishes at or below `limit` (the sibling's shared prefix).
std::uint64_t restored_by_sibling(const QwenTiledPlanConfig& c, std::uint64_t n, std::uint64_t stable,
                                  std::uint64_t block, std::uint64_t cadence, bool root_on) {
    const auto root = root_on ? prefix_root_capture(stable, n, c.rows) : 0;
    std::uint64_t best = 0;
    for (std::uint64_t p = c.rows; p < n; p += c.rows) {
        if (!qwen_full_t_continuation(c, n, p)) continue;
        if (!prefix_capture_boundary_selected(p, n - 1, block, cadence, p == root)) continue;
        if (p <= stable) best = p;
    }
    return best;
}
}  // namespace

int main() {
    try {
        // 1. root on the T grid.
        require(prefix_root_capture(0, 9000, 2048) == 0, "no prefix, no root");
        require(prefix_root_capture(2048, 9000, 2048) == 0, "prefix of exactly T: strictly inside gives 0");
        require(prefix_root_capture(2049, 9000, 2048) == 2048, "prefix T+1");
        require(prefix_root_capture(6000, 7000, 2048) == 4096, "S 6000");
        require(prefix_root_capture(7000, 7600, 2048) == 6144, "S 7000");
        require(prefix_root_capture(7000, 6145, 2048) == 4096, "bounded by n - 2");
        for (std::uint64_t s = 0; s < 20000; s += 37)
            for (std::uint64_t n = 1; n < 20000; n += 211) {
                const auto r = prefix_root_capture(s, n, 2048);
                require(r % 2048 == 0 && (!r || (r < s && r + 2 <= n)), "root bounds");
                if (s) require(r + 2048 > std::min<std::uint64_t>(s - 1, n > 2 ? n - 2 : 0), "root is the deepest");
            }

        // 2. every nonzero root is already a certified full-T boundary of the qualified plan.
        const auto c = qualified_plan();
        std::uint64_t roots = 0;
        for (std::uint64_t n = 2; n <= 16384; n += 7)
            for (std::uint64_t s = 1; s <= n + 64; s += 509) {
                const auto r = prefix_root_capture(s, n, c.rows);
                if (!r) continue;
                ++roots;
                require(qwen_full_t_continuation(c, n, r).has_value(), "root is a full-T continuation n=" +
                        std::to_string(n) + " s=" + std::to_string(s));
                require(!qwen_full_t_certificate(c, n, r).empty(), "root has a full-T certificate");
            }
        require(roots > 1000, "sweep covered roots");

        // 3. selector: requested=false is the previous formula; requested publishes.
        for (std::uint64_t tile : {1024U, 2048U})
            for (std::uint64_t cadence : {2048U, 4096U, 6144U})
                for (std::uint64_t covered = 0; covered < 20000; covered += 113)
                    for (std::uint64_t p = 0; p < 20000; p += 128) {
                        require(prefix_capture_boundary_selected(p, covered, tile, cadence) ==
                                previous_selected(p, covered, tile, cadence), "unrequested selection unchanged");
                        require(prefix_capture_boundary_selected(p, covered, tile, cadence, true), "requested selected");
                    }
        require(!prefix_capture_boundary_selected(6144, 7599, 1024, 0, true), "no cadence: capture off");

        // 4. sibling replay (token counts only, no model): HTTP grid 1024, cadence prefix_capture_cadence(2048,0).
        const auto cadence = prefix_capture_cadence(c.rows, 0);
        require(cadence == 4096, "qualified cadence");
        std::uint64_t gain_rows = 0, cases = 0;
        for (std::uint64_t s = 2100; s <= 14000; s += 100) {
            const auto n = s + 600;  // the producer's own user turn
            const auto off = restored_by_sibling(c, n, s, 1024, cadence, false);
            const auto on = restored_by_sibling(c, n, s, 1024, cadence, true);
            require(on == std::max(off, prefix_root_capture(s, n, c.rows)), "root restore is the root mark");
            gain_rows += on - off;
            ++cases;
            if (s == 6000 || s == 7000 || s == 11000)
                std::printf("root-replay S=%llu n=%llu restored off=%llu on=%llu\n", static_cast<unsigned long long>(s),
                            static_cast<unsigned long long>(n), static_cast<unsigned long long>(off),
                            static_cast<unsigned long long>(on));
        }
        std::printf("root-replay mean gain rows=%llu over %llu prefixes\n",
                    static_cast<unsigned long long>(gain_rows / cases), static_cast<unsigned long long>(cases));
        std::printf("qwen prefix root tests PASS\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
