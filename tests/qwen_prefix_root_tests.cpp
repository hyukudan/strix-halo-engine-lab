// HIP Qwen root mark (include/strix/qwen/cache_admission.hpp prefix_root_capture + prefix_capture_boundary_selected):
// 1. the root is session::root_mark on the T grid: the deepest multiple of T strictly inside the system + tools prefix
//    and at most n - 2;
// 2. on the qualified tiled plan (T 2048, tails 1024/256/64/16, schedule 12) every nonzero root is a full-T
//    continuation boundary with a certificate, so publishing it changes no plan, certificate or row (exhaustive sweep);
// 3. the selector with requested=false is the previous formula (exhaustive); a requested boundary is selected;
// 4. no-model replay: producer publishes the selected full-T boundaries, a sibling with the same prefix restores the
//    deepest one inside the prefix (only boundaries the capture order admits); restored rows with and without the
//    root request;
// 5. capture order (prefix_capture_fence): the incumbent same-T-owner rule is unchanged; a requested boundary whose
//    next unit is another owner is fenced (event on a known tail-owner stream, host otherwise), a cadence one refused;
// 6. the served case (system 6258 of n 6277): the root 6144 is followed by a tail owner; under the incumbent order
//    the sibling restores 4096 (the device failure), fenced it restores 6144.
#include "strix/qwen/cache_admission.hpp"
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

// The capture order at full-T boundary p, as prepare_capture (apps/hip_prefix_state_cache.hpp) sees the bridge's
// order: the producer is the T owner's stream, the predictor stream is shared, a tail owner's stream is known.
// fence_on=false: the incumbent rule (another owner next always refuses).
PrefixCaptureFence capture_fence(const QwenTiledPlanConfig& c, std::uint64_t n, std::uint64_t p, bool requested,
                                 bool fence_on = true) {
    auto cursor = *qwen_full_t_continuation(c, n, p);
    const auto next = qwen_prompt_next(c, cursor, p, n);
    if (!next) return PrefixCaptureFence::refused;
    const bool same = next->t_owner && next->owner == c.block;
    if (!same && !fence_on) return PrefixCaptureFence::refused;
    const bool known = same || (next->kind == QwenPlanKind::tail_owner && !next->t_owner);
    return prefix_capture_fence(same, known, same, true, requested);
}

// Deepest boundary a producer of n tokens publishes at or below `limit` (the sibling's shared prefix).
std::uint64_t restored_by_sibling(const QwenTiledPlanConfig& c, std::uint64_t n, std::uint64_t stable,
                                  std::uint64_t block, std::uint64_t cadence, bool root_on, bool fence_on = true) {
    const auto root = root_on ? prefix_root_capture(stable, n, c.rows) : 0;
    std::uint64_t best = 0;
    for (std::uint64_t p = c.rows; p < n; p += c.rows) {
        if (!qwen_full_t_continuation(c, n, p)) continue;
        if (!prefix_capture_boundary_selected(p, n - 1, block, cadence, p == root)) continue;
        if (capture_fence(c, n, p, p == root, fence_on) == PrefixCaptureFence::refused) continue;
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

        // 5. ordering rule. Incumbent: admitted iff the next unit is this T owner on the producer stream and the
        //    predictor stream is shared; that stays exactly the `stream` class.
        for (int bits = 0; bits < 32; ++bits) {
            const bool same = (bits & 1) != 0, known = (bits & 2) != 0, producer = (bits & 4) != 0;
            const bool pred = (bits & 8) != 0, req = (bits & 16) != 0;
            const auto f = prefix_capture_fence(same, known, producer, pred, req);
            if (same)
                require((f == PrefixCaptureFence::stream) == (known && producer && pred) &&
                            (f == PrefixCaptureFence::stream || f == PrefixCaptureFence::refused),
                        "same owner incumbent");
            else if (!pred || !req)
                require(f == PrefixCaptureFence::refused, "cadence / foreign predictor refused");
            else
                require(f == (!known      ? PrefixCaptureFence::host
                              : producer ? PrefixCaptureFence::stream
                                         : PrefixCaptureFence::event),
                        "requested fenced");
        }
        // Every root of the sweep is admitted; a cadence boundary followed by another owner never is.
        std::uint64_t fenced = 0, hosted = 0;
        for (std::uint64_t n = 2; n <= 16384; n += 7)
            for (std::uint64_t s = 1; s <= n + 64; s += 509) {
                const auto r = prefix_root_capture(s, n, c.rows);
                if (!r) continue;
                const auto f = capture_fence(c, n, r, true);
                require(f != PrefixCaptureFence::refused, "root admitted n=" + std::to_string(n));
                fenced += f == PrefixCaptureFence::event ? 1U : 0U;
                hosted += f == PrefixCaptureFence::host ? 1U : 0U;
                if (f != PrefixCaptureFence::stream)
                    require(capture_fence(c, n, r, false) == PrefixCaptureFence::refused, "cadence cross-owner refused");
            }
        require(fenced > 0, "sweep reached tail-owner roots");
        std::printf("root-fence roots event=%llu host=%llu\n", static_cast<unsigned long long>(fenced),
                    static_cast<unsigned long long>(hosted));

        // 6. the served case: n 6277, system 6258 (sibling prompt 6275).
        require(prefix_root_capture(6258, 6277, c.rows) == 6144, "served root");
        require(capture_fence(c, 6277, 6144, true) == PrefixCaptureFence::event, "served root: tail-owner event fence");
        require(restored_by_sibling(c, 6277, 6258, 1024, cadence, true, false) == 4096, "incumbent order: 4096");
        require(restored_by_sibling(c, 6277, 6258, 1024, cadence, true) == 6144, "fenced: 6144");
        std::printf("qwen prefix root tests PASS\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
