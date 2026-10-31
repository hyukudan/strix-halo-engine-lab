// e4b-eagle device tests: DeviceDrafter vs HostDrafter bit-for-bit (proposals, draft ids, probabilities and ring K/V)
// on random weights through priming, drafting, discard and window wrap; launch counts; optional --bench at the
// production geometry (2560 / 8q2kv x 256 / FFN 5120 / 65,536-row Q8 head) with random weights.
#include "common.hpp"
#include "gemma4/eagle/device.hpp"

#include <chrono>
#include <cstring>
#include <memory>
#include <numeric>

using namespace eagle_test;
namespace eg = strix::gemma4::eagle;

namespace {
struct DeviceTable {
    eg::Matrix host;
    gpu::Buffer device;
    explicit DeviceTable(eg::Matrix m) : host(std::move(m)), device(host.bytes.size()) { device.upload(host.bytes.data(), host.bytes.size()); }
    LinearView view() const { return {device.as<std::uint8_t>(), host.outputs, host.inputs, host.row_bytes(), host.format}; }
};
bool same_bits(float a, float b) { return bit_cast<std::uint32_t>(a) == bit_cast<std::uint32_t>(b); }
std::uint64_t expected_launches(const Config& c, std::uint32_t k) { return 13U + (k - 1U) * (c.concat ? 16U : 15U); }

void device_vs_host(const Config& config, const char* name, WeightFormat format) {
    const auto w = Weights::random(config, 13, format);
    DeviceTable emb(embedding_table(config, 14));
    const auto rope = eg::make_rope_table(1024);
    const eg::RopeView view{rope, 1024};
    eg::HostDrafter host(w, view, emb.host.view());
    auto model = std::make_shared<eg::DeviceModel>(w, view, emb.view());
    eg::DeviceDrafter device(model);
    Scenario s{w.config, 5, 37, 700};
    std::size_t drafts = 0, depths = 0, rows = 0;
    lockstep(s, {&host, &device},
        [&](const Draft& h, const Draft& d, std::size_t, const std::string& label) {
            ++drafts;
            for (std::uint32_t i = 0; i < h.depth; ++i) {
                ++depths;
                EAGLE_CHECK(h.tokens[i] == d.tokens[i] && h.draft_ids[i] == d.draft_ids[i] && same_bits(h.prob[i], d.prob[i]),
                            "%s %s depth %u host %u/%a device %u/%a", name, label.c_str(), i + 1, h.tokens[i], double(h.prob[i]),
                            d.tokens[i], double(d.prob[i]));
            }
            EAGLE_CHECK(device.timings().launches == expected_launches(w.config, h.depth), "%s launches %llu for k %u", name,
                        (unsigned long long)device.timings().launches, h.depth);
        },
        [&](std::span<const std::uint64_t> positions) {
            for (const auto p : positions) {
                ++rows;
                EAGLE_CHECK(host.key(p) == device.key(p) && host.value(p) == device.value(p), "%s ring K/V position %llu", name,
                            (unsigned long long)p);
            }
            EAGLE_CHECK(device.timings().stage_launches == (w.config.concat ? 7U : 6U), "%s stage launches", name);
        });
    std::printf("device == host (%s): %zu drafts, %zu depths, %zu ring rows bit-identical\n", name, drafts, depths, rows);
    // Reuse after reset behaves like a fresh drafter.
    device.reset();
    host.reset();
    Scenario short_run{w.config, 9, 21, 90};
    lockstep(short_run, {&host, &device}, [&](const Draft& h, const Draft& d, std::size_t, const std::string& label) {
        EAGLE_CHECK(h.tokens == d.tokens && h.draft_ids == d.draft_ids, "%s after reset %s", name, label.c_str());
    });
    EAGLE_CHECK(device.reusable(), "reusable");
}

// --------------------------------------------------------------------------- production geometry benchmark
void bench(const std::vector<std::uint32_t>& contexts) {
    using Clock = std::chrono::steady_clock;
    Config c; // production defaults: 2560, 8q/2kv x 256, FFN 5120, Vd 65,536, taps {11, 23, final}
    c.target_vocab = 65536; // smaller embedding table for the bench (the drafter reads one row per depth)
    const auto t0 = Clock::now();
    const auto w = Weights::random(c, 1);
    DeviceTable emb(embedding_table(c, 2));
    const std::uint32_t positions = 4096;
    const auto rope = eg::make_rope_table(positions);
    const eg::RopeView view{rope, positions};
    auto model = std::make_shared<eg::DeviceModel>(w, view, emb.view());
    eg::DeviceDrafter device(model);
    std::printf("bench setup %.1f s; resident %.1f MB; bytes/step %.1f MB, depth-1 %.1f MB, stage %.1f MB\n",
                std::chrono::duration<double>(Clock::now() - t0).count(), double(model->resident_bytes()) / 1e6,
                double(model->step_weight_bytes()) / 1e6, double(model->first_depth_weight_bytes()) / 1e6,
                double(model->stage_weight_bytes()) / 1e6);
    // Production-geometry bit check on a short context (host reference cost stays small).
    {
        eg::HostDrafter host(w, view, emb.host.view());
        Scenario s{c, 3, 2, 0};
        std::vector<std::uint32_t> next{11, 12, 13, 14, 15, 16, 17, 18, 19};
        const auto rows = make_rows(s, 0, std::span(next).first(8));
        host.prime(rows.rows());
        device.prime(rows.rows());
        const auto rows2 = make_rows(s, 8, std::span(next).subspan(8, 1));
        host.prime(rows2.rows());
        device.prime(rows2.rows());
        const auto h = host.draft(19, 9, 3), d = device.draft(19, 9, 3);
        for (std::uint32_t i = 0; i < 3; ++i)
            EAGLE_CHECK(h.tokens[i] == d.tokens[i] && same_bits(h.prob[i], d.prob[i]), "production geometry depth %u", i + 1);
        std::printf("production geometry host == device: %s\n", h.tokens == d.tokens ? "yes" : "NO");
        device.reset();
    }
    for (const auto context : contexts) {
    EAGLE_CHECK(context >= 1 && context + 16 < positions, "bench context %u", context);
    device.reset();
    Scenario s{c, 4, 2, 0};
    Rng r{5};
    std::vector<std::uint32_t> next(context);
    for (auto& t : next) t = r.below(c.target_vocab);
    double publish = 0, stage = 0;
    for (std::uint64_t f = 0; f < context; f += 8) {
        const auto n = std::min<std::uint64_t>(8, context - f);
        const auto rows = make_rows(s, f, std::span(next).subspan(f, n));
        device.stage(rows.rows());
        stage += device.timings().stage_ms;
        device.publish();
        publish += device.timings().publish_ms;
    }
    std::printf("primed %u rows: mean stage(8 rows) %.3f ms, publish %.3f ms (GPU events)\n", context, stage / ((context + 7) / 8),
                publish / ((context + 7) / 8));
    for (std::uint32_t n = 1; n <= 8; n *= 2) {
        const auto rows = make_rows(s, context, std::span(next).first(n));
        double g = 0, wall = 0;
        for (int i = 0; i < 12; ++i) {
            const auto a = Clock::now();
            device.stage(rows.rows());
            const auto b = std::chrono::duration<double, std::milli>(Clock::now() - a).count();
            device.discard();
            if (i >= 2) { g += device.timings().stage_ms; wall += b; }
        }
        std::printf("stage %u rows: GPU %.3f ms, wall %.3f ms, %llu launches\n", n, g / 10, wall / 10,
                    (unsigned long long)device.timings().stage_launches);
    }
    const auto anchor = next[context - 1];
    double ms[8] = {}, walls[8] = {};
    for (std::uint32_t k = 1; k <= eg::max_depth; ++k) {
        double g = 0, wall = 0;
        const int warm = 3, reps = 20;
        for (int i = 0; i < warm + reps; ++i) {
            const auto a = Clock::now();
            (void)device.draft(anchor, context, k);
            const auto b = std::chrono::duration<double, std::milli>(Clock::now() - a).count();
            if (i >= warm) { g += device.timings().draft_ms; wall += b; }
        }
        ms[k] = g / reps; walls[k] = wall / reps;
        std::printf("draft k=%u: GPU %.3f ms, wall %.3f ms, %llu launches\n", k, ms[k], walls[k],
                    (unsigned long long)device.timings().launches);
    }
    const double per = (ms[7] - ms[1]) / 6.0, per_wall = (walls[7] - walls[1]) / 6.0;
    std::printf("context %u: depth-1 %.3f ms, each further depth %.3f ms GPU (%.3f wall) = %.1f logical GB/s\n", context, ms[1], per,
                per_wall, double(model->step_weight_bytes()) / (per * 1e6));
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        gpu::require_device();
        if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
            std::vector<std::uint32_t> contexts;
            for (int i = 2; i < argc; ++i) contexts.push_back(std::uint32_t(std::stoul(argv[i])));
            if (contexts.empty()) contexts.push_back(600U);
            bench(contexts);
        } else {
            device_vs_host(small_config(), "taps C concat scale 1/16 q8", WeightFormat::q8_0);
            device_vs_host(small_config({5, 23, 35}, false, 1.0F), "taps A sum scale 1 q8", WeightFormat::q8_0);
            device_vs_host(small_config({5, 11, 17, 23, 35}), "taps dflash concat bf16", WeightFormat::bf16);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL exception: %s\n", e.what());
        return 1;
    }
    std::printf(eagle_test::failures ? "FAILED %d checks\n" : "ALL OK\n", eagle_test::failures);
    return eagle_test::failures ? 1 : 0;
}
