#include "fixture.hpp"
#include "gemma4/dflash/device.hpp"
#include "gemma4/kernels/device.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>

namespace g4 = strix::gemma4;
namespace df = g4::dflash;
using fixture::check;
void device_cycles();
namespace {
void bytes(std::span<const float> a, std::span<const float> b) {
    check(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size_bytes()) == 0, "host/device byte mismatch");
}
std::vector<float> values(std::size_t n, unsigned seed = 0) {
    std::vector<float> result(n);
    for (std::size_t i = 0; i < n; ++i) result[i] = float(int((i * 7 + seed) % 31) - 15) / 16.0F;
    return result;
}
struct Assets {
    std::vector<float> data;
    g4::gpu::Buffer storage;
    g4::LinearView host, device;
    explicit Assets(unsigned width) : data(values(std::size_t(32) * width)), storage(data.size() * 4),
        host{reinterpret_cast<const std::uint8_t*>(data.data()), 32, width, width * 4ULL, g4::WeightFormat::f32} {
        storage.upload(data.data(), storage.size()); device = host; device.data = storage.as<std::uint8_t>();
    }
};
void forward(const df::RopeFile& rope) {
    auto w = fixture::weights();
    Assets assets(w.shape.hidden);
    df::DeviceDrafter device(w, rope.view(), assets.device, assets.device);
    fixture::rejects([&] { (void)device.context_key(0, ~std::uint64_t{0}); });
    fixture::rejects([&] { (void)device.context_value(0, ~std::uint64_t{0}); });
    df::DraftContext host(w.shape);
    const auto input = values(df::block * w.shape.hidden, 3);
    std::array<bool, df::block> seen{};
    for (std::uint64_t p = 0, batch = 0; p < 1040; ++batch) {
        const unsigned n = unsigned(std::min<std::uint64_t>(1 + batch % 8, 1040 - p));
        seen[n - 1] = true;
        const auto taps = values(n * w.shape.tap_width(), unsigned(p));
        std::vector<std::uint64_t> pos(n); std::iota(pos.begin(), pos.end(), p);
        host.append(w, rope.view(), taps, pos); device.append(taps, pos);
        for (unsigned l = 0; l < w.shape.layers; ++l)
            for (auto j : pos) { bytes(host.key(l, j), device.context_key(l, j)); bytes(host.value(l, j), device.context_value(l, j)); }
        p += n;
        if (p < 75 || p > 1015) {
            const auto expected = df::block_forward(w, host, rope.view(), input, p);
            bytes(expected, device.forward(input, p));
            check(device.select(expected) == df::proposals_from_hidden(w, expected, assets.host), "seven head rows mismatch");
            check(device.draft(3, p) == df::proposals_from_hidden(w,
                df::block_forward(w, host, rope.view(), df::embed_block(3, assets.host), p), assets.host), "embedding/block/head mismatch");
        }
    }
    check(std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }), "missing append row bucket");
    fixture::rejects([&] { (void)device.context_key(0, 0); });
    const auto before = device.context_key(0, 1039);
    std::array<std::uint64_t, 1> next{1040};
    auto taps = values(w.shape.tap_width());
    device.stage_append(taps, next); device.discard_append();
    check(device.frontier() == 1040, "discard published frontier"); bytes(before, device.context_key(0, 1039));
    taps[0] = std::numeric_limits<float>::infinity();
    fixture::rejects([&] { device.append(taps, next); });
    check(device.frontier() == 1040, "failed append published frontier"); bytes(before, device.context_key(0, 1039));
    auto bad = input; bad[0] = std::numeric_limits<float>::max();
    fixture::rejects([&] { (void)device.forward(bad, 1040); });
    bytes(df::block_forward(w, host, rope.view(), input, 1040), device.forward(input, 1040));
    df::DeviceDrafter missing(w, rope.view(), assets.device, assets.device);
    next[0] = 1023; taps = values(w.shape.tap_width()); missing.append(taps, next);
    fixture::rejects([&] { (void)missing.forward(input, 1024); });
}
void saturated() {
    auto s = fixture::shape(); s.layers = 1; s.head_dim = 128;
    auto w = fixture::weights(s);
    std::fill(w.layer[0].o.bits.begin(), w.layer[0].o.bits.end(), std::uint16_t{0});
    std::fill(w.layer[0].gate.bits.begin(), w.layer[0].gate.bits.end(), g4::bf16_bits(-16.0F));
    Assets assets(s.hidden); fixture::Tables rope(8, s.head_dim);
    df::DeviceDrafter device(w, rope.view(), assets.device, assets.device);
    df::DraftContext host(s);
    const std::vector<float> input(df::block * s.hidden, 1.0F);
    bytes(df::block_forward(w, host, rope.view(), input, 0), device.forward(input, 0));
    w.logit_scale = 1.25F; w.logit_cap = 0.01F;
    df::DeviceDrafter capped(w, rope.view(), assets.device, assets.device);
    check(capped.select(input) == df::proposals_from_hidden(w, input, assets.host), "draft cap tie mismatch");
}
void shared_model() {
    auto w = fixture::weights(); const auto original = w;
    Assets assets(w.shape.hidden); fixture::Tables rope(32, w.shape.head_dim);
    auto model = std::make_shared<df::DraftModel>(w, rope.view(), assets.device, assets.device);
    check(model->resident_bytes() > w.fc.bits.size() * 2, "missing resident draft weights");
    df::DeviceDrafter a(model), b(model);
    check(&a.model() == &b.model(), "streams must share immutable model");
    // Neither stream construction nor reset may consult the original host data.
    w.context_norm[0] = std::numeric_limits<float>::quiet_NaN();
    rope.cs[0] = std::numeric_limits<float>::quiet_NaN();
    df::DeviceDrafter late(model);
    const auto input = values(df::block * w.shape.hidden, 5);
    const auto empty = b.forward(input, 0);
    bytes(empty, late.forward(input, 0));
    const auto taps = values(w.shape.tap_width());
    const std::array<std::uint64_t, 1> pos{0};
    a.append(taps, pos);
    check(a.frontier() == 1 && b.frontier() == 0, "peer frontier changed");
    bytes(empty, b.forward(input, 0));
    const auto retained = a.context_key(0, 0);
    b.stage_append(taps, pos); b.reset();
    fixture::rejects([&] { b.publish_append(); });
    fixture::rejects([&] { (void)b.context_key(0, 0); });
    check(b.frontier() == 0 && b.timings().block_ms == 0, "reset metadata");
    bytes(empty, b.forward(input, 0)); bytes(retained, a.context_key(0, 0));
    a.reset(); bytes(empty, a.forward(input, 0));
    // Leases retain the immutable owner after the factory's reference is gone.
    model.reset(); bytes(empty, late.forward(input, 0));
    fixture::Tables valid(32, original.shape.head_dim);
    df::DeviceDrafter copied(original, valid.view(), assets.device, assets.device);
    bytes(empty, copied.forward(input, 0));
}
struct ScriptTxn final : g4::KvTxn {
    std::size_t committed = 0;
    std::uint64_t frontier() const override { return 0; }
    void commit(std::size_t n) override { check(!committed && n, "invalid script commit"); committed = n; }
};
struct ScriptTarget final : g4::Target {
    std::vector<unsigned> greedy;
    unsigned width;
    explicit ScriptTarget(std::vector<unsigned> ids, unsigned tap_width) : greedy(std::move(ids)), width(tap_width) {}
    g4::ForwardResult forward_rows(std::span<const g4::RowPlan> rows) override {
        g4::ForwardResult result{rows.size(), 32, width, std::vector<float>(rows.size() * 32, -1.0F), values(rows.size() * width)};
        for (std::size_t r = 0; r < rows.size(); ++r) result.logits[r * 32 + greedy[r]] = 1.0F;
        return result;
    }
};
void scripted() {
    const auto w = fixture::weights(); Assets assets(w.shape.hidden); fixture::Tables rope(16, w.shape.head_dim);
    df::DeviceDrafter probe(w, rope.view(), assets.device, assets.device);
    const auto proposals = probe.draft(1, 0);
    for (unsigned k = 1; k <= 7; ++k)
        for (unsigned reject = 0; reject <= k; ++reject)
            for (unsigned budget : {1U, 3U, 8U}) {
                df::DeviceDrafter device(w, rope.view(), assets.device, assets.device);
                std::vector<unsigned> greedy(k + 1, 31);
                std::copy_n(proposals.begin(), k, greedy.begin());
                if (reject < k) greedy[reject] = (greedy[reject] + 1) % 32;
                // Budget admission can shorten k, so the expected bonus is the
                // target greedy at that same processed row, exactly as serial.
                ScriptTarget target(greedy, unsigned(w.shape.tap_width())); ScriptTxn txn;
                df::Acceptance stats;
                const auto result = df::run_device_cycle(target, txn, device, 1, k, budget, {}, &stats);
                const auto admitted = std::min(k, budget - 1);
                const auto expected = df::verify_greedy(std::span(proposals).first(admitted), std::span(greedy).first(admitted + 1), budget);
                check(result.verdict.emitted == expected.emitted && txn.committed == expected.processed &&
                    device.frontier() == expected.processed && stats.emitted == expected.processed, "scripted device prefix/budget");
            }
    df::DeviceDrafter device(w, rope.view(), assets.device, assets.device);
    std::vector<unsigned> greedy(proposals.begin(), proposals.end()); greedy.push_back(31);
    ScriptTarget target(greedy, unsigned(w.shape.tap_width())); ScriptTxn txn;
    const std::array<unsigned, 1> eos{proposals[0]};
    const auto result = df::run_device_cycle(target, txn, device, 1, 7, 8, eos);
    check(result.verdict.stopped && result.verdict.emitted.size() == 1 && txn.committed == 1 && device.frontier() == 1,
        "device EOS publication");
}
}
int main(int argc, char** argv) {
    try {
        check(argc == 2, "fixture directory required");
        g4::gpu::require_device();
        forward(df::RopeFile::open(std::filesystem::path(argv[1]) / "rope8-p0-p2063.g4dr"));
        saturated(); shared_model(); scripted(); device_cycles();
        std::cout << "EXACT: context R1..8, ring/mask boundaries, block/head and device cycles\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
