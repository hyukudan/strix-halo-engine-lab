// MTP assistant drafter host tests. usage: strix-gemma4-eagle-mtp-tests [fixture_dir assistant_dir]
// Without arguments the HF fixture comparison is skipped (it needs the real checkpoint).
#include "mtp_common.hpp"
#include "gemma4/eagle/mtp/session.hpp"

#include <limits>

using namespace mtp_test;

namespace {
void recipes() {
    Rng r{3};
    TargetKv kv(300, 5);
    std::vector<float> q(512);
    for (auto& x : q) x = r.uniform();
    const std::pair<std::uint64_t, std::uint64_t> ranges[] = {{0, 0}, {0, 31}, {5, 299}, {0, 299}};
    for (const auto& [first, last] : ranges) {
        std::vector<float> out(512);
        mt::attend_head(q.data(), 512, kv.full(), 1, first, last, out.data());
        std::vector<double> s;
        double m = -1e300, den = 0;
        for (auto j = first; j <= last; ++j) {
            double d = 0;
            for (int i = 0; i < 512; ++i) d += double(q[i]) * kv.kf[j * 1024 + 512 + i];
            s.push_back(d); m = std::max(m, d);
        }
        for (auto& x : s) { x = std::exp(x - m); den += x; }
        double worst = 0;
        for (int i = 0; i < 512; ++i) {
            double acc = 0;
            for (std::size_t j = 0; j < s.size(); ++j) acc += s[j] * kv.vf[(first + j) * 1024 + 512 + i];
            worst = std::max(worst, std::fabs(acc / den - out[i]));
        }
        EAGLE_CHECK(worst < 1e-5, "attention [%llu,%llu] max err %g", (unsigned long long)first, (unsigned long long)last, worst);
    }
    std::vector<float> c(2048, 0.0F);
    c[7] = 3; c[5] = 3; c[100] = 2;
    const auto top = mt::top_clusters(c, 4);
    EAGLE_CHECK(top[0] == 5 && top[1] == 7 && top[2] == 100 && top[3] == 0, "top-k order/ties %u %u %u %u", top[0], top[1], top[2], top[3]);
}

void fixture(const std::filesystem::path& dir, const std::filesystem::path& model, const Table& table) {
    Fixture f;
    if (!Fixture::load(dir, f) || !std::filesystem::exists(model / "model.safetensors")) {
        std::printf("SKIP fixture (no %s or no checkpoint)\n", dir.string().c_str());
        return;
    }
    auto w = mt::Weights::load(model);
    w.config.sliding_keys = f.sliding_keys;
    w.config.top_k = 32; // HF reference selects config.centroid_intermediate_top_k = 32
    Rope rope(1024);
    mt::HostDrafter d(w, rope.view(), table.view());
    // Teacher-forced single steps: each depth's HF input (token, h) at position p, one engine depth, same argmax.
    std::size_t same = 0, total = 0, ties = 0, chain = 0, chain_total = 0;
    for (std::size_t ci = 0; ci < f.cases.size(); ++ci) {
        const auto& c = f.cases[ci];
        for (std::size_t i = 0; i < c.tokens.size(); ++i) {
            const auto h = std::span<const float>(f.step_hidden).subspan((ci * f.K + i) * 2560, 2560);
            const auto out = d.draft({c.inputs[i], c.p, h, f.kv.full_sliding(), f.kv.full()}, 1);
            ++total;
            if (out.tokens[0] == c.tokens[i]) { ++same; continue; }
            if (c.margins[i] < 1e-3) { ++ties; continue; }
            EAGLE_CHECK(false, "p %llu depth %zu engine %u HF %u (HF margin %g)", (unsigned long long)c.p, i + 1, out.tokens[0],
                        c.tokens[i], c.margins[i]);
        }
        // Free-running chain (informational: random synthetic KV makes long chains chaotic).
        const auto run = d.draft({c.anchor, c.p, std::span<const float>(f.hidden).subspan((c.p - 1) * 2560, 2560), f.kv.full_sliding(),
                                  f.kv.full()}, std::uint32_t(c.tokens.size()));
        for (std::size_t i = 0; i < c.tokens.size() && run.tokens[i] == c.tokens[i]; ++i) ++chain;
        chain_total += c.tokens.size();
    }
    std::printf("HF fixture (real weights): %zu/%zu teacher-forced steps identical (%zu near ties); free-running chain prefix %zu/%zu\n",
                same, total, ties, chain, chain_total);
    // The engine's 512-slot tagged ring equals the full array when sliding_keys = 512.
    w.config.sliding_keys = 512;
    mt::HostDrafter d512(w, rope.view(), table.view());
    for (const std::uint64_t p : {100ULL, 512ULL, 513ULL, 650ULL}) {
        mt::Inputs a{77, p, std::span<const float>(f.hidden).subspan((p - 1) * 2560, 2560), f.kv.full_sliding(), f.kv.full()};
        auto b = a;
        b.sliding = f.kv.ring(p);
        const auto x = d512.draft(a, 4), y = d512.draft(b, 4);
        EAGLE_CHECK(x.tokens == y.tokens && x.prob == y.prob, "ring view differs at p %llu", (unsigned long long)p);
    }
}

void session(const Table& table) {
    const auto w = mt::Weights::random(mt::Config{}, 9);
    Rope rope(1024);
    TargetKv kv(600, 4);
    mt::Session s(std::make_unique<mt::HostDrafter>(w, rope.view(), table.view()), {22, 23});
    std::vector<float> hidden(3 * 2560, 0.5F), taps(3 * 5 * 2560, 0.0F);
    const std::uint32_t next[] = {10, 11, 12};
    const std::uint64_t positions[] = {497, 498, 499};
    EAGLE_CHECK(s.propose_window(12, 500, 3) == (std::array<std::uint32_t, 7>{}), "no hidden yet: empty proposal");
    s.stage_rows(taps, hidden, next, positions);
    s.publish();
    EAGLE_CHECK(s.propose_window(12, 500, 3) == (std::array<std::uint32_t, 7>{}), "no KV source: empty proposal");
    s.set_source([&](std::uint32_t layer, mt::KvLayer& out) {
        out = layer == 22 ? kv.full_sliding() : layer == 23 ? kv.full() : mt::KvLayer{};
        return layer == 22 || layer == 23;
    });
    EAGLE_CHECK(s.propose_window(13, 500, 3) == (std::array<std::uint32_t, 7>{}), "anchor mismatch: empty proposal");
    const auto p = s.propose_window(12, 500, 3);
    EAGLE_CHECK(p[0] == s.last_draft().tokens[0] && s.last_draft().depth == 3 && p[3] == 0, "proposal window 3");
    EAGLE_CHECK(throws([&] { s.stage(taps, positions); }), "legacy stage refuses");
    ModelConfig t;
    t.layer.resize(42);
    for (std::uint32_t l = 0; l < 42; ++l) {
        t.layer[l].attention = (l % 6 == 5) ? AttentionKind::global : AttentionKind::sliding;
        t.layer[l].producer = l < 24 ? l : (t.layer[l].attention == AttentionKind::global ? 23 : 22);
    }
    const auto layers = mt::shared_kv_layers(t);
    EAGLE_CHECK(layers.first == 22 && layers.second == 23, "E4B shared KV layers %u %u", layers.first, layers.second);
}
// Release layout: <model dir>/drafter-mtp/model.safetensors; absent drafter -> empty path (decode runs exact).
void layout() {
    namespace fs = std::filesystem;
    const auto root = fs::temp_directory_path() / "e4b-mtp-layout-test";
    fs::remove_all(root);
    fs::create_directories(root / "model" / mt::drafter_subdirectory);
    fs::create_directories(root / "empty");
    std::ofstream(root / "model" / mt::drafter_subdirectory / "model.safetensors") << "x";
    std::ofstream(root / "model" / "Nemotron.gguf") << "x";
    const auto want = root / "model" / mt::drafter_subdirectory;
    EAGLE_CHECK(mt::resolve_drafter(want) == want, "drafter dir itself");
    EAGLE_CHECK(mt::resolve_drafter(root / "model") == want, "model dir");
    EAGLE_CHECK(mt::resolve_drafter(root / "model" / "Nemotron.gguf") == want, "model file");
    EAGLE_CHECK(mt::resolve_drafter(root / "empty").empty() && mt::resolve_drafter(root / "missing").empty(), "absent drafter");
    EAGLE_CHECK(throws([&] { (void)mt::Weights::load(root / "empty"); }), "load refuses an absent drafter");
    fs::remove_all(root);
}
} // namespace

int main(int argc, char** argv) {
    try {
        recipes();
        layout();
        const Table table;
        if (argc > 2) fixture(argv[1], argv[2], table);
        else std::printf("SKIP fixture (usage: fixture_dir assistant_dir)\n");
        session(table);
        for (int i = 3; i < argc; ++i) { // optional extra checkpoints (e.g. fine-tuned exports): load + validate
            const auto w = mt::Weights::load(argv[i]);
            std::printf("loaded %s: hidden %u ffn %u full=%d%d%d%d scalars %.4g %.4g %.4g %.4g\n", argv[i], w.config.hidden,
                        w.config.ffn, w.config.full[0], w.config.full[1], w.config.full[2], w.config.full[3], double(w.layer[0].scalar),
                        double(w.layer[1].scalar), double(w.layer[2].scalar), double(w.layer[3].scalar));
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL exception: %s\n", e.what());
        return 1;
    }
    std::printf(eagle_test::failures ? "FAILED %d checks\n" : "ALL OK\n", eagle_test::failures);
    return eagle_test::failures ? 1 : 0;
}
