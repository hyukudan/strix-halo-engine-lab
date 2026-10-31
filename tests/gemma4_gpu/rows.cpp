#include "gemma4/kernels/device.hpp"
#include "fixture.hpp"
#include <array>

using namespace strix::gemma4;
using namespace strix::gemma4::gpu;
namespace {
template <class Stream>
ForwardResult ingest(Target& target, Stream& stream, unsigned n, unsigned token, bool permute = false) {
    auto tx = stream.begin();
    std::vector<RowPlan> plans;
    for (unsigned r = 0; r < n; ++r)
        plans.push_back({13, 7, stream.frontier() + r, (token + r) % 37, tx.get(), r, true});
    if (permute)
        std::reverse(plans.begin(), plans.end());
    const auto out = target.forward_rows(plans);
    tx->commit(n);
    return out;
}
void compare(const ForwardResult& a, const ForwardResult& b) {
    CHECK(a.rows == b.rows);
    bytes(a.logits, b.logits);
    bytes(a.taps, b.taps);
}
void timing_tests() {
    const auto model = fixture::tiny(1);
    fixture::Tables tables;
    const auto weights = bind_host_weights(model);
    HostTarget host(weights, tables.view(), 2);
    DeviceTarget device(weights, tables.view(), {true, true, false, true, true});
    DeviceStream stream(model.config, 64);
    HostStream reference(model.config);
    for (unsigned step = 0; step < 3; ++step) {
        compare(ingest(device, stream, 1, 3 + step), ingest(host, reference, 1, 3 + step));
        CHECK(device.stats().gpu_pass_ms > 0.0);
        CHECK(stream.digest() == reference.digest());
    }
    refuses([&] { DeviceTarget invalid(weights, tables.view(), {true, true, true, true, true}); });
}
void graph_tests(unsigned count, bool quant) {
    const auto model = fixture::tiny(count, quant);
    fixture::Tables tables;
    HostTarget host(bind_host_weights(model), tables.view(), 2);
    DeviceTarget device(bind_host_weights(model), tables.view());
    DeviceStream batch(model.config, 1024), serial(model.config, 1024);
    HostStream reference(model.config);
    for (unsigned n = 1; n <= 8; ++n) {
        const auto out = ingest(device, batch, n, 2 + n, true), expected = ingest(host, reference, n, 2 + n, true);
        compare(out, expected);
        const auto selected = device.selected();
        for (unsigned r = 0; r < n; ++r) {
            const auto* logits = out.logits.data() + std::size_t(r) * model.config.vocab;
            unsigned best = 0;
            for (unsigned i = 1; i < model.config.vocab; ++i)
                if (better(logits[i], i, logits[best], best))
                    best = i;
            CHECK(selected[r].id == best);
        }
        for (unsigned r = 0; r < n; ++r) {
            const auto single = ingest(device, serial, 1, 2 + n + r);
            CHECK(std::memcmp(single.logits.data(), out.logits.data() + r * model.config.vocab,
                              single.logits.size() * 4) == 0);
            CHECK(std::memcmp(single.taps.data(), out.taps.data() + r * 5 * model.config.hidden,
                              single.taps.size() * 4) == 0);
        }
        CHECK(batch.digest() == serial.digest());
        CHECK(batch.digest() == reference.digest());
        if (count == 42)
            for (unsigned t = 0; t < 5; ++t) {
                bool nonzero = false;
                for (unsigned i = 0; i < model.config.hidden; ++i)
                    nonzero |= out.taps[t * model.config.hidden + i] != 0;
                CHECK(nonzero);
            }
    }
    DeviceStream peer_a(model.config, 64), peer_b(model.config, 64), solo(model.config, 64);
    (void)ingest(device, peer_b, 3, 17);
    auto a = peer_a.begin(), b = peer_b.begin();
    const RowPlan peers[] = {
        {4, 2, 0, 3, a.get(), 2, true}, {999, 999, 65536, 999, nullptr, 0, false}, {5, 3, 3, 17, b.get(), 0, true}};
    const auto out = device.forward_rows(peers);
    a->commit(1);
    b->commit(1);
    const auto expected = ingest(device, solo, 1, 3);
    CHECK(std::memcmp(out.logits.data() + 2 * model.config.vocab, expected.logits.data(), expected.logits.size() * 4) ==
          0);
    CHECK(peer_a.digest() == solo.digest());
    const auto before = peer_a.digest();
    {
        auto tx = peer_a.begin();
        const RowPlan row{4, 2, 1, 2, tx.get(), 0, true};
        (void)device.forward_rows({&row, 1});
    }
    CHECK(peer_a.digest() == before);
    {
        auto tx = peer_a.begin();
        const RowPlan row{4, 2, 2, 2, tx.get(), 0, true};
        refuses([&] { (void)device.forward_rows({&row, 1}); });
        refuses([&] { tx->commit(0); });
    }
    {
        auto tx = peer_a.begin();
        const RowPlan duplicate[] = {{4, 2, 1, 2, tx.get(), 0, true}, {4, 2, 1, 2, tx.get(), 1, true}};
        refuses([&] { (void)device.forward_rows(duplicate); });
    }
    {
        auto tx = peer_a.begin();
        const RowPlan row{4, 2, 1, 2, tx.get(), 0, true};
        (void)device.forward_rows({&row, 1});
        refuses([&] { (void)device.forward_rows({&row, 1}); });
        refuses([&] { tx->commit(2); });
        tx->commit(0);
        refuses([&] { tx->commit(0); });
    }
    DeviceStream limited(model.config, 1);
    (void)ingest(device, limited, 1, 3);
    {
        auto tx = limited.begin();
        const RowPlan row{0, 0, 1, 2, tx.get(), 0, true};
        refuses([&] { (void)device.forward_rows({&row, 1}); });
    }
    // Published transaction handles may survive while the next owner is active.
    DeviceStream lifetime(model.config, 64);
    HostStream lifetime_ref(model.config);
    auto retained = lifetime.begin();
    RowPlan first{0, 0, 0, 3, retained.get(), 0, true};
    compare(device.forward_rows({&first, 1}), ingest(host, lifetime_ref, 1, 3));
    retained->commit(1);
    {
        auto current = lifetime.begin();
        retained.reset();
        refuses([&] { (void)lifetime.begin(); });
        RowPlan second{0, 0, 1, 4, current.get(), 0, true};
        compare(device.forward_rows({&second, 1}), ingest(host, lifetime_ref, 1, 4));
        current->commit(1);
    }
    compare(ingest(device, lifetime, 1, 5), ingest(host, lifetime_ref, 1, 5));
    CHECK(lifetime.digest() == lifetime_ref.digest());
}
void ring_tests() {
    auto model = fixture::tiny(2);
    model.config.layer[1].attention = AttentionKind::global; // exercise both storage types in two blocks
    fixture::Tables tables;
    HostTarget host(bind_host_weights(model), tables.view(), 2);
    DeviceTarget device(bind_host_weights(model), tables.view());
    // Each accepted-prefix depth has its own stream. Prefill shared across eight streams/pass.
    std::vector<std::unique_ptr<DeviceStream>> streams;
    std::vector<std::unique_ptr<HostStream>> refs;
    for (unsigned i = 0; i < 9; ++i) {
        streams.push_back(std::make_unique<DeviceStream>(model.config, 1024));
        refs.push_back(std::make_unique<HostStream>(model.config));
    }
    for (unsigned p = 0; p < 508; ++p) {
        for (unsigned base = 0; base < 9; base += 8) {
            const unsigned n = std::min(8U, 9 - base);
            std::vector<std::unique_ptr<KvTxn>> dtx, htx;
            std::vector<RowPlan> dp, hp;
            for (unsigned r = 0; r < n; ++r) {
                dtx.push_back(streams[base + r]->begin());
                htx.push_back(refs[base + r]->begin());
                dp.push_back({base + r, 0, p, 2 + (p % 31), dtx.back().get(), r, true});
                hp.push_back({base + r, 0, p, 2 + (p % 31), htx.back().get(), r, true});
            }
            compare(device.forward_rows(dp), host.forward_rows(hp));
            for (auto& t : dtx)
                t->commit(1);
            for (auto& t : htx)
                t->commit(1);
        }
    }
    for (unsigned depth = 0; depth <= 8; ++depth) {
        auto& ds = *streams[depth];
        auto& hs = *refs[depth];
        for (unsigned p = 508; p <= 516; ++p) {
            const auto before = ds.digest();
            auto tx = ds.begin();
            auto ht = hs.begin();
            std::vector<RowPlan> dp, hp;
            for (unsigned r = 0; r < 8; ++r) {
                dp.push_back({depth, 0, p + r, 3 + r, tx.get(), r, true});
                hp.push_back({depth, 0, p + r, 3 + r, ht.get(), r, true});
            }
            std::reverse(dp.begin(), dp.end());
            compare(device.forward_rows(dp), host.forward_rows(hp));
            CHECK(ds.digest() == before);
            const unsigned accepted = p == 516 ? depth : 0;
            tx->commit(accepted);
            ht->commit(accepted);
            CHECK(ds.digest() == hs.digest());
            if (p < 516) {
                compare(ingest(device, ds, 1, 17), ingest(host, hs, 1, 17));
            }
        }
        CHECK(ds.frontier() == 516 + depth);
    }
}
} // namespace
int main(int argc, char**) {
    try {
        require_device();
        timing_tests();
        graph_tests(1, false);
        graph_tests(42, false);
        graph_tests(1, true);
        if (argc == 1)
            ring_tests();
        std::cout
            << "device rows: host bytes, R=1..8/serial, taps, aliases, streams, slots, rollback, ring edges passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
