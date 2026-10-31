#define fixture draft_fixture
#include "fixture.hpp"
#undef fixture
#include "../gemma4_gpu/fixture.hpp"
#include "gemma4/dflash/device.hpp"
#include "gemma4/kernels/device.hpp"

namespace g4 = strix::gemma4;
namespace df = g4::dflash;

void device_cycles() {
    const auto model = fixture::tiny(1);
    const auto hw = g4::bind_host_weights(model);
    fixture::Tables tables(80);
    g4::gpu::DeviceTarget target(hw, tables.view());
    g4::gpu::Buffer embedding(std::size_t(hw.embedding.outputs) * hw.embedding.row_bytes);
    embedding.upload(hw.embedding.data, embedding.size());
    auto view = hw.embedding; view.data = embedding.as<std::uint8_t>();
    auto s = draft_fixture::shape(); s.hidden = hw.config.hidden;
    auto w = draft_fixture::weights(s); draft_fixture::Tables rope(80, s.head_dim);
    for (unsigned k = 0; k <= 7; ++k) {
        g4::gpu::DeviceStream speculative(hw.config, 80), serial(hw.config, 80);
        df::DeviceDrafter drafter(w, rope.view(), view, view);
        unsigned anchor = 1, greedy_anchor = 1, emitted = 0;
        df::Acceptance stats;
        while (emitted < 12) {
            auto txn = speculative.begin();
            const auto cycle = df::run_device_cycle(target, *txn, drafter, anchor, k, 12 - emitted, {}, &stats);
            for (auto token : cycle.verdict.emitted) {
                auto st = serial.begin();
                g4::RowPlan row{0, 0, st->frontier(), greedy_anchor, st.get(), 0, true};
                const auto out = target.forward_rows(std::span(&row, 1));
                unsigned best = 0;
                for (unsigned id = 1; id < out.vocab; ++id)
                    if (g4::better(out.logits[id], id, out.logits[best], best)) best = id;
                CHECK(token == best); st->commit(1); greedy_anchor = best;
            }
            anchor = cycle.next_anchor; emitted += unsigned(cycle.verdict.emitted.size());
            CHECK(speculative.digest() == serial.digest()); CHECK(drafter.frontier() == serial.frontier());
        }
        CHECK(stats.emitted == 12);
    }
}
