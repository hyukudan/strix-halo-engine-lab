#include "qualification.hpp"
#include "gemma4/dflash/device.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace strix::gemma4::prefill {
namespace {
void equal(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size() || std::memcmp(a.data(), b.data(), a.size_bytes()))
        throw std::runtime_error("post-WMMA qualification byte mismatch");
}
unsigned greedy(std::span<const float> x) {
    unsigned best = 0;
    for (unsigned i = 1; i < x.size(); ++i) if (better(x[i], i, x[best], best)) best = i;
    return best;
}
ForwardResult decode(gpu::DeviceTarget& target, gpu::DeviceStream& stream, unsigned token) {
    auto tx = stream.begin(); const RowPlan row{0, 1, tx->frontier(), token, tx.get(), 0, true};
    auto out = target.forward_rows({&row, 1}); tx->commit(1); return out;
}
} // namespace
void check_c4(DevicePrefill& prefill, gpu::DeviceTarget& decoder, const ModelConfig& config,
              std::span<const unsigned> tokens, unsigned steps) {
    if (tokens.empty() || !steps || tokens.size() + steps > config.max_positions)
        throw std::invalid_argument("C4 qualification context bounds");
    std::array<std::unique_ptr<gpu::DeviceStream>, 4> batch, solo;
    std::array<unsigned, 4> anchor{};
    for (unsigned i = 0; i < 4; ++i) {
        const auto count = std::max<std::size_t>(1, tokens.size() - std::min<std::size_t>(i, tokens.size() - 1));
        batch[i] = std::make_unique<gpu::DeviceStream>(config, unsigned(count) + steps);
        solo[i] = std::make_unique<gpu::DeviceStream>(config, unsigned(count) + steps);
        const auto a = prefill.run(*batch[i], tokens.first(count)), b = prefill.run(*solo[i], tokens.first(count));
        equal(a.last_logits, b.last_logits); equal(a.tail_taps, b.tail_taps);
        if (batch[i]->digest() != solo[i]->digest()) throw std::runtime_error("C4 starting-state mismatch");
        anchor[i] = greedy(a.last_logits);
    }
    for (unsigned step = 0; step < steps; ++step) {
        std::array<std::unique_ptr<KvTxn>, 4> tx;
        std::vector<RowPlan> rows;
        for (unsigned i = 0; i < 4; ++i) {
            tx[i] = batch[i]->begin(); rows.push_back({i, 9, tx[i]->frontier(), anchor[i], tx[i].get(), 3 - i, true});
        }
        const auto out = decoder.forward_rows(rows);
        for (unsigned i = 0; i < 4; ++i) {
            tx[i]->commit(1); const auto reference = decode(decoder, *solo[i], anchor[i]);
            equal(reference.logits, std::span(out.logits).subspan(std::size_t(3 - i) * out.vocab, out.vocab));
            equal(reference.taps, std::span(out.taps).subspan(std::size_t(3 - i) * out.tap_width, out.tap_width));
            if (batch[i]->digest() != solo[i]->digest()) throw std::runtime_error("C4 committed-state mismatch");
            anchor[i] = greedy(reference.logits);
        }
    }
    std::cout << "{\"gate\":\"post-WMMA-C4-C1\",\"streams\":4,\"decode_rows_per_stream\":" << steps << ",\"verdict\":\"EXACT\"}\n";
}
void check_dflash(DevicePrefill& prefill, gpu::DeviceTarget& decoder, const HostWeights& weights,
                  std::span<const unsigned> tokens, unsigned steps,
                  const std::filesystem::path& draft_model, const std::filesystem::path& draft_rope) {
    namespace df = strix::gemma4::dflash;
    const auto capacity = tokens.size() + steps + df::block;
    if (tokens.empty() || !steps || capacity > weights.config.max_positions)
        throw std::invalid_argument("DFlash qualification context bounds");
    const auto file = df::DraftFile::open(draft_model);
    const auto rope = df::RopeFile::open(draft_rope);
    if (capacity > rope.positions) throw std::invalid_argument("DFlash qualification rotary reservation");
    gpu::Buffer embedding(std::size_t(weights.embedding.outputs) * weights.embedding.row_bytes);
    embedding.upload(weights.embedding.data, embedding.size());
    auto view = weights.embedding; view.data = embedding.as<std::uint8_t>();
    gpu::DeviceStream serial(weights.config, unsigned(capacity));
    const auto initial = prefill.run(serial, tokens);
    auto anchor = greedy(initial.last_logits);
    std::vector<unsigned> expected;
    for (unsigned i = 0; i < steps; ++i) {
        anchor = greedy(decode(decoder, serial, anchor).logits); expected.push_back(anchor);
    }
    const auto digest = serial.digest();
    for (unsigned k : {0U, 1U, 7U}) {
        gpu::DeviceStream stream(weights.config, unsigned(capacity));
        df::DeviceDrafter draft(file.weights(), rope.view(), view, view);
        const auto prompt = prefill.run(stream, tokens, &draft);
        equal(prompt.last_logits, initial.last_logits); equal(prompt.tail_taps, initial.tail_taps);
        anchor = greedy(prompt.last_logits);
        df::Acceptance stats;
        std::vector<unsigned> actual;
        while (actual.size() < steps) {
            auto tx = stream.begin();
            const auto cycle = df::run_device_cycle(decoder, *tx, draft, anchor, k, steps - actual.size(), {}, &stats);
            if (cycle.verdict.emitted.empty()) throw std::runtime_error("DFlash qualification made no progress");
            actual.insert(actual.end(), cycle.verdict.emitted.begin(), cycle.verdict.emitted.end()); anchor = cycle.next_anchor;
            if (draft.frontier() != stream.frontier()) throw std::runtime_error("post-WMMA target/draft frontier mismatch");
        }
        if (k && !stats.attempted) throw std::runtime_error("NO-REFERENCE: DFlash proposals were disabled");
        if (actual != expected || stream.digest() != digest) throw std::runtime_error("post-WMMA DFlash token/state mismatch");
        std::cout << "{\"gate\":\"post-WMMA-DFlash\",\"draft_identity\":\"" << df::arithmetic_id << "\",\"k\":" << k
                  << ",\"tokens\":" << steps << ",\"attempted\":" << stats.attempted << ",\"accepted\":" << stats.accepted
                  << ",\"verdict\":\"EXACT\"}\n";
    }
}
} // namespace strix::gemma4::prefill
