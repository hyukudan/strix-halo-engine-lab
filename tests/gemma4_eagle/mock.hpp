#pragma once
// Row-invariant toy target (greedy and taps are functions of the token history) and an oracle drafter.
#include "common.hpp"

#include <memory>

namespace eagle_test {
// ---------------------------------------------------------------- mock target for cycle exactness
// Row-invariant toy target: greedy(row p) = f(hash of all tokens at positions <= p); taps/hidden from the same hash.
struct MockStream;
struct MockTxn final : KvTxn {
    MockStream* s;
    std::uint64_t start;
    std::vector<std::uint32_t> tokens;
    bool done = false;
    MockTxn(MockStream* stream, std::uint64_t f) : s(stream), start(f) {}
    std::uint64_t frontier() const override { return start; }
    void reserve_rows(std::size_t) override {}
    void commit(std::size_t n) override;
};
struct MockStream {
    std::vector<std::uint32_t> tokens;
    std::vector<std::uint64_t> hashes{0x1234567ULL};
    std::unique_ptr<KvTxn> begin() { return std::make_unique<MockTxn>(this, tokens.size()); }
    static std::uint64_t mix(std::uint64_t h, std::uint32_t t) {
        h ^= t + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
        h *= 0xff51afd7ed558ccdULL;
        return h ^ (h >> 33U);
    }
};
inline void MockTxn::commit(std::size_t n) {
    if (done || n > tokens.size()) throw std::runtime_error("mock commit");
    for (std::size_t i = 0; i < n; ++i) {
        s->tokens.push_back(tokens[i]);
        s->hashes.push_back(MockStream::mix(s->hashes.back(), tokens[i]));
    }
    done = true;
}
struct MockTarget final : Target {
    std::uint32_t hidden = 256, vocab = 1024, eos = 1;
    std::uint64_t rows = 0;
    std::uint32_t greedy(std::uint64_t h) const { return h % 61 == 0 ? eos : std::uint32_t(2 + h % (vocab - 2)); }
    ForwardResult forward_rows(std::span<const RowPlan> plans) override {
        ForwardResult f;
        f.rows = plans.size(); f.vocab = vocab; f.tap_width = 5 * hidden;
        f.logits.assign(f.rows * vocab, 0.0F); f.taps.resize(f.rows * f.tap_width);
        bool want = false;
        for (const auto& p : plans) want = want || p.export_hidden;
        if (want) f.hidden.assign(f.rows * hidden, 0.0F);
        for (const auto& p : plans) {
            auto* tx = static_cast<MockTxn*>(p.transaction);
            if (p.position != tx->start + tx->tokens.size()) throw std::runtime_error("mock rows must be contiguous");
            tx->tokens.push_back(p.token);
            std::uint64_t h = tx->s->hashes.back();
            for (const auto t : tx->tokens) h = MockStream::mix(h, t);
            f.logits[p.output_row * vocab + greedy(h)] = 1.0F;
            Rng r{h};
            for (std::uint32_t i = 0; i < f.tap_width; ++i) f.taps[p.output_row * f.tap_width + i] = 3 * r.uniform();
            if (p.export_hidden) for (std::uint32_t i = 0; i < hidden; ++i) f.hidden[p.output_row * hidden + i] = r.uniform();
            ++rows;
        }
        return f;
    }
};
// Oracle drafter: proposes the mock target's true greedy continuation, with injected errors.
class Oracle final : public Drafter {
public:
    Oracle(const MockTarget& t, MockStream& s, Config c) : t_(t), s_(s), c_(std::move(c)) {}
    const Config& config() const override { return c_; }
    std::uint64_t frontier() const override { return f_; }
    const eagle::Timings& timings() const override { return times_; }
    void reset() override { f_ = 0; }
    void stage(const Rows& r) override { pending_ = r.positions.back() + 1; }
    void publish() override { f_ = pending_; }
    void discard() noexcept override {}
    Draft draft(std::uint32_t anchor, std::uint64_t position, std::uint32_t k) override {
        Draft d;
        d.depth = k;
        std::uint64_t h = MockStream::mix(s_.hashes.at(position), anchor);
        for (std::uint32_t i = 0; i < k; ++i) {
            const auto g = t_.greedy(h);
            d.tokens[i] = (++n_ % 5 == 0) ? (g + 7) % t_.vocab : g;
            h = MockStream::mix(h, d.tokens[i]);
        }
        return d;
    }
private:
    const MockTarget& t_;
    MockStream& s_;
    Config c_;
    eagle::Timings times_;
    std::uint64_t f_ = 0, pending_ = 0, n_ = 0;
};

} // namespace eagle_test
