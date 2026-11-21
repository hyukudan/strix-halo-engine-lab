#include "inkling2/inkling2_mini_state.hpp"
#include "strix/session/growing_visit_checkpoint.hpp"
#include "strix/spec/greedy_continuation.hpp"

#include <iostream>
#include <numeric>

namespace {

using namespace strix;
using namespace strix::inkling2;
constexpr std::uint32_t eos = 9999U;

void require(bool condition, const char* error) {
    if (!condition) throw std::runtime_error(error);
}

// Keep the real Mini visit schedule, window and four recurrent streams.
// Shrink the row widths so rollback is testable without model weights.
struct FixtureTraits : MiniCacheTraits {
    static constexpr auto layout(LayerVisit visit, unsigned context, unsigned max_chunk) {
        const auto mini = MiniCacheTraits::layout(visit, context, max_chunk);
        return session::make_visit_cache_layout(context, max_chunk, 2U, mini.window,
                                                std::array{2U, 2U, 2U, 2U}, mini.history_rows);
    }
};

class CpuFixture final : public spec::VerifyTarget {
    using Window = session::GrowingVisitWindow<FixtureTraits, float>;

public:
    CpuFixture(std::span<const std::uint32_t> truth, unsigned capacity) : plan_(capacity, 1U), truth_(truth) {}

    std::uint32_t step(std::uint32_t token) {
        for (const auto visit : plan_.visits()) {
            const auto& layout = plan_.visit(visit);
            auto& cache = storage_[FixtureTraits::index(visit)];
            cache.prepare(layout, position_);
            const auto slot = attn::cache_rows::row_offset(layout.kv, position_);
            cache.keys[slot] = static_cast<float>(token + 1U);
            cache.keys[slot + 1U] = static_cast<float>(position_ + 1U);
            cache.values[slot] = -cache.keys[slot];
            cache.values[slot + 1U] = -cache.keys[slot + 1U];
            for (auto& history : cache.history) {
                std::rotate(history.begin(), history.begin() + 2, history.end());
                history[history.size() - 2U] = cache.keys[slot];
                history.back() = cache.values[slot + 1U];
            }
        }
        ++position_;
        return position_ < truth_.size() ? truth_[position_] : 0U;
    }

    void verify(std::span<const std::uint32_t> inputs, std::vector<std::uint32_t>& greedy) override {
        const auto drafts = inputs.subspan(1U);
        saw_eos_draft_ |= std::ranges::find(drafts, eos) != drafts.end();
        window_.verify(plan_, storage_, position_, inputs, greedy, [this](auto token) { return step(token); });
    }

    void commit(std::uint32_t rows) override {
        position_ -= window_.commit(storage_, rows);
    }

    bool same_state(const CpuFixture& other) const {
        if (position_ != other.position_) return false;
        return std::ranges::equal(storage_, other.storage_, [](const auto& a, const auto& b) {
            return a.keys == b.keys && a.values == b.values && a.history == b.history;
        });
    }

    bool saw_eos_draft() const { return saw_eos_draft_; }

private:
    Window::Plan plan_;
    std::span<const std::uint32_t> truth_;
    Window::Storage storage_;
    Window window_;
    unsigned position_{};
    bool saw_eos_draft_{};
};

class FixtureChain final : public spec::ChainDrafter {
public:
    explicit FixtureChain(std::span<const std::uint32_t> remaining) : remaining_(remaining) {}

    void draft(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out) override {
        const auto tail = remaining_.subspan(prefix.size(), k);
        out.assign(tail.begin(), tail.end());
    }

    void accept(std::span<const std::uint32_t> emitted) override {
        require(std::ranges::equal(remaining_.first(emitted.size()), emitted), "chain saw a rejected suffix");
        remaining_ = remaining_.subspan(emitted.size());
    }

private:
    std::span<const std::uint32_t> remaining_;
};

void check_rollback() {
    const unsigned start = kSlidingWindow - 2U;
    CpuFixture serial({}, start + 8U), staged({}, start + 8U);
    for (unsigned pos = 0; pos < start; ++pos) {
        serial.step(pos + 19U);
        staged.step(pos + 19U);
    }
    const std::array<std::uint32_t, 6> inputs{3, 4, 5, 6, 7, 8};
    std::vector<std::uint32_t> greedy;
    staged.verify(inputs, greedy);
    // The suffix overwrote old sliding slots at and beyond the wrap.
    staged.commit(2U);
    serial.step(inputs[0]);
    serial.step(inputs[1]);
    require(staged.same_state(serial), "rejecting wrapped rows changed committed CPU state");

    staged.verify(inputs, greedy);
    staged.commit(3U);
    for (const auto token : std::span(inputs).first(3)) serial.step(token);
    require(staged.same_state(serial), "committing a wrapped prefix changed CPU state");
}

void check_continuation(std::span<const std::uint32_t> truth, std::uint32_t prompt_size,
                        std::uint32_t max_new, unsigned capacity, spec::Source expected_arm,
                        bool expect_eos_draft = false) {
    CpuFixture plain(truth, capacity), staged(truth, capacity);
    const auto prompt = truth.first(prompt_size);
    for (const auto token : prompt) {
        plain.step(token);
        staged.step(token);
    }
    const SerialTokenContinuationRequest request{truth[prompt_size], prompt_size, 10000U, capacity, max_new, eos};
    const auto plain_result = execute_serial_token_continuation(request,
        [&](auto token, auto) { return plain.step(token); });

    spec::BlockConfig config;
    config.max_block = 8;
    config.fixed_costs = true;
    spec::set_context_cost(config.arbiter, "off");
    FixtureChain chain(truth.subspan(prompt_size + 1U));
    config.chain = expected_arm == spec::Source::model;
    config.copy = !config.chain;
    config.arbiter.charge_probe = !config.chain;  // the chain fixture must exercise a deep EOS window
    spec::BlockDrafter drafter(config, config.chain ? &chain : nullptr);
    // Synthetic sublinear timings exercise the shared policy independently of
    // the CPU oracle's actual serial cost and the test runner's clock.
    for (std::uint32_t row = 1; row <= config.max_block; ++row)
        drafter.seed_verify_ms(row, 1.0 + 0.05 * static_cast<double>(row - 1U));
    std::vector<std::uint32_t> streamed;
    const auto spec_result = spec::execute_block_token_continuation(
        request, prompt, drafter, staged, config.max_block, [&](auto token) { streamed.push_back(token); });
    require(plain_result.generated_tokens == spec_result.generated_tokens, "spec greedy IDs differ");
    require(streamed == plain_result.generated_tokens, "spec streamed IDs differ");
    require(plain_result.final_position == spec_result.final_position, "committed frontier differs");
    require(plain_result.steps_executed == spec_result.steps_executed, "logical step count differs");
    require(plain_result.stop_reason == spec_result.stop_reason, "stop reason differs");
    require(staged.same_state(plain), "spec final KV or history differs");
    if (expected_arm == spec::Source::serial) {
        require(drafter.stats().source_passes[0] == drafter.stats().passes, "prose used speculative rows");
    } else {
        const auto arm = static_cast<std::size_t>(expected_arm);
        require(drafter.stats().source_accepted[arm] > 0U, "expected drafter accepted no drafts");
        require(drafter.stats().passes < plain_result.steps_executed, "expected drafter saved no passes");
    }
    if (expect_eos_draft) require(staged.saw_eos_draft(), "EOS was never drafted");
}

}  // namespace

int main() {
    try {
        check_rollback();
        std::vector<std::uint32_t> copy(320);
        for (std::size_t i = 0; i < copy.size(); ++i) copy[i] = static_cast<std::uint32_t>(i % 64U) + 1U;
        check_continuation(copy, 128U, 96U, 320U, spec::Source::lookup);
        // A structured repeat with a mutation exercises mismatch correction.
        copy[187] = 234U;
        check_continuation(copy, 144U, 160U, 219U, spec::Source::lookup);
        // A novel continuation shares vocabulary with the prompt but lacks a
        // matching continuation and remains on the plain arm.
        std::vector<std::uint32_t> prose(200);
        std::iota(prose.begin(), prose.end(), 100U);
        for (std::size_t i = 64; i + 1U < prose.size(); i += 3) {
            prose[i] = 102U;
            prose[i + 1U] = 103U;
        }
        check_continuation(prose, 64U, 120U, 200U, spec::Source::serial);
        copy[187] = 60U;
        copy[63] = copy[127] = copy[191] = 9999U;
        check_continuation(copy, 128U, 96U, 320U, spec::Source::lookup);
        copy[130] = eos;
        check_continuation(copy, 128U, 96U, 320U, spec::Source::model, true);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
