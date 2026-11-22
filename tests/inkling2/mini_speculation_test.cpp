#include "inkling2/inkling2_mini_state.hpp"
#include "strix/session/growing_visit_checkpoint.hpp"
#include "strix/spec/engine_target.hpp"
#include "strix/spec/greedy_continuation.hpp"

#include <functional>
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

// Nine visible positions plus an eight-row append exactly fill the
// 16-slot ring. The last staged row can overwrite the old visible tail.
struct ResidentFixtureTraits : FixtureTraits {
    static constexpr auto layout(LayerVisit visit, unsigned context, unsigned max_chunk) {
        const auto mini = FixtureTraits::layout(visit, context, max_chunk);
        return session::make_visit_cache_layout(context, max_chunk, mini.kv.width,
            mini.window == 0U ? 0U : 9U, mini.convolution_widths, mini.history_rows);
    }
};

class ResidentFixture {
    using Plan = session::VisitCachePlan<ResidentFixtureTraits>;

    struct Backend {
        std::vector<std::function<void()>> queued;
        bool fail_finish{};

        template <class T>
        void copy(std::span<const T> input, std::span<T> output) {
            queued.emplace_back([=] { std::ranges::copy(input, output.begin()); });
        }

        void finish() {
            auto work = std::exchange(queued, {});
            for (const auto& op : work) op();
            if (fail_finish) throw std::runtime_error("fixture finish failure");
        }
    };

public:
    using Rows = std::span<const std::uint32_t>;
    enum class LogitRows { all };

    ResidentFixture(Rows truth, unsigned capacity)
        : truth_(truth), plan_(capacity, 8U), keys_(plan_.kv_elements_per_plane()), values_(keys_.size()),
          histories_(plan_.convolution_elements()), saved_histories_(histories_.size()),
          saved_kv_(plan_.kv_checkpoint_elements()), state_(plan_, {keys_, values_}, histories_) {}

    std::uint32_t decode(std::uint32_t token) {
        append({&token, 1U}, false);
        return reply_[0];
    }

    Rows forward_rows(const std::uint32_t* tokens, unsigned rows, LogitRows) {
        const Rows inputs{tokens, rows};
        append(inputs, true);
        std::ranges::copy(inputs, pending_.begin());
        ++widths_[rows];
        return std::span(reply_).first(rows);
    }

    void read_top(Rows top, unsigned rows, std::uint32_t* out) const {
        require(state_.staged() && top.size() == rows, "resident reply was not staged");
        std::ranges::copy(top, out);
    }

    void commit(unsigned keep) {
        state_.commit_staged(backend_, keep, [&](unsigned rows) {
            ++replays_;
            append(std::span(pending_).first(rows), false);
        });
    }

    void sync() const { require(backend_.queued.empty(), "resident reply has pending copies"); }
    unsigned frontier() const { return state_.committed(); }
    unsigned widths(unsigned rows) const { return widths_[rows]; }
    unsigned replays() const { return replays_; }
    void fail_finish() { backend_.fail_finish = true; }
    bool poisoned() const { return state_.poisoned(); }

    bool same_state(const ResidentFixture& plain) const {
        if (frontier() != plain.frontier() || histories_ != plain.histories_) return false;
        for (const auto visit : plan_.visits()) {
            const auto& layout = plan_.visit(visit);
            const auto first = layout.window == 0U ? 0U : frontier() - std::min(layout.window, frontier());
            for (auto pos = first; pos < frontier(); ++pos) {
                const auto offset = layout.kv_offset + attn::cache_rows::row_offset(layout.kv, pos);
                for (unsigned i = 0; i < layout.kv.width; ++i) {
                    if (keys_[offset + i] != plain.keys_[offset + i] ||
                        values_[offset + i] != plain.values_[offset + i]) return false;
                }
            }
        }
        return true;
    }

private:
    void append(Rows tokens, bool stage) {
        const session::KvAppendChunk chunk{frontier(), static_cast<unsigned>(tokens.size())};
        state_.append(backend_, chunk, saved_histories_, saved_kv_, [&] {
            backend_.queued.emplace_back([&, chunk, tokens] {
                for (unsigned row = 0; row < chunk.rows; ++row) {
                    const auto position = chunk.first_position + row;
                    for (const auto visit : plan_.visits()) {
                        const auto& layout = plan_.visit(visit);
                        auto slices = state_.visit(visit);
                        const auto offset = attn::cache_rows::row_offset(layout.kv, position);
                        const auto marker = static_cast<std::uint16_t>(tokens[row] + visit.logical_layer + 1U);
                        slices.kv.keys[offset] = marker;
                        slices.kv.keys[offset + 1U] = static_cast<std::uint16_t>(position + 1U);
                        slices.kv.values[offset] = static_cast<std::uint16_t>(3U * marker);
                        slices.kv.values[offset + 1U] = static_cast<std::uint16_t>(5U * marker);
                        for (std::size_t i = 0; i < slices.history.size(); ++i) {
                            const auto history = slices.history[i];
                            std::rotate(history.begin(), history.begin() + 2, history.end());
                            history[history.size() - 2U] = static_cast<float>(marker + i);
                            history.back() = static_cast<float>(position + i);
                        }
                    }
                    reply_[row] = position + 1U < truth_.size() ? truth_[position + 1U] : 0U;
                }
            });
        }, [&] { sync(); }, nullptr, stage);
    }

    Rows truth_;
    Plan plan_;
    std::vector<std::uint16_t> keys_, values_;
    std::vector<float> histories_, saved_histories_;
    std::vector<std::uint16_t> saved_kv_;
    Backend backend_;
    session::KvStore::RowAppendState<Plan> state_;
    std::array<std::uint32_t, 8> pending_{}, reply_{};
    std::array<unsigned, 9> widths_{};
    unsigned replays_{};
};

void check_resident_rollback() {
    ResidentFixture plain({}, 64U), staged({}, 64U);
    spec::EngineVerifyTarget target(staged);
    for (unsigned pos = 0; pos < 16U; ++pos) {
        plain.decode(pos);
        staged.decode(pos);
    }
    const std::array<std::uint32_t, 8> inputs{31, 32, 33, 34, 35, 36, 37, 38};
    std::vector<std::uint32_t> greedy;
    target.verify(inputs, greedy);
    require(staged.frontier() == 16U && greedy.size() == inputs.size(), "verify published a frontier");
    try {
        staged.decode(39U);
        throw std::logic_error("append admitted with a pending verify");
    } catch (const std::invalid_argument&) {}
    target.commit(0U);
    require(staged.same_state(plain) && staged.replays() == 0U, "discard lost an endangered ring row");
    target.verify(inputs, greedy);
    target.commit(2U);
    for (const auto token : std::span(inputs).first(2U)) plain.decode(token);
    require(staged.same_state(plain) && staged.replays() == 1U, "partial prefix left rejected KV or history");
    target.verify(inputs, greedy);
    target.commit(static_cast<unsigned>(inputs.size()));
    for (const auto token : inputs) plain.decode(token);
    require(staged.same_state(plain) && staged.replays() == 1U, "full acceptance replayed or lost rows");
    target.verify(inputs, greedy);
    staged.fail_finish();
    try {
        target.commit(2U);
        throw std::logic_error("failed restore published a prefix");
    } catch (const std::runtime_error&) {}
    require(staged.poisoned() && staged.frontier() == plain.frontier(), "restore failure left a live session");
}

void check_resident_continuation(std::span<const std::uint32_t> truth) {
    constexpr unsigned prompt_rows = 144U;
    constexpr unsigned limit = 96U;
    const auto prompt = truth.first(prompt_rows);
    ResidentFixture plain(truth, prompt_rows + limit), staged(truth, prompt_rows + limit);
    for (auto token : prompt) {
        plain.decode(token);
        staged.decode(token);
    }
    const SerialTokenContinuationRequest request{truth[prompt_rows], prompt_rows, 10000U,
                                                 prompt_rows + limit, limit, eos};
    const auto ordinary = execute_serial_token_continuation(request, [&](auto token, auto) { return plain.decode(token); });
    spec::BlockConfig config;
    config.max_block = 8U;
    config.fixed_costs = true;
    spec::set_context_cost(config.arbiter, "off");
    spec::BlockDrafter drafter(config);
    for (unsigned row = 1U; row <= config.max_block; ++row) drafter.seed_verify_ms(row, 1.0);
    spec::EngineVerifyTarget target(staged);
    const auto result = spec::execute_block_token_continuation(request, prompt, drafter, target,
                                                               config.max_block, [](auto) {});
    require(result.generated_tokens == ordinary.generated_tokens, "resident verify changed greedy IDs");
    require(result.stop_reason == ordinary.stop_reason && staged.same_state(plain), "resident continuation changed state");
    require(staged.widths(1U) != 0U && staged.widths(8U) != 0U && staged.replays() != 0U,
            "resident fixture did not exercise serial, wide and partial routes");
}

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
        check_resident_rollback();
        std::vector<std::uint32_t> copy(320);
        for (std::size_t i = 0; i < copy.size(); ++i) copy[i] = static_cast<std::uint32_t>(i % 64U) + 1U;
        check_continuation(copy, 128U, 96U, 320U, spec::Source::lookup);
        // A structured repeat with a mutation exercises mismatch correction.
        copy[187] = 234U;
        check_resident_continuation(copy);
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
