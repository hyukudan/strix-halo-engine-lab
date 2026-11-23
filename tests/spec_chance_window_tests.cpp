#include "strix/spec/block_parallel.hpp"
#include "strix/spec/drafting.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
namespace sp = strix::spec;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check_pick(const sp::Decision& d, sp::Source source, std::uint32_t k, std::uint32_t rows,
                double tokens, double ms, const char* message) {
    check(d.source == source && d.k == k && d.rows == rows && !d.probe &&
          d.padded == (rows > k + 1U) && d.expected_tokens == tokens && d.expected_ms == ms, message);
}

sp::ArbiterConfig chance_config() {
    sp::ArbiterConfig config;
    config.mode = sp::ArbiterMode::ChanceWindow;
    config.windows = {1, 2, 3, 4};
    config.chance_window.calibration = {{0.0f, 0.125}, {0.5f, 0.5}, {0.875f, 1.0}};
    for (const double ms : {1.0, 1.5, 1.75, 2.0}) {
        std::array<double, sp::context_bins> row;
        row.fill(ms);
        config.chance_window.verify_ms.push_back(row);
    }
    return config;
}

void window_cases() {
    auto config = chance_config();
    auto& costs = config.chance_window.verify_ms;
    costs[1][1] = 1.25;
    costs[2][1] = 2.5;
    costs[3][1] = 4.0;
    sp::Arbiter arbiter(config);
    constexpr std::array high{1.0f, 0.875f, 1.0f};
    constexpr std::array early_miss{1.0f, 0.0f, 1.0f};
    constexpr std::array first_low{0.0f, 1.0f, 1.0f};
    constexpr std::array flat_low{0.125f, 0.125f, 0.125f};
    sp::StepInput input;
    input.model_confidence = high;
    // 1/1, 2/1.5, 3/1.75, 4/2.
    check_pick(arbiter.decide(input), sp::Source::model, 3, 4, 4.0, 2.0, "all confident uses max window");
    input.model_confidence = early_miss;
    // After the first low-confidence row, cumulative survival at both depths is 1/8.
    check_pick(arbiter.decide(input), sp::Source::model, 1, 2, 2.0, 1.5, "cut before the first low-confidence draft");
    input.model_confidence = first_low;
    check_pick(arbiter.decide(input), sp::Source::serial, 0, 1, 1.0, 1.0, "a low first draft cuts to one verify row");
    input.model_confidence = flat_low;
    check_pick(arbiter.decide(input), sp::Source::serial, 0, 1, 1.0, 1.0, "flat low confidence self-disables");

    input.model_confidence = high;
    input.context = 2047;
    check_pick(arbiter.decide(input), sp::Source::model, 3, 4, 4.0, 2.0, "short context cost");
    input.context = 2048;
    // Same probabilities. The new context bin prices T=2,3,4 at 1.25, 2.5, 4.
    check_pick(arbiter.decide(input), sp::Source::model, 1, 2, 2.0, 1.25, "verify table flips the choice");
    arbiter.observe({sp::Source::model, 0, 1, 1, 2, 99.0, 0.0, input.context}, {});
    check(arbiter.verify_ms(2, input.context) == 1.25, "the supplied cost table stays fixed");

    input.context = 0;
    input.remaining = 2;
    input.max_rows = 3;
    config.windows = {1, 3, 4};
    sp::Arbiter sparse(config);
    check_pick(sparse.decide(input), sp::Source::model, 1, 3, 2.0, 1.75, "budget and sparse windows require padding");
    input.max_rows = 2;
    check_pick(sparse.decide(input), sp::Source::serial, 0, 1, 1.0, 1.0, "respect physical row capacity");
    input.max_rows = 0;
    check_pick(sparse.decide(input), sp::Source::serial, 0, 0, 0.0, 0.0, "no physical room");

    input.max_rows = 4;
    input.remaining = 4;
    config.windows = {1, 2, 3, 4};
    costs[0][0] = 2.0;
    costs[1][0] = 2.0;
    costs[2][0] = 3.0;
    costs[3][0] = 4.0;
    check_pick(sp::Arbiter(config).decide(input), sp::Source::model, 1, 2, 2.0, 2.0, "draft ties use smaller T");
    costs[0][0] = 1.0;
    check_pick(sp::Arbiter(config).decide(input), sp::Source::serial, 0, 1, 1.0, 1.0, "ties include T=1");

    // A low row does not end the scan. T=4 can still win if its verify cost is lower.
    input.model_confidence = early_miss;
    costs[3][0] = 1.0;
    check_pick(sp::Arbiter(config).decide(input), sp::Source::model, 3, 4, 2.25, 1.0, "score the whole prefix curve");
}

void lookup_case() {
    auto config = chance_config();
    config.lookup_prior = {1.0, 1.0};
    config.lookup_sd = 0.0;
    config.min_lookup_evidence = 1.0;
    config.switch_in = 2.0;
    config.chance_window.verify_ms[1][1] = 1.25;
    config.chance_window.verify_ms[2][1] = 2.5;
    config.chance_window.verify_ms[3][1] = 4.0;
    sp::Arbiter arbiter(config);
    const sp::Proposal lookup{{11, 22, 33}, 8};
    constexpr std::array high{1.0f, 1.0f, 1.0f};
    constexpr std::array medium{0.5f, 0.5f, 0.5f};
    sp::StepInput input;
    input.lookup = &lookup;
    input.model_confidence = high;
    check_pick(arbiter.decide(input), sp::Source::model, 3, 4, 4.0, 2.0, "unobserved lookup stays ineligible");
    constexpr std::array<std::uint32_t, 4> emitted{11, 22, 33, 44};
    arbiter.observe({sp::Source::model, 0, 3, 3, 4, 2.0, 0.0, 0}, emitted);
    input.context = 2048;
    input.model_confidence = medium;
    // The observed lookup has a bounded cumulative survival of 2/3 at each depth.
    // At T=2 it scores (1+2/3)/1.25, above the model's (1+1/2)/1.25 and serial's 1/1.
    check_pick(arbiter.decide(input), sp::Source::lookup, 1, 2, 1.0 + 2.0 / 3.0, 1.25, "lookup wins on expected rate");
    check_pick(arbiter.last_arms()[std::size_t(sp::Source::model)], sp::Source::model, 1, 2, 1.5, 1.25,
               "lookup and model use the same table");
}

void block_confidence_case() {
    struct Forward final : sp::BlockForward {
        bool has_confidence = false;
        void forward(const sp::BlockRequest& req, std::vector<std::uint32_t>& proposals,
                     std::vector<float>& confidence) override {
            check(req.k == 3 && confidence.empty(), "block starts with absent confidence");
            proposals = {9, 8, 7};
            if (has_confidence) confidence = {1.0f, 0.125f, 1.0f};
        }
        void accept(std::span<const std::uint32_t>) override {}
    } forward;
    sp::BlockPolicy policy;
    policy.block = 4;
    policy.ring = 4;
    policy.draft_layers = 1;
    policy.target_layer_count = 1;
    sp::ParallelBlockDrafter drafter(policy, forward);
    sp::Arbiter arbiter(chance_config());
    drafter.start(6, 0);
    std::vector<std::uint32_t> ids;
    for (const bool has_confidence : {true, false}) {
        forward.has_confidence = has_confidence;
        drafter.draft({}, 3, ids);
        check(ids == std::vector<std::uint32_t>{9, 8, 7}, "confidence leaves draft IDs intact");
        sp::StepInput input;
        input.model_confidence = drafter.last_confidence();
        if (has_confidence) {
            check(input.model_confidence.size() == ids.size() && input.model_confidence[1] == 0.125f,
                  "confidence aligns with the corresponding IDs");
            check_pick(arbiter.decide(input), sp::Source::model, 1, 2, 2.0, 1.5, "price the block forward result");
        } else {
            check(input.model_confidence.empty(), "absent confidence has no stale rows");
            check_pick(arbiter.decide(input), sp::Source::serial, 0, 1, 1.0, 1.0, "do not invent model probabilities");
        }
    }
}

template <typename Fn>
void refused(std::string_view field, Fn run) {
    try {
        run();
    } catch (const std::invalid_argument& e) {
        check(std::string_view(e.what()).find(field) != std::string_view::npos, "refusal names the invalid field");
        return;
    }
    throw std::runtime_error("invalid ChanceWindow data was accepted");
}

void refusals() {
    const auto valid = chance_config();
    auto bad = valid;
    const auto construct = [&] { (void)sp::Arbiter(bad); };
    bad.chance_window.calibration.clear();
    refused("ChanceWindow.calibration", construct);
    for (const float edge : {0.0f, -0.125f, 1.125f, std::numeric_limits<float>::quiet_NaN()}) {
        bad = valid;
        bad.chance_window.calibration[1].lower = edge;
        refused("ChanceWindow.calibration", construct);
    }
    for (const double q : {-0.125, 1.125, std::numeric_limits<double>::quiet_NaN()}) {
        bad = valid;
        bad.chance_window.calibration[0].acceptance = q;
        refused("ChanceWindow.calibration", construct);
    }
    bad = valid;
    bad.chance_window.verify_ms.pop_back();
    refused("ChanceWindow.verify_ms", construct);
    for (const double ms : {0.0, -0.125, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
        bad = valid;
        bad.chance_window.verify_ms.back().back() = ms;
        refused("ChanceWindow.verify_ms", construct);
    }
    sp::Arbiter arbiter(valid);
    sp::StepInput input;
    for (const float p : {-0.125f, 1.125f, std::numeric_limits<float>::quiet_NaN()}) {
        input.model_confidence = std::span(&p, 1);
        refused("ChanceWindow.model_confidence", [&] { (void)arbiter.decide(input); });
    }
    const std::array<float, sp::max_proposal + 1> too_many{};
    input.model_confidence = too_many;
    refused("ChanceWindow.model_confidence", [&] { (void)arbiter.decide(input); });
}

struct LegacyTrace {
    std::uint64_t decisions = 14695981039346656037ULL;
    std::uint64_t values = 14695981039346656037ULL;
};

LegacyTrace legacy_trace(std::uint32_t pricing, std::uint32_t offer) {
    sp::ArbiterConfig config;
    if (pricing != 0) {
        config.learn_costs = false;
        constexpr std::array modes{"", "online", "off", "0.08"};
        sp::set_context_cost(config, modes[pricing]);
    }
    config.windows = {1, 2, 3, 5, 8};
    config.min_model_passes = 4;
    sp::Arbiter arbiter(config);
    for (const auto rows : config.windows) arbiter.seed_cost(rows, 1.0 + 0.25 * double(rows - 1));

    std::uint32_t random = 0x73a51b2dU;
    const auto next = [&] {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        return random;
    };
    LegacyTrace trace;
    const auto mix = [](std::uint64_t& hash, std::uint64_t value) {
        hash = (hash ^ value) * 1099511628211ULL;
    };
    constexpr std::array<std::uint64_t, 5> lengths{768, 3072, 12500, 50000, 151000};
    constexpr std::array draft_ms{0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375};
    for (std::uint32_t step = 0; step < 384; ++step) {
        sp::Proposal lookup;
        lookup.tokens.resize(next() % 9U);
        for (auto& token : lookup.tokens) token = 1U + next() % 101U;
        lookup.match_length = next() % 24U;
        lookup.run = step / 3U + 1U;
        lookup.resumed = step % 3U != 0;
        const bool lookup_hit = next() % 4U != 0;
        const auto model_accept = next() % 8U;
        sp::StepInput input;
        input.lookup = &lookup;
        input.model_max_k = next() % 8U;
        input.model_draft_ms = draft_ms;
        if (offer == 1) {
            input.model_pass_tokens = 1.3125 + double(next() % 24U) / 8.0;
            input.model_pass_ms = 1.875 + double(next() % 16U) / 16.0;
        }
        input.force_lookup = offer == 2;
        input.remaining = next() % 12U;
        input.max_rows = next() % 10U;
        input.context = lengths[next() % lengths.size()];
        const double wall_noise = double(next() % 31U) / 64.0;
        const auto decision = arbiter.decide(input);
        mix(trace.decisions, std::uint64_t(decision.source));
        mix(trace.decisions, decision.k);
        mix(trace.decisions, decision.rows);
        mix(trace.decisions, decision.padded);
        mix(trace.decisions, decision.probe);
        mix(trace.values, std::bit_cast<std::uint64_t>(decision.expected_tokens));
        mix(trace.values, std::bit_cast<std::uint64_t>(decision.expected_ms));
        if (decision.rows == 0) continue;
        const auto accepted = decision.source == sp::Source::lookup
            ? (lookup_hit ? decision.k : 0U) : std::min(decision.k, model_accept);
        std::array<std::uint32_t, 8> emitted{};
        if (lookup_hit) std::copy_n(lookup.tokens.begin(), std::min(lookup.tokens.size(), emitted.size()), emitted.begin());
        const double verify_ms = 0.875 + 0.1875 * double(decision.rows) + wall_noise;
        arbiter.observe({decision.source, lookup.bucket(), decision.k, accepted, decision.rows, verify_ms,
                         decision.source == sp::Source::model ? draft_ms[decision.k - 1U] : 0.0, input.context},
                        std::span(emitted).first(accepted + 1U));
    }
    return trace;
}
void legacy_cases() {
    // Captured by compiling this trace against 1dcb06a21847b1fcb8130766ba4300b4836e92c9.
    // Cost modes: learned, online fit, off, fixed. Offers: per-depth, measured-rate, forced lookup.
    constexpr LegacyTrace golden[4][3] = {
        {{0x5528d4d119766544ULL, 0x159ac1fa2b7fb3adULL}, {0x4200251a5ffbdb2dULL, 0xcfbeda2d3b9513e3ULL},
         {0x5b2aa1e43ffd45dcULL, 0x40423899b06b4bbcULL}},
        {{0x32ed4410497c8acbULL, 0xb6de522e72c932b6ULL}, {0xfb40444fea9ea0efULL, 0x7100b7272e59ed78ULL},
         {0x5b2aa1e43ffd45dcULL, 0xd0fecb9b59a49cbeULL}},
        {{0x6d7b5a949feda0e4ULL, 0xe72f9ca7dd94cf7eULL}, {0xab65c394bb5e019fULL, 0x78db573eb721563cULL},
         {0x5b2aa1e43ffd45dcULL, 0x6e6be040e0eedf25ULL}},
        {{0x5092347f2153633bULL, 0x6839194d89bc71c8ULL}, {0x3e6cca15abc92337ULL, 0x2716e5108b11ed75ULL},
         {0x5b2aa1e43ffd45dcULL, 0xfd27fc40e0eedf25ULL}},
    };
    for (std::uint32_t pricing = 0; pricing < 4; ++pricing) {
        for (std::uint32_t offer = 0; offer < 3; ++offer) {
            const auto trace = legacy_trace(pricing, offer);
            check(trace.decisions == golden[pricing][offer].decisions, "legacy decision trace differs from base");
#if defined(_MSC_VER) && defined(_M_X64) && !defined(__clang__)
            // The raw IEEE values also include the MSVC x64 host math library and compiler rounding.
            check(trace.values == golden[pricing][offer].values, "legacy decision values differ bitwise from base");
#endif
        }
    }
}
}  // namespace

int main() {
    try {
        window_cases();
        lookup_case();
        block_confidence_case();
        refusals();
        legacy_cases();
        std::puts("PASS spec ChanceWindow arbiter");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL spec ChanceWindow arbiter: %s\n", e.what());
        return 1;
    }
}
