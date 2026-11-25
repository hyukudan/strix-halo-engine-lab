// Inkling 2 Mini binding of the shared block-candidate lever: all-OFF defaults, EXACT greedy only,
// and the same outcome (success for OFF, the same refusal reason otherwise) through the switch, the
// environment and, when the app header is available, the real CLI parser. The server start-up arm
// runs through the production argument parser in mini_serve_test.cpp.
#include "strix/profiles/inkling2_mini.hpp"
#include "inkling2/inkling2_mini_block_candidate_cuda.hpp"
#include "inkling2/inkling2_mini_weight_options.hpp"
#include "strix/spec/block_draft_policy.hpp"
#if defined(STRIX_INKLING2_APP_CLI_TEST)
#include "inkling2_mini_cli.hpp"
#include <vector>
#endif
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace sp = strix::spec;
namespace pf = strix::profile;
namespace mini = strix::inkling2;
namespace {
void check(bool ok, const std::string& what) { if (!ok) throw std::runtime_error(what); }

// The reason each tier must give for each value; empty = the value must resolve.
std::string_view expected_reason(pf::Name tier, std::string_view value) {
    if (value == "off") return {};
    if (value == "dflash2:8:greedy") return "native forward and verify identity are not bound";
    if (value == "dflash2:8:sampled")
        return tier == pf::Name::standard ? "EXACT permits only greedy-identical verification"
                                          : "sampled rejection hook unqualified and unbound";
    if (value == "dflash2:7:greedy") return "requires block 8";
    return "expects off or dflash2";
}

template <class Resolve>
std::string outcome(Resolve&& resolve) {
    try {
        resolve();
        return {};
    } catch (const std::exception& e) {
        return e.what();
    }
}

void check_outcome(const std::string& got, std::string_view want, const std::string& arm) {
    if (want.empty()) check(got.empty(), arm + ": explicit value must resolve, got '" + got + "'");
    else check(got.find(want) != std::string::npos, arm + ": expected '" + std::string(want) + "', got '" + got + "'");
}

#if defined(STRIX_INKLING2_APP_CLI_TEST)
std::string cli_outcome(std::string_view tier, std::string_view value) {
    std::vector<std::string> storage{"mini", "model", "1", "--generate", "4", "--profile", std::string(tier),
                                     "--block-drafter", std::string(value)};
    std::vector<char*> argv;
    for (auto& item : storage) argv.push_back(item.data());
    const strix::serve::SerialGenerationFamily family{"mini", "Mini", "MODEL",
        8U, 8U, 16U, 0U, nullptr, 8U, true, false,
        [](const std::string&) -> std::shared_ptr<const strix::serve::BpeTokenizer> { return {}; },
        true, [](std::string_view) { return std::string{}; }, true};
    return outcome([&] {
        const auto options = strix::inkling2::cli::parse(static_cast<int>(argv.size()), argv.data(), family);
        check(!sp::block_draft_selection(options.tier).enabled(), "CLI selection must stay OFF");
    });
}
#endif

void policy_tests() {
    const auto& engine = strix::inkling2::kMiniProfile;
    const std::pair<pf::Name, std::string_view> tiers[]{
        {pf::Name::standard, "exact"}, {pf::Name::fast, "fast"}, {pf::Name::turbo, "turbo"}};
    for (const auto& [tier, tier_name] : tiers) {
        pf::Inputs cli;
        cli.profile = tier;
        cli.env = [](const char*) -> const char* { return nullptr; };
        check(!sp::block_draft_selection(pf::resolve(engine, cli)).enabled(), "all defaults OFF");
        auto previous = engine;
        previous.table = engine.table.first(engine.table.size() - 1);
        check(pf::resolve(previous, cli).fingerprint() == pf::resolve(engine, cli).fingerprint(),
              "OFF preserves previous profile fingerprint");
        for (const auto* value : {"off", "dflash2:8:greedy", "dflash2:8:sampled", "dflash2:7:greedy", "unknown:8:greedy"}) {
            const auto want = expected_reason(tier, value);
            const std::string arm = std::string(tier_name) + " " + value;
            strix::inkling2::MiniWeightOptions switch_options, env_options;
            switch_options.tier.profile = tier;
            switch_options.tier.env = cli.env;
            check(strix::inkling2::parse_mini_weight_option(switch_options, "--block-drafter", [&] { return value; }, true),
                  "--block-drafter consumed by the common model option parser");
            env_options.tier.profile = tier;
            env_options.tier.env = [&](const char* key) -> const char* {
                return std::string_view(key) == "STRIX_BLOCK_DRAFTER" ? value : nullptr;
            };
            const auto resolve = [](const strix::inkling2::MiniWeightOptions& input) {
                return outcome([&] {
                    const auto resolution = strix::inkling2::resolve_mini_profile(input, true);
                    check(!sp::block_draft_selection(resolution).enabled(), "a resolving value must be OFF");
                });
            };
            check_outcome(resolve(switch_options), want, arm + " (switch)");
            check_outcome(resolve(env_options), want, arm + " (environment)");
#if defined(STRIX_INKLING2_APP_CLI_TEST)
            check_outcome(cli_outcome(tier_name, value), want, arm + " (CLI parser)");
#endif
        }
    }
    std::cout << "PASS resolver: common row, OFF resolves, refusal reason per tier, switch / environment"
#if defined(STRIX_INKLING2_APP_CLI_TEST)
                 " / CLI parser"
#endif
                 " agree, OFF in all tiers\n";
}

void cuda_candidate_step_tests() {
    const auto& engine = mini::kMiniProfile;
    pf::Inputs options;
    options.env = [](const char*) -> const char* { return nullptr; };
    const auto off = [&] {
        return !mini::MiniCudaBlockCandidateStep::prepare(pf::resolve(engine, options), {}, 0U, 0U);
    };
    for (auto tier : {pf::Name::standard, pf::Name::fast, pf::Name::turbo}) {
        options.profile = tier;
        options.cli.clear();
        check(off(), "default OFF must bypass CUDA candidate preparation");
        options.cli.emplace_back(sp::kBlockDraftLeverName, "off");
        check(off(), "explicit OFF must bypass CUDA candidate preparation");
    }

    options.cli.front().second = "dflash2:8:greedy";
    const auto greedy = pf::resolve(engine, options);
    const std::array ids{11U, 17U, 23U};
    const auto step = mini::MiniCudaBlockCandidateStep::prepare(greedy, ids, mini::kMaxContext - 4U, mini::kMaxContext);
    check(step.has_value(), "explicit greedy selection must prepare a descriptor");
    const auto& block = step->descriptor();
    check(block.h.rows == 4U && block.h.valid_mask == 7U && block.h.feed_mask == 0U,
          "three forced rows need a width-four descriptor with masked padding");
    check(block.h.step == mini::kMaxContext - 4U, "descriptor must carry the absolute first position");
    for (unsigned row = 0; row < ids.size(); ++row) {
        const auto& data = block.row[row];
        check(strix::exec::step_row_valid(block, row) && data.token == ids[row] && data.position == block.h.step + row &&
              data.slot == 0U && data.flags == strix::exec::step_valid &&
              data.parent == (row == 0U ? strix::exec::kStepNoParent : row - 1U) &&
              data.ancestors == (1U << (row + 1U)) - 1U, "forced-row token, position and causal order");
    }
    check(!strix::exec::step_row_valid(block, 3U), "padding must not write features or state");
    check_outcome(outcome([&] {
        (void)mini::MiniCudaBlockCandidateStep::prepare(greedy, ids, mini::kMaxContext - 3U, mini::kMaxContext);
    }), "padded verify window exceeds session capacity", "CUDA candidate at the context cap");

    options.cli.front().second = "dflash2:8:sampled";
    check_outcome(outcome([&] {
        (void)mini::MiniCudaBlockCandidateStep::prepare(pf::resolve(engine, options), ids, 0U, 8U);
    }), "sampled rejection hook unqualified and unbound", "CUDA sampled candidate");
}
}  // namespace
int main() {
    try {
        policy_tests();
        cuda_candidate_step_tests();
    } catch (const std::exception& e) {
        std::cerr << "inkling2-mini-block-drafter: " << e.what() << '\n';
        return 1;
    }
    std::cout << "inkling2-mini-block-drafter: ok\n";
    return 0;
}
