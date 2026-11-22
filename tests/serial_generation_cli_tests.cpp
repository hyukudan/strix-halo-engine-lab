#include "strix/serial_request.hpp"
#include "strix/serve/serial_generation_cli.hpp"
#include "strix/session/prompt_windows.hpp"
#include "strix/spec/block_drafter.hpp"
#include "strix/token_sampling.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

using namespace strix::serve;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

constexpr std::array scores{0.0F, 2.0F, -100.0F, 1.0F};
unsigned loads = 0;
std::vector<std::uint32_t> inputs;

class Decoder final : public SerialGenerationDecoder, public strix::spec::VerifyTarget {
public:
    explicit Decoder(unsigned rows) : rows_(rows) {}

    std::uint32_t prefill(std::span<const std::uint32_t> ids, std::span<float> logits) override {
        inputs.assign(ids.begin(), ids.end());
        if (!logits.empty()) std::ranges::copy(scores, logits.begin());
        next_ = ids.size() >= 4 ? 1U : 2U;
        return 1;
    }
    std::uint32_t decode(std::uint32_t id, std::span<float> logits) override {
        inputs.push_back(id);
        // The next row picks EOS under both greedy and sampling-v1.
        if (!logits.empty()) std::ranges::copy(std::array{-100.0F, -100.0F, 2.0F, -100.0F}, logits.begin());
        return next_;
    }

    std::uint32_t max_verify_rows(CliSpeculation) const override { return rows_; }
    strix::spec::VerifyTarget* verify_target() override { return this; }
    void verify(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) override {
        check(tokens.size() <= rows_, "verifier width bounded by admission");
        staged_.assign(tokens.begin(), tokens.end());
        greedy.assign(tokens.size(), next_);
    }
    void commit(std::uint32_t keep) override {
        inputs.insert(inputs.end(), staged_.begin(), staged_.begin() + keep);
        staged_.clear();
    }

private:
    unsigned rows_, next_{};
    std::vector<std::uint32_t> staged_;
};

const SerialGenerationFamily family{"fixture", "Fixture", "CHECKPOINT", 9, 4, 64, 2,
    [](const SerialGenerationLoad& load) -> std::unique_ptr<SerialGenerationDecoder> {
        ++loads;
        check(load.checkpoint == "checkpoint" && load.prefill_chunk == 0 && !load.packed_decode, "family binding");
        check(load.positions >= 2 && load.positions <= 64, "context binding");
        return std::make_unique<Decoder>(std::min(8U, load.verify_rows));
    }, 0U, false, false, nullptr, false, nullptr, true};

struct Outcome { int code; std::string out, error; };

Outcome run(std::initializer_list<std::string> args) {
    std::vector<std::string> owned{"fixture", "checkpoint"};
    owned.insert(owned.end(), args.begin(), args.end());
    std::vector<char*> argv;
    for (auto& arg : owned) argv.push_back(arg.data());
    std::ostringstream out, error;
    const int code = run_serial_generation_cli(static_cast<int>(argv.size()), argv.data(), family, out, error);
    return {code, out.str(), error.str()};
}

}  // namespace

int main() try {
    const auto greedy = run({"8", "4", "--generate", "4"});
    check(greedy.code == 0 && greedy.out.starts_with("next_token 1\ngenerated 1 2\n"), "greedy IDs and EOS");
    check(greedy.out.find(" decode_tokens 1 ") != std::string::npos, "count decode calls only");
    check(inputs == std::vector<std::uint32_t>{8, 4, 1}, "consume the selected ID once, leaving EOS pending");

    const auto probe = run({"8", "4"});
    check(probe.code == 0 && probe.out.starts_with("next_token 1\ntiming ") && inputs.size() == 2,
          "zero generation still selects the prompt head");
    const auto one = run({"8", "4", "--generate", "1"});
    check(one.code == 0 && one.out.starts_with("next_token 1\ngenerated 1\n") && inputs.size() == 2,
          "leave the last budgeted token pending");

    std::uint64_t seed = 0;
    for (; seed < 100; ++seed) {
        strix::TokenSampler sampler({1.0F, 2, 1.0F, 0.0F, seed}, strix::TokenSamplerOptions{});
        if (sampler.decide(scores, 2, nullptr).token == 3) break;
    }
    check(seed < 100, "fixture's shared sampler must select a non-greedy ID");
    const auto sampled = run({"8", "4", "--generate", "4", "--top-k", "2", "--seed", std::to_string(seed)});
    check(sampled.code == 0 && sampled.out.starts_with("next_token 3\ngenerated 3 2\n") && inputs.back() == 3,
          "top-k uses shared sampling-v1 and feeds its selected token back");
    const auto disabled = run({"8", "4", "--top-k", "2", "--temperature", "0", "--seed", std::to_string(seed)});
    check(disabled.code == 0 && disabled.out.starts_with("next_token 1\n"), "explicit zero temperature stays greedy");

    const auto repeated = [](std::string mode) {
        return run({"1", "1", "1", "1", "1", "1", "1", "1", "--generate", "16",
                    "--temperature", "0", "--speculation", mode});
    };
    const auto plain = repeated("off");
    const auto plain_inputs = inputs;
    const auto lookup = repeated("lookup");
    const auto generated_end = plain.out.find("timing ");
    check(plain.code == 0 && lookup.code == 0 && generated_end != std::string::npos &&
          lookup.out.starts_with(plain.out.substr(0, generated_end)) && inputs == plain_inputs,
          "forced lookup matches complete greedy IDs and the committed prefix");
    const auto receipt = lookup.out.find(" acceptance ");
    check(receipt != std::string::npos, "lookup receipt present");
    std::istringstream metrics(lookup.out.substr(receipt));
    std::uint64_t accepted{}, drafted{}, passes{};
    std::string label;
    char slash{};
    double mean{};
    metrics >> label >> accepted >> slash >> drafted >> label >> passes >> label >> mean;
    check(metrics && slash == '/' && drafted > 0 && accepted == drafted && passes > 0 &&
          std::abs(mean - static_cast<double>(accepted) / static_cast<double>(passes)) < 1e-5,
          "report actual accepted and drafted counts and mean per proposal");
    const auto automatic = repeated("auto");
    check(automatic.code == 0 && automatic.out.starts_with(plain.out.substr(0, generated_end)) &&
          inputs == plain_inputs && automatic.out.find("strix-spec-stats ") != std::string::npos,
          "greedy auto runs the shared verifier and matches serial IDs and commit");
    const auto miss = run({"8", "4", "--generate", "4", "--speculation", "lookup"});
    check(miss.code == 0 && miss.out.starts_with("next_token 1\ngenerated 1 2\n") &&
          miss.out.find(" acceptance 0/0 draft_passes 0 mean_accepted_length 0") != std::string::npos,
          "lookup with no match preserves the EOS continuation");
    const auto sampled_auto = run({"8", "4", "--generate", "4", "--speculation", "auto", "--temperature", "1"});
    check(sampled_auto.code == 0 && sampled_auto.out.find("strix-spec-stats ") == std::string::npos,
          "sampled auto stays on the plain route");

    const auto previous = loads;
    check(run({"9", "--generate", "1"}).code == 1, "input vocabulary bounds");
    check(run({"1", "--generate", "64"}).code == 1, "context bounds");
    check(run({"1", "--prefill-chunk", "1"}).code == 1, "reject unsupported native flag");
    check(run({"1", "--temperature", "nan"}).code == 1, "reject nonfinite sampling policy");
    check(run({"1", "--generate", "4", "--speculation", "lookup", "--temperature", "1"}).error.find(
              "--speculation lookup requires greedy decoding") != std::string::npos,
          "reject sampled lookup before load");
    check(loads == previous, "reject invalid requests before load");

    unsigned decodes = 0;
    const auto full = strix::execute_serial_token_continuation({1, 8, 4, 8, 10, 2},
        [&](std::uint32_t, std::uint64_t) { ++decodes; return 2U; });
    check(full.stop_reason == strix::SerialRequestStopReason::context_capacity && full.final_position == 8 &&
          full.generated_tokens == std::vector<std::uint32_t>{1} && decodes == 0,
          "context boundary publishes the pending token without target work");

    constexpr std::array prompt{1U, 2U, 3U, 4U, 5U};
    std::vector<unsigned> ordered;
    std::vector<std::size_t> sizes;
    strix::session::prompt_windows::append_chunks(std::span<const unsigned>(prompt), 2,
        [&](std::span<const unsigned> chunk, bool last) {
            sizes.push_back(chunk.size());
            ordered.insert(ordered.end(), chunk.begin(), chunk.end());
            check(last == (ordered.size() == prompt.size()), "head on final chunk only");
        });
    check(sizes == std::vector<std::size_t>{2, 2, 1} && std::ranges::equal(ordered, prompt), "ordered chunk tail");
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
