#include "inkling2/inkling2_mini_checkpoint.hpp"
#include "inkling2/inkling2_mini_host.hpp"
#include "strix/head_argmax.hpp"
#include "strix/serve/cohort_decode.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {
namespace sv = strix::serve;
using strix::inkling2::MiniHostDecoder;
constexpr unsigned kContext = 16;

unsigned next_token(MiniHostDecoder& decoder, unsigned token) {
    const auto pick = strix::head_argmax::reference(decoder.step(token, true));
    if (pick.nonfinite != 0 || pick.index == strix::head_argmax::kNone)
        throw std::runtime_error("CPU reference returned invalid logits");
    return pick.index;
}

// The CPU model reference evaluates one row at a time. This binding tests the
// cohort's greedy decisions against independent serial reference sessions.
class ModelReferenceCohort final : public sv::CohortBackend {
public:
    explicit ModelReferenceCohort(std::string checkpoint) : checkpoint_(std::move(checkpoint)) {}
    std::uint32_t lanes() const override { return 2U; }
    std::uint32_t rows_per_forward() const override { return 2U; }
    std::uint32_t max_positions() const override { return kContext; }
    void prefill(std::uint32_t lane, std::span<const std::uint32_t> ids, bool first) override {
        if (first) decoders_.at(lane) = std::make_unique<MiniHostDecoder>(checkpoint_, kContext);
        for (auto id : ids) (void)decoders_.at(lane)->step(id, false);
    }
    void verify(std::span<const sv::CohortGroup> groups, std::span<const std::uint32_t> ids,
                std::span<std::uint32_t> top) override {
        peak = std::max(peak, groups.size());
        for (std::size_t i = 0; i < groups.size(); ++i) {
            if (groups[i].rows != 1U) throw std::logic_error("test only supports greedy rows");
            top[i] = next_token(*decoders_.at(groups[i].lane), ids[i]);
        }
    }
    void commit(std::span<const std::uint32_t>) override {}
    void release(std::uint32_t lane) override { decoders_.at(lane).reset(); }
    std::size_t peak = 0;
private:
    std::string checkpoint_;
    std::array<std::unique_ptr<MiniHostDecoder>, 2> decoders_;
};

// A provided row (the splice path of image rows) takes the token rows' arithmetic: step_row with the BF16 embedding
// row of token t gives the bits of step(t), at the first position and after a token row.
void provided_row_equals_token_row(const std::string& checkpoint) {
    constexpr unsigned token = 80377;
    strix::inkling2::MiniTextWeights weights;
    if (const auto status = weights.open(checkpoint); !status) throw std::runtime_error(status.detail);
    std::vector<float> widened(strix::inkling2::kHiddenWidth);
    if (const auto status = weights.read("model.llm.embed.weight", widened,
                                         std::size_t{token} * strix::inkling2::kHiddenWidth); !status)
        throw std::runtime_error(status.detail);
    std::vector<std::uint16_t> row(widened.size());
    for (std::size_t i = 0; i < row.size(); ++i) {
        const auto bits = std::bit_cast<std::uint32_t>(widened[i]);
        if ((bits & 0xFFFFU) != 0U) throw std::runtime_error("embedding table is not BF16");
        row[i] = static_cast<std::uint16_t>(bits >> 16U);
    }
    for (const bool after_token : {false, true}) {
        MiniHostDecoder by_token(checkpoint, kContext);
        MiniHostDecoder by_row(checkpoint, kContext);
        if (after_token) {
            (void)by_token.step(220, false);
            (void)by_row.step(220, false);
        }
        const auto expected = by_token.step(token, true);
        const auto provided = by_row.step_row(row, true);
        if (expected.size() != provided.size() ||
            std::memcmp(expected.data(), provided.data(), expected.size() * sizeof(float)) != 0)
            throw std::runtime_error("a provided embedding row differs from its token row");
    }
}
} // namespace

int main(int argc, char** argv) try {
    if (argc != 2) throw std::invalid_argument("expected checkpoint path");
    provided_row_equals_token_row(argv[1]);
    const std::array requests{sv::CohortRequest{0, {220, 80377}, 3, {}},
                              sv::CohortRequest{1, {11, 220, 4896}, 3, {}}};
    ModelReferenceCohort backend(argv[1]);
    sv::CohortConfig config;
    config.context_draft = false;
    config.max_draft = 0;
    config.prompt_share = 0;
    sv::CohortDecoder cohort(backend, config);
    for (const auto& request : requests) cohort.submit(request);
    cohort.run();
    const auto finished = cohort.take_finished();
    if (finished.size() != requests.size() || backend.peak != 2U)
        throw std::runtime_error("CPU reference cohort did not execute both requests together");
    for (const auto& result : finished) {
        const auto& request = requests.at(static_cast<std::size_t>(result.id));
        MiniHostDecoder solo(argv[1], kContext);
        for (const auto id : std::span(request.prompt).first(request.prompt.size() - 1U))
            (void)solo.step(id, false);
        std::vector<unsigned> expected;
        unsigned pending = request.prompt.back();
        for (unsigned index = 0; index < request.max_tokens; ++index) {
            pending = next_token(solo, pending);
            expected.push_back(pending);
        }
        if (!result.error.empty() || result.ids != expected)
            throw std::runtime_error("cohort ids differ from the serial CPU model reference");
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
