#include "inkling2_mini_cli.hpp"

#include "inkling2/inkling2_mini_host.hpp"
#if defined(STRIX_INKLING2_MEDIA)
#include "inkling2_mini_media.hpp"
#endif
#include "strix/serve/inkling2_chat.hpp"
#include "strix/session/embed_splice.hpp"
#include "strix/serial_request.hpp"
#include "strix/session/prompt_windows.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>

namespace {

class CpuCliDecoder final : public strix::serve::SerialGenerationDecoder {
public:
    CpuCliDecoder(const std::string& checkpoint, unsigned max_positions, const char* dspark = nullptr)
        : decoder_(checkpoint, max_positions, dspark) {}

    unsigned prefill(std::span<const unsigned> prompt, std::span<float> logits) override {
        unsigned next{};
        strix::session::prompt_windows::append_chunks(prompt, 1U, [&](auto chunk, bool last) {
            const auto row = decoder_.step(chunk.front(), last);
            if (last) next = select(row, logits);
        });
        return next;
    }

    unsigned decode(unsigned token, std::span<float> logits) override {
        return select(decoder_.step(token, true), logits);
    }

    // Host reference of a prompt with provided rows: each spliced row passes embed_norm like a token row.
    unsigned prefill_embedded(std::span<const unsigned> ids,
                              const std::shared_ptr<const strix::session::EmbedTable>& table,
                              std::span<float> logits) override {
        if (!table) return prefill(ids, logits);
        const auto run = strix::session::resolve_embed_run(ids, table, {strix::inkling2::kHiddenWidth});
        run.splice.validate(run.tokens.size(), strix::inkling2::kHiddenWidth);
        std::vector<const std::uint16_t*> provided(run.tokens.size(), nullptr);
        for (const auto& segment : run.splice.segments)
            for (std::uint32_t r = 0; r < segment.rows; ++r)
                provided[std::size_t{segment.run_row} + r] = segment.data + std::size_t{r} * strix::inkling2::kHiddenWidth;
        unsigned next{};
        for (std::size_t i = 0; i < run.tokens.size(); ++i) {
            const bool last = i + 1U == run.tokens.size();
            const auto row = provided[i]
                ? decoder_.step_row({provided[i], strix::inkling2::kHiddenWidth}, last)
                : decoder_.step(run.tokens[i], last);
            if (last) next = select(row, logits);
        }
        return next;
    }

    strix::spec::VerifyTarget* verify_target() override { return &decoder_; }

    std::uint32_t max_verify_rows(strix::serve::CliSpeculation mode) const override {
        // The CPU oracle computes each verify row serially.
        using strix::serve::CliSpeculation;
        return mode == CliSpeculation::verify || mode == CliSpeculation::lookup ||
               (mode == CliSpeculation::automatic && decoder_.has_dspark()) ? 8U : 0U;
    }

    strix::spec::ChainDrafter* chain_drafter(std::uint32_t anchor, std::uint32_t position) override {
        return decoder_.start_dspark(anchor, position);
    }

private:
    static unsigned select(std::span<const float> row, std::span<float> logits) {
        if (!logits.empty()) std::ranges::copy(row, logits.begin());
        return strix::select_greedy_token(row, strix::inkling2::kOutputVocabulary);
    }

    strix::inkling2::MiniHostDecoder decoder_;
};

}  // namespace

int main(int argc, char** argv) {
    using namespace strix::inkling2;
    namespace serve = strix::serve;
    serve::SerialGenerationFamily family{
        "strix-inkling2-mini-cpu", "Inkling 2 Mini CPU", "MODEL.safetensors",
        kVocabulary, kOutputVocabulary, kMaxContext, kMiniEosToken,
        [](const serve::SerialGenerationLoad& load) -> std::unique_ptr<serve::SerialGenerationDecoder> {
            return std::make_unique<CpuCliDecoder>(load.checkpoint, load.positions);
        }, 0U, false, false,
        [](const std::string& path) {
            return serve::BpeTokenizer::load_hf_json(path, serve::inkling2_mini_tokenizer_identity);
        },
        false, serve::render_inkling2_chat_request, true};
    // speculation_probe_loss: the fixed interval (models/inkling2-mini.toml has no cpu row).
#if defined(STRIX_INKLING2_MEDIA)
    family.render_media_messages = serve::render_inkling2_chat_media_request;
#endif
    if (argc < 3) {
        std::cerr << "usage: strix-inkling2-mini-cpu MODEL.safetensors TOKEN [TOKEN ...] [--profile exact|fast|turbo] [--generate N] [--logits FILE]"
                     " [--speculation off|auto|verify|lookup] [--dspark FILE] [--temperature T] [--top-k K] [--top-p P] [--min-p P] [--seed N]\n";
        return 2;
    }
    try {
        auto options = cli::parse(argc, argv, family);
#if defined(STRIX_INKLING2_MEDIA)
        family.media = cli::mini_media_opener(options.tier, std::cerr);  // image recipe: this run's resolution
#endif
        if (const auto* dspark = options.dspark) {
            family.load = [dspark](const serve::SerialGenerationLoad& load) {
                return std::make_unique<CpuCliDecoder>(load.checkpoint, load.positions, dspark);
            };
        }
        serve::run_serial_generation(argv[1], std::move(options.generation), family, std::cout,
            [&] { cli::write_profile(options.tier, std::cerr, false); });
    } catch (const std::exception& error) {
        std::cerr << "Inkling 2 Mini CPU: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
