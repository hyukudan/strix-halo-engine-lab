#include "strix/deepseek41_dspark_target.hpp"
#include "strix/ds41_dspark_driver.hpp"
#include "strix/ds41_dspark_files.hpp"
#include "strix/ds41_dspark_vocab.hpp"

#include <cstdlib>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Greedy decoding of a (trimmed) deepseek41 model on the host forward, with or without DSpark drafts.
//   strix-deepseek41-dspark MODEL.gguf DRAFT.gguf --layers N --prompt-ids a,b,c --generate G [--compare] [--vocab-core K]
// STRIX_DS41_DSPARK=1 enables the drafter (host reference drafter, bitwise the device drafter);
// STRIX_DS41_DSPARK_WINDOW=fixed:N | auto[:c0,..,c5] | survival[:c0,..,c5] picks the verified drafts per window;
// STRIX_DS41_DSPARK_VOCAB=full | <N> | auto:<dir> | files:<a>,<b> the draft vocabulary (auto: chosen from the prompt).
// --compare also runs target-only greedy decoding and fails unless the tokens are identical.
int main(int argc, char** argv) try {
    using namespace strix;
    std::string model, draft_file, prompt;
    std::uint32_t layers = 5, generate = 16, vocab_core = ds41::dspark::vocab;
    bool compare = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&] {
            if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
            return std::string(argv[++i]);
        };
        if (a == "--layers") layers = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--prompt-ids") prompt = next();
        else if (a == "--generate") generate = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--compare") compare = true;
        else if (a == "--vocab-core") vocab_core = static_cast<std::uint32_t>(std::stoul(next()));
        else if (model.empty()) model = a;
        else if (draft_file.empty()) draft_file = a;
        else throw std::invalid_argument("unknown argument " + a);
    }
    if (model.empty() || draft_file.empty() || prompt.empty())
        throw std::invalid_argument("usage: strix-deepseek41-dspark MODEL.gguf DRAFT.gguf --layers N --prompt-ids a,b,c [--generate G] [--compare]");
    std::vector<std::uint32_t> ids;
    std::stringstream ss(prompt);
    for (std::string t; std::getline(ss, t, ',');) ids.push_back(static_cast<std::uint32_t>(std::stoul(t)));
    deepseek41::HostForwardOptions options;
    options.layers = layers;
    const auto print = [](const char* label, const std::vector<std::uint32_t>& tokens) {
        std::printf("%s:", label);
        for (const auto t : tokens) std::printf(" %u", t);
        std::printf("\n");
    };
    std::vector<std::uint32_t> plain;
    if (compare || !ds41::dspark::dspark_enabled_from_env()) {
        const auto t0 = std::chrono::steady_clock::now();
        deepseek41::HostForward forward(model, options);
        deepseek41::HostForwardTarget target(forward);
        plain = ds41::dspark::generate_greedy(target, ids, generate);
        print("target-only", plain);
        std::printf("target-only: %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        if (!ds41::dspark::dspark_enabled_from_env()) return 0;
    }
    const auto weights = ds41::dspark::load_weights(model, draft_file);
    ds41::dspark::Vocab vocabulary;
    vocabulary.core = vocab_core;
    if (const char* spec = std::getenv("STRIX_DS41_DSPARK_VOCAB"); spec && *spec) {
        const auto choice = ds41::dspark::choose_vocabulary(spec, ids);
        vocabulary = choice.vocabulary;
        std::printf("draft vocabulary: %s%c", choice.label.c_str(), 10);
    }
    ds41::dspark::HostDrafter drafter(weights->weights, vocabulary);
    deepseek41::HostForward forward(model, options);
    deepseek41::HostForwardTarget target(forward);
    ds41::dspark::SpeculativeStats stats;
    const auto policy = ds41::dspark::window_policy_from_env();
    const auto tokens = ds41::dspark::generate_speculative(target, drafter, ids, generate, policy, &stats);
    print("dspark", tokens);
    std::printf("dspark: %u windows, %u of %u drafts accepted, %u target steps, accepted by depth", stats.windows, stats.accepted,
                stats.verified, stats.target_steps);
    for (std::uint32_t j = 0; j < ds41::dspark::block_size; ++j) std::printf(" %u/%u", stats.accepted_at_depth[j], stats.verified_at_depth[j]);
    std::printf("; per-position survival P(d_j ok | d_<j ok)");
    for (std::uint32_t j = 0; j < ds41::dspark::block_size; ++j)
        std::printf(" %.3f", stats.reached_at_depth[j] ? static_cast<double>(stats.accepted_at_depth[j]) / stats.reached_at_depth[j] : 0.0);
    std::printf("; drafter %.1f s, target %.1f s\n", stats.draft_seconds, stats.target_seconds);
    if (compare) {
        const bool same = tokens == plain;
        std::printf("tokens with DSpark %s target-only greedy\n", same ? "IDENTICAL to" : "DIFFER from");
        return same ? 0 : 3;
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << "strix-deepseek41-dspark: " << error.what() << '\n';
    return 1;
}
