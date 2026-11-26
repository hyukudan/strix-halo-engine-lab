#pragma once
// Hy4-flash serve binding: the family's options, its shared-API binding and its ServiceFactory for strix-serve.
//
// ONE RESOLVER: the native rank and the service use the same profile parser and resolver.
// strix-serve passes the arguments after `--` verbatim. Host inspection has no device facts.
// Request-level values (reasoning effort, sampling, stop, tool grammar) are resolved per request by the shared HTTP layer
// (src/serve/api.cpp) from the ApiConfig that bind_api fills; nothing here is a second defaults branch.
#include "strix/hy4/placement_policy.hpp"
#include "strix/serve/api.hpp"
#include "strix/serve/hy4_chat.hpp"
#include "strix/serve/json.hpp"
#include "strix/serve/run.hpp"
#include <cstdint>
#include <optional>
#include <ostream>
#include <span>
#include <string>

namespace strix::hy4::serve {

namespace sv = strix::serve;

inline constexpr const char* kFamily = "hy4_flash";       // registry family and ServiceFactory key
inline constexpr const char* kArchitecture = "hy_v4_flash";  // GGUF general.architecture

struct Options {
    ::strix::profile::Inputs profile;         // same profile and explicit lever inputs as the native rank
    std::string gguf;                       // --gguf PATH: first part of the pack (tokenizer, chat template, weights)
    std::uint64_t ctx = 0;                  // --ctx N: context cap; 0 = the pack's context_length
    std::optional<std::uint32_t> prompt_rows;
    std::optional<bool> head_q8;
    std::uint32_t max_tokens = 4096;        // --max-tokens N: default reply cap (shared ApiConfig::default_max_tokens)
    std::string model_id = "hy4-flash";     // --model-id NAME: the id the API reports
    bool ctx_given = false, max_tokens_given = false, model_id_given = false;
    bool print_effective_config = false;    // --print-effective-config: print effective_config and exit
    bool help = false;                      // --help
    bool no_mtp = false;                    // --no-mtp: the shared speculation opt-out (as the native rank)
};

// The family argument parser (argument 0 is NOT a program name). Throws std::invalid_argument on an unknown option,
// a missing value or a malformed number.
[[nodiscard]] Options parse_options(std::span<const std::string> args);

// What the pack contributes without device work (header read only): its chat dialect and context length.
struct PackFacts {
    sv::Hy4ChatDialect dialect{};          // official ":opensource" release when no pack is given
    bool dialect_from_pack = false;
    std::uint64_t context_length = 0;       // 0 = unknown (no pack)
    bool next_token_block = false;          // <arch>.nextn_predict_layers > 0: the pack ships the MTP drafter block
    // The model's sampling card: generation_config.json, or the pack's general.sampling.* when present. Documented in the
    // effective config; requests use them only when the client sends them (the default stays exact greedy).
    double card_temperature = 0.9, card_top_p = 1.0;
    std::int64_t card_top_k = -1;
    bool card_from_pack = false;
};
[[nodiscard]] PackFacts read_pack_facts(const Options& options);

// The effective values, each with its origin, plus every lever where serving differs from the CLI and why.
[[nodiscard]] sv::Json effective_config(const Options& options, const PackFacts& pack);

// Fills the family hooks of the shared HTTP layer (renderer, parser, tool marker, stop, thinking, sampling policy).
void bind_api(sv::ApiConfig& config, const Options& options, const PackFacts& pack);

[[nodiscard]] const char* usage();

struct SessionAssets {
    std::shared_ptr<const sv::ApiTokenizer> tokenizer;
    std::unique_ptr<sv::GenerationEngine> engine;
};
using SessionAssetLoader = std::function<SessionAssets(const Options&, const PackFacts&)>;
[[nodiscard]] sv::Json placement_config(const model_description::ModelDescription&, const md::Query&);
void install_session_service(sv::ServiceFactories& factories, SessionAssetLoader loader);
void install_session_service(sv::ServiceFactories& factories);
#ifdef STRIX_HY4_SERVE_HIP
SessionAssets load_hip_service(const Options&, const PackFacts&);
#endif

// GGUF admission (strix/model_family_registry.hpp) for architecture hy_v4_flash, validated by the pack's tensor naming
// map resolver (strix/weight/tensor_map.hpp hy4_flash_map). Idempotent.
void register_family();

}  // namespace strix::hy4::serve
