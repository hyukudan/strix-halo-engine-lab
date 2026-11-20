#pragma once
// Hy4-flash serve binding: the family's options, its shared-API binding and its ServiceFactory for strix-serve.
//
// ONE RESOLVER: parse_options is the only parser of the family arguments. strix-serve passes the arguments after `--`
// verbatim, and a Hy4 CLI must call the same function, so both end with the same Options and the same effective_config.
// Request-level values (reasoning effort, sampling, stop, tool grammar) are resolved per request by the shared HTTP layer
// (src/serve/api.cpp) from the ApiConfig that bind_api fills; nothing here is a second defaults branch.
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
    std::string gguf;                       // --gguf PATH: first part of the pack (tokenizer, chat template, weights)
    std::uint64_t ctx = 0;                  // --ctx N: context cap; 0 = the pack's context_length
    std::uint32_t max_tokens = 4096;        // --max-tokens N: default reply cap (shared ApiConfig::default_max_tokens)
    std::string model_id = "hy4-flash";     // --model-id NAME: the id the API reports
    bool ctx_given = false, max_tokens_given = false, model_id_given = false;
    bool print_effective_config = false;    // --print-effective-config: print effective_config and exit
    bool help = false;                      // --help
};

// The family argument parser (argument 0 is NOT a program name). Throws std::invalid_argument on an unknown option,
// a missing value or a malformed number.
[[nodiscard]] Options parse_options(std::span<const std::string> args);

// What the pack contributes without device work (header read only): its chat dialect and context length.
struct PackFacts {
    sv::Hy4ChatDialect dialect{};          // official ":opensource" release when no pack is given
    bool dialect_from_pack = false;
    std::uint64_t context_length = 0;       // 0 = unknown (no pack)
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

// strix-serve factory "hy4_flash": validate/inspect/early-exit need no device; create refuses until a device binding
// (hy4_hipbind / hy4_cudabind) exists.
void install_session_service(sv::ServiceFactories& factories);

// GGUF admission (strix/model_family_registry.hpp) for architecture hy_v4_flash, validated by the pack's tensor naming
// map resolver (strix/weight/tensor_map.hpp hy4_flash_map). Idempotent.
void register_family();

}  // namespace strix::hy4::serve
