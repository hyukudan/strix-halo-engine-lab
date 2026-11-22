#pragma once

#include <cstdint>
#include <span>
#include <string_view>

namespace strix::spec {

enum class DraftKind : std::uint8_t { chain, block, lookup, table, model };
enum class HeadRows : std::uint8_t { full, subset, shortlist };
enum class DraftAttention : std::uint8_t { causal, block_noncausal };
enum class LookupSource : std::uint8_t { context, prompt, ngram, frame };

struct HeadDecl {
    HeadRows rows = HeadRows::full;
    std::uint32_t size = 0;
    std::string_view packs{}, candidates{}, image{};
};

struct DraftTap {
    std::string_view value;
    std::uint32_t width = 0;  // 0 uses the named value's natural width.
};

// Views refer to immutable model-declaration storage. Model loading resolves
// graph, weights, taps and shared state before binding a runner.
struct DraftDecl {
    std::string_view id{};
    DraftKind kind = DraftKind::chain;
    std::string_view graph{}, file{}, tensors{}, feed{};
    std::uint32_t max_k = 0, block = 0, layers = 0;  // Block width includes the anchor row.
    std::span<const std::int32_t> taps{};
    std::span<const DraftTap> tap_values{};
    HeadDecl head{};
    bool shares_kv = false;
    std::span<const std::string_view> shares{};
    float cut_first = 0, cut_rest = 0;
    std::uint64_t identity = 0;  // Proposal identity used with depth by the controller.

    std::uint32_t noise_token = 0;
    DraftAttention attention = DraftAttention::block_noncausal;
    std::string_view confidence{}, chain{};  // Named confidence emission and candidate policy.
    bool anchor_proposes = false;
    bool prefix_in_block = false;
    std::span<const LookupSource> sources{};
};

}  // namespace strix::spec
