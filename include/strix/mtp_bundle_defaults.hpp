#pragma once
#include <array>
#include <string_view>
#include <utility>

namespace strix {
// The production MTP bundle's options that do not depend on which qualified draft is used
// (include/strix/mtp_draft_profiles.hpp holds the draft's identity, provenance and min-p; the target's
// pins come from the target file). A drafter found by discovery gets exactly these budgets, so the shipped
// recipe `recipes/qwen-distill-g-yarn.mtp-bundle.json` and a discovered profile are the same bundle; a host
// test compares the two. A value "true" is a switch.
inline constexpr std::array<std::pair<std::string_view, std::string_view>, 31> mtp_production_bundle_options{{
    {"--mtp-conversion", "effective-norm-embedding-first-v1"},
    {"--mtp-normalized-rounding", "bf16-rne"},
    {"--mtp-projection-rounding", "bf16-rne"},
    {"--mtp-sidecar-residency", "register"},
    {"--mtp-header-bytes", "33554432"},
    {"--mtp-sidecar-retained-bytes", "3221225472"},
    {"--mtp-sidecar-setup-bytes", "4294967296"},
    {"--mtp-golden-max-bytes", "6442450944"},
    {"--mtp-replay-host-bytes", "6442450944"},
    {"--mtp-runtime-bytes", "2418528231"},
    {"--mtp-target-capture-bytes", "16777216"},
    {"--mtp-policy-id", "1"},
    {"--mtp-max-hidden-error", "0"},
    {"--mtp-max-logit-error", "0"},
    {"--mtp-draft-chain", "true"},
    {"--mtp-draft-context-vocab", "4096"},
    {"--mtp-draft-q4", "true"},
    {"--mtp-candidate-depth", "7"},
    {"--mtp-target-window-rows", "8"},
    {"--mtp-target-window-rows-set", "2,4,8"},
    {"--mtp-window-prefix", "true"},
    {"--mtp-window-hyper-mix", "true"},
    {"--mtp-window-qsa-rows", "true"},
    {"--mtp-window-expert-union", "true"},
    {"--mtp-window-expert-union-bytes", "65536"},
    {"--mtp-window-arena-bytes", "2147483648"},
    {"--mtp-window-capture-bytes", "67108864"},
    {"--mtp-window-checkpoint-bytes", "8589934592"},
    {"--mtp-window-total-bytes", "17179869184"},
    {"--mtp-window-history-intervals", "8192"},
    // The draft vocabulary directory ("auto:DIR") is added only where `draft-vocab/` exists beside the model.
    {"--mtp-draft-vocab", "auto:draft-vocab"},
}};
} // namespace strix
