#pragma once
#include <array>
#include <cstdint>
#include <string_view>

namespace strix {
// Qualified MTP draft sidecars and their decode profile (docs/engine/speculative/decode-defaults.md, D-055).
//
// The engine still takes every asset explicitly: a profile never chooses a file. It does two things:
//   - names a qualified draft, so `--mtp-draft-profile NAME` fills that draft's identity and provenance pins
//     (logical name, bytes, SHA-256, source repository/revision/path, converter revision and SHA-256s) and only
//     the local path (`--mtp-sidecar`) remains to be given; an explicit pin that disagrees is an error;
//   - gives the draft early-stop probability that measured best for that draft (the default of
//     `--mtp-spec-min-p` when the sidecar's SHA-256 is a profiled draft; an explicit value always wins).
// The probability only changes how many drafts a cycle proposes and which verify window runs; the target
// verifies every token, so tokens are those of the same command with the explicit value.
struct MtpDraftProfile {
    std::string_view name;
    std::string_view sidecar_sha256;
    std::string_view logical_name;
    std::uint64_t sidecar_bytes;
    std::string_view repository, revision, source_path;
    std::string_view converter_revision, converter_sha256, norm_converter_sha256;
    float min_probability;
};

inline constexpr std::array<MtpDraftProfile, 3> mtp_draft_profiles{{
    // The published draft sidecar (D-022: min-p 0.85).
    {"base", "5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6",
     "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf", 2786568256ULL,
     "unsloth/Qwen3.8-Flash-Next-GGUF", "38bb39ee97821de2c9009abb7e93950eec396e66",
     "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf",
     "6fcaa16f4b360649933a54d1f91ad40ed35c0e11",
     "6dab74eb519ea3b12787a28cc3fd5039ff65135ff14ba44321c9855edf1f8289",
     "c53604b9df72214ebfe492d05899042dd375a1e9d9aa03a5de87ab01f97d7730", 0.85F},
    // Distilled drafts (branch mtp/distill, round 1 = D, round 2 = G): min-p 0.7.
    {"distill-d", "47cb2cbd35184726065eec9a21ea89d121988852eea037cb3464be4818966017",
     "mtp-Qwen3.8-Flash-Next-shared-distillD-Q8_0.gguf", 2786568256ULL,
     "local/mtp-distill", "f9eccbc789e860ca1b8402980830e410594eda52",
     "export/mtp-Qwen3.8-Flash-Next-shared-distillD-Q8_0.gguf",
     "f9eccbc789e860ca1b8402980830e410594eda52",
     "5dfead6ba7840f708945c1da3a288a3292e22e4dceb70630ccefab00cbba1366",
     "4ea7af3e68357e554eb75f47e40275d882f48406fcaced18af1eba71acbfac0c", 0.7F},
    {"distill-g", "5516d3ce407783a424315b6a2a0d8b06341665bb3541b98e5db0fe944add4c3c",
     "mtp-Qwen3.8-Flash-Next-shared-distillG-Q8_0.gguf", 2786568256ULL,
     "local/mtp-distill", "1a0eed49be3995d8ea15eeb6dd370dd16d9c04db",
     "export/mtp-Qwen3.8-Flash-Next-shared-distillG-Q8_0.gguf",
     "1a0eed49be3995d8ea15eeb6dd370dd16d9c04db",
     "5dfead6ba7840f708945c1da3a288a3292e22e4dceb70630ccefab00cbba1366",
     "4ea7af3e68357e554eb75f47e40275d882f48406fcaced18af1eba71acbfac0c", 0.7F},
}};

// The recommended draft (final campaign: +10 % decode over the base draft at min-p 0.85).
inline constexpr std::string_view mtp_recommended_draft_profile = "distill-g";

inline constexpr const MtpDraftProfile* find_mtp_draft_profile(std::string_view name) noexcept {
    for (const auto& p : mtp_draft_profiles) if (p.name == name) return &p;
    return nullptr;
}
inline constexpr const MtpDraftProfile* find_mtp_draft_profile_by_sha256(std::string_view sha256) noexcept {
    for (const auto& p : mtp_draft_profiles) if (p.sidecar_sha256 == sha256) return &p;
    return nullptr;
}
} // namespace strix
