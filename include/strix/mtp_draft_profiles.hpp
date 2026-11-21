#pragma once
#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace strix {
// Qualified MTP draft sidecars and their decode profile (docs/engine/speculative/decode-defaults.md, D-055).
//
// The engine still takes every asset explicitly: a profile never chooses a file. It does two things:
//   - names a qualified draft, so `--mtp-draft-profile NAME` fills that draft's identity and provenance pins
//     (logical name, bytes, SHA-256, source repository/revision/path, converter revision and SHA-256s) and only
//     the local path (`--mtp-sidecar`) remains to be given; an explicit pin that disagrees is an error;
//   - gives the draft early-stop probability per backend (the default of `--mtp-spec-min-p` on HIP when the
//     sidecar's SHA-256 is a profiled draft; the CUDA serve resolves no draft identity yet, so its `--min-p`
//     default is the CUDA unprofiled value; an explicit value always wins). Each backend column holds only a value
//     qualified on that backend, and a caller selects a column only from a sidecar identity it verified:
//     HIP carries the measured values (D-022 base 0.85, D-055 distilled 0.7); CUDA keeps its legacy 0.7 for
//     every draft until an RTX receipt qualifies another value (no HIP number is promoted to CUDA).
// The probability only changes how many drafts a cycle proposes and which verify window runs; the target
// verifies every token, so tokens are those of the same command with the explicit value.
struct MtpDraftProfile {
    std::string_view name;
    std::string_view sidecar_sha256;
    std::string_view logical_name;
    std::uint64_t sidecar_bytes;
    std::string_view repository, revision, source_path;
    std::string_view converter_revision, converter_sha256, norm_converter_sha256;
    float hip_min_probability;   // qualified on HIP (Strix)
    float cuda_min_probability;  // qualified on CUDA (RTX serve); legacy 0.7 until an RTX receipt
};

inline constexpr std::array<MtpDraftProfile, 3> mtp_draft_profiles{{
    // The published draft sidecar (D-022: min-p 0.85).
    {"base", "5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6",
     "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf", 2786568256ULL,
     "unsloth/Qwen3.8-Flash-Next-GGUF", "38bb39ee97821de2c9009abb7e93950eec396e66",
     "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf",
     "6fcaa16f4b360649933a54d1f91ad40ed35c0e11",
     "6dab74eb519ea3b12787a28cc3fd5039ff65135ff14ba44321c9855edf1f8289",
     "c53604b9df72214ebfe492d05899042dd375a1e9d9aa03a5de87ab01f97d7730", 0.85F, 0.7F},
    // Distilled drafts (branch mtp/distill, round 1 = D, round 2 = G): min-p 0.7.
    {"distill-d", "47cb2cbd35184726065eec9a21ea89d121988852eea037cb3464be4818966017",
     "mtp-Qwen3.8-Flash-Next-shared-distillD-Q8_0.gguf", 2786568256ULL,
     "local/mtp-distill", "f9eccbc789e860ca1b8402980830e410594eda52",
     "export/mtp-Qwen3.8-Flash-Next-shared-distillD-Q8_0.gguf",
     "f9eccbc789e860ca1b8402980830e410594eda52",
     "5dfead6ba7840f708945c1da3a288a3292e22e4dceb70630ccefab00cbba1366",
     "4ea7af3e68357e554eb75f47e40275d882f48406fcaced18af1eba71acbfac0c", 0.7F, 0.7F},
    {"distill-g", "5516d3ce407783a424315b6a2a0d8b06341665bb3541b98e5db0fe944add4c3c",
     "mtp-Qwen3.8-Flash-Next-shared-distillG-Q8_0.gguf", 2786568256ULL,
     "local/mtp-distill", "1a0eed49be3995d8ea15eeb6dd370dd16d9c04db",
     "export/mtp-Qwen3.8-Flash-Next-shared-distillG-Q8_0.gguf",
     "1a0eed49be3995d8ea15eeb6dd370dd16d9c04db",
     "5dfead6ba7840f708945c1da3a288a3292e22e4dceb70630ccefab00cbba1366",
     "4ea7af3e68357e554eb75f47e40275d882f48406fcaced18af1eba71acbfac0c", 0.7F, 0.7F},
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

enum class MtpDraftBackend : unsigned char { hip, cuda };
// Default draft early-stop probability when the draft identity is not a profiled draft: HIP 0.5 (D-055),
// CUDA 0.7 (the serve default before per-draft profiles).
inline constexpr float mtp_unprofiled_draft_min_probability(MtpDraftBackend backend) noexcept {
    return backend == MtpDraftBackend::cuda ? 0.7F : 0.5F;
}
// The one resolver both backends use for an implicit draft min-p: the profile's column for that backend,
// else the backend's unprofiled default. Callers apply an explicit value before calling this.
inline constexpr float mtp_draft_backend_min_probability(const MtpDraftProfile* profile,
                                                         MtpDraftBackend backend) noexcept {
    if (profile == nullptr) return mtp_unprofiled_draft_min_probability(backend);
    return backend == MtpDraftBackend::cuda ? profile->cuda_min_probability : profile->hip_min_probability;
}

// Where a resolved draft min-p came from: an explicit value (0 included), the selected profile's backend
// column, or the backend's unprofiled default. A profile here is the identity the caller resolved (HIP: the
// sidecar SHA-256; the CUDA serve passes none until it keys a profile on the loaded sidecar's SHA-256).
struct MtpDraftMinProbability {
    float value;
    std::string_view source;  // "explicit" | "profile" | "unprofiled"
};
inline MtpDraftMinProbability resolve_mtp_draft_min_probability(std::optional<float> explicit_value,
                                                                const MtpDraftProfile* profile,
                                                                MtpDraftBackend backend) {
    if (explicit_value) {
        if (!(*explicit_value >= 0.0F && *explicit_value <= 1.0F))
            throw std::invalid_argument("draft min-p must be a finite value in [0, 1]");
        return {*explicit_value, "explicit"};
    }
    return {mtp_draft_backend_min_probability(profile, backend), profile ? "profile" : "unprofiled"};
}
} // namespace strix
