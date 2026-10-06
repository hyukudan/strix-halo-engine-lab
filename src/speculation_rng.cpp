#include "strix/speculation_rng.hpp"

#include <bit>
#include <stdexcept>

namespace strix {
namespace {

constexpr std::uint64_t fnv_offset =
    14695981039346656037ULL;
constexpr std::uint64_t fnv_prime =
    1099511628211ULL;

void hash_u64(
    std::uint64_t& hash,
    std::uint64_t value) noexcept {
    for (unsigned shift = 0;
         shift < 64;
         shift += 8) {
        hash ^=
            (value >> shift) &
            0xffULL;
        hash *=
            fnv_prime;
    }
}

void validate_identity(
    const SpeculationRngIdentity& identity) {
    if (identity.request_identity == 0 ||
        identity.sampling_policy_identity == 0) {
        throw std::invalid_argument(
            "speculation RNG requires non-zero request and sampling-policy identities");
    }
}

std::uint64_t splitmix64(
    std::uint64_t value) noexcept {
    value +=
        0x9e3779b97f4a7c15ULL;
    value =
        (value ^
         (value >> 30U)) *
        0xbf58476d1ce4e5b9ULL;
    value =
        (value ^
         (value >> 27U)) *
        0x94d049bb133111ebULL;
    return value ^
           (value >> 31U);
}

std::uint64_t counter_key(
    const SpeculationRngIdentity& identity,
    std::uint64_t absolute_position,
    SpeculationRandomDrawClass draw_class,
    std::uint32_t draw_index) noexcept {
    std::uint64_t hash =
        fnv_offset;
    hash_u64(
        hash,
        identity.seed);
    hash_u64(
        hash,
        identity.request_identity);
    hash_u64(
        hash,
        identity.sampling_policy_identity);
    hash_u64(
        hash,
        absolute_position);
    hash_u64(
        hash,
        static_cast<std::uint64_t>(
            draw_class));
    hash_u64(
        hash,
        draw_index);
    return hash;
}

}  // namespace

std::uint64_t speculation_rng_identity(
    const SpeculationRngIdentity& identity) {
    validate_identity(
        identity);
    std::uint64_t hash =
        fnv_offset;
    hash_u64(
        hash,
        identity.seed);
    hash_u64(
        hash,
        identity.request_identity);
    hash_u64(
        hash,
        identity.sampling_policy_identity);
    if (hash == 0) {
        throw std::logic_error(
            "speculation RNG identity must be non-zero");
    }
    return hash;
}

std::uint64_t speculation_random_u64(
    const SpeculationRngIdentity& identity,
    std::uint64_t absolute_position,
    SpeculationRandomDrawClass draw_class,
    std::uint32_t draw_index) {
    validate_identity(
        identity);
    return splitmix64(
        counter_key(
            identity,
            absolute_position,
            draw_class,
            draw_index));
}

double speculation_uniform01(
    const SpeculationRngIdentity& identity,
    std::uint64_t absolute_position,
    SpeculationRandomDrawClass draw_class,
    std::uint32_t draw_index) {
    const auto bits =
        speculation_random_u64(
            identity,
            absolute_position,
            draw_class,
            draw_index);
    // Use the upper 53 random bits so every returned value is exactly
    // representable as k / 2^53 and lies in [0, 1).
    const auto mantissa =
        bits >> 11U;
    return static_cast<double>(
               mantissa) *
           (1.0 /
            9007199254740992.0);
}

}  // namespace strix
