#include "strix/kernel_registry.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strix::kernels {
namespace {

int qualification_rank(QualificationStatus status) noexcept {
    switch (status) {
    case QualificationStatus::experimental:
        return 0;
    case QualificationStatus::oracle_matched:
        return 1;
    case QualificationStatus::target_qualified:
        return 2;
    case QualificationStatus::production:
        return 3;
    case QualificationStatus::rejected:
        return -1;
    }
    return -1;
}

bool shape_matches(const ShapeClass& variant, const ShapeClass& query) noexcept {
    return variant.logical_rows == query.logical_rows &&
           variant.request_concurrency == query.request_concurrency &&
           variant.hidden_size == query.hidden_size &&
           variant.auxiliary_size == query.auxiliary_size;
}

void hash_bytes(std::uint64_t& hash, std::string_view value) noexcept {
    constexpr std::uint64_t prime = 1099511628211ULL;
    for (const unsigned char byte : value) {
        hash ^= static_cast<std::uint64_t>(byte);
        hash *= prime;
    }
}

void hash_u64(std::uint64_t& hash, std::uint64_t value) noexcept {
    constexpr std::uint64_t prime = 1099511628211ULL;
    for (unsigned shift = 0; shift < 64; shift += 8) {
        hash ^= (value >> shift) & 0xffULL;
        hash *= prime;
    }
}

}  // namespace

bool valid_variant(const KernelVariant& variant) noexcept {
    if (variant.id.empty() || variant.shape.logical_rows == 0 ||
        variant.shape.request_concurrency == 0 ||
        variant.resources.workgroup_threads == 0 ||
        variant.resources.workgroup_threads > 1024 ||
        variant.resources.workgroup_threads %
                static_cast<std::uint32_t>(variant.wave) !=
            0 ||
        variant.qualification == QualificationStatus::rejected ||
        variant.input_layout.empty() || variant.output_layout.empty() ||
        variant.compiler_identity.empty() ||
        !std::isfinite(variant.measured_seconds) ||
        variant.measured_seconds < 0.0) {
        return false;
    }

    if (variant.numerics.input_precision.empty() ||
        variant.numerics.accumulator_precision.empty() ||
        variant.numerics.output_precision.empty()) {
        return false;
    }

    return true;
}

bool matches(const KernelVariant& variant, const KernelQuery& query) noexcept {
    if (!valid_variant(variant) || variant.family != query.family ||
        variant.phase != query.phase || !shape_matches(variant.shape, query.shape) ||
        qualification_rank(variant.qualification) <
            qualification_rank(query.minimum_qualification)) {
        return false;
    }

    if (query.require_graph_safe && !variant.graph_safe) {
        return false;
    }
    if (query.require_row_invariant && !variant.numerics.row_invariant) {
        return false;
    }
    if (query.require_deterministic && !variant.numerics.deterministic) {
        return false;
    }
    if (query.required_wave.has_value() && variant.wave != *query.required_wave) {
        return false;
    }
    return true;
}

void KernelRegistry::add(KernelVariant variant) {
    if (!valid_variant(variant)) {
        throw std::invalid_argument("invalid kernel variant");
    }
    if (find(variant.id) != nullptr) {
        throw std::invalid_argument("duplicate kernel id");
    }
    variants_.push_back(std::move(variant));
}

const KernelVariant* KernelRegistry::find(std::string_view id) const noexcept {
    const auto found = std::find_if(
        variants_.begin(), variants_.end(),
        [&](const KernelVariant& variant) { return variant.id == id; });
    return found == variants_.end() ? nullptr : &*found;
}

const KernelVariant* KernelRegistry::select(const KernelQuery& query) const noexcept {
    const KernelVariant* best = nullptr;
    for (const auto& variant : variants_) {
        if (!matches(variant, query)) {
            continue;
        }
        if (best == nullptr) {
            best = &variant;
            continue;
        }

        const bool variant_measured = variant.measured_seconds > 0.0;
        const bool best_measured = best->measured_seconds > 0.0;
        if (variant_measured != best_measured) {
            if (variant_measured) {
                best = &variant;
            }
            continue;
        }

        if (variant_measured && best_measured &&
            variant.measured_seconds != best->measured_seconds) {
            if (variant.measured_seconds < best->measured_seconds) {
                best = &variant;
            }
            continue;
        }

        const auto variant_rank = qualification_rank(variant.qualification);
        const auto best_rank = qualification_rank(best->qualification);
        if (variant_rank != best_rank) {
            if (variant_rank > best_rank) {
                best = &variant;
            }
            continue;
        }

        if (variant.id < best->id) {
            best = &variant;
        }
    }
    return best;
}

std::uint64_t plan_identity(
    std::span<const KernelVariant* const> selected) noexcept {
    constexpr std::uint64_t offset = 14695981039346656037ULL;
    std::uint64_t hash = offset;
    for (const auto* variant : selected) {
        if (variant == nullptr) {
            hash_u64(hash, 0);
            continue;
        }
        hash_bytes(hash, variant->id);
        hash_u64(hash, static_cast<std::uint64_t>(variant->family));
        hash_u64(hash, static_cast<std::uint64_t>(variant->phase));
        hash_u64(hash, variant->shape.logical_rows);
        hash_u64(hash, variant->shape.request_concurrency);
        hash_u64(hash, variant->shape.hidden_size);
        hash_u64(hash, variant->shape.auxiliary_size);
        hash_u64(hash, static_cast<std::uint64_t>(variant->wave));
        hash_bytes(hash, variant->compiler_identity);
        hash_u64(hash, std::bit_cast<std::uint64_t>(variant->measured_seconds));
    }
    return hash;
}

}  // namespace strix::kernels
