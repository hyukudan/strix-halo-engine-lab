#include "strix/kernel_registry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

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
        variant.measured_seconds < 0.0 ||
        (qualification_rank(variant.qualification) >= 2 &&
         variant.measured_seconds <= 0.0)) {
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

TunedKernelSelection KernelRegistry::select_tuned(const KernelQuery& query,
    const OfflineTuningKey& key, std::span<const OfflineTuningRecord> records) const noexcept {
    const auto fallback = select(query);
    const auto digest = [](const std::string& value) {
        return value.size() == 64 && std::any_of(value.begin(), value.end(), [](char c) { return c != '0'; }) &&
            std::all_of(value.begin(), value.end(), [](char c) {
                return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
            });
    };
    if (key.gfx_target.empty() || key.device_sku.empty() || !key.compute_units ||
        key.compiler_identity.empty() || key.hip_identity.empty() || !digest(key.binary_sha256) ||
        !digest(key.model_content_sha256) || !digest(key.compiler_flags_sha256) ||
        key.input_precision.empty() || key.accumulator_precision.empty() || key.output_precision.empty() ||
        key.weight_format.empty() || key.input_layout.empty() || key.output_layout.empty() || key.state_layout.empty() ||
        key.context_bucket.empty() || key.memory_regime.empty() || !key.m || !key.n || !key.k ||
        !key.numerical_contract_identity || key.family != query.family || key.phase != query.phase ||
        key.shape != query.shape || (query.required_wave && key.wave != *query.required_wave))
        return {fallback, TuningSelectionReason::invalid_key};
    // Repeated records for one candidate require offline reconciliation rather
    // than choosing whichever duplicate happens to report the fastest sample.
    for (std::size_t i = 0; i < records.size(); ++i) {
        if (records[i].key != key) continue;
        for (std::size_t j = 0; j < i; ++j)
            if (records[j].key == key && records[j].kernel_id == records[i].kernel_id)
                return {fallback, TuningSelectionReason::ambiguous_records};
    }
    const KernelVariant* best = nullptr;
    double seconds = 0;
    bool baseline_measured = false;
    for (const auto& record : records) {
        if (record.key != key || !record.complete_workload || !record.owners_and_resources_qualified ||
            record.samples < 3 || !digest(record.qualification_receipt_sha256) ||
            !std::isfinite(record.median_complete_seconds) || record.median_complete_seconds <= 0) continue;
        const auto* candidate = find(record.kernel_id);
        if (!candidate || !matches(*candidate, query) ||
            (candidate->qualification != QualificationStatus::target_qualified &&
             candidate->qualification != QualificationStatus::production) ||
            candidate->wave != key.wave || candidate->compiler_identity != key.compiler_identity ||
            candidate->numerical_contract_identity != key.numerical_contract_identity ||
            candidate->input_layout != key.input_layout || candidate->output_layout != key.output_layout ||
            candidate->numerics.input_precision != key.input_precision ||
            candidate->numerics.accumulator_precision != key.accumulator_precision ||
            candidate->numerics.output_precision != key.output_precision) continue;
        if (candidate == fallback) baseline_measured = true;
        if (!best || record.median_complete_seconds < seconds ||
            (record.median_complete_seconds == seconds && candidate->id < best->id)) {
            best = candidate;
            seconds = record.median_complete_seconds;
        }
    }
    return best && baseline_measured ? TunedKernelSelection{best, TuningSelectionReason::qualified_record} :
        TunedKernelSelection{fallback, TuningSelectionReason::no_qualified_record};
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
        hash_bytes(hash, variant->input_layout);
        hash_bytes(hash, variant->output_layout);
        hash_bytes(hash, variant->numerics.input_precision);
        hash_bytes(hash, variant->numerics.accumulator_precision);
        hash_bytes(hash, variant->numerics.output_precision);
        hash_u64(hash, variant->numerics.row_invariant ? 1ULL : 0ULL);
        hash_u64(hash, variant->numerics.deterministic ? 1ULL : 0ULL);
        hash_u64(hash, variant->graph_safe ? 1ULL : 0ULL);
    }
    return hash;
}

}  // namespace strix::kernels
