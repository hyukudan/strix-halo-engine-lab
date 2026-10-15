#include "strix/uma_owner_inventory.hpp"
#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace strix {
namespace {
std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a)
        throw std::overflow_error("UMA inventory byte accounting overflow");
    return a + b;
}
void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
using Key = std::tuple<std::uint64_t,std::uint64_t,std::uint64_t>;
struct Group {
    UmaBackingSummary summary;
    std::uint64_t logical_bytes = 0;
    std::vector<std::pair<std::uint64_t,std::uint64_t>> slices;
};
void validate_observation(const UmaMemoryObservation& observed) {
    const bool has_fields = observed.mem_total_bytes || observed.mem_available_bytes ||
        observed.process_rss_bytes || observed.gtt_reported_bytes || observed.process_peak_bytes;
    require(!has_fields || observed.observation_identity, "UMA observation values need an observation identity");
    if (observed.mem_total_bytes)
        require(*observed.mem_total_bytes != 0, "observed MemTotal must be positive");
    if (observed.mem_total_bytes && observed.mem_available_bytes)
        require(*observed.mem_available_bytes <= *observed.mem_total_bytes,
                "observed MemAvailable exceeds MemTotal");
    // RSS/GTT/peak are kept separate: their overlap and measurement domains are
    // not established by these scalar observations. No cross-field sum/check.
}
} // namespace
UmaOwnerInventory build_uma_owner_inventory(std::span<const UmaOwnerView> views,
                                            UmaMemoryObservation observation) {
    validate_observation(observation);
    UmaOwnerInventory result;
    result.observation = std::move(observation);
    std::set<std::uint64_t> view_ids;
    std::map<Key,Group> groups;
    std::map<std::string,std::uint64_t> categories;
    for (const auto& view : views) {
        require(view.view_identity && view_ids.insert(view.view_identity).second,
                "UMA inventory requires unique nonzero view IDs");
        require(view.allocation_pin && !view.category.empty() && view.allocation_bytes,
                "UMA inventory requires retained actual lease, category and allocation extent");
        require(view.kind == UmaBackingKind::allocated_storage || view.kind == UmaBackingKind::file_mapping,
                "UMA inventory backing kind is invalid");
        require(view.view_offset <= view.allocation_bytes &&
                view.view_bytes <= view.allocation_bytes - view.view_offset,
                "UMA inventory view lies outside allocation");
        result.logical_view_bytes = add(result.logical_view_bytes, view.view_bytes);
        categories[view.category] = add(categories[view.category], view.view_bytes);
        const auto& key = view.backing;
        const bool unknown = !key.issuer && !key.allocation && !key.generation;
        require(unknown || (key.issuer && key.allocation && key.generation),
                "UMA inventory backing identity must be complete or unknown");
        if (unknown) {
            result.unknown_backing_views = add(result.unknown_backing_views, 1);
            result.reservation_upper_bound_bytes = add(result.reservation_upper_bound_bytes, view.allocation_bytes);
            continue;
        }
        auto [where, inserted] = groups.try_emplace(Key{key.issuer,key.allocation,key.generation});
        auto& group = where->second;
        if (inserted) {
            group.summary.identity = key;
            group.summary.kind = view.kind;
            group.summary.allocation_bytes = view.allocation_bytes;
        } else {
            require(group.summary.kind == view.kind && group.summary.allocation_bytes == view.allocation_bytes,
                    "UMA same backing lease has contradictory extent/kind");
        }
        group.summary.views = add(group.summary.views, 1);
        group.logical_bytes = add(group.logical_bytes, view.view_bytes);
        if (view.view_bytes)
            group.slices.emplace_back(view.view_offset, view.view_offset + view.view_bytes);
    }
    std::uint64_t known_reserved = 0;
    for (auto& [key, group] : groups) {
        static_cast<void>(key);
        auto& slices = group.slices;
        std::sort(slices.begin(), slices.end());
        std::uint64_t begin = 0, end = 0, covered = 0;
        for (const auto& slice : slices) {
            if (slice.first > end) {
                covered = add(covered, end - begin);
                begin = slice.first; end = slice.second;
            } else if (slice.second > end) end = slice.second;
        }
        covered = add(covered, end - begin);
        group.summary.covered_view_bytes = covered;
        group.summary.alias_view_bytes = group.logical_bytes - covered;
        known_reserved = add(known_reserved, group.summary.allocation_bytes);
        if (group.summary.kind == UmaBackingKind::allocated_storage)
            result.known_allocated_storage_bytes = add(result.known_allocated_storage_bytes, group.summary.allocation_bytes);
        else result.known_file_mapping_bytes = add(result.known_file_mapping_bytes, group.summary.allocation_bytes);
        result.known_covered_view_bytes = add(result.known_covered_view_bytes, covered);
        result.known_alias_view_bytes = add(result.known_alias_view_bytes, group.summary.alias_view_bytes);
        result.known_backings.push_back(group.summary);
    }
    result.reservation_upper_bound_bytes = add(result.reservation_upper_bound_bytes, known_reserved);
    if (!result.unknown_backing_views) result.unique_backing_reserved_bytes = known_reserved;
    for (auto& [category, logical] : categories)
        result.logical_categories.push_back(UmaCategoryViews{category, logical});
    result.pinned_views.assign(views.begin(), views.end());
    return result;
}
} // namespace strix
