#include "strix/ple_rolling.hpp"
#include "strix/residency/expert_residency.hpp"

#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
void require(bool success, const char* message) {
    if (!success) throw std::runtime_error(message);
}

void pressure() {
    constexpr std::uint64_t page = 4096;
    constexpr std::uint64_t budget = 2U * page;
    std::vector<std::uint64_t> retired_pages;
    strix::ple_reclaim::RollingPages::Io io;
    io.reclaimable = [](auto) { return true; };
    io.resident = [] { return std::uint64_t{0}; };
    io.evict = [&](auto pages) { retired_pages.insert(retired_pages.end(), pages.begin(), pages.end()); };
    auto manager = std::make_shared<strix::ple_reclaim::ResidencyManager>(
        "pressure", 1, 1, 1, page, 0, page, 8, 1, budget, budget, 1, io);
    auto turn = manager->begin();
    turn->source(1, 1, std::array<std::uint64_t, 2>{1, 2}, [] {});
    turn->source(1, 1, std::array<std::uint64_t, 2>{3, 4}, [] {});
    turn->source(1, 1, std::array<std::uint64_t, 2>{3, 5}, [] {});
    require(retired_pages == std::vector<std::uint64_t>{1, 2, 4}, "ordered PLE evictions under pressure");
    require(manager->stats().evicted_pages == retired_pages.size(), "PLE eviction accounting");

    strix::residency::ExpertRefill refill;
    std::array<std::int32_t, 8> slot_of{0, -1, -1, -1, -1, -1, -1, -1};
    std::array<std::int32_t, 2> occupants{-1, -1};
    std::array<std::uint64_t, 2> last_used{};
    constexpr std::array<std::uint8_t, 8> bytes{0, 1, 2, 3, 4, 5, 6, 7};
    std::array<const std::uint8_t*, 8> sources{};
    for (std::size_t i = 1; i < 7; ++i) sources[i] = &bytes[i];
    const strix::residency::RefillLayer layer{1, slot_of, occupants, last_used, sources, page};
    std::vector<std::uint64_t> retired_experts;
    std::vector<int> operations;
    const auto wait = [&] { operations.push_back(-1); };
    const auto copy = [&](std::size_t slot, const std::uint8_t* source) {
        const auto evicted = occupants.at(slot - 1U);
        if (evicted >= 0) retired_experts.push_back(static_cast<std::uint64_t>(evicted));
        operations.push_back(*source);
    };

    refill.begin_token(2);
    require(refill.apply(layer, std::array<std::uint32_t, 3>{0, 1, 2}, wait, copy), "fill both expert slots");
    require(occupants == std::array<std::int32_t, 2>{1, 2}, "empty slots filled in ascending order");
    require(refill.stats().evictions == 0 && refill.stats().copied_bytes == budget, "fill accounting");

    refill.begin_token(2);
    require(refill.apply(layer, std::array<std::uint32_t, 4>{3, 4, 6, 7}, wait, copy), "replace both equal-age experts");
    const auto over_budget = refill.stats();
    require(over_budget.remote == 4 && over_budget.copies == 2 && over_budget.evictions == 2 &&
            over_budget.copied_bytes == budget && over_budget.over_budget == 1 && over_budget.not_in_source == 1,
            "expert budget, missing source and eviction accounting");

    refill.begin_token(2);
    require(refill.apply(layer, std::array<std::uint32_t, 3>{5, 3, 6}, wait, copy), "replace the single unprotected expert");
    const auto protected_slots = refill.stats();
    require(protected_slots.remote == 2 && protected_slots.dynamic_hits == 1 && protected_slots.copies == 1 &&
            protected_slots.evictions == 1 && protected_slots.no_slot == 1 && protected_slots.copied_bytes == page,
            "expert protected-slot accounting");
    require(slot_of == std::array<std::int32_t, 8>{0, -1, -1, 1, -1, 2, -1, -1}, "published expert table");
    require(operations == std::vector<int>{-1, 1, 2, -1, 3, 4, -1, 5}, "one wait before each layer's ordered copies");
    require(retired_experts == retired_pages, "PLE and expert adapters agree on the protected LRU order");
    turn->finish();
}
}  // namespace

int main() try {
    pressure();
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
