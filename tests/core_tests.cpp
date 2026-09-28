#include "strix/memory_plan.hpp"
#include "strix/oracles.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename Function>
void expect_overflow(Function&& function, const char* message) {
    try {
        function();
        expect(false, message);
    } catch (const std::overflow_error&) {
    }
}

}  // namespace

int main() {
    const auto model = strix::ModelContract::qwen38_flash_next();
    expect(model.validate().empty(), "official model shape contract validates");
    expect(strix::base_kv_bytes_per_token(model) == 24ULL * 1024ULL,
           "base KV is 24 KiB/token/session");
    expect(strix::mtp_kv_bytes_per_token(model) == 2ULL * 1024ULL,
           "MTP KV is 2 KiB/token/session");
    expect(strix::provisional_indexer_bytes_per_token(model) == 768ULL,
           "provisional indexer estimate is 768 B/token/session");
    expect(strix::gdn_state_bytes_per_session(model) == 108ULL * 1024ULL * 1024ULL,
           "minimal GDN FP32 state is 108 MiB/session");

    strix::MemoryPlanRequest request;
    request.context_tokens = 262'144;
    request.sessions = 1;
    request.include_provisional_indexer = false;
    const auto one_session = strix::build_memory_plan(model, request);
    expect(one_session.base_kv == strix::gib(6), "262K base KV is 6 GiB");
    expect(one_session.mtp_kv == strix::gib(0) + 512ULL * 1024ULL * 1024ULL,
           "262K MTP KV is 512 MiB");

    request.sessions = 4;
    request.rollback = strix::RollbackPolicy::full_gdn_snapshots;
    request.draft_tokens = 7;
    const auto c4 = strix::build_memory_plan(model, request);
    expect(c4.base_kv == strix::gib(24), "C4 base KV is 24 GiB");
    expect(c4.rollback == 3ULL * strix::gib(1) - 48ULL * 1024ULL * 1024ULL,
           "C4 seven-snapshot rollback is 3024 MiB");

    const strix::HostBudget host{strix::gib(128), strix::gib(2), strix::gib(18)};
    auto small = strix::MemoryPlanRequest{};
    small.resident_model_bytes = strix::gib(68);
    const auto small_plan = strix::build_memory_plan(model, small);
    expect(strix::decide_admission(host, small_plan).admitted,
           "default 32K C1 plan fits host budget");

    request.resident_model_bytes = strix::gib(93);
    request.bounded_cache_bytes = strix::gib(4);
    const auto large_plan = strix::build_memory_plan(model, request);
    expect(!strix::decide_admission(host, large_plan).admitted,
           "large C4 full-rollback plan is rejected");

    expect_overflow(
        [&model] {
            auto overflow = strix::MemoryPlanRequest{};
            overflow.context_tokens = std::numeric_limits<std::uint64_t>::max();
            overflow.sessions = 2;
            static_cast<void>(strix::build_memory_plan(model, overflow));
        },
        "overflow is rejected");

    const float tied_scores[] = {4.0F, 9.0F, 9.0F, -1.0F, 8.0F};
    const auto top = strix::oracle::top_k_lower_index(tied_scores, 3);
    expect(top == std::vector<std::size_t>({1, 2, 4}),
           "candidate top-k keeps ties and orders them by lower index");
    try {
        const float invalid_scores[] = {1.0F, std::nanf("")};
        static_cast<void>(strix::oracle::top_k_lower_index(invalid_scores, 1));
        expect(false, "candidate top-k rejects NaN");
    } catch (const std::invalid_argument&) {
    }

    if (failures == 0) {
        std::cout << "all core tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
