#include "strix/memory_plan.hpp"

#include <iostream>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();

    expect(strix::mtp_indexer_bytes_per_token(model) == 256,
           "one MTP predictor layer stores one BF16 128-wide raw index key per token");

    strix::MemoryPlanRequest request;
    request.context_tokens = 16;
    request.sessions = 2;
    request.include_mtp_kv = false;
    request.include_indexer = false;
    request.include_mtp_draft_state = false;

    const auto plan =
        strix::build_memory_plan(
            model,
            request);
    expect(plan.mtp_indexer == 8192,
           "memory plan accounts MTP indexer state across sessions and context");

    auto without = request;
    without.include_mtp_indexer = false;
    const auto no_indexer =
        strix::build_memory_plan(
            model,
            without);
    expect(no_indexer.mtp_indexer == 0 &&
               plan.total - no_indexer.total ==
                   plan.mtp_indexer,
           "MTP indexer is an independently visible admission category");

    if (failures == 0) {
        std::cout << "MTP memory state tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
