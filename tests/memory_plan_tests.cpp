#include "strix/memory_plan.hpp"

#include <iostream>
#include <stdexcept>

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

    expect(
        strix::mtp_draft_state_bytes_per_session(
            model,
            2) == 20'480,
        "BF16 planning scenario accounts one 10240-wide MTP state row");

    expect(
        strix::mtp_draft_state_bytes_per_session(
            model,
            4) == 40'960,
        "FP32 planning scenario scales MTP draft state independently");

    strix::MemoryPlanRequest request;
    request.context_tokens = 16;
    request.sessions = 3;
    request.include_mtp_kv = false;
    request.include_indexer = false;
    request.mtp_draft_state_element_bytes = 2;

    const auto plan =
        strix::build_memory_plan(
            model,
            request);
    expect(plan.mtp_draft_state == 61'440,
           "memory plan accounts one bounded MTP draft state per session");

    auto without = request;
    without.include_mtp_draft_state = false;
    const auto no_draft_state =
        strix::build_memory_plan(
            model,
            without);
    expect(no_draft_state.mtp_draft_state == 0 &&
               plan.total - no_draft_state.total ==
                   plan.mtp_draft_state,
           "MTP draft state can be isolated as a planning category");

    bool zero_rejected = false;
    try {
        static_cast<void>(
            strix::mtp_draft_state_bytes_per_session(
                model,
                0));
    } catch (const std::invalid_argument&) {
        zero_rejected = true;
    }
    expect(zero_rejected,
           "zero-byte MTP state element size is rejected");

    if (failures == 0) {
        std::cout << "memory plan tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
