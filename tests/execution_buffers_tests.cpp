#include "strix/execution_buffers.hpp"

#include <cstdint>
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

    strix::ExecutionBufferRequest request;
    request.logical_rows = 1;

    const auto c1 =
        strix::build_serial_execution_buffer_plan(
            model, request);

    expect(c1.expanded_hidden_bytes ==
               4ULL * 2560ULL * 2ULL,
           "C1 expanded hidden buffer matches hc_count * hidden BF16");
    expect(c1.hidden_bytes == 2560ULL * 2ULL,
           "C1 hidden buffer matches one BF16 hidden row");
    expect(c1.injection_weight_bytes == 4ULL * 4ULL,
           "C1 injection weights use four FP32 controls");
    expect(c1.route_index_bytes == 10ULL * 4ULL &&
               c1.route_weight_bytes == 10ULL * 4ULL,
           "C1 route metadata matches top-10");
    expect(c1.logits_bytes == 248320ULL * 4ULL,
           "C1 full logits materialization is explicit");
    expect(c1.ple_workspace_bytes == 138'496ULL,
           "C1 PLE workspace uses canonical aligned layout");
    expect(c1.mlp_intermediate_bytes ==
               2ULL * 640ULL * 4ULL,
           "C1 MoE temporary stores FP32 gate/up intermediates");
    expect(c1.gdn_workspace_bytes ==
               22'624ULL * 4ULL,
           "C1 FP32 analytical GDN workspace covers qkv z a b and core");
    expect(c1.expanded_hidden_buffers == 2 &&
               c1.hidden_buffers == 3,
           "serial plan reserves ping-pong hyper state and MoE branch buffers");
    expect(c1.peak_with_materialized_logits_bytes >
               c1.peak_internal_bytes,
           "full logits can dominate internal decode scratch");

    request.logical_rows = 8;
    const auto r8 =
        strix::build_serial_execution_buffer_plan(
            model, request);
    expect(r8.expanded_hidden_bytes ==
               8ULL * c1.expanded_hidden_bytes &&
               r8.logits_bytes ==
                   8ULL * c1.logits_bytes &&
               r8.mlp_intermediate_bytes ==
                   8ULL * c1.mlp_intermediate_bytes &&
               r8.gdn_workspace_bytes ==
                   8ULL * c1.gdn_workspace_bytes,
           "buffer planner scales explicitly with logical row class");

    bool invalid = false;
    try {
        auto bad = request;
        bad.logical_rows = 0;
        static_cast<void>(
            strix::build_serial_execution_buffer_plan(
                model, bad));
    } catch (const std::invalid_argument&) {
        invalid = true;
    }
    expect(invalid,
           "zero-row execution buffer plan is rejected");

    if (failures == 0) {
        std::cout << "execution buffer tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
