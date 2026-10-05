#include "strix/direct_transport_rearm.hpp"

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

strix::transport::TransferDescriptor descriptor(
    std::uint64_t epoch,
    std::uint64_t sequence) {
    strix::transport::TransferDescriptor value;
    value.connection_epoch = epoch;
    value.model_plan_id = 1;
    value.request_id = 2;
    value.boundary_id = 3;
    value.sequence = sequence;
    value.shape_fingerprint = 4;
    value.dtype_id = 1;
    value.byte_length = 4096;
    return value;
}

}  // namespace

int main() {
    using namespace strix::transport;

    DirectRingRearmQualification gate(3);
    for (std::uint64_t cycle = 0;
         cycle < 3;
         ++cycle) {
        const auto epoch =
            100ULL + cycle;
        const auto sequence =
            500ULL + cycle;
        const auto desc =
            descriptor(
                epoch,
                sequence);
        gate.begin_cycle(
            epoch,
            sequence);
        gate.observe_first_completion(
            desc,
            Completion{
                10ULL + cycle,
                CompletionKind::receive_complete,
                desc.byte_length,
            });
        gate.end_cycle();
    }
    expect(
        gate.qualified() &&
            gate.completed_cycles() == 3,
        "re-arm gate requires all repeated first-completion cycles");

    DirectLinkLifecycle link;
    link.completion_rearm_qualified = false;
    publish_direct_rearm_qualification(
        link,
        gate);
    expect(
        link.completion_rearm_qualified,
        "qualified re-arm gate publishes lifecycle capability");

    DirectRingRearmQualification stale(1);
    stale.begin_cycle(
        7,
        9);
    bool stale_rejected = false;
    try {
        const auto desc =
            descriptor(
                8,
                9);
        stale.observe_first_completion(
            desc,
            Completion{
                1,
                CompletionKind::receive_complete,
                desc.byte_length,
            });
    } catch (const std::invalid_argument&) {
        stale_rejected = true;
    }
    expect(
        stale_rejected &&
            !stale.qualified(),
        "stale connection epoch cannot satisfy first-completion gate");
    stale.abort_cycle();

    DirectRingRearmQualification missing(1);
    missing.begin_cycle(
        9,
        11);
    bool missing_rejected = false;
    try {
        missing.end_cycle();
    } catch (const std::logic_error&) {
        missing_rejected = true;
    }
    expect(
        missing_rejected,
        "ring cycle cannot qualify without observing first completion");

    if (failures == 0) {
        std::cout
            << "direct transport re-arm tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
