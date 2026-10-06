#include "strix/speculation_rng.hpp"

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
    strix::SpeculationRngIdentity identity;
    identity.seed = 1234;
    identity.request_identity = 0x1111ULL;
    identity.sampling_policy_identity =
        0x2222ULL;

    const auto rng_id =
        strix::speculation_rng_identity(
            identity);
    expect(
        rng_id != 0,
        "speculation RNG has a stable non-zero identity");

    const auto serial_10 =
        strix::speculation_random_u64(
            identity,
            10,
            strix::SpeculationRandomDrawClass::
                serial_target);
    const auto serial_11 =
        strix::speculation_random_u64(
            identity,
            11,
            strix::SpeculationRandomDrawClass::
                serial_target);

    // Exercise speculative-only draws in between. Stateless addressing means
    // these can never advance a cursor used by the serial target stream.
    static_cast<void>(
        strix::speculation_random_u64(
            identity,
            10,
            strix::SpeculationRandomDrawClass::
                draft_acceptance));
    static_cast<void>(
        strix::speculation_random_u64(
            identity,
            10,
            strix::SpeculationRandomDrawClass::
                residual_fallback));
    for (std::uint32_t index = 0;
         index < 16;
         ++index) {
        static_cast<void>(
            strix::speculation_random_u64(
                identity,
                10,
                strix::SpeculationRandomDrawClass::
                    draft_acceptance,
                index));
    }

    expect(
        strix::speculation_random_u64(
            identity,
            10,
            strix::SpeculationRandomDrawClass::
                serial_target) ==
                serial_10 &&
            strix::speculation_random_u64(
                identity,
                11,
                strix::SpeculationRandomDrawClass::
                    serial_target) ==
                serial_11,
        "rejected-draft RNG work cannot perturb serial target draws");

    expect(
        serial_10 != serial_11 &&
            serial_10 !=
                strix::speculation_random_u64(
                    identity,
                    10,
                    strix::SpeculationRandomDrawClass::
                        draft_acceptance),
        "absolute position and draw class occupy distinct RNG counters");

    auto other_request =
        identity;
    ++other_request.request_identity;
    expect(
        strix::speculation_random_u64(
            other_request,
            10,
            strix::SpeculationRandomDrawClass::
                serial_target) !=
            serial_10,
        "request identity separates random streams");

    auto other_policy =
        identity;
    ++other_policy.sampling_policy_identity;
    expect(
        strix::speculation_random_u64(
            other_policy,
            10,
            strix::SpeculationRandomDrawClass::
                serial_target) !=
            serial_10,
        "sampling-policy identity separates random streams");

    const auto unit =
        strix::speculation_uniform01(
            identity,
            10,
            strix::SpeculationRandomDrawClass::
                serial_target);
    expect(
        unit >= 0.0 &&
            unit < 1.0,
        "speculation RNG maps deterministically into [0,1)");

    bool invalid_rejected = false;
    try {
        strix::SpeculationRngIdentity invalid;
        invalid.seed = 1;
        invalid.request_identity = 0;
        invalid.sampling_policy_identity = 2;
        static_cast<void>(
            strix::speculation_rng_identity(
                invalid));
    } catch (const std::invalid_argument&) {
        invalid_rejected = true;
    }
    expect(
        invalid_rejected,
        "speculation RNG rejects missing request identity");

    if (failures == 0) {
        std::cout
            << "speculation RNG tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
