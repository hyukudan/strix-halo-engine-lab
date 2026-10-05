#include "strix/direct_transport_rearm.hpp"

#include <stdexcept>

namespace strix::transport {

DirectRingRearmQualification::
DirectRingRearmQualification(
    std::uint32_t required_cycles)
    : required_cycles_(required_cycles) {
    if (required_cycles_ == 0) {
        throw std::invalid_argument(
            "direct ring re-arm qualification requires at least one cycle");
    }
}

void DirectRingRearmQualification::begin_cycle(
    std::uint64_t connection_epoch,
    std::uint64_t expected_first_sequence) {
    if (active_) {
        throw std::logic_error(
            "direct ring re-arm cycle already active");
    }
    if (connection_epoch == 0) {
        throw std::invalid_argument(
            "direct ring re-arm cycle requires non-zero epoch");
    }
    epoch_ = connection_epoch;
    expected_sequence_ =
        expected_first_sequence;
    active_ = true;
    first_completion_seen_ = false;
}

void DirectRingRearmQualification::observe_first_completion(
    const TransferDescriptor& descriptor,
    const Completion& completion) {
    if (!active_ ||
        first_completion_seen_) {
        throw std::logic_error(
            "direct ring first completion is not expected");
    }
    if (descriptor.connection_epoch !=
            epoch_ ||
        descriptor.sequence !=
            expected_sequence_ ||
        descriptor.byte_length == 0 ||
        completion.handle == 0 ||
        completion.kind !=
            CompletionKind::receive_complete ||
        completion.bytes !=
            descriptor.byte_length) {
        throw std::invalid_argument(
            "direct ring first completion does not match armed epoch/sequence");
    }
    first_completion_seen_ = true;
}

void DirectRingRearmQualification::end_cycle() {
    if (!active_ ||
        !first_completion_seen_) {
        throw std::logic_error(
            "direct ring re-arm cycle cannot complete before first completion");
    }
    ++completed_cycles_;
    active_ = false;
    first_completion_seen_ = false;
    epoch_ = 0;
    expected_sequence_ = 0;
}

void DirectRingRearmQualification::abort_cycle() noexcept {
    active_ = false;
    first_completion_seen_ = false;
    epoch_ = 0;
    expected_sequence_ = 0;
}

void publish_direct_rearm_qualification(
    DirectLinkLifecycle& link,
    const DirectRingRearmQualification& qualification) {
    if (!qualification.qualified() ||
        qualification.cycle_active()) {
        throw std::logic_error(
            "direct link cannot publish incomplete re-arm qualification");
    }
    link.completion_rearm_qualified = true;
}

}  // namespace strix::transport
