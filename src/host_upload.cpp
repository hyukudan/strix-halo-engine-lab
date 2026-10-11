#include "strix/host_upload.hpp"

#include <atomic>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strix {
namespace {

std::uint64_t next_pool_namespace() {
    static std::atomic<std::uint64_t> next{1};
    auto candidate = next.load(std::memory_order_relaxed);
    for (;;) {
        if (candidate == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("host upload pool namespace exhausted");
        }
        if (next.compare_exchange_weak(candidate, candidate + 1,
                                       std::memory_order_relaxed)) return candidate;
    }
}

bool same_destination(const ExecutionBufferAddress& left,
                      const ExecutionBufferAddress& right) noexcept {
    return left.address == right.address && left.bytes == right.bytes &&
           left.allocation_generation == right.allocation_generation && left.use == right.use;
}

void validate_destination(const ExecutionBufferAddress& destination,
                          std::uint64_t bytes) {
    if (destination.address == 0 || destination.allocation_generation == 0 || destination.use.generation == 0 ||
        bytes == 0 || bytes > destination.bytes ||
        destination.bytes > std::numeric_limits<std::uintptr_t>::max() - destination.address) {
        throw std::invalid_argument("host upload destination is unbound, undersized or overflowing");
    }
}

}  // namespace

HostUploadPool::HostUploadPool(HostUploadBackend& backend,
                             AllocationGenerationSource& generations,
                             HostUploadPolicy policy)
    : backend_(&backend), generations_(&generations), policy_(policy) {
    if (policy.maximum_slots == 0 || policy.bytes_per_slot == 0 ||
        policy.maximum_total_bytes == 0 || policy.maximum_polls == 0 ||
        policy.bytes_per_slot > std::numeric_limits<std::size_t>::max() ||
        policy.bytes_per_slot > policy.maximum_total_bytes / policy.maximum_slots) {
        throw std::invalid_argument("host upload pool requires bounded slots, bytes and progress");
    }
    pool_generation_ = next_pool_namespace();
    slots_.resize(policy.maximum_slots);
}

HostUploadPool::~HostUploadPool() {
    for (auto& slot : slots_) {
        if (slot.handle != 0) cancel_noexcept(slot);
    }
    // Backend-owned operations still retain every pending source. Dropping the
    // pool's references cannot free a source that a copy may still read.
}

HostUploadTicket HostUploadPool::start(std::span<const std::byte> bytes,
                                      const ExecutionBufferAddress& destination) {
    if (failed_) throw std::logic_error("failed host upload pool cannot accept new work");
    if (bytes.empty() || bytes.size() > policy_.bytes_per_slot) {
        throw std::invalid_argument("host upload exceeds bounded slot bytes");
    }
    validate_destination(destination, bytes.size());
    for (const auto& pending : slots_) {
        if (pending.handle != 0 &&
            destination.address < pending.destination.address + pending.destination.bytes &&
            pending.destination.address < destination.address + destination.bytes) {
            throw std::invalid_argument("host upload overlaps a destination still owned by a pending copy");
        }
    }
    std::uint32_t index = 0;
    while (index < slots_.size() && slots_[index].handle != 0) ++index;
    if (index == slots_.size()) {
        throw std::runtime_error("host upload pool has no reusable slot");
    }
    auto& slot = slots_[index];
    if (slot.sequence == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("host upload slot sequence exhausted");
    }
    if (!slot.source) {
        const auto generation = generations_->next();
        slot.source_generation = generation;
        slot.source = backend_->allocate_source(policy_.bytes_per_slot, generation);
        if (!slot.source || slot.source->allocation_generation() != generation ||
            slot.source->writable_bytes().size() != policy_.bytes_per_slot ||
            slot.source->bytes().size() != policy_.bytes_per_slot ||
            slot.source->bytes().data() == nullptr ||
            slot.source->bytes().data() != slot.source->writable_bytes().data()) {
            failed_ = true;
            throw std::invalid_argument("host upload allocator returned foreign source binding");
        }
    }
    if (slot.source->allocation_generation() != slot.source_generation) {
        failed_ = true;
        throw std::invalid_argument("host upload source allocation generation changed");
    }
    std::memcpy(slot.source->writable_bytes().data(), bytes.data(), bytes.size());
    slot.destination = destination;
    slot.cancel_requested = false;
    ++slot.sequence;
    try {
        slot.handle = backend_->prepare(destination, slot.source, bytes.size());
        if (slot.handle == 0) throw std::runtime_error("host upload backend returned zero handle");
        backend_->submit(slot.handle);
    } catch (...) {
        failed_ = true;
        if (slot.handle != 0) cancel_noexcept(slot);
        throw;
    }
    return {pool_generation_, index, slot.source_generation, slot.sequence};
}

HostUploadPool::Slot& HostUploadPool::require_ticket(const HostUploadTicket& ticket) {
    if (ticket.pool_generation != pool_generation_ || ticket.slot >= slots_.size()) {
        throw std::invalid_argument("host upload ticket belongs to a foreign pool");
    }
    auto& slot = slots_[ticket.slot];
    if (slot.handle == 0 || !slot.source || ticket.sequence != slot.sequence ||
        ticket.source_allocation_generation != slot.source_generation ||
        slot.source_generation != slot.source->allocation_generation()) {
        throw std::invalid_argument("host upload ticket is stale or has foreign allocation generation");
    }
    return slot;
}

void HostUploadPool::cancel_noexcept(Slot& slot) noexcept {
    if (slot.cancel_requested) return;
    slot.cancel_requested = true;
    try { backend_->cancel(slot.handle); } catch (...) { failed_ = true; }
}

HostUploadStatus HostUploadPool::poll(const HostUploadTicket& ticket,
                                    const ExecutionBufferAddress& current_destination) {
    auto& slot = require_ticket(ticket);
    if (!same_destination(slot.destination, current_destination)) {
        failed_ = true;
        cancel_noexcept(slot);
        throw std::invalid_argument("host upload destination binding changed while copy was pending");
    }
    HostUploadStatus status = HostUploadStatus::failed_retained;
    try { status = backend_->poll(slot.handle); }
    catch (...) {
        failed_ = true;
        cancel_noexcept(slot);
        throw;
    }
    if (status == HostUploadStatus::complete || status == HostUploadStatus::canceled) {
        try { backend_->retire(slot.handle); }
        catch (...) { failed_ = true; cancel_noexcept(slot); throw; }
        slot.handle = 0;
    } else if (status == HostUploadStatus::failed_retained) {
        failed_ = true;
        cancel_noexcept(slot);
    }
    return status;
}

void HostUploadPool::wait(const HostUploadTicket& ticket,
                          const ExecutionBufferAddress& current_destination) {
    for (std::uint64_t count = 0; count < policy_.maximum_polls; ++count) {
        const auto status = poll(ticket, current_destination);
        if (status == HostUploadStatus::complete) return;
        if (status == HostUploadStatus::canceled) {
            throw std::runtime_error("host upload was canceled after source release");
        }
        if (status == HostUploadStatus::failed_retained) {
            throw std::runtime_error("host upload failed with source retained");
        }
    }
    failed_ = true;
    cancel_noexcept(require_ticket(ticket));
    throw std::runtime_error("host upload completion poll budget exhausted; source retained");
}

void HostUploadPool::cancel(const HostUploadTicket& ticket) {
    auto& slot = require_ticket(ticket);
    if (slot.cancel_requested) return;
    slot.cancel_requested = true;
    try { backend_->cancel(slot.handle); }
    catch (...) { failed_ = true; throw; }
}

bool HostUploadPool::drain(const HostUploadTicket& ticket,
                          const ExecutionBufferAddress& current_destination) {
    for (std::uint64_t count = 0; count < policy_.maximum_polls; ++count) {
        const auto status = poll(ticket, current_destination);
        if (status == HostUploadStatus::complete || status == HostUploadStatus::canceled) return true;
    }
    return false;
}

std::uint64_t HostUploadPool::allocated_bytes() const noexcept {
    std::uint64_t total = 0;
    for (const auto& slot : slots_) if (slot.source) total += policy_.bytes_per_slot;
    return total;
}

std::uint32_t HostUploadPool::active_uploads() const noexcept {
    std::uint32_t active = 0;
    for (const auto& slot : slots_) if (slot.handle != 0) ++active;
    return active;
}

}  // namespace strix
