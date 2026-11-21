#include "strix/session/kv_capacity.hpp"
#include "strix/session/prefix_cache.hpp"

#include <mutex>
#include <utility>

namespace strix::session {

struct KvCapacity::State {
    State(std::uint64_t bytes, std::size_t slots)
        : owner(make_cache_owner(bytes)), free_slots(slots) {}
    std::shared_ptr<PrefixStateCache> owner;
    Credit fixed;
    std::mutex mutex;
    std::size_t free_slots;
};

KvCapacity::KvCapacity(std::uint64_t budget, std::uint64_t fixed_bytes, std::size_t slots) {
    if (!budget || fixed_bytes > budget || !slots)
        throw std::invalid_argument("invalid session capacity");
    state_ = std::make_shared<State>(budget, slots);
    state_->fixed = state_->owner->reserve(fixed_bytes);
}

KvCapacity::Lease::Execution::Execution(std::shared_ptr<State> state, std::size_t slots,
                                        Credit whole_slots, KvPages<Credit> pages)
    : kv(std::move(pages)), state_(std::move(state)), slots_(slots), whole_slots_(std::move(whole_slots)) {}

KvCapacity::Lease::Execution::~Execution() {
    std::lock_guard lock(state_->mutex);
    kv.clear();
    whole_slots_ = {};
    state_->free_slots += slots_;
}

KvCapacity::Result KvCapacity::reserve(KvPageLayout layout, std::uint64_t context,
                                       std::size_t slots, std::uint64_t retained_bytes) {
    if (!context || !slots || !layout.page_tokens)
        throw std::invalid_argument("session reservation requires context, slots and page geometry");
    std::lock_guard lock(state_->mutex);
    if (slots > state_->free_slots) return {Admission::slot_limit, {}};
    const auto per_slot = 1U + (context - 1U) / layout.page_tokens;
    if (layout.page_bytes > (UINT64_MAX - layout.whole_slot_bytes) / per_slot)
        return {Admission::byte_limit, {}};
    const auto slot_bytes = per_slot * layout.page_bytes + layout.whole_slot_bytes;
    if (slot_bytes > (UINT64_MAX - retained_bytes) / slots) return {Admission::byte_limit, {}};
    const auto needed = retained_bytes + slots * slot_bytes;
    const auto budget = state_->owner->policy().host_byte_budget;
    if (needed > budget - state_->owner->reserved_bytes()) return {Admission::byte_limit, {}};
    auto credit = state_->owner->reserve(needed);
    if (!credit) return {Admission::byte_limit, {}};

    // Reserve the entire extent before creating any page descriptors. A fork
    // shares each credit until its first write obtains a separate page credit.
    KvPages<Credit> pages([state = state_, bytes = layout.page_bytes](const Credit&) {
        std::lock_guard detach_lock(state->mutex);
        auto detached = state->owner->reserve(bytes);
        if (!detached) throw std::length_error("insufficient capacity to detach KV page");
        return std::make_unique<Credit>(std::move(detached));
    });
    if (layout.page_bytes) {
        const auto count = slots * per_slot;
        for (std::uint64_t i = 0; i < count; ++i)
            pages.append(std::make_unique<Credit>(credit.split(layout.page_bytes)));
    }
    auto retained = credit.split(retained_bytes);
    auto execution = std::unique_ptr<Lease::Execution>(
        new Lease::Execution(state_, slots, std::move(credit), std::move(pages)));
    state_->free_slots -= slots;
    return {Admission::accepted, {std::move(retained), std::move(execution)}};
}

std::uint64_t KvCapacity::reserved_bytes() const { return state_->owner->reserved_bytes(); }
void KvCapacity::with_optional_cache(const std::function<void(PrefixStateCache&)>& visit) const {
    std::lock_guard lock(state_->mutex);
    visit(*state_->owner);
}

} // namespace strix::session
