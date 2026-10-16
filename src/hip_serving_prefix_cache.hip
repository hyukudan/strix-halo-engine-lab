#include "strix/hip_serving_prefix_cache.hpp"
#include "strix/hip_runtime_allocator.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <limits>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace strix {
namespace {
std::uint64_t sum(std::uint64_t a,std::uint64_t b) {
    if(b>std::numeric_limits<std::uint64_t>::max()-a) throw std::overflow_error("serving prefix reservation overflows");
    return a+b;
}
std::uint64_t product(std::uint64_t a,std::uint64_t b) {
    if(a && b>std::numeric_limits<std::uint64_t>::max()/a) throw std::overflow_error("serving prefix reservation overflows");
    return a*b;
}
void count(std::uint64_t& value,std::uint64_t amount,bool& saturated) noexcept {
    if(amount>std::numeric_limits<std::uint64_t>::max()-value) {value=std::numeric_limits<std::uint64_t>::max();saturated=true;}
    else value+=amount;
}
HipServingPrefixCachePolicy checked_policy(HipServingPrefixCachePolicy p) {
    (void)HipServingPrefixCache::requested_reservation_bytes(p);return p;
}
// Standalone allocator lifetime: cache eviction or the final pinned handle may
// be released on another worker after the originating request is gone.
class SnapshotAllocator final : public RuntimeMemoryAllocator {
public:
    std::shared_ptr<const HipSetupQuiescenceGuard> guard=std::make_shared<HipSetupQuiescenceGuard>();
    HipRuntimeAllocator native{HipHostAllocationMode::pageable_aligned};
    ~SnapshotAllocator() override {bind_noexcept();}
    void bind() const {
        if(current_hip_launch_context()) throw std::invalid_argument("serving prefix cache cannot nest an active HIP scope");
        if(hipSetDevice(guard->device_index())!=hipSuccess) throw std::runtime_error("serving prefix snapshot device is unavailable");
        guard->require_original();
    }
    void bind_noexcept() const noexcept {
        if(hipSetDevice(guard->device_index())!=hipSuccess) std::terminate();
        try {guard->require_original();} catch(...) {std::terminate();}
    }
    std::uintptr_t allocate_device(std::uint64_t n,std::uint64_t alignment) override {bind();return native.allocate_device(n,alignment);}
    std::uintptr_t allocate_host(std::uint64_t n,std::uint64_t alignment) override {bind();return native.allocate_host(n,alignment);}
    void free_device(std::uintptr_t p,std::uint64_t n) noexcept override {bind_noexcept();native.free_device(p,n);}
    void free_host(std::uintptr_t p,std::uint64_t n) noexcept override {bind_noexcept();native.free_host(p,n);}
};
}
struct HipServingPrefixCache::Impl {
    const HipServingPrefixCachePolicy policy;
    const ModelContentIdentity content;
    const std::uint32_t vocabulary;
    const std::shared_ptr<SnapshotAllocator> allocator;
    AllocationGenerationSource generations;
    mutable std::mutex mutex;
    SerialPrefixCache cache;
    HipServingPrefixCacheStats counters;
    Impl(HipServingPrefixCachePolicy p,ModelContentIdentity model,std::uint32_t v)
        :policy(checked_policy(p)),content(model),vocabulary(v),allocator(std::make_shared<SnapshotAllocator>()),
         cache(policy.max_entries,policy.snapshot_byte_budget) {
        if(!content.valid() || !vocabulary) throw std::invalid_argument("serving prefix cache requires its actual model and vocabulary");
    }
};
std::uint64_t HipServingPrefixCache::requested_reservation_bytes(const HipServingPrefixCachePolicy& p) {
    if(!p.max_entries || !p.snapshot_byte_budget || p.capture_peak_bytes<p.snapshot_byte_budget || !p.max_prefix_tokens ||
        p.max_entries==std::numeric_limits<std::size_t>::max() || p.max_prefix_tokens>std::numeric_limits<std::size_t>::max()/sizeof(std::uint32_t))
        throw std::invalid_argument("serving prefix cache needs bounded entries, snapshot bytes, capture peak and prefix length");
    // One extra candidate during capture; 64-byte snapshot alignment adds at
    // most63 bytes to each device backing. Bookkeeping allowance includes the
    // seven-range plan copies, one serial lineage interval and bounded entries.
    // General allocator/driver overhead remains in caller's explicit headroom.
    const auto entry=sum(product(p.max_prefix_tokens,sizeof(std::uint32_t)),4096U+63U);
    return sum(p.capture_peak_bytes,product(static_cast<std::uint64_t>(p.max_entries)+1U,entry));
}
HipServingPrefixCache::HipServingPrefixCache(HipServingPrefixCachePolicy policy,ModelContentIdentity content,std::uint32_t vocabulary)
    :impl_(std::make_unique<Impl>(policy,content,vocabulary)) {}
HipServingPrefixCache::~HipServingPrefixCache()=default;
int HipServingPrefixCache::device_index() const noexcept {return impl_->allocator->guard->device_index();}
const ModelContentIdentity& HipServingPrefixCache::model_content() const noexcept {return impl_->content;}
std::uint32_t HipServingPrefixCache::vocabulary_size() const noexcept {return impl_->vocabulary;}
HipServingPrefixCacheStats HipServingPrefixCache::stats() const {
    std::lock_guard lock(impl_->mutex);auto stats=impl_->counters;
    stats.entries=static_cast<std::uint64_t>(impl_->cache.size());stats.snapshot_bytes=impl_->cache.snapshot_bytes();return stats;
}
std::optional<GreedyPrefixRestore> HipServingPrefixCache::try_restore(const ModelContract& model,const SerialGreedyRequest& request,
    SerialSessionContext& session,const SerialSessionStateLayout& layout,const SessionStateMemoryBinding& state,
    SessionSnapshotTransferBackend& transfer) {
    auto& s=*impl_;s.allocator->bind();
    if(request.vocab_size!=s.vocabulary || model.vocab_size!=s.vocabulary || request.prefix_tokens.empty() ||
        request.context_capacity!=layout.context_tokens || session.health!=SerialSessionHealth::ready ||
        session.next_position || session.committed_tokens || session.arithmetic_lineage.tainted)
        throw std::invalid_argument("serving prefix restore requires a fresh compatible request frontier");
    std::shared_ptr<const SerialPrefixCacheEntry> handle;
    {
        std::lock_guard lock(s.mutex);count(s.counters.lookups,1U,s.counters.saturated);
        handle=s.cache.find_longest_greedy(request.prefix_tokens,s.content,s.vocabulary,session.execution_identity,
            session.runtime_compatibility_identity,session.state_layout_identity);
        if(!handle) return std::nullopt;
        count(s.counters.hits,1U,s.counters.saturated);
    }
    const auto n=handle->prefix().prefix_tokens.size();
    try {
        // The handle pins snapshot+allocator even while another worker inserts
        // or evicts. Its frontier and exact source lineage are immutable.
        handle->restore(session,model,layout,state,transfer,request.prefix_tokens.first(n));
        std::lock_guard lock(s.mutex);count(s.counters.restored_prefix_tokens,static_cast<std::uint64_t>(n),s.counters.saturated);
        return GreedyPrefixRestore{static_cast<std::uint64_t>(n),n==request.prefix_tokens.size()?
            std::optional<std::uint32_t>(handle->greedy_frontier()->pending_token):std::nullopt};
    } catch(...) {
        std::lock_guard lock(s.mutex);count(s.counters.failures,1U,s.counters.saturated);throw;
    }
}
void HipServingPrefixCache::capture(const ModelContract& model,std::span<const std::uint32_t> prefix,std::uint32_t pending,
    const SerialSessionContext& session,const SerialSessionStateLayout& layout,const SessionStateMemoryBinding& state,
    SessionSnapshotTransferBackend& transfer) {
    auto& s=*impl_;s.allocator->bind();
    if(model.vocab_size!=s.vocabulary || pending>=s.vocabulary || prefix.empty() ||
        std::any_of(prefix.begin(),prefix.end(),[&](auto token){return token>=s.vocabulary;}) || session.health!=SerialSessionHealth::ready ||
        session.arithmetic_lineage.tainted || session.next_position!=prefix.size() || session.committed_tokens!=prefix.size() ||
        !arithmetic_lineage_serial_only(session.arithmetic_lineage,session.execution_identity,session.next_position))
        throw std::invalid_argument("serving prefix capture requires a completed serial finite-scalar frontier");
    std::lock_guard lock(s.mutex);count(s.counters.capture_attempts,1U,s.counters.saturated);
    if(prefix.size()>s.policy.max_prefix_tokens) {count(s.counters.skipped,1U,s.counters.saturated);return;}
    auto existing=s.cache.find_longest_greedy(prefix,s.content,s.vocabulary,session.execution_identity,
        session.runtime_compatibility_identity,session.state_layout_identity);
    if(existing && existing->prefix().prefix_tokens.size()==prefix.size()) {
        if(existing->greedy_frontier()->pending_token!=pending) throw std::invalid_argument("serving prefix repeated frontier differs");
        count(s.counters.skipped,1U,s.counters.saturated);return;
    }
    existing.reset(); // Do not accidentally pin a shorter entry during replacement.
    SerialSessionSnapshotRequest request;request.compiled_execution_identity=session.execution_identity;
    request.runtime_compatibility_identity=session.runtime_compatibility_identity;request.next_position=session.next_position;
    request.committed_tokens=session.committed_tokens;request.state=session.state;
    const auto snapshot=build_serial_session_snapshot_plan(model,layout,request);
    if(snapshot.ranges.size()!=7U) throw std::invalid_argument("serving prefix snapshot must cover all seven state components");
    std::array<bool,7> components{};
    for(const auto& range:snapshot.ranges) {
        const auto index=static_cast<std::size_t>(range.component);
        const auto* allocation=find_session_state_allocation(layout,range.component);
        if(index>=components.size() || components[index] || !allocation || range.bytes!=allocation->bytes ||
            range.storage!=allocation->storage || range.source_offset!=allocation->offset)
            throw std::invalid_argument("serving prefix snapshot omits or truncates a state component");
        components[index]=true;
    }
    if(session.arithmetic_lineage.intervals.size()!=1U) {count(s.counters.skipped,1U,s.counters.saturated);return;}
    const auto bytes=sum(snapshot.device_bytes,snapshot.host_bytes);
    if(bytes>s.policy.snapshot_byte_budget || bytes>s.policy.capture_peak_bytes ||
        s.cache.snapshot_bytes()>s.policy.capture_peak_bytes-bytes) {count(s.counters.skipped,1U,s.counters.saturated);return;}
    try {
        auto entry=SerialPrefixCacheEntry::capture(model,prefix,session,layout,state,*s.allocator,s.generations,transfer,64U,
            SerialPrefixGreedyFrontier{s.content,s.vocabulary,pending});
        const auto admission=s.cache.insert(std::move(entry),s.allocator);
        if(admission==SerialPrefixCacheAdmission::inserted) count(s.counters.inserted,1U,s.counters.saturated);
        else count(s.counters.skipped,1U,s.counters.saturated);
    } catch(...) {count(s.counters.failures,1U,s.counters.saturated);throw;}
}
} // namespace strix
