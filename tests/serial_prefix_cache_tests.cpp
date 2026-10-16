#include "strix/serial_prefix_cache.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

class VectorAllocator final
    : public strix::RuntimeMemoryAllocator {
public:
    std::uintptr_t allocate_device(
        std::uint64_t bytes,
        std::uint64_t) override {
        return allocate(bytes);
    }

    std::uintptr_t allocate_host(
        std::uint64_t bytes,
        std::uint64_t) override {
        return allocate(bytes);
    }

    void free_device(
        std::uintptr_t address,
        std::uint64_t) noexcept override {
        storage_.erase(address);
    }

    void free_host(
        std::uintptr_t address,
        std::uint64_t) noexcept override {
        storage_.erase(address);
    }

private:
    std::uintptr_t allocate(
        std::uint64_t bytes) {
        auto buffer =
            std::vector<std::uint8_t>(
                static_cast<std::size_t>(
                    bytes));
        const auto address =
            reinterpret_cast<std::uintptr_t>(
                buffer.data());
        storage_.emplace(
            address,
            std::move(buffer));
        return address;
    }

    std::unordered_map<
        std::uintptr_t,
        std::vector<std::uint8_t>> storage_;
};

class MemcpyTransferBackend final
    : public strix::SessionSnapshotTransferBackend {
public:
    bool fail_restore = false;
    void capture_device(
        std::uintptr_t source,
        std::uintptr_t destination,
        std::uint64_t bytes) override {
        copy(source, destination, bytes);
    }

    void capture_host(
        std::uintptr_t source,
        std::uintptr_t destination,
        std::uint64_t bytes) override {
        copy(source, destination, bytes);
    }

    void restore_device(
        std::uintptr_t source,
        std::uintptr_t destination,
        std::uint64_t bytes) override {
        if (fail_restore) { throw std::runtime_error("injected restore failure"); }
        copy(source, destination, bytes);
    }

    void restore_host(
        std::uintptr_t source,
        std::uintptr_t destination,
        std::uint64_t bytes) override {
        copy(source, destination, bytes);
    }

private:
    static void copy(
        std::uintptr_t source,
        std::uintptr_t destination,
        std::uint64_t bytes) {
        std::memcpy(
            reinterpret_cast<void*>(destination),
            reinterpret_cast<const void*>(source),
            static_cast<std::size_t>(bytes));
    }
};

strix::PersistentState all_state() {
    using strix::PersistentState;
    return
        PersistentState::attention_kv |
        PersistentState::recurrent_matrix |
        PersistentState::convolution_history |
        PersistentState::sparse_index |
        PersistentState::ngram_history |
        PersistentState::position_history;
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::
            qwen38_flash_next();

    strix::SerialSessionStateLayoutRequest
        layout_request;
    layout_request.context_tokens = 4;
    layout_request.device_alignment = 64;
    layout_request.host_alignment = 64;
    const auto layout =
        strix::build_serial_session_state_layout(
            model,
            layout_request);

    std::vector<std::uint8_t> device_state(
        static_cast<std::size_t>(
            layout.device_total_bytes));
    std::vector<std::uint8_t> host_state(
        static_cast<std::size_t>(
            layout.host_total_bytes));

    for (std::size_t index = 0;
         index < device_state.size();
         ++index) {
        device_state[index] =
            static_cast<std::uint8_t>(
                (index * 11U + 1U) & 0xffU);
    }
    for (std::size_t index = 0;
         index < host_state.size();
         ++index) {
        host_state[index] =
            static_cast<std::uint8_t>(
                (index * 5U + 9U) & 0xffU);
    }

    strix::SessionStateMemoryBinding
        state_memory;
    state_memory.layout_identity =
        layout.layout_identity;
    state_memory.device_base =
        reinterpret_cast<std::uintptr_t>(
            device_state.data());
    state_memory.device_bytes =
        device_state.size();
    state_memory.device_allocation_generation = 3;
    state_memory.host_base =
        reinterpret_cast<std::uintptr_t>(
            host_state.data());
    state_memory.host_bytes =
        host_state.size();
    state_memory.host_allocation_generation = 4;

    const std::vector<std::uint32_t>
        prefix_tokens{101, 102};

    strix::SerialSessionContext session;
    session.execution_identity =
        0xaaa1ULL;
    session.runtime_compatibility_identity =
        0xaaa2ULL;
    session.state_layout_identity =
        layout.layout_identity;
    session.next_position =
        prefix_tokens.size();
    session.committed_tokens =
        prefix_tokens.size();
    session.state.valid =
        all_state();
    for (auto& generation :
         session.state.generations) {
        generation = 7;
    }
    for (std::uint64_t position = 0; position < prefix_tokens.size(); ++position) {
        session.arithmetic_lineage = strix::prepare_serial_row_lineage(
            session.arithmetic_lineage, position, session.execution_identity, 0x1234ULL);
    }

    VectorAllocator allocator;
    strix::AllocationGenerationSource
        generations;
    MemcpyTransferBackend transfer;

    auto cache =
        strix::SerialPrefixCacheEntry::capture(
            model,
            prefix_tokens,
            session,
            layout,
            state_memory,
            allocator,
            generations,
            transfer);

    expect(
        cache.valid() &&
            cache.prefix().prefix_tokens ==
                prefix_tokens &&
            cache.memory().allocation_generation != 0,
        "prefix cache owns metadata and compact snapshot allocation");

    std::fill(
        device_state.begin(),
        device_state.end(),
        0);
    std::fill(
        host_state.begin(),
        host_state.end(),
        0);
    session.next_position = 0;
    session.committed_tokens = 0;
    session.state = {};
    session.health =
        strix::SerialSessionHealth::poisoned;

    cache.restore(
        session,
        model,
        layout,
        state_memory,
        transfer,
        prefix_tokens);

    expect(
        session.health ==
            strix::SerialSessionHealth::ready &&
        session.next_position ==
            prefix_tokens.size() &&
        session.committed_tokens ==
            prefix_tokens.size() &&
        session.state.valid ==
            all_state(),
        "prefix cache restores exact committed session state");

    bool restored_bytes_match = true;
    const auto& snapshot =
        cache.prefix().snapshot;
    for (const auto& range :
         snapshot.ranges) {
        const auto* state_base =
            range.storage ==
                    strix::SessionStateStorage::device
                ? device_state.data()
                : host_state.data();
        const auto snapshot_base =
            range.storage ==
                    strix::SessionStateStorage::device
                ? cache.memory().
                      device_section_base
                : cache.memory().
                      host_section_base;
        if (std::memcmp(
                state_base + range.source_offset,
                reinterpret_cast<const void*>(
                    snapshot_base +
                    range.destination_offset),
                static_cast<std::size_t>(
                    range.bytes)) != 0) {
            restored_bytes_match = false;
            break;
        }
    }
    expect(
        restored_bytes_match,
        "prefix cache retains compact bytes needed for later restore");
    expect(strix::arithmetic_lineage_matches(session.arithmetic_lineage,
        cache.prefix().arithmetic_lineage, prefix_tokens.size()), "owned cache restores the recorded full arithmetic lineage");
    auto wrong_history = session.arithmetic_lineage;
    ++wrong_history.intervals.front().reduction_plan_identity;
    wrong_history.history_identity = strix::arithmetic_history_identity(wrong_history.intervals);
    expect(!cache.matches(prefix_tokens, session.execution_identity, session.runtime_compatibility_identity,
        session.state_layout_identity, &wrong_history), "full requested history mismatch is an owned-cache miss");
    const auto before_miss = session;
    expect(!cache.try_restore(session, model, layout, state_memory, transfer, prefix_tokens, &wrong_history) &&
        session.health == before_miss.health && session.next_position == before_miss.next_position,
        "try-restore turns an arithmetic-history mismatch into an ordinary cache miss without state transfer");
    bool wrong_history_rejected = false;
    try { cache.restore(session, model, layout, state_memory, transfer, prefix_tokens, &wrong_history); }
    catch (const std::invalid_argument&) { wrong_history_rejected = true; }
    expect(wrong_history_rejected && session.next_position == before_miss.next_position &&
        session.arithmetic_lineage.history_identity == before_miss.arithmetic_lineage.history_identity,
        "history miss fails before transfer or mutation of the destination session");
    auto unknown = session;
    unknown.arithmetic_lineage = {};
    bool unknown_capture_rejected = false;
    try { static_cast<void>(strix::SerialPrefixCacheEntry::capture(model, prefix_tokens, unknown,
        layout, state_memory, allocator, generations, transfer)); }
    catch (const std::invalid_argument&) { unknown_capture_rejected = true; }
    expect(unknown_capture_rejected, "position counters alone cannot create a reusable owned prefix cache");
    transfer.fail_restore = true;
    bool restore_failed = false;
    try { cache.restore(session, model, layout, state_memory, transfer, prefix_tokens); }
    catch (const std::runtime_error&) { restore_failed = true; }
    expect(restore_failed && session.health == strix::SerialSessionHealth::poisoned &&
        session.arithmetic_lineage.tainted && session.arithmetic_lineage.history_identity == 0,
        "failed prefix transfer poisons session and deletes reusable producer history");
    transfer.fail_restore = false;
    cache.restore(session, model, layout, state_memory, transfer, prefix_tokens);
    expect(strix::arithmetic_lineage_matches(session.arithmetic_lineage, cache.prefix().arithmetic_lineage,
        prefix_tokens.size()), "independently valid prefix restore recovers full producer history after failed transfer");

    auto wrong_prefix =
        prefix_tokens;
    ++wrong_prefix.back();
    bool wrong_prefix_rejected = false;
    try {
        cache.restore(
            session,
            model,
            layout,
            state_memory,
            transfer,
            wrong_prefix);
    } catch (const std::invalid_argument&) {
        wrong_prefix_rejected = true;
    }
    expect(
        wrong_prefix_rejected,
        "prefix cache rejects reuse for a different token sequence");

    auto foreign_session =
        session;
    ++foreign_session.runtime_compatibility_identity;
    bool foreign_runtime_rejected = false;
    try {
        cache.restore(
            foreign_session,
            model,
            layout,
            state_memory,
            transfer,
            prefix_tokens);
    } catch (const std::invalid_argument&) {
        foreign_runtime_rejected = true;
    }
    expect(
        foreign_runtime_rejected,
        "prefix cache rejects restore into a foreign runtime");

    // Actual snapshot allocations, including eviction while another request
    // holds a handle. Token variants isolate cache keys, not model arithmetic.
    auto cache_allocator = std::make_shared<VectorAllocator>();
    const std::weak_ptr<VectorAllocator> allocator_lifetime = cache_allocator;
    const auto capture_variant = [&](unsigned variant) {
        auto tokens = prefix_tokens;
        tokens.back() ^= variant;
        return strix::SerialPrefixCacheEntry::capture(model, tokens, session,
            layout, state_memory, *cache_allocator, generations, transfer);
    };
    auto first_entry = capture_variant(0);
    const auto snapshot_bytes = first_entry.memory().device_section_bytes + first_entry.memory().host_section_bytes;
    std::shared_ptr<const strix::SerialPrefixCacheEntry> surviving_handle;
    auto tokens_b = prefix_tokens; tokens_b.back() ^= 1U;
    auto tokens_c = prefix_tokens; tokens_c.back() ^= 2U;
    {
        strix::SerialPrefixCache pool(2, snapshot_bytes * 2U);
        expect(pool.insert(std::move(first_entry), cache_allocator) == strix::SerialPrefixCacheAdmission::inserted,
            "bounded cache admits the first complete snapshot");
        auto pinned = pool.find(prefix_tokens, session.execution_identity,
            session.runtime_compatibility_identity, session.state_layout_identity);
        expect(pinned != nullptr, "cache lookup pins a complete snapshot");
        expect(pool.insert(capture_variant(1), cache_allocator) == strix::SerialPrefixCacheAdmission::inserted &&
            pool.insert(capture_variant(2), cache_allocator) == strix::SerialPrefixCacheAdmission::inserted,
            "cache evicts an unpinned snapshot to admit another key");
        expect(!pool.find(tokens_b, session.execution_identity, session.runtime_compatibility_identity,
            session.state_layout_identity), "eviction removes the unpinned key");
        surviving_handle = pool.find(tokens_c, session.execution_identity,
            session.runtime_compatibility_identity, session.state_layout_identity);
        expect(surviving_handle && pool.size() == 2U && pool.snapshot_bytes() == snapshot_bytes * 2U,
            "cache accounts both retained physical snapshot sections");
        expect(pool.insert(capture_variant(3), cache_allocator) == strix::SerialPrefixCacheAdmission::pinned_capacity &&
            pool.size() == 2U && pool.evict_unpinned() == 0U,
            "capacity rejection preserves both in-flight snapshots");
        expect(pool.insert(capture_variant(0), cache_allocator) == strix::SerialPrefixCacheAdmission::already_present,
            "duplicate capture does not replace a pinned snapshot");
        pinned.reset();
        expect(pool.evict_unpinned() == 1U && pool.snapshot_bytes() == snapshot_bytes,
            "unpinning enables eviction and updates the byte budget");
        cache_allocator.reset();
        expect(!allocator_lifetime.expired(), "cache handles retain the actual snapshot allocator");
    }
    expect(!allocator_lifetime.expired() && surviving_handle,
        "a lookup handle retains its snapshot and allocator after cache destruction");
    if (surviving_handle) {
        surviving_handle->restore(session, model, layout, state_memory, transfer, tokens_c);
    }
    surviving_handle.reset();
    expect(allocator_lifetime.expired(), "snapshot buffers release before the final allocator owner");

    // Scalar frontier is attached to the same complete snapshot, not a
    // parallel cache keyed only by a hash or a token count.
    {
        auto retained_allocator=std::make_shared<VectorAllocator>();
        strix::ModelContentIdentity content;content.bytes[0]=7U;
        const auto original_session=session;
        auto short_session=session;short_session.next_position=prefix_tokens.size();short_session.committed_tokens=prefix_tokens.size();
        short_session.arithmetic_lineage={};
        for(std::uint64_t pos=0;pos<prefix_tokens.size();++pos)
            short_session.arithmetic_lineage=strix::prepare_serial_row_lineage(short_session.arithmetic_lineage,pos,session.execution_identity,0x1234U);
        const auto frontier=strix::SerialPrefixGreedyFrontier{content,model.vocab_size,103U};
        auto short_entry=strix::SerialPrefixCacheEntry::capture(model,prefix_tokens,short_session,layout,state_memory,
            *retained_allocator,generations,transfer,64U,frontier);
        const auto bytes=short_entry.memory().device_section_bytes+short_entry.memory().host_section_bytes;
        strix::SerialPrefixCache pool(2U,bytes*2U);
        expect(pool.insert(std::move(short_entry),retained_allocator)==strix::SerialPrefixCacheAdmission::inserted,"greedy cache retains the original snapshot allocation");
        auto longer=prefix_tokens;longer.push_back(104U);auto longer_session=short_session;
        longer_session.next_position=longer.size();longer_session.committed_tokens=longer.size();
        longer_session.arithmetic_lineage=strix::prepare_serial_row_lineage(short_session.arithmetic_lineage,
            prefix_tokens.size(),session.execution_identity,0x1234U);
        auto long_entry=strix::SerialPrefixCacheEntry::capture(model,longer,longer_session,layout,state_memory,
            *retained_allocator,generations,transfer,64U,strix::SerialPrefixGreedyFrontier{content,model.vocab_size,105U});
        expect(pool.insert(std::move(long_entry),retained_allocator)==strix::SerialPrefixCacheAdmission::inserted,"longer complete greedy frontier admitted");
        auto request=longer;request.push_back(106U);
        auto hit=pool.find_longest_greedy(request,content,model.vocab_size,session.execution_identity,
            session.runtime_compatibility_identity,session.state_layout_identity);
        expect(hit && hit->prefix().prefix_tokens==longer && hit->greedy_frontier()->pending_token==105U,
            "longest exact token prefix retains its unconsumed finite scalar frontier");
        auto different_model=content;different_model.bytes[1]=1U;
        expect(!pool.find_longest_greedy(request,different_model,model.vocab_size,session.execution_identity,
            session.runtime_compatibility_identity,session.state_layout_identity),"greedy lookup rejects a different model SHA");
        expect(!pool.find_longest_greedy(request,content,model.vocab_size-1U,session.execution_identity,
            session.runtime_compatibility_identity,session.state_layout_identity),"greedy lookup rejects a different vocabulary");
        expect(!pool.find_longest_greedy(request,content,model.vocab_size,session.execution_identity,
            session.runtime_compatibility_identity+1U,session.state_layout_identity),"greedy lookup rejects a different runtime compatibility identity");
        auto different_tokens=request;different_tokens[0]^=1U;
        expect(!pool.find_longest_greedy(different_tokens,content,model.vocab_size,session.execution_identity,
            session.runtime_compatibility_identity,session.state_layout_identity),"greedy lookup compares actual tokens, not only prefix lengths");
        auto pinned_short=pool.find_longest_greedy(prefix_tokens,content,model.vocab_size,session.execution_identity,
            session.runtime_compatibility_identity,session.state_layout_identity);
        auto other_tokens=prefix_tokens;other_tokens[0]^=2U;
        auto other=strix::SerialPrefixCacheEntry::capture(model,other_tokens,short_session,layout,state_memory,
            *retained_allocator,generations,transfer,64U,frontier);
        expect(pool.insert(std::move(other),retained_allocator)==strix::SerialPrefixCacheAdmission::pinned_capacity,
            "in-flight greedy restores pin both entries against eviction");
        bool invalid=false;
        try {auto bad=strix::SerialPrefixCacheEntry::capture(model,prefix_tokens,short_session,layout,state_memory,
            *retained_allocator,generations,transfer,64U,strix::SerialPrefixGreedyFrontier{content,model.vocab_size,static_cast<std::uint32_t>(model.vocab_size)});}
        catch(const std::invalid_argument&) {invalid=true;}
        expect(invalid,"capture rejects an out-of-vocabulary scalar frontier before any snapshot copy");
        {
            auto chunk_session=short_session;
            chunk_session.arithmetic_lineage=strix::prepare_prefill_chunk_lineage({},0U,prefix_tokens.size(),
                session.execution_identity,0x1234U,session.execution_identity);
            auto chunk=strix::SerialPrefixCacheEntry::capture(model,prefix_tokens,chunk_session,layout,state_memory,
                *retained_allocator,generations,transfer,64U,frontier);
            strix::SerialPrefixCache general(1U,bytes);
            expect(general.insert(std::move(chunk),retained_allocator)==strix::SerialPrefixCacheAdmission::inserted,
                "general cache preserves explicitly recorded chunk-producer snapshots");
            expect(general.find(prefix_tokens,session.execution_identity,session.runtime_compatibility_identity,
                session.state_layout_identity,&chunk_session.arithmetic_lineage)!=nullptr,
                "explicit chunk history still matches the general cache contract");
            expect(!general.find_longest_greedy(prefix_tokens,content,model.vocab_size,session.execution_identity,
                session.runtime_compatibility_identity,session.state_layout_identity),
                "greedy serving must miss a nonserial producer even when its numerical execution ID matches");
        }
        session=original_session;
    }

    if (failures == 0) {
        std::cout
            << "serial prefix cache tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
