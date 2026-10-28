#include "strix/prefix_state_cache.hpp"
#include "strix/model_content_hash.hpp"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <atomic>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

strix::PrefixCacheBuffer heap(std::uint64_t bytes) {
    std::shared_ptr<std::byte> data(new std::byte[static_cast<std::size_t>(bytes ? bytes : 1)],
        std::default_delete<std::byte[]>());
    return {std::move(data), bytes};
}

strix::ModelContentIdentity identity_of(const char* text) {
    return strix::sha256_bytes(std::as_bytes(std::span<const char>(text, std::strlen(text))));
}

constexpr std::uint32_t T = 4;
constexpr std::uint64_t segment_bytes = 64, fixed_bytes = 256;

// Payload bytes are a function of (block or position, salt) so restores can be checked.
std::byte pattern(std::uint64_t a, std::uint64_t i) { return static_cast<std::byte>((a * 131U + i * 7U) & 255U); }

// Builds the checkpoints of `tokens` at every block boundary, sharing segments like the engine does.
std::vector<std::shared_ptr<const strix::PrefixCacheCheckpoint>> build(strix::PrefixStateCache& cache,
    const strix::ModelContentIdentity& identity, const std::vector<std::uint32_t>& tokens) {
    std::vector<std::shared_ptr<const strix::PrefixCacheCheckpoint>> out;
    std::vector<std::shared_ptr<const strix::PrefixCacheSegment>> segments;
    auto previous = identity;
    for (std::uint64_t k = 0; k < tokens.size() / T; ++k) {
        const std::span<const std::uint32_t> block(tokens.data() + k * T, T);
        auto segment = std::make_shared<strix::PrefixCacheSegment>();
        segment->identity = identity;
        segment->parent = previous;
        segment->digest = strix::prefix_cache_block_digest(previous, block);
        segment->block_index = k;
        segment->block_rows = T;
        segment->tokens.assign(block.begin(), block.end());
        segment->payload = cache.allocate(segment_bytes);
        for (std::uint64_t i = 0; i < segment_bytes; ++i) segment->payload.data.get()[i] = pattern(k + tokens[k * T], i);
        previous = segment->digest;
        segments.push_back(segment);
        auto checkpoint = std::make_shared<strix::PrefixCacheCheckpoint>();
        checkpoint->identity = identity;
        checkpoint->digest = previous;
        checkpoint->block_rows = T;
        checkpoint->position = (k + 1) * T;
        checkpoint->segments = segments;
        checkpoint->fixed = cache.allocate(fixed_bytes);
        for (std::uint64_t i = 0; i < fixed_bytes; ++i) checkpoint->fixed.data.get()[i] = pattern(1000 + checkpoint->position, i);
        checkpoint->metadata.assign(3, static_cast<std::byte>(k));
        out.push_back(checkpoint);
    }
    return out;
}

bool same_payloads(const strix::PrefixCacheCheckpoint& a, const strix::PrefixCacheCheckpoint& b) {
    if (a.position != b.position || a.segments.size() != b.segments.size() || a.fixed.bytes != b.fixed.bytes ||
        a.metadata != b.metadata || std::memcmp(a.fixed.data.get(), b.fixed.data.get(), a.fixed.bytes) != 0)
        return false;
    for (std::size_t k = 0; k < a.segments.size(); ++k)
        if (a.segments[k]->tokens != b.segments[k]->tokens || a.segments[k]->payload.bytes != b.segments[k]->payload.bytes ||
            std::memcmp(a.segments[k]->payload.data.get(), b.segments[k]->payload.data.get(), a.segments[k]->payload.bytes) != 0)
            return false;
    return true;
}
}  // namespace

void prefix_dma_tests();

int main() {
    const auto dir = std::filesystem::temp_directory_path() /
        ("strix-prefix-cache-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        const auto id = identity_of("identity A"), other = identity_of("identity B");
        std::vector<std::uint32_t> tokens;
        for (std::uint32_t i = 0; i < 18; ++i) tokens.push_back(100 + i);  // 4 full blocks + 2

        // Digests chain and stop at full blocks.
        const auto digests = strix::prefix_cache_block_digests(id, tokens, T, 100);
        require(digests.size() == 4, "digest count");
        require(digests[1] == strix::prefix_cache_block_digest(digests[0], std::span(tokens).subspan(4, 4)), "digest chain");
        require(strix::prefix_cache_block_digests(other, tokens, T, 100)[0] != digests[0], "identity changes digests");

        {
            strix::PrefixStateCache cache({1ULL << 20, dir, 0}, heap);
            const auto checkpoints = build(cache, id, tokens);
            require(cache.find_longest(id, tokens, T, 17) == nullptr, "empty cache misses");
            for (const auto& checkpoint : checkpoints) require(cache.insert(checkpoint), "insert");
            // Shared segments are counted once: 4 segments + 4 fixed blobs + metadata.
            require(cache.stats().host_bytes == 4 * segment_bytes + 4 * (fixed_bytes + 3), "shared segment accounting");
            auto hit = cache.find_longest(id, tokens, T, 17);
            require(hit && hit->position == 16 && same_payloads(*hit, *checkpoints[3]), "longest hit");
            hit = cache.find_longest(id, tokens, T, 15);
            require(hit && hit->position == 12, "limit bounds the hit");
            // A different token in block 3 hits block 2 only.
            auto changed = tokens;
            changed[9] = 7;
            hit = cache.find_longest(id, changed, T, 17);
            require(hit && hit->position == 8 && same_payloads(*hit, *checkpoints[1]), "diverging tokens hit the common prefix");
            // A different identity never hits.
            require(cache.find_longest(other, tokens, T, 17) == nullptr, "identity mismatch misses");
            // A different block size never hits (different digests and positions).
            require(cache.find_longest(id, tokens, 8, 17) == nullptr, "block size mismatch misses");
            cache.flush();
            require(cache.stats().disk_write_failures == 0 && cache.stats().disk_writes == 4, "disk writes");
        }
        {
            // A fresh store finds the checkpoints on disk, byte-identical, and loads them into the host tier.
            strix::PrefixStateCache cache({1ULL << 20, dir, 0}, heap);
            strix::PrefixStateCache reference({1ULL << 20, {}, 0}, heap);
            const auto expected = build(reference, id, tokens);
            auto hit = cache.find_longest(id, tokens, T, 17);
            require(hit && hit->position == 16 && same_payloads(*hit, *expected[3]), "disk hit restores payloads");
            require(cache.stats().disk_hits == 1 && cache.stats().host_checkpoints == 1, "disk hit is retained in host tier");
            hit = cache.find_longest(id, tokens, T, 17);
            require(cache.stats().host_hits == 1, "second lookup is a host hit");
            require(cache.find_longest(other, tokens, T, 17) == nullptr, "disk identity mismatch misses");
        }
        {
            strix::PrefixStateCache cache({1ULL<<20,dir,0},heap);
            auto hit=cache.find_longest(id,tokens,T,17,false);
            require(hit && hit->position==16,"no-store disk lookup allowed");
            const auto after=cache.counters();
            require(after.host_bytes==fixed_bytes+4*segment_bytes && after.host_segments==4 && after.host_checkpoints==0 &&
                after.inserted==0 && after.disk_writes==0,"no-store disk hit does not promote");
            require(after.lookup_tokens==17 && after.hit_tokens==16 && after.restored_tokens==16,"token counters");
            cache.record_no_store_skip();
            require(cache.counters().no_store_skips==1 && cache.counters().captures==0,"suppressed capture counters");
        }
        {
            // A corrupted fixed blob on disk is rejected (counted) and the next shorter prefix is used.
            const auto digests16 = strix::prefix_cache_block_digests(id, tokens, T, 4);
            const auto path = dir / ("ckpt-" + strix::model_content_identity_hex(digests16[3]) + ".tmpc");
            {
                std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
                file.seekp(-1, std::ios::end);
                file.put('x');
            }
            strix::PrefixStateCache cache({1ULL << 20, dir, 0}, heap);
            const auto hit = cache.find_longest(id, tokens, T, 17);
            require(hit && hit->position == 12 && cache.stats().disk_read_failures == 1, "corrupt checkpoint is skipped");
        }
        {
            // Startup sweep: old temporaries and unreferenced segments go, referenced segments stay.
            const auto stale = std::filesystem::file_time_type::clock::now() - std::chrono::hours(1);
            const auto orphan = dir / ("seg-" + std::string(64, 'a') + ".tmps");
            const auto partial = dir / "ckpt-partial.tmpc.tmp.1";
            { std::ofstream(orphan) << "x"; std::ofstream(partial) << "x"; }
            std::filesystem::last_write_time(orphan, stale);
            std::filesystem::last_write_time(partial, stale);
            std::size_t before = 0;
            for (const auto& entry : std::filesystem::directory_iterator(dir)) before += entry.path().extension() == ".tmps";
            strix::PrefixStateCache cache({1ULL << 20, dir, 0}, heap);
            std::size_t after = 0;
            for (const auto& entry : std::filesystem::directory_iterator(dir)) after += entry.path().extension() == ".tmps";
            require(!std::filesystem::exists(orphan) && !std::filesystem::exists(partial) && after + 1 == before,
                "sweep removes only stale orphans");
            require(cache.find_longest(id, tokens, T, 13) != nullptr, "referenced segments survive the sweep");
        }
        {
            // LRU within the byte budget: room for two checkpoints of one prompt (shared segments).
            const std::uint64_t budget = 3 * segment_bytes + 2 * (fixed_bytes + 3);
            strix::PrefixStateCache cache({budget, {}, 0, strix::PrefixCacheRetention::lru}, heap);
            auto checkpoints = build(cache, id, tokens);
            require(cache.insert(checkpoints[0]) && cache.insert(checkpoints[1]), "insert within budget");
            checkpoints[0].reset(); // Release the fixed-state lease before expecting reclaimed capacity.
            require(cache.find_longest(id, tokens, T, 9) != nullptr, "touch block 2");
            require(cache.insert(checkpoints[2]), "insert evicts");
            const auto stats = cache.stats();
            require(stats.evicted == 1 && stats.host_bytes <= budget, "evicted least recently used");
            // Block 1 was evicted, block 2 and 3 are retained.
            require(cache.find_longest(id, tokens, T, 4) == nullptr, "evicted entry misses");
            require(cache.find_longest(id, tokens, T, 12)->position == 12, "newest entry hits");
            // A checkpoint larger than the whole budget is rejected.
            strix::PrefixStateCache tiny({fixed_bytes, {}, 0}, heap);
            require(!tiny.insert(checkpoints[3]) && tiny.stats().rejected_over_budget == 1, "over budget rejected");
        }
        {
            // A lone conversation borrows; the first competing conversation reclaims that borrowing.
            strix::PrefixStateCache cache({6*(segment_bytes+fixed_bytes+3),{},0},heap);
            std::vector<strix::ModelContentIdentity> noisy;
            for (std::uint32_t n=0;n<6;++n) {
                auto c=std::make_shared<strix::PrefixCacheCheckpoint>(*build(cache,id,{1,2,3,n+10})[0]);
                c->conversation="noisy"; noisy.push_back(c->digest);
                require(cache.insert(c),"lone conversation borrows");
            }
            require(cache.counters().host_checkpoints==6 && cache.counters().evicted==0,"lone conversation fills budget without eviction");
            auto quiet=std::make_shared<strix::PrefixCacheCheckpoint>(*build(cache,id,{8,9,10,11})[0]);
            quiet->conversation="quiet";
            require(cache.insert(quiet),"new conversation admitted");
            require(cache.contains(quiet->digest) && cache.counters().evictions_conversation_cap==1 && !cache.contains(noisy.front()),"borrowed bytes reclaimed first");
            require(cache.counters().host_bytes==cache.stats().host_bytes,"cheap gauges reconcile without leases");
        }
        {
            // Different conversations share a physical trunk without sharing their fixed state.
            strix::PrefixStateCache cache({1024,{},0},heap);
            auto chain=build(cache,id,{1,2,3,4,5,6,7,8});
            auto a=std::make_shared<strix::PrefixCacheCheckpoint>(*chain[0]); a->conversation="A";
            auto b=std::make_shared<strix::PrefixCacheCheckpoint>(*chain[1]); b->conversation="B";
            require(cache.insert(a) && cache.insert(b),"shared trunk insert");
            const auto expected=2*segment_bytes+2*(fixed_bytes+3);
            require(cache.counters().host_bytes==expected && cache.counters().host_segments==2,"unique payload accounting");
            auto lease=cache.find_longest(id,std::vector<std::uint32_t>{1,2,3,4,5,6,7,8},T,8);
            auto c=build(cache,other,{9,10,11,12})[0]; require(cache.insert(c),"pressure insert");
            require(lease && same_payloads(*lease,*b),"eviction preserves leased rows and shared trunk");
            chain.clear(); a.reset(); b.reset(); c.reset(); lease.reset();
            require(cache.counters().host_bytes==cache.stats().host_bytes,"released leases conserve gauges");
        }
        {
            // Many divergent branches cannot displace a quiet conversation or free its trunk.
            strix::PrefixStateCache cache({2048,{},0},heap);
            auto root=build(cache,id,{1,2,3,4})[0]->segments[0];
            auto quiet=std::make_shared<strix::PrefixCacheCheckpoint>(*build(cache,id,{1,2,3,4,70,71,72,73})[1]);
            quiet->segments[0]=root; quiet->conversation="quiet";
            require(cache.insert(quiet),"quiet shared-prefix insert");
            const auto quiet_digest=quiet->digest;
            quiet.reset(); // Policy, rather than a checkpoint lease, must protect this conversation.
            for (std::uint32_t branch=0;branch<12;++branch) {
                auto chain=build(cache,id,{1,2,3,4,branch+100,20,21,22});
                auto c=std::make_shared<strix::PrefixCacheCheckpoint>(*chain[1]);
                c->segments[0]=root; c->conversation="branches";
                (void)cache.insert(c);
            }
            require(cache.contains(quiet_digest),"noisy branches preserve quiet conversation");
            require(cache.find_segment(root->digest)==root && root->payload.data.get()[0]==pattern(1,0),"shared leading prefix survives branch eviction");
            require(cache.counters().evictions_conversation_cap>0,"branch conversation byte cap enforced");
        }
        {
            // Retained content can belong to multiple named conversations without duplication.
            strix::PrefixStateCache cache({4096,{},0},heap);
            auto c=build(cache,id,{1,2,3,4})[0];
            require(cache.insert(c),"membership seed");
            const auto before=cache.counters();
            require(cache.find_longest(id,std::vector<std::uint32_t>{1,2,3,4},T,4,true,"named")==c,"explicit conversation lookup");
            const auto after=cache.counters();
            require(before.host_bytes==after.host_bytes && before.host_segments==after.host_segments && after.captures==1,"membership is not a second payload or capture");
        }
        {
            // One complete deep checkpoint remains useful even above B/3.
            strix::PrefixStateCache cache({800,{},0},heap);
            auto deep=build(cache,id,tokens).back();
            require(cache.insert(deep),"one checkpoint exception");
            require(cache.counters().host_checkpoints==1,"one checkpoint retained");
        }
        {
            // Spare capacity is useful even when multiple conversations exceed nominal caps.
            strix::PrefixStateCache cache({4096,{},0},heap);
            for (std::uint32_t n=0;n<8;++n) {
                auto c=std::make_shared<strix::PrefixCacheCheckpoint>(*build(cache,id,{1,2,3,n+10})[0]);
                c->conversation=n<7 ? "busy" : "quiet";
                require(cache.insert(c),"spare-budget insert");
            }
            require(cache.counters().host_checkpoints==8 && cache.counters().evicted==0,
                "fair caps only order pressure victims");
        }
        {
            // Pre-allocation admission uses the arriving conversation to order pressure victims.
            strix::PrefixStateCache cache({6*(segment_bytes+fixed_bytes+3),{},0},heap);
            std::vector<strix::ModelContentIdentity> keys;
            for (std::uint32_t n=0;n<6;++n) {
                auto c=std::make_shared<strix::PrefixCacheCheckpoint>(*build(cache,id,{1,2,3,n+10})[0]);
                c->conversation="busy"; keys.push_back(c->digest);
                require(cache.insert(c),"capture preflight fill");
            }
            auto next=std::make_shared<strix::PrefixCacheCheckpoint>(*build(cache,id,{8,9,10,11})[0]);
            next->conversation="quiet";
            require(cache.admit_capture(*next),"capture pressure reservation");
            require(!cache.contains(keys.front()) && cache.counters().evictions_conversation_cap==1,
                "capture admission reclaims borrowed bytes first");
            require(cache.insert(next),"capture inserted after reservation");
        }
        for (const auto retention : {strix::PrefixCacheRetention::lru, strix::PrefixCacheRetention::fair}) {
            strix::PrefixStateCache cache({1ULL<<20,{},0,retention},heap);
            std::vector<strix::ModelContentIdentity> keys;
            for (std::uint32_t n=0;n<260;++n) {
                auto c=build(cache,id,{1,2,3,n+10})[0]; keys.push_back(c->digest);
                require(cache.insert(c),"directory pressure evicts rather than refuses");
            }
            const auto stats=cache.counters();
            require(stats.host_checkpoints==(retention==strix::PrefixCacheRetention::lru ? 260U : 256U),
                "directory cap applies only to fair");
            require(stats.evicted==(retention==strix::PrefixCacheRetention::lru ? 0U : 4U),"entry cap eviction count");
            require(retention==strix::PrefixCacheRetention::lru || !cache.contains(keys.front()),"fair tie uses oldest victim");
        }
        {
            strix::PrefixStateCache cache({1ULL<<20,{},0},heap);
            std::vector<std::shared_ptr<const strix::PrefixCacheCheckpoint>> leases;
            for (std::uint32_t n=0;n<256;++n) {
                leases.push_back(build(cache,id,{1,2,3,n+10})[0]);
                require(cache.insert(leases.back()),"leased directory fill");
            }
            require(!cache.insert(build(cache,id,{1,2,3,999})[0]),"all leased directory refuses");
            require(cache.counters().host_checkpoints==256 && cache.counters().evicted==0,"leased entries survive cap refusal");
        }
        {
            unsigned allocations=0;
            strix::PrefixStateCache cache({100,{},0},[&](std::uint64_t) -> strix::PrefixCacheBuffer {
                ++allocations; throw std::bad_alloc();
            });
            strix::PrefixCacheCheckpoint estimate; estimate.fixed.bytes=101;
            require(!cache.admit_capture(estimate) && allocations==0 && cache.counters().budget_skips==1,
                "impossible checkpoint skips before allocator");
            estimate.fixed.bytes=100;
            require(cache.admit_capture(estimate),"fitting capture estimate admitted");
            bool failed=false;
            try { (void)cache.allocate(100); }
            catch (const strix::PrefixCacheAllocationFailure&) { cache.record_capture_failure(); failed=true; }
            require(failed && allocations==1 && cache.counters().capture_failures==1 && cache.counters().host_bytes==0,
                "allocator throws typed capture failure without publishing partial state");
            strix::PrefixStateCache invalid({100,{},0},[](std::uint64_t) { return strix::PrefixCacheBuffer{}; });
            failed=false;
            try { (void)invalid.allocate(100); } catch (const strix::PrefixCacheAllocationFailure&) { failed=true; }
            require(failed && invalid.counters().host_bytes==0,"empty allocator result has same failure seam");
            strix::PrefixStateCache fatal({100,{},0},[](std::uint64_t) -> strix::PrefixCacheBuffer {
                throw strix::PrefixCacheFatalError("device broken");
            });
            failed=false;
            try { (void)fatal.allocate(100); } catch (const strix::PrefixCacheFatalError&) { failed=true; }
            require(failed,"fatal allocator errors preserve propagation");

        }
        {
            const auto tiny=[&](strix::PrefixStateCache& cache, unsigned token, bool allocate) {
                auto seg=std::make_shared<strix::PrefixCacheSegment>();
                seg->identity=id; seg->parent=id; seg->tokens={token}; seg->block_rows=1;
                seg->digest=strix::prefix_cache_block_digest(id,seg->tokens);
                seg->payload=allocate ? cache.allocate(10) : strix::PrefixCacheBuffer{{},10};
                auto c=std::make_shared<strix::PrefixCacheCheckpoint>();
                c->identity=id; c->digest=seg->digest; c->block_rows=1; c->position=1;
                c->segments={seg}; c->fixed=allocate ? cache.allocate(60) : strix::PrefixCacheBuffer{{},60};
                return c;
            };
            const auto diskdir=dir/"lease-pressure";
            { strix::PrefixStateCache writer({100,diskdir,0},heap);
              require(writer.insert(tiny(writer,1,true)),"disk seed"); writer.flush(); }
            for (const bool promote : {false,true}) {
                std::uint64_t live=0, peak=0;
                strix::PrefixStateCache reader({100,diskdir,0},[&](std::uint64_t n) {
                    live+=n; peak=std::max(peak,live);
                    return strix::PrefixCacheBuffer{std::shared_ptr<std::byte>(new std::byte[n?n:1], [&,n](std::byte* p) { live-=n; delete[] p; }), n};
                });
                auto resident=tiny(reader,2,true);
                require(reader.insert(resident),"resident seed"); reader.flush();
                require(!reader.find_longest(id,std::vector<std::uint32_t>{1},1,1,promote),"disk read must reserve before allocating");
                require(live==70 && peak==70 && reader.counters().host_bytes==70 && reader.counters().budget_skips==1,"disk refusal physical accounting");
            }
            {
                strix::PrefixStateCache reader({100,diskdir,0},heap);
                auto transient=reader.find_longest(id,std::vector<std::uint32_t>{1},1,1,false);
                require(transient && reader.counters().host_checkpoints==0 && reader.counters().host_bytes==70,"unpromoted read charged");
                auto credits=reader.reserve(20);
                require(bool(credits) && !reader.reserve(11),"outstanding reservation charged");
                auto payload=reader.allocate(20,credits);
                require(reader.counters().host_bytes==90 && !reader.reserve(11),"conversion does not double count or lose charge");
                payload={}; transient.reset();
                require(reader.counters().host_bytes==0 && bool(reader.reserve(100)),"transient last lease releases charge");
            }
            {
                const auto shorter_dir=dir/"shorter-reservation";
                { strix::PrefixStateCache writer({1000,shorter_dir,0},heap);
                  for (const auto& c : build(writer,id,{1,2,3,4,5,6,7,8}))
                      require(writer.insert(c),"shorter seed");
                  writer.flush(); }
                strix::PrefixStateCache reader({320,shorter_dir,0},heap);
                auto shorter=reader.find_longest(id,std::vector<std::uint32_t>{1,2,3,4,5,6,7,8},T,8,false);
                require(shorter && shorter->position==4 && reader.counters().budget_skips==1 &&
                    reader.counters().host_bytes==320,"oversized disk candidate falls back to shorter reserved payload");
            }
            strix::PrefixStateCache disk({100,diskdir,0},heap);
            auto held=disk.find_longest(id,std::vector<std::uint32_t>{1},1,1);
            require(held && disk.counters().host_bytes==70,"disk promotion physical charge");
            require(!disk.admit_capture(*tiny(disk,2,false)),"disk hit lease blocks incoming reservation");
            require(!disk.insert(tiny(disk,2,true)),"disk hit lease blocks insertion");
            require(disk.contains(held->digest) && disk.counters().host_bytes==70,"disk hit protects directory and fixed payload");
            disk.flush();
            // A separate checkpoint alias still owns registered fixed/segment payloads
            // after its directory object is evicted. Admission must use the ledger.
            strix::PrefixStateCache aliases({100,{},0},heap);
            auto reader=tiny(aliases,3,true);
            auto directory=std::make_shared<strix::PrefixCacheCheckpoint>(*reader);
            require(aliases.insert(directory),"alias seed"); directory.reset();
            require(!aliases.admit_capture(*tiny(aliases,4,false)),"fixed alias remains charged after eviction");
            require(aliases.counters().host_checkpoints==0 && aliases.counters().host_bytes==70,"evicted fixed lease ledger");
            require(!aliases.insert(tiny(aliases,4,true)),"ledger rejects second retained payload");
            reader.reset();
            require(aliases.counters().host_bytes==0,"last fixed lease releases physical charge");
            strix::PrefixStateCache legacy({100,{},0,strix::PrefixCacheRetention::lru,true},heap);
            auto old_reader=tiny(legacy,5,true);
            require(legacy.insert(old_reader) && legacy.insert(tiny(legacy,6,true)),"baseline keeps old LRU admission with reader");
            const auto old_stats=legacy.counters();
            require(old_stats.evicted==1 && old_stats.inserted==2 && old_stats.host_bytes==80,"baseline legacy gauge and eviction projection");
        }
        {
            // A segment's bytes and count must be observed as one ledger update, even
            // when the final payload owner releases outside the store mutex.
            strix::PrefixStateCache cache({1024,{},0,strix::PrefixCacheRetention::lru},heap);
            for (unsigned n=0;n<200;++n) {
                auto c=std::make_shared<strix::PrefixCacheCheckpoint>();
                auto seg=std::make_shared<strix::PrefixCacheSegment>();
                seg->identity=id; seg->parent=id; seg->tokens={n}; seg->block_rows=1;
                seg->digest=strix::prefix_cache_block_digest(id,seg->tokens); seg->payload=cache.allocate(64);
                c->identity=id; c->digest=seg->digest; c->block_rows=1; c->position=1; c->segments={seg};
                require(cache.insert(c),"concurrent ledger seed");
                seg.reset();
                // Force directory removal while the checkpoint remains leased.
                strix::PrefixCacheCheckpoint estimate; estimate.fixed.bytes=1024;
                require(!cache.admit_capture(estimate),"lease remains charged after LRU removal");
                std::thread release([lease=std::move(c)]() mutable { lease.reset(); });
                bool coherent=true;
                for(unsigned k=0;k<20;++k) { const auto s=cache.counters();
                    coherent=coherent && s.host_bytes==64*s.host_segments; }
                release.join();
                require(coherent,"coherent ledger gauges");
                require(cache.counters().host_bytes==0,"released ledger reconciles");
            }
        }
        {
            strix::PrefixStateCache legacy({100,{},0,strix::PrefixCacheRetention::lru,true},[](std::uint64_t)->strix::PrefixCacheBuffer {
                throw std::runtime_error("prefix state cache pinned host allocation failed");
            });
            bool old=false; try { (void)legacy.allocate(1); }
            catch(const strix::PrefixCacheAllocationFailure&) { require(false,"baseline reclassified allocator error"); }
            catch(const std::runtime_error& e) {old=std::string(e.what())=="prefix state cache pinned host allocation failed";}
            require(old && legacy.counters().capture_failures==0,"legacy error text and counters");
            strix::PrefixStateCache arbitrary({100,{},0},[](std::uint64_t)->strix::PrefixCacheBuffer { throw std::logic_error("allocator bug"); });
            old=false; try { (void)arbitrary.allocate(1); } catch(const std::logic_error&) {old=true;}
            require(old,"non-resource allocator failures propagate");
        }
        {
            strix::PrefixStateCache named({100,{},0,strix::PrefixCacheRetention::lru},heap);
            require(named.park_session("delete",heap(60),2) && named.delete_session("delete"),"explicit deletion");
            const auto stats=named.counters();
            require(stats.evicted==0 && stats.evictions_explicit==1,"explicit reason does not change legacy evicted");
        }
        {
            // Candidate validation owns a physical lease and runs outside the store mutex.
            // Refusal neither promotes/touches nor counts a hit; continue to shorter prefixes.
            strix::PrefixStateCache cache({1U<<20,dir/"certified",0},heap);
            std::vector<std::uint32_t> candidate_tokens(21,7);
            auto checkpoints=build(cache,id,candidate_tokens);
            for(const auto& cp:checkpoints) {
                auto c=std::make_shared<strix::PrefixCacheCheckpoint>(*cp);
                c->producer_certificate={c->position==16?1U:2U};
                require(cache.insert(c),"certified seed");
            }
            const auto eligible=[&](const strix::PrefixCacheCheckpoint& c) {
                (void)cache.counters(); // Must not deadlock under a store lock.
                return c.producer_certificate==std::vector<std::uint64_t>{1};
            };
            auto hit=cache.find_longest(id,candidate_tokens,T,20,false,{},eligible);
            require(hit && hit->position==16,"incompatible longest falls back to compatible shorter");
            auto stats=cache.counters();
            require(stats.plan_misses==1 && stats.host_hits==1 && stats.restored_tokens==16,"accepted-only accounting");
            hit=cache.find_longest(id,candidate_tokens,T,20,false,{},[](const auto&) {return false;});
            stats=cache.counters();
            require(!hit && stats.plan_misses==6 && stats.host_hits==1 && stats.misses==1 && stats.restored_tokens==16,
                "all refused is a miss, never a false restore");
            cache.flush();
            require(cache.counters().disk_writes==0,"certified checkpoints never enter v1 disk writer");
        }
        {
            const auto tier=dir/"legacy-v1-refusal";
            const std::vector<std::uint32_t> disk_tokens(9,3);
            {
                strix::PrefixStateCache writer({1U<<20,tier,0},heap);
                for(const auto& c:build(writer,id,disk_tokens)) require(writer.insert(c),"legacy disk seed");
                writer.flush();
            }
            strix::PrefixStateCache reader({1U<<20,tier,0},heap);
            const auto refused=reader.find_longest(id,disk_tokens,T,8,false,{},[](const auto&) {return true;});
            require(!refused && reader.counters().disk_reads==0 && reader.counters().disk_hits==0,
                "certified candidate API cannot consume old v1 disk files");
            const auto legacy_hit=reader.find_longest(id,disk_tokens,T,8);
            require(legacy_hit && reader.counters().disk_hits==1,"explicit legacy disk API remains functional");
        }
        std::filesystem::remove_all(dir);
        prefix_dma_tests();
        std::cout << "prefix state cache tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove_all(dir);
        std::cerr << "prefix state cache tests failed: " << error.what() << '\n';
        return 1;
    }
}
