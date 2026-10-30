#include "strix/ple_direct.hpp"
#include "strix/ple_rolling.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>
#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

using namespace strix::ple_reclaim;
namespace {
void require(bool b, const char* why) { if (!b) throw std::runtime_error(why); }
template<class F> void refuses(F f) {
    bool caught = false;
    try { f(); } catch (const std::exception&) { caught = true; }
    require(caught, "expected clean refusal");
}
constexpr std::uint64_t page = 4096;
std::byte value(std::uint64_t offset) { return static_cast<std::byte>((offset * 17U + offset / page) % 251U); }
DirectPageCache::Read synthetic() {
    return [](int, void* out, std::size_t size, std::uint64_t offset) {
        require(reinterpret_cast<std::uintptr_t>(out) % page == 0 && offset % page == 0 && size % page == 0 &&
            size <= (128U << 10U), "all direct reads are aligned and coalesced runs are bounded");
        for (std::size_t i = 0; i < size; ++i) static_cast<std::byte*>(out)[i] = value(offset + i);
        return static_cast<std::int64_t>(size);
    };
}
void check_row(DirectPageCache& cache, std::uint64_t offset) {
    std::array<std::byte, 90> row{};
    cache.copy(offset, row);
    for (std::size_t i = 0; i < row.size(); ++i) require(row[i] == value(offset+i), "straddling row byte identity");
}
void setup_prime() {
    constexpr std::uint64_t cap = 16U * page;
    auto cache = std::make_shared<DirectPageCache>(-1, 128U * page, page, cap + 2U * page, synthetic());
    std::promise<void> entered, release;
    std::atomic<bool> first{true};
    auto released = release.get_future().share();
    RollingPages::Io io;
    io.reclaimable = [](auto) { return true; };
    io.resident = [] { return std::uint64_t{0}; };
    io.evict = [cache](auto p) { cache->erase(p); };
    io.prefetch = [&, cache](auto p) {
        if (first.exchange(false)) { entered.set_value(); released.wait(); }
        return cache->ensure(p, true);
    };
    io.prepare = [cache](auto p) { cache->ensure(p, false); };
    io.owned = cache;
    std::vector<std::uint32_t> rows(1024);
    for (std::size_t i = 0; i < rows.size(); ++i) rows[i] = static_cast<std::uint32_t>(i % 128U);
    RollingPipeline pipeline(256, page, 0, page, 1, rows, cap + 2U * page, 2U * page, cap, 512, io);
    require(entered.get_future().wait_for(std::chrono::seconds(10)) == std::future_status::ready,
        "setup primes direct I/O before the first demand/gather");
    auto cancelled = std::async(std::launch::async, [&] { pipeline.finish(); });
    const bool drains = cancelled.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout;
    release.set_value(); cancelled.get();
    require(drains && cache->stats().peak_used_bytes <= cap && pipeline.stats().peak_owned_bytes <= cap &&
        cache->stats().used_bytes == 0, "cancelled setup prime drains in-flight I/O and retires every capped page");
    refuses([&] { pipeline.gather(0, 1, [] {}); });
}
void logic_and_faults() {
    require(!DirectReadScope::background(), "foreground source wait by default");
    {
        const DirectReadScope hidden(true);
        require(DirectReadScope::background(), "c1 source wait classified as background");
        { const DirectReadScope exposed(false); require(!DirectReadScope::background(), "nested wait classification"); }
        require(DirectReadScope::background(), "nested scope restores background classification");
    }
    require(!DirectReadScope::background(), "scope restores foreground classification");
    DirectPageCache cache(-1, 2U << 20U, page, 128U * page, synthetic());
    std::vector<std::uint64_t> pages(128);
    for (std::size_t i = 0; i < pages.size(); ++i) pages[i] = i;
    require(cache.ensure(pages, true) == 128, "all pages initially missing");
    require(cache.stats().reads == 5, "128 contiguous pages coalesced into four 128 KiB reads plus probe");
    std::reverse(pages.begin(), pages.end());
    pages.push_back(1); pages.push_back(1);
    require(cache.ensure(pages, false) == 0 && cache.stats().reads == 5, "sorted horizon deduplication and hits");
    check_row(cache, page-43); check_row(cache, 17);
    refuses([&] { cache.ensure(std::array<std::uint64_t, 1>{128}, false); });
    require(cache.stats().used_bytes == 128U * page && cache.stats().peak_owned_bytes == 128U * page,
        "reservation refusal never exceeds owned arena cap");
    {
        const auto view = cache.view(pages);
        std::vector<std::future<void>> copies;
        for (unsigned worker = 0; worker < 8; ++worker) copies.push_back(std::async(std::launch::async, [&, worker] {
            for (unsigned i = 0; i < 10000; ++i) {
                const auto offset = (i * 90U + worker * page) % (127U * page);
                std::array<std::byte, 90> row{};
                view->copy(offset, row);
                for (std::size_t j = 0; j < row.size(); ++j) require(row[j] == value(offset + j), "parallel pinned bytes");
            }
        }));
        for (auto& copy : copies) copy.get();
        refuses([&] { cache.erase(std::array<std::uint64_t, 1>{1}); });
        require(cache.stats().used_bytes == 128U * page, "pinned view refusal preserves capacity");
        std::array<std::byte, 90> row{};
        refuses([&] { view->copy(128U * page - 40U, row); });
    }
    refuses([&] { (void)cache.view(std::array<std::uint64_t, 1>{128}); });
    cache.erase(pages);
    refuses([&] { check_row(cache, 17); });
    require(cache.stats().used_bytes == 0, "retired cache pages no longer readable");
    for (std::uint64_t i = 0; i < 1000; ++i) {
        const std::array<std::uint64_t, 2> wave{i % 511U, i % 511U + 1U};
        cache.ensure(wave, false); check_row(cache, wave[0]*page + page-43); cache.erase(wave);
        require(cache.stats().peak_owned_bytes <= 128U * page, "adversarial stream stays inside fixed allocation");
    }
    refuses([&] { cache.ensure(std::array<std::uint64_t, 1>{512}, false); });

    std::atomic<int> calls{0};
    auto base = synthetic();
    DirectPageCache partial(-1, 16U * page, page, 8U * page,
        [&](int f, void* out, std::size_t n, std::uint64_t offset) {
            if (calls.fetch_add(1) == 0) { errno = EINTR; return std::int64_t{-1}; }
            return base(f, out, std::min<std::size_t>(n, page), offset);
        });
    DirectActivity activity;
    { const DirectReadScope measured(false, &activity);
      partial.ensure(std::array<std::uint64_t, 4>{0,1,2,3}, false); }
    require(activity.bytes == 4U * page && activity.wait_ns > 0, "scope counts issued bytes and caller read completion wait");
    check_row(partial, page-43);
    require(partial.stats().reads == 6, "EINTR retried and aligned short reads completed");
    for (const int fault : {EIO, EINVAL, 0, -17}) {
        std::atomic<bool> fail{false};
        DirectPageCache broken(-1, 16U*page, page, 8U*page,
            [&](int f, void* out, std::size_t n, std::uint64_t offset) {
                if (!fail) return base(f, out, n, offset);
                if (fault > 0) { errno = fault; return std::int64_t{-1}; }
                return static_cast<std::int64_t>(-fault); // EOF or unaligned short read
            });
        fail = true;
        refuses([&] { broken.ensure(std::array<std::uint64_t, 4>{0,1,2,3}, false); });
        require(!broken.healthy(), "I/O failure poisons owned source certificate");
        refuses([&] { check_row(broken, 0); });
        broken.erase(std::array<std::uint64_t, 4>{0,1,2,3}); // completed failed jobs still drain
    }
    refuses([&] { DirectPageCache unsupported(-1, page, page, page,
        [](int, void*, std::size_t, std::uint64_t) { errno=EOPNOTSUPP; return std::int64_t{-1}; }); });

    DirectPageCache tail(-1, page+137U, page, 2U*page,
        [&](int f, void* out, std::size_t n, std::uint64_t offset) {
            (void)base(f, out, n, offset);
            return static_cast<std::int64_t>(std::min<std::uint64_t>(n, page+137U-offset));
        });
    tail.ensure(std::array<std::uint64_t, 2>{0,1}, false);
    check_row(tail, page+47U);
    refuses([&] { check_row(tail, page+48U); });
}
void concurrent_reads() {
    std::promise<void> release, started;
    auto gate = release.get_future().share();
    std::atomic<bool> block{false}, signalled{false};
    auto base = synthetic();
    DirectPageCache cache(-1, 1024U*page, page, 256U*page,
        [&](int f, void* out, std::size_t n, std::uint64_t offset) {
            if (block) {
                if (!signalled.exchange(true)) started.set_value();
                gate.wait();
            }
            return base(f, out, n, offset);
        });
    std::vector<std::uint64_t> pages(256);
    for (std::size_t i=0; i<pages.size(); ++i) pages[i]=i*2U;
    block = true;
    auto first = std::async(std::launch::async, [&] { return cache.ensure(pages,true); });
    started.get_future().wait();
    auto second = std::async(std::launch::async, [&] { return cache.ensure(pages,false); });
    const auto wait = second.wait_for(std::chrono::milliseconds(20));
    release.set_value();
    require(first.get() == 256 && second.get() == 0 && wait == std::future_status::timeout,
        "duplicate source joins pending owned reads");
    require(cache.stats().reads == 257 && cache.stats().used_bytes == 256U*page,
        "bounded queue and active workers issue each scattered page once");
}
template<class F> void eventually(F condition, const char* why) {
    const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!condition() && std::chrono::steady_clock::now() < limit) std::this_thread::yield();
    require(condition(), why);
}
void critical_order(bool emulate) {
    constexpr std::size_t critical_count = 1024, ahead_count = 64;
    std::atomic<bool> armed{false}, signalled{false};
    std::atomic<std::size_t> completed{0};
    std::promise<void> entered, release;
    const auto gate = release.get_future().share();
    auto base = synthetic();
    DirectPageCache cache(-1, 4096U*page, page, (critical_count+ahead_count)*page,
        [&](int f, void* out, std::size_t n, std::uint64_t offset) {
            if (armed) {
                if (!signalled.exchange(true)) entered.set_value();
                gate.wait();
                if (offset / page >= 3072U)
                    require(completed == critical_count, "lookahead cannot issue while any critical page remains beyond device depth");
            }
            const auto result = base(f, out, n, offset);
            if (armed && offset / page < 3072U) ++completed;
            return result;
        }, {0, emulate});
    std::vector<std::uint64_t> critical(critical_count), ahead(ahead_count);
    for (std::size_t i=0; i<critical.size(); ++i) critical[i]=2U*i;
    for (std::size_t i=0; i<ahead.size(); ++i) ahead[i]=3072U+2U*i;
    armed=true;
    auto foreground=std::async(std::launch::async,[&] { return cache.ensure(critical,false,true); });
    entered.get_future().wait();
    eventually([&] { return cache.stats().peak_inflight==(emulate ? 512U : 32U); },"all configured read slots filled before release");
    auto future=std::async(std::launch::async,[&] { return cache.ensure(ahead,true,false); });
    eventually([&] { return cache.stats().used_bytes==(critical_count+ahead_count)*page; }, "both priorities reserved inside cap");
    release.set_value();
    require(foreground.get()==critical_count && future.get()==ahead_count,"both priorities finish exact unique pages");
    require(cache.stats().peak_inflight==(emulate ? 512U : 32U),"reader maintains configured device/pool depth");
    for (auto pg:critical) check_row(cache,pg*page+17);
    require(cache.stats().peak_used_bytes==(critical_count+ahead_count)*page,"critical and lookahead respect owned cap");
}
void queued_promotion() {
    std::atomic<bool> armed{false}, critical_done{false};
    std::atomic<unsigned> entered{0};
    std::promise<void> release;
    const auto gate=release.get_future().share();
    auto base=synthetic();
    DirectPageCache cache(-1,4096U*page,page,1024U*page,
        [&](int f,void* out,std::size_t n,std::uint64_t offset) {
            const auto ordinal=armed ? entered.fetch_add(1) : 0U;
            if (armed) {
                gate.wait();
                if (ordinal>=32U && offset!=2046U*page)
                    require(critical_done,"queued shared critical read precedes remaining lookahead");
            }
            const auto result=base(f,out,n,offset);
            if (armed && offset==2046U*page) critical_done=true;
            return result;
        });
    std::vector<std::uint64_t> ahead(1024);
    for (std::size_t i=0;i<ahead.size();++i) ahead[i]=i*2U;
    armed=true;
    auto background=std::async(std::launch::async,[&] { return cache.ensure(ahead,true,false); });
    eventually([&] { return entered==32U; },"pread pool filled before critical promotion");
    auto critical=std::async(std::launch::async,[&] { return cache.ensure(std::array<std::uint64_t,1>{2046},false,true); });
    eventually([&] { return cache.stats().pending_critical==1U; },"shared queued run promoted without duplicate I/O");
    release.set_value();
    require(critical.get()==0 && background.get()==1024 && cache.stats().reads==1025,"promoted page read exactly once");
}
void ring_faults() {
    auto base=synthetic();
    for (int code:{ENOSYS,EPERM,EINVAL}) {
        DirectPageCache cache(-1,16U*page,page,8U*page,base,{code,false});
        require(std::strcmp(cache.reader(),"pread-pool")==0 && cache.fallback_errno()==code,"typed setup fallback witness");
        cache.ensure(std::array<std::uint64_t,2>{0,1},false); check_row(cache,page-43);
    }
    std::atomic<int> calls{0};
    DirectPageCache partial(-1,16U*page,page,8U*page,
        [&](int f,void* out,std::size_t n,std::uint64_t at) {
            const auto call=calls.fetch_add(1);
            if (call==1 || call==2) { errno=call==1 ? EAGAIN : EINTR; return std::int64_t{-1}; }
            return base(f,out,std::min<std::size_t>(n,page),at);
        },{0,true});
    DirectActivity activity;
    { const DirectReadScope scope(false,&activity); partial.ensure(std::array<std::uint64_t,4>{0,1,2,3},false); }
    require(activity.reads==6 && activity.bytes==4U*page && activity.end_ns>activity.begin_ns,"CQ retries/short completions have actual operation/time telemetry");
    check_row(partial,page-43);
    for (const int fault:{EIO,EINVAL,0,-17}) {
        std::atomic<bool> fail{false};
        DirectPageCache broken(-1,16U*page,page,8U*page,
            [&](int f,void* out,std::size_t n,std::uint64_t at) {
                if (!fail) return base(f,out,n,at);
                if (fault>0) { errno=fault; return std::int64_t{-1}; }
                return static_cast<std::int64_t>(-fault);
            },{0,true});
        fail=true;
        refuses([&] { broken.ensure(std::array<std::uint64_t,4>{0,1,2,3},false); });
        require(!broken.healthy(),"bad CQE poisons source certificate");
        broken.erase(std::array<std::uint64_t,4>{0,1,2,3});
    }
    DirectPageCache tail(-1,page+137U,page,2U*page,
        [&](int f,void* out,std::size_t n,std::uint64_t at) {
            (void)base(f,out,n,at); return static_cast<std::int64_t>(std::min<std::uint64_t>(n,page+137U-at));
        },{0,true});
    tail.ensure(std::array<std::uint64_t,2>{0,1},false); check_row(tail,page+47U);
    refuses([&] { check_row(tail,page+48U); });
}
void block_order() {
    constexpr std::uint64_t block=1024, cap=4096U*page;
    auto cache=std::make_shared<DirectPageCache>(-1,8192U*page,page,cap,synthetic(),DirectPageCache::TestRing{0,true});
    std::atomic<unsigned> calls{0};
    RollingPages::Io io;
    io.owned=cache; io.reclaimable=[](auto) { return true; }; io.resident=[] { return std::uint64_t{0}; };
    io.evict=[cache](auto p) { cache->erase(p); };
    const auto issue=[&](auto pages,bool critical) {
        const auto call=calls.fetch_add(1);
        require(pages.size()==block,"whole block submitted as one high-depth wave");
        require(pages.front()==call*block*2U,"critical block completes before next block lookahead starts");
        if (call==0) require(critical,"setup block zero receives critical priority");
        return cache->ensure(pages,true,critical);
    };
    io.critical=[&](auto pages) { return issue(pages,true); };
    io.prefetch=[&](auto pages) { return issue(pages,false); };
    io.prepare=[cache](auto pages) { cache->ensure(pages,false,true); };
    std::vector<std::uint32_t> rows(3U*block);
    for (std::size_t i=0;i<rows.size();++i) rows[i]=static_cast<std::uint32_t>(i*2U);
    RollingPipeline pipeline(256,page,0,page,1,rows,cap,0,cap,2U*block,io,block);
    eventually([&] { return calls>=2U; },"setup starts block-zero reads and ordered lookahead before demand");
    pipeline.gather_range(0,block,[] (auto,auto) {});
    const auto metrics=pipeline.block_io(0,block);
    require(metrics.critical.bytes==block*page && metrics.critical.reads==block &&
        metrics.critical.end_ns>metrics.critical.begin_ns,"setup critical reads attributed to block zero");
    pipeline.advance(block);
    pipeline.gather_range(block,block,[] (auto,auto) {});
    pipeline.gather_range(2U*block,block,[] (auto,auto) {});
    pipeline.finish();
    require(cache->stats().peak_used_bytes<=cap,"ordered full blocks remain capped");
}
void owned_decode_lru() {
    constexpr std::uint64_t cap = 16U * page, quota = 4U * page;
    auto cache = std::make_shared<DirectPageCache>(-1, 4096U*page, page, cap+quota, synthetic());
    RollingPages::Io io;
    io.owned=cache; io.reclaimable=[](auto) { return true; };
    io.resident=[] { return std::uint64_t{0}; };
    io.prefetch=[cache](auto p) { return cache->ensure(p,true); };
    io.prepare=[cache](auto p) { (void)cache->ensure(p,false); };
    io.evict=[cache](auto p) { cache->erase(p); };
    const std::array<std::uint32_t,1> prompt{0};
    RollingPipeline pipeline(1,page,0,90,1,prompt,cap+quota,quota,cap,1,io);
    pipeline.gather(0,1,[&] { pipeline.source(std::array<std::uint64_t,1>{0},[&] { check_row(*cache,0); }); });
    for (std::uint64_t i=0; i<10000; ++i) {
        const auto row = (i*1823U) % 180000U;
        pipeline.source(std::array<std::uint64_t,1>{row},[&] { check_row(*cache,row*90U); });
        require(cache->stats().used_bytes <= cap+quota, "decode replacement remains bounded beyond quota lifetime");
    }
    pipeline.finish();
    require(cache->stats().used_bytes == 0, "decode finish drains cache ownership");

    auto manager = std::make_shared<ResidencyManager>("direct-lru",8192,1,1,page,0,90,180000,1,
        cap+quota,quota,1,io);
    auto turn = manager->begin();
    for (std::uint64_t i=0; i<10000; ++i) {
        const auto row = (i*1823U) % 180000U;
        turn->source(8192,1,std::array<std::uint64_t,1>{row},[&] {
            std::array<std::byte,90> out{}; turn->read_row(row,out);
            for (std::size_t k=0; k<out.size(); ++k) require(out[k]==value(row*90U+k),"MTP LRU row bytes");
        });
        require(cache->stats().used_bytes <= quota, "MTP owned LRU obeys fixed decode quota");
    }
    turn->finish();
    require(cache->stats().used_bytes==0,"MTP turn retires owned bytes");
}
void adaptive_replay() {
    constexpr std::uint64_t quota=16U*page;
    auto cache=std::make_shared<DirectPageCache>(-1,4096U*page,page,quota,synthetic());
    RollingPages::Io io;
    io.owned=cache; io.reclaimable=[](auto) {return true;}; io.resident=[] {return std::uint64_t{0};};
    io.prepare=[cache](auto p) {(void)cache->ensure(p,false);}; io.evict=[cache](auto p) {cache->erase(p);};
    auto manager=std::make_shared<ResidencyManager>("adaptive-replay",8192,1,256,page,0,90,180000,1,quota,quota,1,io);
    auto turn=manager->begin();
    const std::array<std::uint64_t,8> first{45,227,409,591,773,955,1137,1319};
    auto check=[&](std::span<const std::uint64_t> rows) {
        for (auto row:rows) {
            std::array<std::byte,90> out{}; turn->read_row(row,out);
            for (std::size_t k=0;k<out.size();++k) require(out[k]==value(row*90U+k),"adaptive replay exact stored bytes");
        }
    };
    turn->verify_window(8192,1,8,first,[&] {
        check(first);
        // Fill the complete fixed quota: a nested pressure source cannot evict
        // even the last padded row before the outer window copy has joined.
        refuses([&] {turn->source(8192,1,std::array<std::uint64_t,1>{50000},[] {});});
        check(first);
    });
    const auto before=cache->stats().reads;
    for (std::uint64_t cycle=0;cycle<1024;++cycle) {
        const auto n=2U<<(cycle%3U);
        std::vector<std::uint64_t> ids;
        for (std::uint64_t r=0;r<n;++r) ids.push_back(2000U+((cycle*8U+r)*91U)%170000U);
        turn->verify_window(8192,1,n,ids,[&] {check(ids);});
        require(cache->stats().used_bytes<=quota,"adaptive windows wrap a fixed quota independent of G");
    }
    const auto replay_reads=cache->stats().reads;
    require(replay_reads>before,"diverse adaptive stream forces cache replacement");
    refuses([&] {check(first);}); // evicted pages cannot be silently read
    turn->verify_window(8192,1,8,first,[&] {check(first);});
    require(cache->stats().reads>replay_reads,"rejected old window is re-fetched before replay");
    turn->source(8192,1,first,[&] {check(first);}); // serial replay after restored state
    bool touched=false;
    refuses([&] {turn->verify_window(8192,1,9,first,[&] {touched=true;});});
    refuses([&] {turn->verify_window(8192,1,4,first,[&] {touched=true;});});
    require(!touched && manager->healthy(),"invalid or incomplete horizons refuse before touch");
    turn->finish(); require(cache->stats().used_bytes==0,"adaptive/replay turn drains all sources");
}
#if defined(__linux__)
void filesystem() {
    char path[]="/tmp/strix-ple-direct-XXXXXX";
    const int fd=::mkstemp(path);
    require(fd>=0,"temporary direct fixture");
    std::vector<std::byte> bytes(8U<<20U);
    for (std::size_t i=0; i<bytes.size(); ++i) bytes[i]=value(i);
    std::size_t at=0;
    while (at<bytes.size()) {
        const auto n=::write(fd,bytes.data()+at,bytes.size()-at);
        require(n>0,"temporary fixture write"); at+=static_cast<std::size_t>(n);
    }
    require(::fsync(fd)==0,"fixture sync");
    const int direct=::open(path,O_RDONLY|O_CLOEXEC|O_DIRECT);
    void* aligned=nullptr;
    require(::posix_memalign(&aligned,page,page)==0,"direct probe alignment");
    const bool supported=direct>=0 && ::pread(direct,aligned,page,0)==static_cast<ssize_t>(page);
    std::free(aligned);
    ::unlink(path);
    if (supported) {
        DirectPageCache cache(direct,bytes.size(),page,1024U*page);
        std::vector<std::uint64_t> pages(128);
        for (std::size_t i=0; i<pages.size(); ++i) pages[i]=i;
        const auto probes = cache.stats().reads;
        cache.ensure(pages,false); check_row(cache,page-43);
        require(cache.stats().reads==probes+4,"real O_DIRECT coalesced reads");
        cache.erase(pages);
        std::vector<std::uint64_t> sparse(1024);
        for (std::size_t i=0; i<sparse.size(); ++i) sparse[i]=i*2U;
        cache.ensure(sparse,false);
        for (auto pg:sparse) check_row(cache,pg*page+17);
        require(cache.stats().reads==probes+4U+sparse.size(),"real sparse CQ completions across multiple queue depths");
        require(cache.stats().peak_inflight<=512U && cache.stats().peak_used_bytes<=1024U*page,"real direct queue/cap bounded");
        std::cout << "real temporary-file O_DIRECT: PASS backend=" << cache.reader()
                  << " fallback-errno=" << cache.fallback_errno() << " peak-inflight=" << cache.stats().peak_inflight << '\n';
    } else {
        if (direct>=0) ::close(direct);
        std::cout << "temporary filesystem lacks O_DIRECT: explicit aligned test seam exercised\n";
    }
    // A buffered descriptor must never masquerade as a production reader.
    const int buffered=::dup(fd);
    refuses([&] { DirectPageCache rejected(buffered,bytes.size(),page,page); });
    ::close(fd);
}
#endif
}
int main() {
    try {
        setup_prime(); logic_and_faults(); concurrent_reads(); critical_order(false); critical_order(true);
        queued_promotion(); ring_faults(); block_order(); owned_decode_lru(); adaptive_replay();
#if defined(__linux__)
        filesystem();
#endif
        std::cout << "PLE direct CPU tests passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
