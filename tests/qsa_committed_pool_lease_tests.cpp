// Run the production service with a deterministic queued-event backend. No HIP
// loader/device is needed: only allocation and stream/event primitives are replaced.
#include "strix/hip_indexed_attention_launcher.hpp"
#include "strix/qsa_committed_pool.hpp"
#include "rotary_mock.hpp"
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
namespace qsa_pool_fake {
struct Stream {std::vector<std::function<void()>> work;std::size_t completed=0;};
struct Event {Stream* stream=nullptr;std::size_t point=0;};
using hipStream_t=Stream*;using hipEvent_t=Event*;using hipError_t=int;
constexpr int hipSuccess=0,hipEventDisableTiming=1;
int current_device=0,allocations=0,events=0;
bool fail_allocation=false;int fail_event_after=-1;
std::function<void()> device_hook;
void complete(Stream* stream,std::size_t point) {
    while(stream && stream->completed<point) {auto work=stream->work[stream->completed++];work();}
}
int hipGetDevice(int* out){*out=current_device;if(device_hook){auto hook=std::move(device_hook);device_hook={};hook();}return 0;}
int hipGetLastError(){return 0;}
int hipMalloc(void** out,std::uint64_t bytes){if(fail_allocation)return 1;*out=std::malloc(static_cast<std::size_t>(bytes));if(!*out)return 1;++allocations;return 0;}
int hipFree(void* p){std::free(p);--allocations;return 0;}
int hipEventCreateWithFlags(Event** out,int){if(fail_event_after==0)return 1;if(fail_event_after>0)--fail_event_after;*out=new Event;++events;return 0;}
int hipEventDestroy(Event* e){delete e;--events;return 0;}
int hipEventRecord(Event* e,Stream* stream){*e={stream,stream->work.size()};return 0;}
int hipEventSynchronize(Event* e){complete(e->stream,e->point);return 0;}
int hipStreamWaitEvent(Stream* stream,Event* e,int){const auto copy=*e;stream->work.push_back([copy]{complete(copy.stream,copy.point);});return 0;}
void check_hip(int status,const char* what){if(status)throw std::runtime_error(what);}
void require(bool ok,const char* what){if(!ok){std::fprintf(stderr,"FAIL: %s\n",what);std::exit(1);}}
template<class F>void refuses(F f){bool refused=false;try{f();}catch(const std::exception&){refused=true;}require(refused,"unsafe lease refused");}
}
using namespace qsa_pool_fake;
namespace strix {
// Storage view only; the production service/frontier/lease algorithms below are
// included unchanged from the engine. Device scorer fixtures use the real view.
struct QsaPrefillPool {
    std::uint32_t mtp_role=13U;
    float* keys=nullptr;std::uint16_t* packed=nullptr;std::uint32_t capacity=0,valid=0;
    std::shared_ptr<const HipQwenRotaryTable> rotary_table;
};
#include "../src/hip_qsa_committed_pool.inc"
}
int main() {
    using namespace strix;
    const auto plan=mtp_qsa_pool_plan(128);
    fail_allocation=true;require(!HipQsaCommittedPool::create(plan),"allocation failure declines");fail_allocation=false;
    fail_event_after=3;require(!HipQsaCommittedPool::create(plan),"partial event allocation declines");fail_event_after=-1;
    require(allocations==0 && events==0,"failed factory releases all partial ownership");
    auto owner=HipQsaCommittedPool::create(plan);auto other=HipQsaCommittedPool::create(plan);
    Stream a,b;auto model=ModelContract::qwen38_flash_next();RuntimeTensorBinding norm{};
    norm.address=0x100000;norm.bytes=512;norm.allocation_generation=1;
    {
        auto tables=HipQsaCommittedPool::create(plan);auto table_model=model;table_model.qwen_rotary_table=mock_rotary(2);
        const auto first_table=table_model.qwen_rotary_table;
        for(unsigned role=0;role<13;++role) {
            const auto raw=std::uintptr_t{0x1000000}+role*std::uintptr_t{0x100000};
            auto access=tables->before_rewrite(role==12,raw,128*256,1,norm,0,table_model,100,1,&a,true);
            require(access->key().role==role,"all thirteen table-backed roles retain their own keys");
            tables->checked(4,0,role);
        }
        table_model.qwen_rotary_table=mock_rotary(2);
        refuses([&]{(void)tables->before_rewrite(false,0x1000000,128*256,1,norm,0,table_model,101,1,&a,true);});
        tables->begin_epoch();
        {auto rebound=tables->before_rewrite(false,0x1000000,128*256,2,norm,0,table_model,100,1,&a,true);
            require(rebound->key().epoch==2,"drained epoch admits a different retained table owner");}
        require(first_table!=table_model.qwen_rotary_table,"fixture uses distinct same-payload table owners");
    }
    const auto lease=[&](std::shared_ptr<HipQsaCommittedPool> pool,Stream* stream,std::uint64_t generation=1) {
        return pool->before_rewrite(false,0x200000,128*256,generation,norm,0,model,100,1,stream,true);
    };
    std::uint16_t* first=nullptr;bool produced=false,consumed=false;
    {
        auto writer=lease(owner,&a);auto* storage=static_cast<QsaPrefillPool*>(writer->storage());
        require(!storage->keys && storage->packed && storage->capacity==32,"packed-only exact entry envelope");
        first=storage->packed;a.work.push_back([&]{first[0]=71;produced=true;});
        const auto descriptor=writer->key();require(descriptor.producer_event && descriptor.epoch==1,"current producer completion fence");
    }
    require(!produced,"recording a lease completion does not synchronize the writer");
    {
        auto reader=lease(owner,&b);require(reader->key().packed==first,"all windows/serial borrow the same entry");
        b.work.push_back([&]{require(produced && first[0]==71,"reader follows cross-stream producer");consumed=true;first[0]=92;});
    }
    require(!consumed,"cross-stream dependency is queued");
    owner->invalidate(false,100);
    require(consumed && first[0]==92,"invalidation drains readers before external raw restore");
    {
        auto second=lease(other,&a);require(second->key().packed!=first,"separate sessions never alias keys");
    }
    refuses([&]{auto bad=lease(owner,&a,2);}); // no rebinding a live epoch
    auto changed=model;changed.rope_theta*=2;
    refuses([&]{(void)owner->before_rewrite(false,0x200000,128*256,1,norm,0,changed,100,1,&a,true);});
    owner->begin_epoch();{auto rebound=lease(owner,&a,2);require(rebound->key().epoch==2,"idle epoch permits allocation-generation rebinding");}
    owner->cancel();refuses([&]{auto stopped=lease(owner,&a,2);});refuses([&]{(void)owner->selector_lease(&a);});
    owner->begin_epoch();
    {auto scratch=owner->selector_lease(&a);auto* p=scratch->scratch();require(p!=nullptr,"K1 exclusive scratch lease");a.work.push_back([p]{p[0]=123;});}
    {auto scratch=owner->selector_lease(&b);auto* p=scratch->scratch();b.work.push_back([p]{require(p[0]==123,"scratch reuse waits on its own previous consumer");});}
    owner->begin_epoch(); // also drains outstanding scratch consumers
    current_device=1;
    refuses([&]{auto wrong_device=lease(owner,&a);});
    refuses([&]{(void)owner->selector_lease(&a);});
    current_device=0;
    device_hook=[&]{owner->begin_epoch();};
    refuses([&]{auto stale=lease(owner,&a);}); // epoch changes between admission and entry lock
    device_hook=[&]{owner->begin_epoch();};
    refuses([&]{(void)owner->selector_lease(&a);});
    const auto weak=std::weak_ptr<HipQsaCommittedPool>(owner);
    auto retained=lease(owner,&a);auto* final=static_cast<QsaPrefillPool*>(retained->storage())->packed;
    bool final_write=false;a.work.push_back([&]{final[0]=101;final_write=true;});
    owner.reset();require(!weak.expired() && !final_write,"in-flight lease retains allocation through enqueue");
    retained.reset();require(weak.expired() && final_write,"teardown completes queued consumers before free");
    other.reset();require(allocations==0 && events==0,"all allocations and completion events reclaimed");
    std::puts("PASS production service with fake events: stream handoffs, restore/cancel, epochs, identities, scratch and retained destruction");
}
