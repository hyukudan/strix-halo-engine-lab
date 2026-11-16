#include <strix/weight/cache.hpp>
#include <strix/weight/source.hpp>
#include <atomic>
#include <cstring>
#include <future>
#include <thread>
namespace {
using namespace strix::weight;
struct Memory final : Source {
    std::array<Byte,16> bytes{};
    Status read(ByteRange r, std::span<Byte> out) const override {
        if (r.offset>bytes.size() || r.bytes>bytes.size()-r.offset || out.size()!=r.bytes) return {Error::malformed,"range"};
        std::memcpy(out.data(),bytes.data()+r.offset,out.size()); return {Error::ok,{}};
    }
    std::uint64_t size() const noexcept override { return bytes.size(); }
    Digest256 identity() const override { return digest_bytes(bytes); }
};
}
bool wfmt_cache_test() {
    auto source=std::make_shared<Memory>();
    WeightView view{}; view.tensor=1; view.storage={"plain.f32",1}; view.shape={{1,4,0,0},2}; view.payload={source,{0,16}};
    Plan plan{}; plan.selected.storage=view.storage; plan.selected.arithmetic={"cpu-q8a8-tree32",1};
    plan.selected.image={"image.raw",1}; plan.selected.backend=Backend::cpu; plan.selected.strategy=Strategy::native;
    plan.selected.resources.image_bytes=16;
    auto context=cpu_execution_context();
    ImageKey key{}; if (!make_image_key(view,plan,"host","test@1",key)) return false;
    ImageCache cache(16); ImageLease first;
    if (!acquire_image(cache,key,plan,view,context,first) || !first || cache.bytes_reserved()!=16) return false;
    ImageLease second; if (!acquire_image(cache,key,plan,view,context,second) || first.data()!=second.data()) return false;
    if (cache.evict_completed(16)!=0) return false;
    auto wrong=context; wrong.device_index=1;
    if (record_last_use(first,wrong).code!=Error::unsupported_backend) return false;
    first=ImageLease{}; second=ImageLease{};
    if (cache.evict_completed(16)!=16 || cache.bytes_reserved()!=0) return false;
    ImageKey changed=key; changed.metadata.bytes[0]^=1;
    if (acquire_image(cache,changed,plan,view,context,first).code!=Error::stale_source) return false;
    auto sidecar_source=std::make_shared<Memory>();
    view.sidecars.push_back({"codebook",Scalar::f32,{{4,0,0,0},1},{sidecar_source,{0,16}}});
    if (!make_image_key(view,plan,"host","test@1",changed) || changed==key) return false;
    auto before=changed; sidecar_source->bytes[0]=Byte{1};
    if (!make_image_key(view,plan,"host","test@1",changed) || changed==before) return false;
    if (acquire_image(cache,before,plan,view,context,first).code!=Error::stale_source) return false;
    ImageCache too_small(15); if (acquire_image(too_small,changed,plan,view,context,first).code!=Error::memory_budget) return false;
    std::atomic<int> conversions=0, allocations=0, retired=0;
    std::atomic<bool> complete=false;
    auto callbacks=std::make_shared<RuntimeCallbacks>(*context.callbacks);
    auto convert=callbacks->convert; auto allocate=callbacks->allocate; auto retire=callbacks->retire;
    callbacks->allocate=[&](std::uint64_t bytes,void*& data) { ++allocations; return allocate(bytes,data); };
    callbacks->convert=[&](const WeightView& v,const Plan& p,void* data,ExecutionContext& c) { ++conversions; std::this_thread::yield(); return convert(v,p,data,c); };
    callbacks->event_complete=[&](void*) { return complete.load(); };
    callbacks->retire=[&](void* data,std::vector<void*> events,ExecutionContext c) { ++retired; retire(data,std::move(events),std::move(c)); };
    context.callbacks=callbacks;
    ImageCache concurrent(16);
    auto acquire=[&] { ImageLease lease; auto local=context; if (!acquire_image(concurrent,changed,plan,view,local,lease)) return ImageLease{}; return lease; };
    auto a=std::async(std::launch::async,acquire); auto b=std::async(std::launch::async,acquire);
    auto la=a.get(); auto lb=b.get();
    if (!la || !lb || la.data()!=lb.data() || conversions!=1 || allocations!=1) return false;
    la=ImageLease{}; lb=ImageLease{};
    if (concurrent.evict_completed(16)!=0) return false;
    complete=true; if (concurrent.evict_completed(16)!=16 || retired!=1) return false;
    // Every failure releases its reservation and allows a subsequent successful acquisition.
    callbacks->convert=[](const WeightView&,const Plan&,void*,ExecutionContext&) -> Status { return {Error::device_failure,"injected conversion failure"}; };
    ImageCache failed(16);
    if (acquire_image(failed,changed,plan,view,context,first).code!=Error::device_failure || failed.bytes_reserved()!=0) return false;
    callbacks->convert=convert;
    if (!acquire_image(failed,changed,plan,view,context,first)) return false;
    first=ImageLease{}; failed.set_budget(0);
    if (failed.bytes_reserved()!=0) return false;
    callbacks->record_event=[](ExecutionContext&,void*&) -> Status { return {Error::device_failure,"injected event failure"}; };
    ImageCache event_failure(16);
    if (acquire_image(event_failure,changed,plan,view,context,first).code!=Error::device_failure || event_failure.bytes_reserved()!=0) return false;
    callbacks->record_event=cpu_execution_context().callbacks->record_event;
    callbacks->allocate=[](std::uint64_t,void*&) -> Status { return {Error::memory_budget,"injected allocation failure"}; };
    ImageCache allocation_failure(16);
    return acquire_image(allocation_failure,changed,plan,view,context,first).code==Error::memory_budget && allocation_failure.bytes_reserved()==0;
}
