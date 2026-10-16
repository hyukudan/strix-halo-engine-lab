#include "strix/hip_gpu_visibility_probe.hpp"
#include <cassert>
#include <array>
#include <iostream>
#include <string_view>
using namespace strix::transport;
struct Owner {
 void* data=nullptr;
 explicit Owner(std::size_t bytes){assert(hipMalloc(&data,bytes)==hipSuccess);}
 ~Owner(){assert(hipFree(data)==hipSuccess);}
};
template<class F> void rejected(F&& fn){bool caught=false;try{fn();}catch(const std::exception&){caught=true;}assert(caught);}
static std::shared_ptr<Owner> owner;
static std::unique_ptr<HipGpuVisibilityProbe> probe;
static GpuVisibilityPattern pattern{17,9,123};
static std::array<GpuVisibilityRange,2> ranges;
static void setup(){
 owner=std::make_shared<Owner>(8200);
 ranges={GpuVisibilityRange{reinterpret_cast<std::uintptr_t>(owner->data),4097},GpuVisibilityRange{reinterpret_cast<std::uintptr_t>(owner->data)+4100,4096}};
 probe=std::make_unique<HipGpuVisibilityProbe>(owner,[]{return &fake_hip::stream;},[]{},HipGpuVisibilityProbePolicy{16384,std::chrono::milliseconds{5}});
 assert(fake_hip::event_flags==static_cast<int>(hipEventDisableTiming));
}
static void check_success(){
 probe->produce(ranges,pattern);auto r=probe->verify(ranges,pattern);
 assert(valid_gpu_visibility_result(r,pattern,r.nonce,8193));
 assert(fake_hip::copied_bytes==64);
 std::uint64_t digest=0,consumed=0;
 for(std::uint64_t p=0;p<8193;++p){auto b=gpu_visibility_byte(p,pattern);digest^=gpu_visibility_mix(p^b);consumed+=gpu_visibility_consume(p,b);}
 assert(r.observed_digest==digest&&r.consumed_digest==consumed);
}
int main(int argc,char** argv){
 assert(argc==2);std::string_view c=argv[1];setup();
 if(c=="full_bytes") {check_success();assert(probe->negative_control(ranges,pattern,4097));auto r=probe->verify(ranges,pattern);assert(valid_gpu_visibility_result(r,pattern,r.nonce,8193));}
 else if(c=="epoch_replay") {probe->produce(ranges,pattern);auto other=pattern;++other.epoch;auto r=probe->verify(ranges,other);assert(r.mismatches>0);r=probe->verify(ranges,pattern);assert(valid_gpu_visibility_result(r,pattern,r.nonce,8193));}
 else if(c=="single_byte") {probe->produce(ranges,pattern);static_cast<std::uint8_t*>(owner->data)[8195]^=1;auto r=probe->verify(ranges,pattern);assert(r.mismatches==1&&r.first_mismatch==8192);}
 else if(c=="bounds") {
  auto bad=ranges;bad[0].bytes=0;rejected([&]{probe->produce(bad,pattern);});
  bad=ranges;bad[1].address=bad[0].address;rejected([&]{probe->produce(bad,pattern);});
  bad=ranges;bad[1].bytes=10000;rejected([&]{probe->produce(bad,pattern);});
  auto old=pattern;old.epoch=0;rejected([&]{probe->produce(ranges,old);});
  rejected([&]{(void)probe->negative_control(ranges,pattern,8193);});
  check_success();
 }
 else if(c=="foreign_device") {fake_hip::allocations.at(reinterpret_cast<std::uintptr_t>(owner->data)).device=1;rejected([&]{probe->produce(ranges,pattern);});fake_hip::allocations.at(reinterpret_cast<std::uintptr_t>(owner->data)).device=0;}
 else if(c=="captured") {fake_hip::stream.captured=true;rejected([&]{probe->produce(ranges,pattern);});fake_hip::stream.captured=false;}
 else if(c=="stale_result") {probe->produce(ranges,pattern);fake_hip::corrupt_result=true;rejected([&]{(void)probe->verify(ranges,pattern);});}
 else if(c=="failed_record") {
  fake_hip::record_fail=1;fake_hip::stream.busy=true;
  rejected([&]{probe->produce(ranges,pattern);});auto frees=fake_hip::free_calls;
  assert(!probe->close()&&fake_hip::free_calls==frees&&owner.use_count()==2);
  fake_hip::stream.busy=false;assert(probe->close());
 }
 else if(c=="timeout") {
  fake_hip::never_complete=true;fake_hip::stream.busy=true;rejected([&]{probe->produce(ranges,pattern);});
  assert(!probe->close());fake_hip::stream.busy=false;fake_hip::never_complete=false;assert(probe->close());
 }
 else if(c=="ambiguous_cleanup") {
  // Retention is intentional. _Exit leaves handles/owner retained; this
  // checks refusal to retry, not successful reclamation or leak freedom.
  check_success();fake_hip::event_free_fail=1;assert(!probe->close());
  auto calls=fake_hip::event_free_calls;assert(!probe->close());assert(fake_hip::event_free_calls==calls&&owner.use_count()==2);
  std::cout<<"PASS "<<c<<" (intentional fail-closed retention)\n";
  std::cout.flush();std::_Exit(0); // no destructor makes this success by retrying.
 }
 else {return 2;}
 assert(probe->close());probe.reset();owner.reset();assert(fake_hip::allocations.empty());std::cout<<"PASS "<<c<<"\n";
}
