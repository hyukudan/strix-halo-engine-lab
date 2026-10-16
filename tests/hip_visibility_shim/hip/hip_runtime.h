#pragma once
// Test double only: sequential C++ execution is NOT GPU cache/DMA validation.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <chrono>
#define __global__
#define __host__
#define __device__
struct dim3 { unsigned x,y,z; explicit dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
inline dim3 blockIdx(0), blockDim(1), threadIdx(0), gridDim(1);
using hipError_t=int;
using hipDevice_t=int;
struct FakeHipStream { int device=0; bool busy=false; bool captured=false; };
struct FakeHipEvent { bool recorded=false; };
using hipStream_t=FakeHipStream*;
using hipEvent_t=FakeHipEvent*;
inline constexpr int hipSuccess=0,hipErrorNotReady=1,hipErrorUnknown=2;
inline constexpr unsigned hipEventDisableTiming=2,hipHostMallocDefault=0;
inline constexpr int hipMemoryTypeDevice=1,hipMemoryTypeHost=2;
enum hipMemcpyKind { hipMemcpyDeviceToHost,hipMemcpyDeviceToDevice,hipMemcpyHostToDevice };
enum hipStreamCaptureStatus { hipStreamCaptureStatusNone,hipStreamCaptureStatusActive };
struct hipPointerAttribute_t { int type=0,device=0,isManaged=0; };
namespace fake_hip {
struct Allocation { std::size_t size; bool host=false; int device=0; bool managed=false; };
inline std::map<std::uintptr_t,Allocation> allocations;
inline int device=0,event_flags=-1,record_fail=0,query_error=0,launch_error=0,allocation_fail_after=-1;
inline int free_fail=0,event_free_fail=0,host_free_fail=0,free_calls=0,event_free_calls=0;
inline bool never_complete=false,corrupt_result=false;
inline std::uint64_t copied_bytes=0;
inline FakeHipStream stream;
inline const char* selected_case="";
inline auto lookup(std::uintptr_t address) {
 auto it=allocations.upper_bound(address);
 if(it==allocations.begin()) return allocations.end();
 --it;
 return address-it->first<it->second.size?it:allocations.end();
}
inline hipError_t allocate(void** out,std::size_t bytes,bool host) {
 if(allocation_fail_after==0){allocation_fail_after=-1;return hipErrorUnknown;}
 if(allocation_fail_after>0)--allocation_fail_after;
 *out=std::calloc(1,bytes); if(!*out)return hipErrorUnknown;
 allocations.emplace(reinterpret_cast<std::uintptr_t>(*out),Allocation{bytes,host,device,false});return hipSuccess;
}
inline hipError_t deallocate(void* p,int& fail) {
 ++free_calls;
 if(fail){--fail;return hipErrorUnknown;}
 auto it=allocations.find(reinterpret_cast<std::uintptr_t>(p));
 if(it==allocations.end())return hipErrorUnknown;
 allocations.erase(it);std::free(p);return hipSuccess;
}
}
inline const char* hipGetErrorString(hipError_t){return "simulated error";}
inline hipError_t hipMalloc(void** p,std::size_t n){return fake_hip::allocate(p,n,false);}
inline hipError_t hipHostMalloc(void** p,std::size_t n,unsigned){return fake_hip::allocate(p,n,true);}
inline hipError_t hipFree(void* p){return fake_hip::deallocate(p,fake_hip::free_fail);}
inline hipError_t hipHostFree(void* p){return fake_hip::deallocate(p,fake_hip::host_free_fail);}
inline hipError_t hipGetDevice(int* d){*d=fake_hip::device;return hipSuccess;}
inline hipError_t hipPointerGetAttributes(hipPointerAttribute_t* out,void* p){
 auto it=fake_hip::lookup(reinterpret_cast<std::uintptr_t>(p));if(it==fake_hip::allocations.end())return hipErrorUnknown;
 *out={it->second.host?hipMemoryTypeHost:hipMemoryTypeDevice,it->second.device,it->second.managed};return hipSuccess;
}
inline hipError_t hipMemGetAddressRange(void** p,std::size_t* n,void* address){
 auto it=fake_hip::lookup(reinterpret_cast<std::uintptr_t>(address));if(it==fake_hip::allocations.end())return hipErrorUnknown;
 *p=reinterpret_cast<void*>(it->first);*n=it->second.size;return hipSuccess;
}
inline hipError_t hipStreamGetDevice(hipStream_t s,hipDevice_t* d){if(!s)return hipErrorUnknown;*d=s->device;return hipSuccess;}
inline hipError_t hipStreamIsCapturing(hipStream_t s,hipStreamCaptureStatus* c){if(!s)return hipErrorUnknown;*c=s->captured?hipStreamCaptureStatusActive:hipStreamCaptureStatusNone;return hipSuccess;}
inline hipError_t hipStreamQuery(hipStream_t s){return s&&!s->busy?hipSuccess:hipErrorNotReady;}
inline hipError_t hipEventCreateWithFlags(hipEvent_t* e,unsigned flags){fake_hip::event_flags=static_cast<int>(flags);*e=new FakeHipEvent;return hipSuccess;}
inline hipError_t hipEventRecord(hipEvent_t e,hipStream_t){if(fake_hip::record_fail){--fake_hip::record_fail;return hipErrorUnknown;}e->recorded=true;return hipSuccess;}
inline hipError_t hipEventQuery(hipEvent_t e){
 if(fake_hip::query_error)return fake_hip::query_error;
 return e->recorded&&!fake_hip::never_complete?hipSuccess:hipErrorNotReady;
}
inline hipError_t hipEventDestroy(hipEvent_t e){++fake_hip::event_free_calls;if(fake_hip::event_free_fail){--fake_hip::event_free_fail;return hipErrorUnknown;}delete e;return hipSuccess;}
inline hipError_t hipGetLastError(){auto e=fake_hip::launch_error;fake_hip::launch_error=0;return e;}
inline hipError_t hipMemcpyAsync(void* to,const void* from,std::size_t bytes,hipMemcpyKind kind,hipStream_t){
 if(kind==hipMemcpyDeviceToHost)fake_hip::copied_bytes+=bytes;
 std::memcpy(to,from,bytes);
 if(fake_hip::corrupt_result&&bytes==64)static_cast<std::uint64_t*>(to)[2]^=1;
 return hipSuccess;
}
#define hipLaunchKernelGGL(kernel,grid,block,shared,stream,...) do { (void)(grid);(void)(block);(void)(shared);(void)(stream); kernel(__VA_ARGS__); } while(0)
inline unsigned long long atomicAdd(unsigned long long* p,unsigned long long v){auto old=*p;*p+=v;return old;}
inline unsigned long long atomicMin(unsigned long long* p,unsigned long long v){auto old=*p;if(v<*p)*p=v;return old;}
inline unsigned long long atomicXor(unsigned long long* p,unsigned long long v){auto old=*p;*p^=v;return old;}
