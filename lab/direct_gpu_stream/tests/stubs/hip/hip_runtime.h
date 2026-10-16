#pragma once
// Deterministic CPU API double. Not a GPU, cache or DMA simulator.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <set>
#define __global__
#define __host__
#define __device__
struct dim3 { unsigned x, y, z; explicit dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
inline dim3 blockIdx, blockDim, threadIdx, gridDim;
using hipStream_t = void*;
using hipEvent_t = void*;
using hipError_t = int;
inline constexpr hipError_t hipSuccess=0,hipErrorNotReady=1,hipErrorUnknown=2;
inline constexpr unsigned hipHostMallocDefault=0,hipEventDisableTiming=2,hipEventReleaseToSystem=0x80000000U;
enum hipMemcpyKind { hipMemcpyHostToDevice, hipMemcpyDeviceToHost, hipMemcpyDeviceToDevice };
namespace fake_hip {
inline std::set<void*> allocations;
inline unsigned events=0,event_flags=0;
inline int query_status=hipSuccess,queries_until_ready=0,fail_next=0;
inline std::size_t readback_bytes=0;
inline bool omit_verify=false;
inline bool fail() { if(fail_next && --fail_next==0)return true;return false; }
template<class Function,class...Args> void launch(Function function,dim3 grid,dim3 block,Args...args) {
    gridDim=grid;blockDim=block;
    if(omit_verify&&block.x==256)return;
    for(unsigned b=0;b<grid.x;++b)for(unsigned t=0;t<block.x;++t){blockIdx.x=b;threadIdx.x=t;function(args...);}
}
}
#define hipLaunchKernelGGL(kernel,grid,block,shared,stream,...) \
    do { (void)(shared); (void)(stream); fake_hip::launch(kernel,grid,block,__VA_ARGS__); } while(0)
inline unsigned long long atomicAdd(unsigned long long* p,unsigned long long n){auto old=*p;*p+=n;return old;}
inline unsigned long long atomicMin(unsigned long long* p,unsigned long long n){auto old=*p;*p=std::min(*p,n);return old;}
inline hipError_t hipMalloc(void** p,std::size_t n){if(fake_hip::fail())return hipErrorUnknown;*p=std::malloc(n);assert(*p);fake_hip::allocations.insert(*p);return hipSuccess;}
inline hipError_t hipHostMalloc(void** p,std::size_t n,unsigned){return hipMalloc(p,n);}
inline hipError_t hipFree(void* p){if(fake_hip::fail())return hipErrorUnknown;assert(fake_hip::allocations.erase(p));std::free(p);return hipSuccess;}
inline hipError_t hipHostFree(void* p){return hipFree(p);}
inline hipError_t hipEventCreateWithFlags(hipEvent_t* e,unsigned f){if(fake_hip::fail())return hipErrorUnknown;*e=new unsigned{f};++fake_hip::events;fake_hip::event_flags=f;return hipSuccess;}
inline hipError_t hipEventDestroy(hipEvent_t e){if(fake_hip::fail())return hipErrorUnknown;delete static_cast<unsigned*>(e);--fake_hip::events;return hipSuccess;}
inline hipError_t hipEventRecord(hipEvent_t,hipStream_t){return fake_hip::fail()?hipErrorUnknown:hipSuccess;}
inline hipError_t hipEventQuery(hipEvent_t){if(fake_hip::queries_until_ready>0){--fake_hip::queries_until_ready;return hipErrorNotReady;}return fake_hip::query_status;}
inline hipError_t hipMemcpyAsync(void* a,const void* b,std::size_t n,hipMemcpyKind kind,hipStream_t){if(fake_hip::fail())return hipErrorUnknown;std::memcpy(a,b,n);if(kind==hipMemcpyDeviceToHost)fake_hip::readback_bytes+=n;return hipSuccess;}
inline hipError_t hipGetLastError(){return fake_hip::fail()?hipErrorUnknown:hipSuccess;}
inline const char* hipGetErrorString(hipError_t){return "injected test error";}

struct hipDeviceProp_t { char gcnArchName[256]; };
hipError_t hipSetDevice(int);
hipError_t hipGetDeviceProperties(hipDeviceProp_t*,int);
