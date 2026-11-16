#include "../../src/weight/native_runtime.hpp"
#include <cstring>
#include <cstdlib>
namespace {
struct Api {
    using Error=int;
    struct Properties { int warpSize=32; std::size_t sharedMemPerMultiprocessor=65536; };
    struct EventData {};
    using Event=EventData*; using Stream=void*;
    static constexpr auto backend=strix::weight::Backend::cuda;
    static constexpr int Success=0,EventDisableTiming=1,MemcpyHostToDevice=2;
    inline static int allocations=0,frees=0,events=0;
    inline static bool complete=false;
    static std::string arch(const Properties&) { return "fake-arch"; }
    static int SetDevice(int) { return 0; }
    static int GetDeviceProperties(Properties* p,int) { *p={}; return 0; }
    static int MemGetInfo(std::size_t* free,std::size_t* total) { *free=*total=1024; return 0; }
    static int Malloc(void** p,std::size_t bytes) { *p=std::malloc(bytes); ++allocations; return *p?0:1; }
    static int Free(void* p) { std::free(p); ++frees; return 0; }
    static int EventCreateWithFlags(Event* e,int) { *e=new EventData; ++events; return 0; }
    static int EventRecord(Event,Stream) { return 0; }
    static int EventQuery(Event) { return complete?0:1; }
    static int EventDestroy(Event e) { delete e; --events; return 0; }
    static int StreamWaitEvent(Stream,Event,int) { return 0; }
    static int Memcpy(void* dst,const void* src,std::size_t bytes,int) { std::memcpy(dst,src,bytes); return 0; }
    static const char* GetErrorString(int) { return "fake device error"; }
};
}
bool wfmt_runtime_test() {
    using namespace strix::weight;
    ExecutionContext context; Hardware hardware;
    using Runtime=detail::NativeRuntime<Api>;
    if (!Runtime::bind(nullptr,0,context,hardware) || context.backend!=Backend::cuda || hardware.arch!="fake-arch") return false;
    if (!hardware.compiler_id.size() || hardware.memory_bytes_per_second!=0) return false;
    void* data=nullptr;
    if (!context.callbacks->allocate(16,data) || !data) return false;
    void* event=nullptr; if (!context.callbacks->record_event(context,event)) return false;
    context.callbacks->retire(data,{event},context);
    if (Api::frees || Api::events!=2) return false;
    Api::complete=true;
    Runtime::collect();
    return Api::allocations==1 && Api::frees==1 && Api::events==0;
}
