// Compile the COMPLETE modified native translation unit with declared API
// doubles; execute only parser and isolated warmup logic. This is not hipcc.
#define main unused_commission_main
#include "../commission_main.hip"
#undef main
#include <cassert>

namespace strix {
std::uint64_t AllocationGenerationSource::next() { static std::uint64_t n=100;return ++n; }
}
namespace strix::transport {
DirectGpuExportDevice HipDirectGpuExportIo::inspect_device() { return {}; }
}

int main(int argc,char** argv) {
    if(argc==2 && std::string_view(argv[1])=="warm") {
        for(const auto bytes:{1U,257U,4097U}) {
            auto owner=std::make_shared<ExecutionOwner>();
            owner->bytes=bytes;
            AllocationGenerationSource generations;
            owner->allocate(generations,true);
            assert(!owner->readback && !owner->readback_event && owner->views.size()==4);
            std::vector<std::uint8_t> raw(bytes);
            TransportBufferView view{reinterpret_cast<std::uintptr_t>(raw.data()),bytes,1000,TransportMemoryDomain::gpu_device,TransportBufferAccess::read_write};
            Options options;options.bytes=bytes;options.epoch=7;options.gpu_visibility=true;
            owner->warm_and_self_test(reinterpret_cast<void*>(1),view,options);
            assert(owner->visibility_self_test && !owner->quiescent);
            assert(!owner->release()); // Whole helper stream not yet certified idle.
            owner->quiescent=true;assert(owner->release());
            assert(fake_hip::allocations.empty()&&!fake_hip::events);
        }
        std::cout<<"PASS warm\n";return 0;
    }
    try {
        const auto options=parse(argc,argv);
        assert(fake_hip::allocations.empty()&&!fake_hip::events);
        std::cout<<"{\"visibility\":"<<(options.gpu_visibility?"true":"false")<<",\"iterations\":"<<options.iterations<<"}\n";
        return 0;
    } catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
