#include "../lab/gpu_queue/native_memory.hpp"
#include <cassert>
#include <iostream>
#include <string>
using namespace strix::transport;
template<class Function> void rejects(Function fn){bool rejected=false;try{fn();}catch(const std::exception&){rejected=true;}assert(rejected);}
int main(int argc,char** argv){
    assert(argc==2);const std::string name=argv[1];
    if(name=="null_lease"){rejects([]{queue_lab::QueueGpuMemory memory({});});}
    else {
        auto lease=std::make_shared<DirectGpuExportImportLease>();
        queue_lab::QueueGpuMemory memory(lease);AllocationGenerationSource generations;
        if(name=="bounds"){
            for(unsigned depth:{0U,1U,9U})rejects([&]{memory.allocate(depth,4097,generations,std::chrono::milliseconds{1000});});
            rejects([&]{memory.allocate(8,16ULL*1024*1024,generations,std::chrono::milliseconds{1000});});
            rejects([&]{memory.allocate(8,4097,generations,std::chrono::milliseconds{0});});
            assert(fake_hip::allocations.empty());
        }else if(name=="partial_allocation"){
            fake_hip::allocation_fail_after=5;
            rejects([&]{memory.allocate(8,4097,generations,std::chrono::milliseconds{1000});});
            assert(memory.close());assert(fake_hip::allocations.empty());
        }else{
            memory.allocate(8,4097,generations,std::chrono::milliseconds{10});
            assert(fake_hip::allocations.size()==32);assert(memory.quiescent());
            if(name=="normal")for(unsigned n=0;n<8;++n){assert(memory.slot(n).stream);memory.wait(n);}
            else if(name=="busy"){
                memory.slot(3).stream->busy=true;assert(!memory.quiescent());assert(!memory.close());
                assert(fake_hip::allocations.size()==32);memory.slot(3).stream->busy=false;
            }else if(name=="capture"){
                memory.slot(2).stream->captured=true;assert(!memory.quiescent());memory.slot(2).stream->captured=false;
            }else if(name=="timeout"){
                fake_hip::never_complete=true;rejects([&]{memory.wait(0);});fake_hip::never_complete=false;
            }else if(name=="device"){
                fake_hip::device=1;assert(!memory.quiescent());fake_hip::device=0;
            }else if(name=="repeat_allocation"){
                rejects([&]{memory.allocate(8,4097,generations,std::chrono::milliseconds{10});});
                assert(fake_hip::allocations.size()==32);
            }else assert(false);
            assert(memory.close());assert(memory.close());assert(fake_hip::allocations.empty());
        }
    }
    std::cout<<"PASS "<<name<<" with HIP/export-observation doubles\n";
}
