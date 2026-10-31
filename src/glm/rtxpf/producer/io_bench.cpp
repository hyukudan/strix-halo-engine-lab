// CPU-only, read-only ABBA microbench. No hash, scan, model copy or CUDA call.
// This executable cannot issue qualification/state/model-identity receipts.
#define STRIX_IO_BENCHMARK
#include "tier_reader.hpp"
#include <chrono>
#include <iostream>
#include <optional>
using namespace strix::glm::rtxpf::producer;
void glm_io_bench_observe(const void*, unsigned long long);
double cpu_ms(){
    FILETIME created,exited,kernel,user;
    io_need(GetProcessTimes(GetCurrentProcess(),&created,&exited,&kernel,&user)!=0,"CPU time");
    ULARGE_INTEGER k{},u{};
    k.LowPart=kernel.dwLowDateTime;k.HighPart=kernel.dwHighDateTime;
    u.LowPart=user.dwLowDateTime;u.HighPart=user.dwHighDateTime;
    return double(k.QuadPart+u.QuadPart)/10000;
}
int main(int argc,char** argv){try{
    io_need(argc==3,"usage: glm_native_io_bench SHARD BYTES_PER_ARM (ABBA; aggregate <=4 GiB)");
    SetPriorityClass(GetCurrentProcess(),IDLE_PRIORITY_CLASS);
    auto bytes=std::stoull(argv[2]);
    // The legacy cursor fills whole 16 MiB windows. A 16 MiB-aligned budget
    // makes both arms read exactly [0, bytes): no cursor overread.
    constexpr std::uint64_t legacy_window=16ULL<<20;
    io_need(bytes&&bytes<=(1ULL<<30)&&bytes%legacy_window==0,"read budget must be a 16 MiB multiple <= 1 GiB");
    auto path=std::filesystem::path(argv[1]);auto size=std::filesystem::file_size(path);
    io_need(bytes<=size,"shard too small");
    VerifiedFile primary(path,size);primary.admit_unhashed_benchmark();
    constexpr std::size_t ticket=32ULL<<20;
    TierReader reader(primary);AlignedMemory ring(16*ticket),staging(io_chunk);
    std::memset(ring.p,0,16*ticket);std::memset(staging.p,0,io_chunk);glm_io_bench_observe(ring.p,16*ticket);
    for(unsigned arm=0;arm<4;++arm){
        const bool direct=arm==1||arm==2;
        // Both arms are prepared before timing: shared prefaulted ring and
        // staging; a legacy arm gets a fresh prefaulted cursor with no cached
        // window, as each direct arm starts from offset 0 with no state.
        std::optional<VerifiedFile::Reader> cursor;if(!direct){cursor.emplace(primary,legacy_window);cursor->prefault();}
        DriveStats before;reader.counters(before);auto legacy_before=primary.benchmark_counters();std::uint64_t requests=0;
        double cpu=cpu_ms();auto start=std::chrono::steady_clock::now();
        if(!direct){
            // Legacy: two 8 MiB reads fill a 16 MiB cursor; copy cursor ->
            // staging -> consumer ring. Finite scan omitted (conservative).
            for(std::uint64_t off=0;off<bytes;off+=io_chunk){
                cursor->read_at(off,staging.p,io_chunk);
                std::memcpy(ring.p,staging.p,io_chunk);
                glm_io_bench_observe(ring.p,io_chunk);++requests;
            }
        }else for(std::uint64_t off=0;off<bytes;){
            std::vector<ss_mfile_io> ios;
            for(unsigned i=0;i<16&&off<bytes;++i){
                auto n=std::min<std::uint64_t>(ticket,bytes-off);
                ios.push_back({static_cast<char*>(ring.p)+i*ticket,off,n});
                off+=n;++requests;
            }
            reader.read(ios);glm_io_bench_observe(ring.p,ios.size()*ticket);
        }
        auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        auto out=strix::serve::Json::object();DriveStats after;reader.counters(after);
        if(!direct){auto drive=path.root_name().string();before.clear();after.clear();before[drive]=legacy_before;after[drive]=primary.benchmark_counters();}
        out.set("arm",direct?"direct-preadv-16":"old-window-two-copies");out.set("sequence",int(arm));
        out.set("bytes",bytes);out.set("wall_ms",ms);out.set("MB_s",double(bytes)/ms/1000);
        out.set("cpu_ms",cpu_ms()-cpu);out.set("logical_requests",requests);
        out.set("drive_io",drive_receipt(after,before));
        out.set("legacy_ReadFile_requests",direct?0:requests);
        out.set("ReadFile_requests",direct?bytes/(4ULL<<20):primary.benchmark_counters()[2]-legacy_before[2]);
        out.set("readfile_count_basis",direct?"inferred from pinned ABI 4 MiB subreads":"instrumented held-handle submissions");
        out.set("hashed",false);out.set("cuda",false);
        out.set("host_memory","prefaulted VirtualAlloc aligned; no CUDA registration");out.set("setup","buffers and cursor prepared and prefaulted before timing; untimed");
        out.set("scope","held-object CPU throughput; no H2D or model qualification");
        std::cout<<out.dump()<<std::endl;
    }
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
