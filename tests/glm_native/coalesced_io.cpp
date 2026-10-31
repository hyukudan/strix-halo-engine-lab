// Synthetic host-only reader qualification. All fixture files stay <100 MB.
#define STRIX_HASH_AUTHORITY_TESTING
#include "glm/rtxpf/producer/tier_reader.hpp"
#include "glm/rtxpf/producer/coalesced_plan.hpp"
#include "glm/rtxpf/producer/read_ahead.hpp"
#include "glm/rtxpf/producer/policy.hpp"
#include "glm/rtxpf/producer/finite_weights.hpp"
#include "xxh64.hpp"
#include <iostream>
#include <random>
#include <condition_variable>
#include <source_location>
using namespace strix::glm::rtxpf::producer;
void check(bool b,std::source_location where=std::source_location::current()){if(!b)throw std::runtime_error("coalesced IO check failed at "+std::to_string(where.line()));}
template<class F> void rejects(F f){bool rejected=false;try{f();}catch(...){rejected=true;}check(rejected);}
// Only the named refusal counts; any other failure is a test failure.
template<class F> void rejects_with(F f,const char* reason){
    try{f();}catch(const std::exception& e){if(std::string(e.what()).find(reason)!=std::string::npos)return;throw std::runtime_error(std::string("unexpected rejection: ")+e.what());}
    throw std::runtime_error(std::string("not rejected: ")+reason);}
// Mount-point reparse points need no symlink privilege (as in verified_io.cpp).
void junction(const std::filesystem::path& link,const std::filesystem::path& target){
    std::filesystem::create_directory(link);
    struct Reparse {DWORD tag;USHORT length,reserved,sub_offset,sub_length,print_offset,print_length;wchar_t paths[4096];} data{};
    auto native=std::filesystem::absolute(target).make_preferred().wstring();
    auto substitute=L"\\??\\"+native;auto print=native;
    data.tag=IO_REPARSE_TAG_MOUNT_POINT;data.sub_length=USHORT(substitute.size()*2);data.print_offset=data.sub_length+2;data.print_length=USHORT(print.size()*2);
    data.length=USHORT(8+data.print_offset+data.print_length+2);check(data.length+8<=sizeof data);
    std::memcpy(data.paths,substitute.c_str(),data.sub_length+2);std::memcpy(reinterpret_cast<char*>(data.paths)+data.print_offset,print.c_str(),data.print_length+2);
    WinHandle handle;handle.h=CreateFileW(link.c_str(),GENERIC_WRITE,0,nullptr,OPEN_EXISTING,FILE_FLAG_OPEN_REPARSE_POINT|FILE_FLAG_BACKUP_SEMANTICS,nullptr);
    DWORD bytes=0;check(handle.h!=INVALID_HANDLE_VALUE&&DeviceIoControl(handle.h,FSCTL_SET_REPARSE_POINT,&data,data.length+8,nullptr,0,&bytes,nullptr));
}
int main(int argc,char** argv){try{
    check(argc==2);auto root=std::filesystem::path(argv[1])/"coalesced-fixtures";std::filesystem::create_directories(root);
    auto path=root/"source.bin",wrong=root/"wrong.bin";std::mt19937_64 rng(97117);constexpr unsigned n=37*1024*1024+124;
    std::vector<char> reference(n);for(auto& x:reference)x=char(rng());
    {std::ofstream f(path,std::ios::binary);f.write(reference.data(),reference.size());
        auto different=reference;different[0]^=1;std::ofstream w(wrong,std::ios::binary);w.write(different.data(),different.size());}
    {
        VerifiedFile file(path,n),other(wrong,n);rejects([&]{TierReader unverified(file);});
        auto digest=file.compute_or_recall(root/"hashes");TierReader reader(file);
        rejects([&]{same_file_object(file.held_handle(),other.held_handle());});
        WinHandle writer;writer.h=CreateFileW(path.c_str(),GENERIC_WRITE,FILE_SHARE_READ,nullptr,OPEN_EXISTING,0,nullptr);
        check(writer.h==INVALID_HANDLE_VALUE&&GetLastError()==ERROR_SHARING_VIOLATION);
        auto replacement=path;replacement+=".replaced";
        check(!std::filesystem::exists(replacement));check(!MoveFileExW(path.c_str(),replacement.c_str(),0));
        // The adapter keeps both serving file and parent namespace immutable.
        auto renamed=root;renamed+="-replaced";check(!std::filesystem::exists(renamed));check(!MoveFileExW(root.c_str(),renamed.c_str(),0));
        WinHandle directory_writer;directory_writer.h=CreateFileW(root.c_str(),GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,
            nullptr,OPEN_EXISTING,FILE_FLAG_BACKUP_SEMANTICS|FILE_FLAG_OPEN_REPARSE_POINT,nullptr);
        check(directory_writer.h==INVALID_HANDLE_VALUE&&GetLastError()==ERROR_SHARING_VIOLATION);
        for(unsigned trial=0;trial<80;++trial){std::vector<SourceRange> ranges;std::uint64_t packed=0;
            for(unsigned j=0;j<1+trial%31;++j){auto source=rng()%(n-1),bytes=1+rng()%std::min<std::uint64_t>(n-source,1ULL<<20);
                ranges.push_back({source,packed,bytes});packed+=bytes+256;}
            CoalescedPlan plan(ranges,n);AlignedMemory raw(std::size_t(plan.bytes));std::vector<ss_mfile_io> ios;
            std::uint64_t last=0;for(auto r:plan.reads){check(r.source>=last&&r.bytes>=CoalescedPlan::minimum&&r.source%65536==0&&r.bytes%4096==0);last=r.source+r.bytes;
                ios.push_back({static_cast<char*>(raw.p)+r.buffer,r.source,r.bytes});}reader.read(ios);
            std::vector<char> actual(std::size_t(packed),0),expected(actual);
            for(auto r:ranges)std::memcpy(expected.data()+r.destination,reference.data()+r.source,std::size_t(r.bytes));
            for(auto s:plan.slices)std::memcpy(actual.data()+s.destination,static_cast<char*>(raw.p)+s.buffer,std::size_t(s.bytes));check(actual==expected);
        }
        // An admitted digest is tied to its object; even an authentic record
        // from another handle must never authorize a wrong expected digest.
        other.compute_or_recall(root/"hashes");
        rejects([&]{other.verify(digest.sha256,root/"hashes");});rejects([&]{TierReader invalid(other);});
        DriveStats stats;reader.counters(stats);check(!stats.empty()&&stats.begin()->second[0]>0);
        // Secondary manifests/xxh64 are not an integrity authority. A valid
        // manifest describing corrupted bytes must still fail primary equality.
        auto secondary=root/"replica";std::filesystem::create_directory(secondary);
        constexpr std::uint64_t chunk=8ULL<<20;
        ts::StripeManifest manifest;manifest.chunk_bytes=chunk;
        manifest.primary_basename="source.bin";manifest.primary_size=n;manifest.primary_sha256=digest.sha256;
        manifest.data_name="source.stripe.bin";manifest.data_size=chunk;manifest.runs={{1,1,0}};
        auto publish=[&](bool corrupt){
            std::vector<char> data(reference.begin()+chunk,reference.begin()+2*chunk);if(corrupt)data[0]^=1;
            std::ofstream f(secondary/manifest.data_name,std::ios::binary);f.write(data.data(),data.size());f.close();
            manifest.hashes={ts::xxh64(data.data(),data.size())};
            std::ofstream(secondary/ts::stripe_manifest_name("source.bin"))<<ts::stripe_manifest_write(manifest);
        };
        publish(false);
        {TierReader multi(file,path_utf8(secondary));AlignedMemory output(4*chunk);
            multi.read({{output.p,0,4*chunk}});check(!std::memcmp(output.p,reference.data(),4*chunk));}
        publish(true);rejects([&]{TierReader corrupt(file,path_utf8(secondary));});
        // Two manifests with different mappings over ONE data FileId. Only the
        // admitted mapping may serve; the library must parse exactly its bytes.
        {
            auto honest=root/"tier-honest",swapped=root/"tier-swapped",sub=swapped/"sub",alias=root/"tier-alias";
            // A failed earlier run may leave fixtures behind; links make reuse fail.
            {std::error_code ec;if(std::filesystem::symlink_status(alias,ec).type()!=std::filesystem::file_type::not_found)RemoveDirectoryW(alias.c_str());}
            for(auto& d:{honest,swapped})std::filesystem::remove_all(d);
            for(auto& d:{honest,sub})std::filesystem::create_directories(d);
            std::vector<char> data(reference.begin()+chunk,reference.begin()+3*chunk);
            {std::ofstream f(sub/"source.stripe.bin",std::ios::binary);f.write(data.data(),data.size());}
            check(CreateHardLinkW((honest/"source.stripe.bin").c_str(),(sub/"source.stripe.bin").c_str(),nullptr)!=0);
            ts::StripeManifest admitted=manifest;admitted.data_size=2*chunk;admitted.runs={{1,2,0}};
            admitted.hashes={ts::xxh64(data.data(),chunk),ts::xxh64(data.data()+chunk,chunk)};
            auto substituted=admitted;substituted.runs={{2,2,0}};  // chunk 2 <- chunk 1's slot: self-consistent xxh64
            auto put=[](const std::filesystem::path& p,const std::string& s){std::ofstream f(p,std::ios::binary);f<<s;};
            const auto canonical_name=ts::stripe_manifest_name("source.bin");
            put(honest/canonical_name,ts::stripe_manifest_write(admitted));put(sub/"admitted.json",ts::stripe_manifest_write(admitted));
            put(sub/canonical_name,ts::stripe_manifest_write(substituted));
            {VerifiedFile a(honest/"source.stripe.bin",2*chunk),b(sub/"source.stripe.bin",2*chunk);same_file_object(a.held_handle(),b.held_handle());}
            // Reads and equality checks never sit inside a permitted-refusal catch:
            // wrong served bytes must FAIL this test.
            auto serves_reference=[&](TierReader& multi){AlignedMemory output(5*chunk);
                multi.read({{output.p,0,(n+4095ULL)/4096*4096}});check(!std::memcmp(output.p,reference.data(),n));};
            {TierReader multi(file,path_utf8(honest));serves_reference(multi);}
            rejects_with([&]{TierReader substitution(file,path_utf8(sub));},"stripe differs from authenticated primary");
            // The library's actual name must be the admitted held object with
            // identical bytes; a sibling directory's manifest never qualifies.
            {VerifiedFile held(honest/canonical_name,std::filesystem::file_size(honest/canonical_name));auto text=held.contents();
                auto primary_name=path_utf8(path);
                TierReader::check_parsed_manifest(path_utf8(honest/"source.stripe.bin"),primary_name,held,text);
                rejects([&]{TierReader::check_parsed_manifest(path_utf8(sub/"source.stripe.bin"),primary_name,held,text);});
                VerifiedFile other(sub/canonical_name,std::filesystem::file_size(sub/canonical_name));
                rejects([&]{TierReader::check_parsed_manifest(path_utf8(sub/"source.stripe.bin"),primary_name,other,text);});}
            // Final-component symlink to an admitted manifest beside a
            // substituted canonical one. Only a missing symlink privilege may
            // skip; a qualification host with symlink rights must run it.
            auto leaf=swapped/canonical_name;check(CreateHardLinkW((swapped/"source.stripe.bin").c_str(),(sub/"source.stripe.bin").c_str(),nullptr)!=0);
            if(CreateSymbolicLinkW(leaf.c_str(),L"sub\\admitted.json",SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE)){
                rejects_with([&]{TierReader link(file,path_utf8(swapped));},"stripe manifest must be a regular non-reparse file");
                check(DeleteFileW(leaf.c_str())!=0);std::cout<<"symlink leaf case PASS\n";
            }else{auto error=GetLastError();check(error==ERROR_PRIVILEGE_NOT_HELD);std::cout<<"symlink leaf case skipped: ERROR_PRIVILEGE_NOT_HELD\n";}
            // A hard-linked alternate name: the only permitted outcome other than
            // serving the admitted mapping is the canonical-name refusal.
            check(CreateHardLinkW(leaf.c_str(),(sub/"admitted.json").c_str(),nullptr)!=0);
            std::unique_ptr<TierReader> linked;
            try{linked=std::make_unique<TierReader>(file,path_utf8(swapped));}
            catch(const std::runtime_error& e){check(std::string(e.what())=="stripe manifest must have its canonical name in the supplied directory");}
            if(linked)serves_reference(*linked);
            std::cout<<"hard-linked manifest name "<<(linked?"served the admitted mapping":"refused by canonical-name check")<<"\n";linked.reset();
            // Junction ancestors are refused even when the target is honest.
            junction(alias,honest);rejects_with([&]{TierReader through(file,path_utf8(alias));},"source directory reparse refused");
            rejects_with([&]{TierReader through(file,path_utf8(alias/"."));},"source directory reparse refused");
            check(RemoveDirectoryW(alias.c_str())!=0);
            // Case-sensitive ancestry is refused: the pinned library lowercases
            // paths, so `Tier` (locked) and `tier` (not) could be different.
            check(!case_sensitivity_refused(true,ERROR_SUCCESS,0));
            check(case_sensitivity_refused(true,ERROR_SUCCESS,FILE_CS_FLAG_CASE_SENSITIVE_DIR));
            for(DWORD unsupported:{DWORD(ERROR_INVALID_PARAMETER),DWORD(ERROR_NOT_SUPPORTED),DWORD(ERROR_INVALID_FUNCTION)})check(!case_sensitivity_refused(false,unsupported,0));
            for(DWORD failure:{DWORD(ERROR_ACCESS_DENIED),DWORD(ERROR_SHARING_VIOLATION),DWORD(ERROR_IO_DEVICE)})check(case_sensitivity_refused(false,failure,0));
            {auto upper=root/"Tier-Case";auto lower=library_name(upper);check(same_local_name(lower,upper)&&path_utf8(lower.filename())=="tier-case");}
            {auto cs=root/"case-sensitive";std::filesystem::remove_all(cs);std::filesystem::create_directory(cs);
                WinHandle h;h.h=CreateFileW(cs.c_str(),FILE_WRITE_ATTRIBUTES|FILE_READ_ATTRIBUTES,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,nullptr,OPEN_EXISTING,FILE_FLAG_BACKUP_SEMANTICS,nullptr);
                check(h.h!=INVALID_HANDLE_VALUE);refuse_case_sensitive(h.h);
                FILE_CASE_SENSITIVE_INFO on{FILE_CS_FLAG_CASE_SENSITIVE_DIR};
                if(SetFileInformationByHandle(h.h,FileCaseSensitiveInfo,&on,sizeof on)){
                    // Locked `Tier` is honest; the unlocked lowercase sibling is not.
                    for(auto name:{"Tier","tier"})std::filesystem::create_directory(cs/name);
                    for(auto name:{"Tier","tier"})check(CreateHardLinkW((cs/name/"source.stripe.bin").c_str(),(sub/"source.stripe.bin").c_str(),nullptr)!=0);
                    put(cs/"Tier"/canonical_name,ts::stripe_manifest_write(admitted));put(cs/"tier"/canonical_name,ts::stripe_manifest_write(substituted));
                    rejects_with([&]{TierReader mixed(file,path_utf8(cs/"Tier"));},"case-sensitive source directory refused");
                    std::cout<<"case-sensitive namespace case PASS\n";
                    CloseHandle(h.h);h.h=INVALID_HANDLE_VALUE;std::filesystem::remove_all(cs);
                }else{auto error=GetLastError();check(error==ERROR_ACCESS_DENIED||error==ERROR_PRIVILEGE_NOT_HELD);
                    std::cout<<"case-sensitive namespace case skipped: setting the flag needs privilege/WSL (error "<<error<<")\n";}
            }
        }
    }
    // Exactly the old finite reject, once at startup. Every byte split of an
    // unaligned scalar is exercised; integer trellis bytes may contain any bits.
    for(auto dtype:{"F16","BF16","F32","I16","I32"}){
        const unsigned width=std::string(dtype)=="F32"||std::string(dtype)=="I32"?4:2;
        const unsigned exponent=std::string(dtype)=="BF16"?0x7f80:width==2?0x7c00:0x7f800000;
        for(unsigned bad:{0u,exponent,exponent|1u})for(unsigned split=1;split<2*width;++split){
            std::array<unsigned char,9> raw{};std::memcpy(raw.data()+1+width,&bad,width);
            auto run=[&]{FiniteWeights scan(dtype);scan.append(raw.data()+1,split);scan.append(raw.data()+1+split,2*width-split);scan.finish();};
            if(bad&&dtype[0]!='I')rejects(run);else run();
        }
    }
    // Sparse file-layer ordinals deliberately have repeated device parity.
    // Two reads start before acquire; slot reuse waits for its H2D fence.
    {
        LayerReadAhead ahead({3,5,9,10,15});std::mutex m;std::condition_variable cv;unsigned started=0;bool fence_open=false,fence_entered=false;unsigned loaded[2]{};
        auto load=[&](unsigned slot,unsigned layer){std::lock_guard lock(m);loaded[slot]=layer;++started;cv.notify_all();};
        ahead.reset(load);{std::unique_lock lock(m);check(cv.wait_for(lock,std::chrono::seconds(5),[&]{return started==2;}));}
        check(ahead.acquire(3)==0);check(ahead.acquire(5)==1);rejects([&]{ahead.acquire(3);});
        rejects([&]{ahead.acquire(9);});
        ahead.release(3,load,[&](unsigned slot){check(slot==0);std::unique_lock lock(m);fence_entered=true;cv.notify_all();cv.wait(lock,[&]{return fence_open;});});
        {std::unique_lock lock(m);bool entered=cv.wait_for(lock,std::chrono::seconds(5),[&]{return fence_entered;});
            bool unchanged=loaded[0]==3;fence_open=true;cv.notify_all();check(entered&&unchanged);}
        ahead.release(5,load,[](unsigned){});check(ahead.acquire(9)==0);ahead.release(9,load,[](unsigned){});
        check(ahead.acquire(10)==1);ahead.release(10,load,[](unsigned){});check(ahead.acquire(15)==0);ahead.release(15,load,[](unsigned){});ahead.drain();
        rejects([&]{ahead.acquire(15);});
        ahead.reset([](unsigned,unsigned){throw std::runtime_error("injected read error");});rejects([&]{ahead.drain();});
    }
    {
        // Later completion may arrive first; acquire must still wait for the
        // demanded ordinal. A failed slot must drain its peer before reset.
        LayerReadAhead ahead({3,4});std::promise<void> unblock,late;auto gate=unblock.get_future().share();
        ahead.reset([&](unsigned slot,unsigned){if(!slot)gate.wait();else late.set_value();});
        late.get_future().wait();check(ahead.acquire(4)==1);
        auto first=std::async(std::launch::async,[&]{return ahead.acquire(3);});
        bool blocked=first.wait_for(std::chrono::milliseconds(20))==std::future_status::timeout;
        unblock.set_value();check(first.get()==0&&blocked);ahead.drain();
        std::atomic<unsigned> completed=0;
        ahead.reset([&](unsigned slot,unsigned){if(!slot)throw std::runtime_error("failed first slot");Sleep(20);++completed;});
        rejects([&]{ahead.drain();});check(completed==1);
    }
    // Cache-mode startup scan: manifest offsets/dtypes per blob, any batch
    // split. A NaN/Inf in a float tensor rejects; integer/padding bytes do not.
    {
        auto manifest=strix::serve::parse_json(R"({"tensors":{
            "a":{"blob":"base.bin","offset":0,"bytes":6,"dtype":"F16"},
            "b":{"blob":"base.bin","offset":256,"bytes":8,"dtype":"F32"},
            "c":{"blob":"base.bin","offset":512,"bytes":4,"dtype":"I16"},
            "d":{"blob":"base.bin","offset":768,"bytes":6,"dtype":"BF16"},
            "e":{"blob":"experts-03.bin","offset":0,"bytes":4,"dtype":"F32"}}})");
        check(blob_finite_ranges(manifest,"base.bin").size()==3&&blob_finite_ranges(manifest,"experts-03.bin").size()==1);
        auto scan_blob=[&](const std::vector<unsigned char>& blob,std::size_t step){
            BlobFiniteScan scan(blob_finite_ranges(manifest,"base.bin"),blob.size());
            for(std::size_t off=0;off<blob.size();off+=step)scan.append(off,blob.data()+off,std::min(step,blob.size()-off));scan.finish();};
        std::vector<unsigned char> blob(1024,0);
        auto poke=[&](std::size_t at,std::uint32_t bits,unsigned width){auto copy=blob;std::memcpy(copy.data()+at,&bits,width);return copy;};
        for(std::size_t step:{1,3,7,257,1024}){
            scan_blob(blob,step);
            rejects([&]{scan_blob(poke(2,0x7e00,2),step);});            // F16 NaN
            rejects([&]{scan_blob(poke(260,0x7f800000u,4),step);});     // F32 Inf
            rejects([&]{scan_blob(poke(257,0x7fc0u<<8,4),step);});      // F32 NaN across a split
            rejects([&]{scan_blob(poke(772,0x7f81,2),step);});          // BF16 NaN
            scan_blob(poke(512,0x7f807f80u,4),step);                    // I16 bits are not floats
            scan_blob(poke(100,0x7f800000u,4),step);                    // padding is not a tensor
        }
        std::vector<unsigned char> blob_bytes(1024,0);
        rejects([&]{BlobFiniteScan scan(blob_finite_ranges(manifest,"base.bin"),1024);scan.append(0,blob_bytes.data(),512);scan.finish();});
        rejects([&]{BlobFiniteScan scan(blob_finite_ranges(manifest,"base.bin"),1024);scan.append(8,blob_bytes.data(),8);});
        rejects([&]{BlobFiniteScan scan(blob_finite_ranges(manifest,"base.bin"),512);});
        rejects([&]{BlobFiniteScan scan({{0,8,"F32"},{4,8,"F32"}},64);});
        std::uint32_t nan=0x7fc00000u;rejects([&]{BlobFiniteScan scan(blob_finite_ranges(manifest,"experts-03.bin"),4);scan.append(0,&nan,4);scan.finish();});
    }
    rejects([]{CoalescedPlan({{UINT64_MAX-1,0,8}},42);});
    check(host_budget(8ULL<<30,32ULL<<30,8)==0);check(!admit(14ULL<<30,host_budget(22ULL<<30,64ULL<<30,8),1ULL<<30));
    check(file_cache_admits(14ULL<<30,{24ULL<<30,24ULL<<30,24ULL<<30},{32ULL<<30,32ULL<<30,32ULL<<30},8));
    check(!file_cache_admits(14ULL<<30,{22ULL<<30,22ULL<<30,22ULL<<30},{32ULL<<30,32ULL<<30,32ULL<<30},8));
    check(!file_cache_admits(1ULL<<30,{24ULL<<30,23ULL<<30,24ULL<<30},{32ULL<<30,32ULL<<30,32ULL<<30},8));
    check(!file_cache_admits(1ULL<<30,{24ULL<<30,24ULL<<30,24ULL<<30},{32ULL<<30,31ULL<<30,32ULL<<30},8));
    auto renamed=root;renamed+="-replaced";std::filesystem::rename(root,renamed);std::filesystem::rename(renamed,root);
    std::filesystem::remove_all(root);std::cout<<"coalesced byte identity 80 random sets; held identity; manifest binding over one data FileId; ring ordering PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
