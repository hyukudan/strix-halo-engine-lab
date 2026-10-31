#pragma once
#include "verified_io.hpp"
#include "strix/serve/json.hpp"
#include "strix_stream/ss.h"
#include "raw_file.hpp"
#include "stripe_manifest.hpp"
#include <map>
#include <array>
#include <climits>
#include <future>

namespace strix::glm::rtxpf::producer {
inline std::string path_utf8(const std::filesystem::path& p){auto s=p.u8string();return {s.begin(),s.end()};}
inline void same_file_object(HANDLE a,HANDLE b){
    FILE_ID_INFO x{},y{};
    io_need(GetFileInformationByHandleEx(a,FileIdInfo,&x,sizeof x)&&GetFileInformationByHandleEx(b,FileIdInfo,&y,sizeof y),"source FileId query");
    io_need(x.VolumeSerialNumber==y.VolumeSerialNumber&&!std::memcmp(&x.FileId,&y.FileId,sizeof x.FileId),"source handle identity mismatch");
}
inline std::filesystem::path held_path(HANDLE h){
    std::wstring s(32768,L'\0');auto n=GetFinalPathNameByHandleW(h,s.data(),DWORD(s.size()),FILE_NAME_NORMALIZED|VOLUME_NAME_DOS);
    io_need(n&&n<s.size(),"held source path");s.resize(n);
    // std::filesystem may treat the extended prefix's device namespace as an
    // additional ancestor. Only local drive paths are admitted by this adapter;
    // strip the prefix before walking actual filesystem directories.
    io_need(s.size()>=7&&s.compare(0,4,L"\\\\?\\")==0&&s[5]==L':'&&s[6]==L'\\',"local held source path required");
    return s.substr(4);
}
inline bool same_local_name(const std::filesystem::path& a,const std::filesystem::path& b){
    auto x=a.lexically_normal().make_preferred().wstring(),y=b.lexically_normal().make_preferred().wstring();
    return x.size()<=INT_MAX&&y.size()<=INT_MAX&&CompareStringOrdinal(x.data(),int(x.size()),y.data(),int(y.size()),TRUE)==CSTR_EQUAL;
}
// 873b004 lowercases every canonical path (mfile.cpp canonical()). Under a
// case-sensitive directory its lowercase name can select a sibling that this
// adapter never locked, so refuse case-sensitive ancestry. Fail closed: only a
// cleared flag, or a filesystem without per-directory case sensitivity, passes.
inline bool case_sensitivity_refused(bool queried,DWORD error,ULONG flags){
    if(queried)return (flags&FILE_CS_FLAG_CASE_SENSITIVE_DIR)!=0;
    return !(error==ERROR_INVALID_PARAMETER||error==ERROR_NOT_SUPPORTED||error==ERROR_INVALID_FUNCTION);
}
inline void refuse_case_sensitive(HANDLE directory){
    FILE_CASE_SENSITIVE_INFO info{};bool queried=GetFileInformationByHandleEx(directory,FileCaseSensitiveInfo,&info,sizeof info)!=0;
    io_need(!case_sensitivity_refused(queried,queried?ERROR_SUCCESS:GetLastError(),info.Flags),"case-sensitive source directory refused");
}
// The exact name the pinned library derives: weakly_canonical(absolute(p)),
// then CharLowerBuffW over the whole path.
inline std::filesystem::path library_name(const std::filesystem::path& p){
    std::error_code ec;auto c=std::filesystem::weakly_canonical(std::filesystem::absolute(p),ec);
    if(ec)c=std::filesystem::absolute(p).lexically_normal();
    auto w=c.wstring();if(!w.empty())CharLowerBuffW(w.data(),DWORD(w.size()));return w;
}
// Before ss_mfile_open: the library's own (lowercased) name must equal the
// bound name case-insensitively and open the same held object, not a reparse.
inline void same_library_object(const std::filesystem::path& library,const std::filesystem::path& bound,HANDLE held){
    io_need(same_local_name(library,bound),"library path differs from bound path");
    WinHandle h;h.h=CreateFileW(library.c_str(),FILE_READ_ATTRIBUTES,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,nullptr,OPEN_EXISTING,FILE_FLAG_BACKUP_SEMANTICS|FILE_FLAG_OPEN_REPARSE_POINT,nullptr);
    io_need(h.h!=INVALID_HANDLE_VALUE,"open library path");FILE_ATTRIBUTE_TAG_INFO info{};
    io_need(GetFileInformationByHandleEx(h.h,FileAttributeTagInfo,&info,sizeof info)&&!(info.FileAttributes&FILE_ATTRIBUTE_REPARSE_POINT),"library path reparse refused");
    same_file_object(held,h.h);
}
// Retain deny-write handles for `dir` and every ancestor, refusing any reparse
// or case-sensitive directory. A held directory cannot gain, lose or rename
// children, or become a reparse.
inline void lock_directory_chain(const std::filesystem::path& dir,std::vector<std::unique_ptr<WinHandle>>& out){
    for(auto p=dir;!p.empty();){
        auto h=std::make_unique<WinHandle>();
        // FILE_LIST_DIRECTORY makes this a sharing-participating read
        // handle: attributes-only handles do not exclude directory writers.
        // Deny writes too, including in-place reparse point installation.
        h->h=CreateFileW(p.c_str(),FILE_LIST_DIRECTORY|FILE_READ_ATTRIBUTES,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_FLAG_BACKUP_SEMANTICS|FILE_FLAG_OPEN_REPARSE_POINT,nullptr);
        io_need(h->h!=INVALID_HANDLE_VALUE,"anchor source directory");FILE_ATTRIBUTE_TAG_INFO info{};
        io_need(GetFileInformationByHandleEx(h->h,FileAttributeTagInfo,&info,sizeof info),"source directory attribute query");
        io_need(!(info.FileAttributes&FILE_ATTRIBUTE_REPARSE_POINT),"source directory reparse refused");
        refuse_case_sensitive(h->h);
        out.push_back(std::move(h));auto next=p.parent_path();if(next==p)break;p=next;
    }
}
// 873b004 has no primary-handle authentication callback. Anchor the final path
// namespace instead: retain deny-write/delete handles for EVERY ancestor, reject
// reparses, and compare a retained deny-write/delete serving guard to the
// authenticated FileId. No ancestor or file can be replaced between this
// comparison, mfile's open and its last read. Never trust a pathname-only probe.
class BoundPath {
    std::vector<std::unique_ptr<WinHandle>> parents_;
public:
    std::filesystem::path path;
    std::unique_ptr<ts::RawFile> guard;
    BoundPath(const VerifiedFile& file):path(held_path(file.held_handle())){
        lock_directory_chain(path.parent_path(),parents_);
        guard=std::make_unique<ts::RawFile>(path_utf8(path),true);
        io_need(guard->direct(),"unbuffered source required");
        same_file_object(file.held_handle(),static_cast<HANDLE>(guard->handle()));
    }
};
using DriveStats=std::map<std::string,std::array<std::uint64_t,3>>;
inline strix::serve::Json drive_receipt(const DriveStats& after,const DriveStats& before={}){
    auto rows=strix::serve::Json::array();for(auto [drive,v]:after){auto old=before.find(drive);if(old!=before.end())for(unsigned i=0;i<3;++i)v[i]-=old->second[i];
        auto row=strix::serve::Json::object();row.set("drive",drive);row.set("bytes",v[0]);row.set("busy_ms",double(v[1])/1e6);row.set("reads",v[2]);rows.push(row);}return rows;
}
class TierReader {
    BoundPath primary_;
    // Secondary verification is byte equality with the HMAC/USN-bound primary,
    // not authority delegated to an unauthenticated xxh64 cache or manifest.
    // Each admitted manifest stays held deny-write/delete with the exact bytes
    // whose mapping was compared, beside the data object it names.
    struct Tier {
        std::unique_ptr<VerifiedFile> manifest,data;
        std::unique_ptr<BoundPath> manifest_anchor,data_anchor;
        std::string text;
    };
    std::vector<Tier> tiers_;
    std::vector<std::unique_ptr<WinHandle>> directories_;
    std::vector<std::unique_ptr<WinHandle>> names_;
    ss_mfile* file_=nullptr;
    unsigned locations_=0;
    static void checked_preadv(ss_mfile* file,const ss_mfile_io* ios,std::uint32_t count){
        if(ss_mfile_preadv(file,ios,count)==SS_OK)return;
#ifdef STRIX_HASH_AUTHORITY_TESTING
        throw std::runtime_error(std::string("bound preadv failed: ")+ss_last_error());
#else
        throw std::runtime_error("bound preadv failed");
#endif
    }
    void bind_name(const std::filesystem::path& path,const VerifiedFile& held){
        auto name=std::make_unique<WinHandle>();name->h=CreateFileW(path.c_str(),FILE_READ_ATTRIBUTES,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_FLAG_OPEN_REPARSE_POINT,nullptr);
        io_need(name->h!=INVALID_HANDLE_VALUE,"retain serving name");FILE_ATTRIBUTE_TAG_INFO info{};
        io_need(GetFileInformationByHandleEx(name->h,FileAttributeTagInfo,&info,sizeof info)&&!(info.FileAttributes&FILE_ATTRIBUTE_REPARSE_POINT),"serving file reparse refused");
        same_file_object(held.held_handle(),name->h);names_.push_back(std::move(name));
    }
public:
    // 873b004 rebuilds `<canonical dir>/<primary>.stripe.json` and parses it
    // itself; its API exposes neither that text nor a held-handle admission.
    // Prove the name it used is the admitted, still-held manifest object (not a
    // reparse) and that the held object's bytes equal the admitted bytes.
    static void check_parsed_manifest(const std::string& location,const std::string& primary_location,
                                      const VerifiedFile& manifest,const std::string& admitted){
        auto name=std::filesystem::u8path(location).parent_path()/
                  std::filesystem::u8path(ts::stripe_manifest_name(path_utf8(std::filesystem::u8path(primary_location).filename())));
        WinHandle h;h.h=CreateFileW(name.c_str(),FILE_READ_ATTRIBUTES,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_FLAG_OPEN_REPARSE_POINT,nullptr);
        io_need(h.h!=INVALID_HANDLE_VALUE,"library stripe manifest name");FILE_ATTRIBUTE_TAG_INFO info{};
        io_need(GetFileInformationByHandleEx(h.h,FileAttributeTagInfo,&info,sizeof info)&&!(info.FileAttributes&FILE_ATTRIBUTE_REPARSE_POINT),"library stripe manifest reparse refused");
        same_file_object(manifest.held_handle(),h.h);
        io_need(manifest.contents()==admitted,"library stripe manifest differs from admitted bytes");
    }
    TierReader(const VerifiedFile& primary,const std::string& directories="",unsigned readers=16):primary_(primary){
        io_need(readers>=8&&readers<=32,"drive readers must be 8..32");
        const auto& digest=primary.admitted_digest();
        io_need(digest.size()==64&&digest.find_first_not_of("0123456789abcdef")==std::string::npos,"admitted source digest required");
        std::string admitted_dirs;
        const auto manifest_name=std::filesystem::u8path(ts::stripe_manifest_name(path_utf8(primary_.path.filename())));
        const auto library_primary=library_name(primary_.path);
        same_library_object(library_primary,primary_.path,primary.held_handle());
        const auto library_manifest_name=std::filesystem::u8path(ts::stripe_manifest_name(path_utf8(library_primary.filename())));
        for(std::size_t pos=0;pos<directories.size();){auto end=directories.find(';',pos);if(end==std::string::npos)end=directories.size();auto item=directories.substr(pos,end-pos);pos=end+1;
            if(item.empty())continue;
            auto dir=std::filesystem::absolute(std::filesystem::u8path(item)).lexically_normal();
            if(!dir.has_filename()&&dir.has_relative_path())dir=dir.parent_path();
            auto manifest_path=dir/manifest_name;std::error_code ec;
            auto status=std::filesystem::symlink_status(manifest_path,ec);if(status.type()==std::filesystem::file_type::not_found)continue;
            io_need(!ec&&status.type()==std::filesystem::file_type::regular,"stripe manifest must be a regular non-reparse file");
            // The supplied directory itself is the namespace the library will
            // search: lock it and every ancestor, refusing junctions/symlinks.
            const auto anchored=directories_.size();lock_directory_chain(dir,directories_);
            const auto library_dir=library_name(dir);same_library_object(library_dir,dir,directories_[anchored]->h);
            io_need(std::filesystem::file_size(manifest_path)<=16ULL<<20,"stripe manifest bound");
            auto manifest=std::make_unique<VerifiedFile>(manifest_path,std::filesystem::file_size(manifest_path));
            // Hold first, then bind the name: a final-component reparse names
            // another object and is refused, like the data file below.
            bind_name(manifest_path,*manifest);
            auto manifest_anchor=std::make_unique<BoundPath>(*manifest);
            io_need(same_local_name(manifest_anchor->path,manifest_path),"stripe manifest must have its canonical name in the supplied directory");
            auto text=manifest->contents();auto m=ts::stripe_manifest_parse(text);
            io_need(m.primary_sha256==digest&&m.primary_size==primary_.guard->size()&&m.primary_basename==path_utf8(primary_.path.filename()),"stripe primary identity mismatch");
            ts::stripe_data_name_validate(m.primary_basename,m.data_name);
            auto data_path=dir/std::filesystem::u8path(m.data_name);
            auto data=std::make_unique<VerifiedFile>(data_path,m.data_size);bind_name(data_path,*data);
            // The library opens <lowercased dir>/<lowercased primary>.stripe.json
            // and <lowercased dir>/<data name>: both must be the held objects.
            same_library_object(library_dir/library_manifest_name,manifest_path,manifest->held_handle());
            same_library_object(library_dir/std::filesystem::u8path(m.data_name),data_path,data->held_handle());
            auto data_anchor=std::make_unique<BoundPath>(*data);
            VerifiedFile::Reader reference(primary,io_chunk);AlignedMemory expected(io_chunk),actual(io_chunk);
            for(auto run:m.runs)for(std::uint64_t k=0;k<run.count;++k){auto logical=(run.first_chunk+k)*m.chunk_bytes,physical=(run.first_slot+k)*m.chunk_bytes;
                auto bytes=std::min(m.chunk_bytes,m.primary_size-logical);
                for(std::uint64_t off=0;off<bytes;){auto n=std::min<std::uint64_t>(io_chunk,bytes-off);reference.read_at(logical+off,expected.p,std::size_t(n));
                    auto got=data_anchor->guard->read(actual.p,CoalescedRound(n),physical+off);
                    io_need(got>=n&&!std::memcmp(expected.p,actual.p,std::size_t(n)),"stripe differs from authenticated primary");off+=n;}
            }
            if(!admitted_dirs.empty())admitted_dirs+=';';admitted_dirs+=path_utf8(manifest_anchor->path.parent_path());
            tiers_.push_back({std::move(manifest),std::move(data),std::move(manifest_anchor),std::move(data_anchor),std::move(text)});
        }
        ss_mfile_options o{};ss_mfile_options_defaults(&o);o.primary_sha256=digest.c_str();o.drive_readers=readers;o.verify_mode=SS_MFILE_VERIFY_CACHED;
        io_need(ss_mfile_open(path_utf8(primary_.path).c_str(),admitted_dirs.c_str(),&o,&file_)==SS_OK,"open bound multi-location reader");
        ss_mfile_counters c{};c.struct_size=sizeof c;
        try {
            io_need(ss_mfile_get_counters(file_,&c)==SS_OK&&c.locations&&c.locations<=SS_MFILE_MAX_LOCATIONS,
                    "multi-location counters invalid");
            io_need((c.direct_mask|c.disabled_mask)==((1u<<c.locations)-1)&&!(c.disabled_mask&1),
                    "buffered tier fallback refused");
            locations_=c.locations;
            // MFile canonicalises paths internally. Check its actual canonical
            // names while the entire namespace is locked. Its own deny-write
            // handles therefore name precisely the admitted immutable objects.
            for(unsigned i=0;i<c.locations;++i){
                if(c.disabled_mask&(1u<<i))continue;
                auto name=ss_mfile_location_path(file_,int(i));io_need(name,"missing serving location");
                ts::RawFile serving(name,true);const Tier* tier=nullptr;
                if(i==0)same_file_object(primary.held_handle(),static_cast<HANDLE>(serving.handle()));
                else for(auto& t:tiers_)try{
                    same_file_object(t.data_anchor->guard->handle(),serving.handle());tier=&t;break;
                }catch(const std::exception&){}
                io_need((i==0||tier)&&serving.direct(),"unbound serving location");
                // The mapping placing this data must be the one admitted with it.
                if(tier)check_parsed_manifest(name,ss_mfile_location_path(file_,0),*tier->manifest,tier->text);
            }
        }catch(...){ss_mfile_close(file_);file_=nullptr;throw;}
    }
    static std::uint64_t CoalescedRound(std::uint64_t n){return (n+4095)/4096*4096;}
    ~TierReader(){if(file_)ss_mfile_close(file_);}
    TierReader(const TierReader&)=delete;
    void read(const std::vector<ss_mfile_io>& ios)const {
        io_need(ios.size()<=UINT32_MAX,"preadv range count");
        if(locations_==1){checked_preadv(file_,ios.data(),std::uint32_t(ios.size()));return;}
        // This pinned ABI serialises large vectors when secondary locations
        // exist. Feed eight concurrent large calls into its shared per-volume
        // pools. 32 MiB tickets span the usual 16 MiB stripe grid, so a ticket
        // can use both homes even before later upstream routing improvements.
        // Every job drains on failure before caller-owned pinned memory returns.
        std::atomic<std::size_t> next=0;std::vector<std::future<void>> jobs;
        std::exception_ptr error;
        try{for(unsigned k=0;k<std::min<std::size_t>(8,ios.size());++k)jobs.push_back(std::async(std::launch::async,[&]{
            for(auto i=next.fetch_add(1);i<ios.size();i=next.fetch_add(1))
                checked_preadv(file_,&ios[i],1);
        }));}catch(...){error=std::current_exception();}
        for(auto& job:jobs)try{job.get();}catch(...){if(!error)error=std::current_exception();}
        if(error)std::rethrow_exception(error);
    }
    void counters(DriveStats& out)const {ss_mfile_counters c{};c.struct_size=sizeof c;io_need(ss_mfile_get_counters(file_,&c)==SS_OK,"reader counters");
        for(unsigned i=0;i<c.locations;++i){auto p=ss_mfile_location_path(file_,int(i));io_need(p!=nullptr,"location path");auto drive=std::filesystem::u8path(p).root_name().string();auto& v=out[drive];v[0]+=c.bytes[i];v[1]+=c.busy_ns[i];v[2]+=c.reads[i];}}
};
}
