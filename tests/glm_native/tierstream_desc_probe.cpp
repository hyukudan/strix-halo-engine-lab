// Separate TU: the real tierstream desc-v1 parser (src/desc.cpp and
// src/json.cpp, compiled read-only from GLM_TIERSTREAM_SRC) applied to a GLM
// descriptor. Kept apart because tierstream's ts::Json differs from ours.
#include "desc.hpp"
#include "json.hpp"
#include "strix_stream/ss.h"
#include <cstdint>
#include <malloc.h>
#include <exception>
#include <string>
std::string tierstream_desc_check(const std::string& text,std::uint64_t* layers,std::uint64_t* units,std::uint64_t* max_unit){
    try{auto d=ts::desc_from_json(ts::json_parse(text));*layers=d.layers.size();*units=std::uint64_t(d.total_units);*max_unit=d.max_unit_bytes;return {};}
    catch(const std::exception& e){std::string m=e.what();return m.empty()?"rejected":m;}
}
// Opens a source through tierstream's own public reader (ss_mfile_open: the
// library's path decoding and file open) and reads `bytes` from offset 0. The
// path is sources[0].path of the parsed descriptor, or `raw_path` when given
// (used to show what an ANSI-encoded name would have opened). Returns "" on success.
std::string tierstream_source_read(const std::string& desc_text,const char* raw_path,const std::string& sha256,std::uint64_t bytes,std::string* data){
    try{
        std::string path;
        if(raw_path)path=raw_path;
        else{auto d=ts::desc_from_json(ts::json_parse(desc_text));if(d.sources.empty())return "descriptor has no sources";path=d.sources[0].path;}
        ss_mfile_options o;ss_mfile_options_defaults(&o);o.direct_io=0;o.primary_sha256=sha256.c_str();
        auto last=[]{const char* m=ss_last_error();return std::string(m?m:"");};
        ss_mfile* f=nullptr;auto st=ss_mfile_open(path.c_str(),nullptr,&o,&f);if(st!=SS_OK||!f)return "ss_mfile_open failed ("+std::to_string(int(st))+"): "+last();
        // Read contract: dst, bytes and offset are multiples of ss_mfile_alignment.
        const std::uint64_t align=ss_mfile_alignment(f);if(!align||bytes%align){ss_mfile_close(f);return "probe read size not aligned";}
        void* buf=_aligned_malloc(std::size_t(bytes),std::size_t(align));if(!buf){ss_mfile_close(f);return "probe buffer";}
        std::uint64_t got=0;st=ss_mfile_pread(f,buf,bytes,0,&got);std::string err=st!=SS_OK?last():std::string();ss_mfile_close(f);
        data->assign(static_cast<const char*>(buf),std::size_t(bytes));_aligned_free(buf);
        if(st!=SS_OK||got!=bytes)return "ss_mfile_pread failed ("+std::to_string(int(st))+"): "+err;return {};
    }catch(const std::exception& e){std::string m=e.what();return m.empty()?"rejected":m;}
}
