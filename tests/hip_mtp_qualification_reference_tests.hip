#include "hip_mtp_qualification_reference.hpp"
#include "strix/model_content_hash.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
namespace {
void require(bool v){if(!v)throw std::runtime_error("qualification file codec fixture failed");}
template<class F>void rejects(F work){bool failed=false;try{work();}catch(const std::invalid_argument&){failed=true;}require(failed);}
struct Temp {
    std::filesystem::path path=std::filesystem::temp_directory_path()/
        ("strix-mtp-reference-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".bin");
    ~Temp(){std::error_code ignored;std::filesystem::remove(path,ignored);}
};
}
int main(){try{
    using namespace strix;using namespace strix::tests;
    MtpQualificationReference g;g.kind=MtpQualificationKind::cycles;
    g.independent_source_content.bytes.fill(1);g.target_content.bytes.fill(2);g.sidecar_content.bytes.fill(3);
    g.independent_reference_identity=4;g.proposal_identity=5;g.plan_identity=6;g.semantic_identity=7;g.native_profile_identity=8;
    // Parser payload only, never a claimed independent model oracle.
    HipMtpGreedyCycleGoldenRow row;row.initial_source_position=11;row.final_source_position=13;
    row.eos_tokens={19};row.emitted_tokens={2,3};row.final_target_wide_bf16={0x3f80,0x4000};
    row.reconciled_kv_bf16={0x4040};row.reconciled_raw_index_bf16={0x4080};g.cycles.cycles.push_back(row);
    const auto original=encode_mtp_qualification_reference(g,4096);Temp file;
    const auto write=[&](const std::vector<std::byte>& data){std::ofstream f(file.path,std::ios::binary|std::ios::trunc);
        f.write(reinterpret_cast<const char*>(data.data()),static_cast<std::streamsize>(data.size()));require(bool(f));};
    write(original);auto decoded=read_mtp_qualification_reference(file.path,sha256_bytes(original),4096);
    require(encode_mtp_qualification_reference(decoded,4096)==original);
    require(decoded.cycles.cycles[0].reconciled_kv_bf16==row.reconciled_kv_bf16 && decoded.cycles.numerical_profile_identity==8);
    auto wrong=sha256_bytes(original);wrong.bytes[0]^=1U;
    rejects([&]{(void)read_mtp_qualification_reference(file.path,wrong,4096);});
    rejects([&]{(void)read_mtp_qualification_reference(file.path,sha256_bytes(original),160);});
    rejects([&]{(void)encode_mtp_qualification_reference(g,160);});
    for(unsigned fault=0;fault<4U;++fault){auto bad=original;
        if(fault==0U)bad[0]^=std::byte{1};
        if(fault==1U)bad[12]=std::byte{99};
        if(fault==2U)for(unsigned j=0;j<8U;++j)bad[160U+j]=std::byte{255};
        if(fault==3U){bad.push_back(std::byte{0});const auto n=bad.size();for(unsigned j=0;j<8U;++j)bad[16U+j]=static_cast<std::byte>((n>>(8U*j))&255U);}
        write(bad);rejects([&]{(void)read_mtp_qualification_reference(file.path,sha256_bytes(bad),4096);});
    }
    std::cout<<"qualification parser pins/bounds/roundtrip passed; no native model execution\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
