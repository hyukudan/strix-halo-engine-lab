#include "strix/mtp_forward_golden_file.hpp"
#include "strix/model_content_hash.hpp"
#include <array>
#include <bit>
#include <string>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
int failures=0;
void expect(bool pass,const char* message) {
    if(!pass) { std::cerr<<"FAIL: "<<message<<'\n'; ++failures; }
}
template<class Work> void rejects(Work&& work,const char* message) {
    bool rejected=false;
    try { work(); } catch(const std::exception&) { rejected=true; }
    expect(rejected,message);
}
void put(std::vector<std::byte>& bytes,std::uint64_t value,unsigned count) {
    for(unsigned i=0;i<count;++i) bytes.push_back(static_cast<std::byte>((value>>(8U*i))&255U));
}
void patch(std::vector<std::byte>& bytes,std::size_t offset,std::uint64_t value,unsigned count) {
    for(unsigned i=0;i<count;++i) bytes.at(offset+i)=static_cast<std::byte>((value>>(8U*i))&255U);
}
std::vector<std::byte> independent_container() {
    // This literal producer does not call the implementation encoder.
    std::vector<std::byte> bytes;
    for(const char c:std::string("STRMTPG1")) put(bytes,static_cast<unsigned char>(c),1);
    put(bytes,1,4); put(bytes,168,4); put(bytes,256,8);
    for(const auto value:{1U,2U,3U}) for(unsigned i=0;i<32;++i) put(bytes,value,1);
    for(const auto id:{11U,12U,13U,14U}) put(bytes,id,8);
    put(bytes,2,4); put(bytes,2,4); put(bytes,3,4); put(bytes,0,4);
    for(unsigned row=0;row<2;++row) {
        put(bytes,row,8); put(bytes,row,4); put(bytes,row+1,4); put(bytes,row==0?1:0,4); put(bytes,0,4);
        for(const auto bf16:{0x8000U,1U,0x3f80U,0xbf80U}) put(bytes,bf16,2);
        const std::array<std::uint32_t,3> logits=row==0?
            std::array<std::uint32_t,3>{0xbf800000U,0x40000000U,0x40000000U}:
            std::array<std::uint32_t,3>{0,0x80000000U,0xbf800000U};
        for(const auto bits:logits) put(bytes,bits,4);
    }
    return bytes;
}
struct Files {
    std::filesystem::path directory,path;
    Files() {
        directory=std::filesystem::temp_directory_path()/
            ("strix-mtp-golden-codec-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        if(!std::filesystem::create_directory(directory)) throw std::runtime_error("fixture output already exists");
        path=directory/"reference.bin";
    }
    ~Files() { std::error_code ignored; std::filesystem::remove(path,ignored); std::filesystem::remove(directory,ignored); }
    void write(const std::vector<std::byte>& bytes) {
        std::ofstream out(path,std::ios::binary|std::ios::trunc);
        out.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));
        if(!out) throw std::runtime_error("cannot write golden codec fixture");
    }
};
} // namespace
int main() {
    using namespace strix;
    try {
        Files file;
        const auto manual=independent_container();
        file.write(manual);
        const auto digest=sha256_bytes(manual);
        const auto golden=read_mtp_forward_golden_file(file.path,digest,manual.size());
        expect(golden.rows.size()==2 && golden.rows[0].selected_token==1 && golden.rows[1].selected_token==0,
               "independent LE container preserves first maximum ties including signed zero");
        expect(golden.rows[0].target_hidden_bf16==std::vector<std::uint16_t>({0x8000,1}) &&
               golden.rows[1].next_hidden_bf16==std::vector<std::uint16_t>({0x3f80,0xbf80}),
               "all source and next wide BF16 values preserved bitwise including subnormal");
        expect(std::bit_cast<std::uint32_t>(golden.rows[1].full_logits[1])==0x80000000U,
               "full FP32 signed-zero payload is preserved");
        expect(golden.independent_source_content.bytes[0]==1 && golden.target_content.bytes[0]==2 &&
               golden.sidecar_content.bytes[0]==3 && golden.independent_reference_identity==11 &&
               golden.structural_plan_identity==12 && golden.semantic_identity==13 && golden.numerical_profile_identity==14,
               "all independent reference/model/semantic/profile pins preserved");
        expect(encode_mtp_forward_golden_file(golden,manual.size())==manual,
               "encoding matches independent literal bytes, not only its own decoder");
        rejects([&] { (void)read_mtp_forward_golden_file(file.path,digest,manual.size()-1); },"file budget before payload allocation");
        rejects([&] { (void)read_mtp_forward_golden_file(file.path,digest,0); },"explicit budget required");
        auto wrong=digest; wrong.bytes[0]^=1;
        rejects([&] { (void)read_mtp_forward_golden_file(file.path,wrong,1024); },"wrong full-file digest rejected");
        rejects([&] { (void)read_mtp_forward_golden_file(file.path,{},1024); },"unknown digest rejected");
        const auto malformed=[&](std::vector<std::byte> bytes,const char* message) {
            file.write(bytes);
            rejects([&] { (void)read_mtp_forward_golden_file(file.path,sha256_bytes(bytes),1024); },message);
        };
        auto bytes=manual; bytes[0]=std::byte{0}; malformed(bytes,"bad magic rejected");
        bytes=manual; patch(bytes,8,2,4); malformed(bytes,"unsupported version rejected");
        bytes=manual; patch(bytes,12,160,4); malformed(bytes,"wrong header length rejected");
        bytes=manual; patch(bytes,16,255,8); malformed(bytes,"wrong declared file length rejected");
        bytes=manual; patch(bytes,164,1,4); malformed(bytes,"header reserved bits rejected");
        bytes=manual; patch(bytes,188,1,4); malformed(bytes,"row reserved bits rejected");
        bytes=manual; patch(bytes,152,4097,4); malformed(bytes,"row limit enforced before decoding allocation");
        bytes=manual; patch(bytes,156,0,4); malformed(bytes,"empty wide shape rejected");
        bytes=manual; patch(bytes,160,0xffffffffU,4); malformed(bytes,"oversized vocabulary rejected");
        bytes=manual; patch(bytes,212,3,8); malformed(bytes,"noncanonical later source position rejected");
        bytes=manual; patch(bytes,176,3,4); malformed(bytes,"source token outside vocabulary rejected");
        bytes=manual; patch(bytes,180,3,4); malformed(bytes,"next token outside vocabulary rejected");
        bytes=manual; patch(bytes,184,2,4); malformed(bytes,"non-first tied selected token rejected");
        bytes=manual; patch(bytes,192,0x7f80,2); malformed(bytes,"BF16 source infinity rejected");
        bytes=manual; patch(bytes,196,0x7fc1,2); malformed(bytes,"BF16 next-state NaN rejected");
        bytes=manual; patch(bytes,200,0x7f800000U,4); malformed(bytes,"full logit infinity rejected");
        bytes=manual; patch(bytes,200,0x7fc00001U,4); malformed(bytes,"full logit NaN rejected");
        bytes=manual; patch(bytes,144,0,8); malformed(bytes,"unknown native numerical profile rejected");
        bytes=manual; for(std::size_t i=24;i<56;++i) bytes[i]=std::byte{0}; malformed(bytes,"unknown reference source digest rejected");
        bytes=manual; bytes.push_back(std::byte{0}); patch(bytes,16,bytes.size(),8); malformed(bytes,"trailing bytes rejected even with matching declared size and SHA");
        for(const std::size_t length:{0U,8U,167U,168U,190U,255U}) {
            bytes=manual; bytes.resize(length); malformed(bytes,"truncation rejected with its own matching SHA");
        }
        auto inconsistent=golden; inconsistent.rows[1].next_hidden_bf16.pop_back();
        rejects([&] { (void)encode_mtp_forward_golden_file(inconsistent,1024); },"encoder rejects inconsistent later wide row");
        rejects([&] { (void)encode_mtp_forward_golden_file(golden,255); },"encoder validates total budget before allocating bytes");
        return failures?1:0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}

