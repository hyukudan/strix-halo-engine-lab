#include "strix/mtp_forward_golden_file.hpp"
#include "strix/model_content_hash.hpp"
#include <array>
#include <bit>
#include <string>
#include <chrono>
#include <cstring>
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
        // Self-generated dump route: writer -> strict reader round trip with its returned pin.
        auto self=golden; self.independent_source_content=mtp_self_generated_golden_source();
        self.independent_reference_identity=mtp_self_generated_golden_reference;
        const auto dumped=file.directory/"dumped.bin";
        const auto pin=write_mtp_forward_golden_file(dumped,self,1024);
        const auto back=read_mtp_forward_golden_file(dumped,pin,1024);
        expect(pin==sha256_bytes(encode_mtp_forward_golden_file(self,1024)),"writer returns the full-file SHA pin");
        expect(back.independent_source_content==self.independent_source_content && back.independent_reference_identity==self.independent_reference_identity &&
               back.target_content==self.target_content && back.sidecar_content==self.sidecar_content &&
               back.structural_plan_identity==12 && back.semantic_identity==13 && back.numerical_profile_identity==14 &&
               back.rows.size()==self.rows.size(),"writer->reader preserves self-generated marker and all bindings");
        for(std::size_t i=0;i<self.rows.size();++i)
            expect(back.rows[i].source_position==i && back.rows[i].source_input_token==self.rows[i].source_input_token &&
                   back.rows[i].next_input_token==self.rows[i].next_input_token && back.rows[i].selected_token==self.rows[i].selected_token &&
                   back.rows[i].target_hidden_bf16==self.rows[i].target_hidden_bf16 && back.rows[i].next_hidden_bf16==self.rows[i].next_hidden_bf16 &&
                   std::memcmp(back.rows[i].full_logits.data(),self.rows[i].full_logits.data(),self.rows[i].full_logits.size()*4)==0,
                   "writer->reader round trip is bitwise per row");
        expect(mtp_self_generated_golden_source()!=golden.independent_source_content && mtp_self_generated_golden_source().valid(),
               "self-generated marker is explicit and distinct");
        rejects([&] { (void)write_mtp_forward_golden_file(dumped,self,1024); },"writer refuses to overwrite an existing golden");
        std::error_code ignored; std::filesystem::remove(dumped,ignored);

        // Version 2: the header appends the target prefill and decode profile identities.
        expect(!golden.target_prefill_profile_identity && !golden.target_decode_profile_identity,
               "a version-1 golden records no target profile pair");
        auto v2=manual;
        patch(v2,8,2,4); patch(v2,12,184,4); patch(v2,16,manual.size()+16U,8);
        std::vector<std::byte> ids; put(ids,0x1111222233334444ULL,8); put(ids,0x5555666677778888ULL,8);
        v2.insert(v2.begin()+168,ids.begin(),ids.end());
        file.write(v2);
        const auto paired=read_mtp_forward_golden_file(file.path,sha256_bytes(v2),v2.size());
        expect(paired.target_prefill_profile_identity==0x1111222233334444ULL &&
               paired.target_decode_profile_identity==0x5555666677778888ULL && paired.rows.size()==2 &&
               paired.rows[1].next_hidden_bf16==golden.rows[1].next_hidden_bf16,
               "version 2 preserves both target profile identities and every row");
        expect(encode_mtp_forward_golden_file(paired,v2.size())==v2,"version 2 encoding matches independent literal bytes");
        bytes=v2; patch(bytes,168,0,8); malformed(bytes,"version 2 with an unrecorded prefill identity rejected");
        bytes=v2; patch(bytes,176,0,8); malformed(bytes,"version 2 with an unrecorded decode identity rejected");
        bytes=v2; patch(bytes,8,1,4); malformed(bytes,"version 1 with the version-2 header length rejected");
        bytes=v2; patch(bytes,12,168,4); malformed(bytes,"version 2 with the version-1 header length rejected");
        auto half=paired; half.target_prefill_profile_identity=0;
        rejects([&] { (void)encode_mtp_forward_golden_file(half,4096); },"encoder rejects a half-recorded target profile pair");
        auto pair_self=self; pair_self.target_prefill_profile_identity=7; pair_self.target_decode_profile_identity=9;
        const auto pin2=write_mtp_forward_golden_file(dumped,pair_self,4096);
        const auto back2=read_mtp_forward_golden_file(dumped,pin2,4096);
        expect(back2.target_prefill_profile_identity==7 && back2.target_decode_profile_identity==9 && pin2!=pin,
               "writer->reader round trip of a version-2 golden keeps the pair and changes the file pin");
        std::filesystem::remove(dumped,ignored);
        for(const auto factor:{2U,4U}) {
            auto v3=pair_self;
            const auto descriptor=RotaryDescriptor::yarn(factor);
            ModelContentIdentity table;table.bytes[0]=static_cast<std::uint8_t>(factor);
            v3.rotary_factor=factor;v3.rotary_descriptor_sha256=descriptor.sha256();v3.rotary_table_sha256=table;
            v3.context_capacity=factor==2?32768U:1048576U;v3.calibration_end=v3.rows.size();
            v3.prompt_prefill_profile_identity=15;
            const auto bytes3=encode_mtp_forward_golden_file(v3,4096);
            expect(bytes3.size()==manual.size()+120U,"v3 exact header extent");
            file.write(bytes3);
            require_mtp_forward_golden_geometry(file.path,v3.rows.size(),v3.rows.front().target_hidden_bf16.size(),
                v3.rows.front().full_logits.size(),v3.context_capacity);
            for(unsigned fault=0;fault<4U;++fault) {
                rejects([&]{require_mtp_forward_golden_geometry(file.path,v3.rows.size()+(fault==0U),
                    v3.rows.front().target_hidden_bf16.size()+(fault==1U),v3.rows.front().full_logits.size()+(fault==2U),
                    v3.context_capacity+(fault==3U));},"bounded header preflight refuses mismatched payload geometry before decoding");
            }
            auto oversized=bytes3;oversized.push_back(std::byte{0});file.write(oversized);
            rejects([&]{require_mtp_forward_golden_geometry(file.path,v3.rows.size(),v3.rows.front().target_hidden_bf16.size(),
                v3.rows.front().full_logits.size(),v3.context_capacity);},"extra encoded storage is refused before allocating file payload");
            file.write(bytes3);
            const auto back3=read_mtp_forward_golden_file(file.path,sha256_bytes(bytes3),4096);
            expect(encode_mtp_forward_golden_file(back3,4096)==bytes3,"v3 bitwise roundtrip");
            const RotaryIdentity rotary(descriptor,table);
            validate_mtp_forward_golden_rotary(back3,rotary,v3.context_capacity,15,7,9);
            rejects([&]{validate_mtp_forward_golden_rotary(golden,rotary,v3.context_capacity);},"legacy golden is not YaRN evidence");
            rejects([&]{validate_mtp_forward_golden_rotary(back3,RotaryIdentity{},v3.context_capacity);},"YaRN cannot replay as native");
            rejects([&]{validate_mtp_forward_golden_rotary(back3,rotary,v3.context_capacity-1U);},"actual C pinned");
            rejects([&]{validate_mtp_forward_golden_rotary(back3,rotary,v3.context_capacity,16,7,9);},"prompt profile pinned");
            rejects([&]{validate_mtp_forward_golden_rotary(back3,rotary,v3.context_capacity,15,8,9);},"calibration profile pinned");
            rejects([&]{validate_mtp_forward_golden_rotary(back3,rotary,v3.context_capacity,15,7,10);},"decode profile pinned");
            auto changed=table;changed.bytes[1]=1;
            rejects([&]{validate_mtp_forward_golden_rotary(back3,RotaryIdentity(descriptor,changed),v3.context_capacity);},"table content pinned");
            auto invalid=v3;invalid.calibration_end++;
            rejects([&]{(void)encode_mtp_forward_golden_file(invalid,4096);},"calibration extent pinned");
            invalid=v3;invalid.rotary_descriptor_sha256.bytes[0]^=1;
            rejects([&]{(void)encode_mtp_forward_golden_file(invalid,4096);},"descriptor pinned");
            expect(back3.independent_source_content==mtp_self_generated_golden_source(),"v3 capture remains self-generated");
        }
        return failures?1:0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}

