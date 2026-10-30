#include "strix/mtp_forward_golden_file.hpp"
#include "strix/model_content_hash.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace strix {
namespace {
constexpr std::array<unsigned char,8> magic={'S','T','R','M','T','P','G','1'};
// Version 1: 168-byte header. Version 2 appends the target prefill and decode profile identities.
constexpr std::uint64_t header_bytes_v1=168,header_bytes_v2=184,header_bytes_v3=288,row_header_bytes=24;
constexpr std::uint32_t max_rows=4096,max_width=1U<<20,max_vocabulary=1U<<24;
static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559);
void require(bool value,const char* message) { if(!value) throw std::invalid_argument(message); }
std::uint64_t add(std::uint64_t a,std::uint64_t b) {
    if(b>std::numeric_limits<std::uint64_t>::max()-a) throw std::overflow_error("MTP golden file size overflow");
    return a+b;
}
std::uint64_t mul(std::uint64_t a,std::uint64_t b) {
    if(b && a>std::numeric_limits<std::uint64_t>::max()/b) throw std::overflow_error("MTP golden file size overflow");
    return a*b;
}
std::uint64_t geometry(std::uint64_t header_bytes,std::uint64_t rows,std::uint64_t width,std::uint64_t vocabulary) {
    require(rows && rows<=max_rows && width && width<=max_width && vocabulary && vocabulary<=max_vocabulary,
        "MTP golden dimensions exceed the closed file format");
    return add(header_bytes,mul(rows,add(row_header_bytes,add(mul(width,4),mul(vocabulary,4)))));
}
void budget(std::uint64_t bytes,std::uint64_t maximum) {
    require(maximum && bytes<=maximum && bytes<=std::numeric_limits<std::size_t>::max() &&
        bytes<=static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()),
        "MTP golden file exceeds the explicit file byte budget");
}
void pins(const HipMtpForwardGolden& golden,std::uint64_t rows=UINT64_MAX) {
    require(golden.independent_source_content.valid() && golden.target_content.valid() && golden.sidecar_content.valid() &&
        golden.independent_reference_identity && golden.structural_plan_identity && golden.semantic_identity &&
        golden.numerical_profile_identity,"MTP golden source/model/plan/profile pins must be explicit");
    if(golden.rotary_factor) {
        const auto descriptor=RotaryDescriptor::yarn(static_cast<std::uint32_t>(golden.rotary_factor));
        require(golden.rotary_factor==2 || golden.rotary_factor==4,"MTP golden v3 factor differs");
        require(golden.context_capacity && golden.context_capacity<=descriptor.effective_max() &&
            golden.rotary_descriptor_sha256==descriptor.sha256() && golden.rotary_table_sha256.valid() &&
            golden.calibration_begin==0 && golden.calibration_end==(rows==UINT64_MAX?golden.rows.size():rows) &&
            golden.calibration_end<golden.context_capacity && golden.prompt_prefill_profile_identity &&
            golden.target_prefill_profile_identity && golden.target_decode_profile_identity,
            "MTP golden v3 rotary/capacity/calibration/profile pins differ");
    }
    if(!golden.rotary_factor) require(!golden.context_capacity && !golden.calibration_begin && !golden.calibration_end &&
        !golden.rotary_descriptor_sha256.valid() && !golden.rotary_table_sha256.valid() && !golden.prompt_prefill_profile_identity,
        "MTP native golden cannot contain partial v3 metadata");
    require((golden.target_prefill_profile_identity==0)==(golden.target_decode_profile_identity==0),
        "MTP golden records both target profile identities or neither");
}
std::uint32_t version_of(const HipMtpForwardGolden& golden) { return golden.rotary_factor ? 3U : golden.target_decode_profile_identity ? 2U : 1U; }
void finite_bf16(std::uint16_t value) {
    require((value&0x7f80U)!=0x7f80U,"MTP golden BF16 values must be finite");
}
struct Reader {
    std::span<const std::byte> data;
    std::size_t offset=0;
    std::uint64_t integer(unsigned bytes) {
        require(offset<=data.size() && bytes<=data.size()-offset,"MTP golden file is truncated");
        std::uint64_t value=0;
        for(unsigned i=0;i<bytes;++i) value|=static_cast<std::uint64_t>(std::to_integer<unsigned char>(data[offset++]))<<(i*8U);
        return value;
    }
    ModelContentIdentity digest() {
        ModelContentIdentity value;
        for(auto& byte:value.bytes) byte=static_cast<std::uint8_t>(integer(1));
        return value;
    }
};
void append(std::vector<std::byte>& bytes,std::uint64_t value,unsigned count) {
    for(unsigned i=0;i<count;++i) bytes.push_back(static_cast<std::byte>((value>>(i*8U))&255U));
}
void append(std::vector<std::byte>& bytes,const ModelContentIdentity& hash) {
    for(const auto byte:hash.bytes) bytes.push_back(static_cast<std::byte>(byte));
}
void validate_row(const HipMtpForwardGoldenRow& row,std::size_t index,std::size_t width,std::size_t vocabulary) {
    require(row.source_position==index && row.source_input_token<vocabulary && row.next_input_token<vocabulary &&
        row.selected_token<vocabulary && row.target_hidden_bf16.size()==width && row.next_hidden_bf16.size()==width &&
        row.full_logits.size()==vocabulary,"MTP golden row metadata or dimensions are inconsistent");
    for(const auto value:row.target_hidden_bf16) finite_bf16(value);
    for(const auto value:row.next_hidden_bf16) finite_bf16(value);
    std::size_t best=0;
    for(std::size_t i=0;i<vocabulary;++i) {
        require(std::isfinite(row.full_logits[i]),"MTP golden logits must be finite");
        if(row.full_logits[i]>row.full_logits[best]) best=i;
    }
    require(best==row.selected_token,"MTP golden selection must be the finite first maximum");
}
HipMtpForwardGolden decode(std::span<const std::byte> bytes) {
    Reader reader{bytes};
    for(const auto value:magic) require(reader.integer(1)==value,"MTP golden file magic differs");
    const auto version=reader.integer(4);
    require((version>=1 && version<=3) && reader.integer(4)==(version==1?header_bytes_v1:version==2?header_bytes_v2:header_bytes_v3),
        "MTP golden file version or header size differs");
    const auto header_bytes=version==1?header_bytes_v1:version==2?header_bytes_v2:header_bytes_v3;
    require(reader.integer(8)==bytes.size(),"MTP golden file declared byte size differs");
    HipMtpForwardGolden golden;
    golden.independent_source_content=reader.digest(); golden.target_content=reader.digest(); golden.sidecar_content=reader.digest();
    golden.independent_reference_identity=reader.integer(8); golden.structural_plan_identity=reader.integer(8);
    golden.semantic_identity=reader.integer(8); golden.numerical_profile_identity=reader.integer(8);
    const auto rows=reader.integer(4),width=reader.integer(4),vocabulary=reader.integer(4);
    require(reader.integer(4)==0,"MTP golden reserved field is nonzero");
    if(version>=2) {
        golden.target_prefill_profile_identity=reader.integer(8); golden.target_decode_profile_identity=reader.integer(8);
        require(golden.target_prefill_profile_identity && golden.target_decode_profile_identity,
            "MTP golden version 2 requires both target profile identities");
    }
    if(version==3) {
        golden.rotary_factor=reader.integer(8); golden.rotary_descriptor_sha256=reader.digest();
        golden.rotary_table_sha256=reader.digest(); golden.context_capacity=reader.integer(8);
        golden.calibration_begin=reader.integer(8); golden.calibration_end=reader.integer(8);
        golden.prompt_prefill_profile_identity=reader.integer(8);
        require(golden.rotary_factor && golden.calibration_end==rows,"MTP golden v3 extent differs");
    }
    pins(golden,rows);
    // Dimensions and v3 pins are checked before decoded row storage is allocated.
    require(geometry(header_bytes,rows,width,vocabulary)==bytes.size(),"MTP golden dimensions, payload size or trailing bytes differ");
    // Pass one validates every row before allocating any decoded payload.
    const auto payload_offset=reader.offset;
    for(std::uint64_t row=0;row<rows;++row) {
        require(reader.integer(8)==row,"MTP golden rows must be the ordered fresh source frontier");
        const auto input=reader.integer(4),next=reader.integer(4),selected=reader.integer(4);
        require(input<vocabulary && next<vocabulary && selected<vocabulary && reader.integer(4)==0,
            "MTP golden row tokens or reserved field differ");
        for(std::uint64_t i=0;i<mul(width,2);++i) finite_bf16(static_cast<std::uint16_t>(reader.integer(2)));
        float best_value=0; std::uint64_t best=0;
        for(std::uint64_t i=0;i<vocabulary;++i) {
            const auto value=std::bit_cast<float>(static_cast<std::uint32_t>(reader.integer(4)));
            require(std::isfinite(value),"MTP golden logits must be finite");
            if(i==0 || value>best_value) { best_value=value; best=i; }
        }
        require(selected==best,"MTP golden selection must be the finite first maximum");
    }
    require(reader.offset==bytes.size(),"MTP golden file has trailing payload");
    reader.offset=payload_offset;
    golden.rows.resize(static_cast<std::size_t>(rows));
    for(auto& row:golden.rows) {
        row.source_position=reader.integer(8); row.source_input_token=static_cast<std::uint32_t>(reader.integer(4));
        row.next_input_token=static_cast<std::uint32_t>(reader.integer(4)); row.selected_token=static_cast<std::uint32_t>(reader.integer(4));
        (void)reader.integer(4);
        row.target_hidden_bf16.resize(static_cast<std::size_t>(width)); row.next_hidden_bf16.resize(static_cast<std::size_t>(width));
        row.full_logits.resize(static_cast<std::size_t>(vocabulary));
        for(auto& value:row.target_hidden_bf16) value=static_cast<std::uint16_t>(reader.integer(2));
        for(auto& value:row.next_hidden_bf16) value=static_cast<std::uint16_t>(reader.integer(2));
        for(auto& value:row.full_logits) value=std::bit_cast<float>(static_cast<std::uint32_t>(reader.integer(4)));
    }
    pins(golden);
    return golden;
}
} // namespace
std::vector<std::byte> encode_mtp_forward_golden_file(const HipMtpForwardGolden& golden,std::uint64_t maximum) {
    pins(golden); require(!golden.rows.empty(),"MTP golden file has no rows");
    const auto width=golden.rows.front().target_hidden_bf16.size(),vocabulary=golden.rows.front().full_logits.size();
    const auto version=version_of(golden);
    const auto bytes=geometry(version==1?header_bytes_v1:version==2?header_bytes_v2:header_bytes_v3,golden.rows.size(),width,vocabulary); budget(bytes,maximum);
    for(std::size_t i=0;i<golden.rows.size();++i) validate_row(golden.rows[i],i,width,vocabulary);
    std::vector<std::byte> result; result.reserve(static_cast<std::size_t>(bytes));
    for(const auto value:magic) append(result,value,1);
    append(result,version,4); append(result,version==1?header_bytes_v1:version==2?header_bytes_v2:header_bytes_v3,4); append(result,bytes,8);
    append(result,golden.independent_source_content); append(result,golden.target_content); append(result,golden.sidecar_content);
    append(result,golden.independent_reference_identity,8); append(result,golden.structural_plan_identity,8);
    append(result,golden.semantic_identity,8); append(result,golden.numerical_profile_identity,8);
    append(result,golden.rows.size(),4); append(result,width,4); append(result,vocabulary,4); append(result,0,4);
    if(version>=2) { append(result,golden.target_prefill_profile_identity,8); append(result,golden.target_decode_profile_identity,8); }
    if(version==3) {
        append(result,golden.rotary_factor,8); append(result,golden.rotary_descriptor_sha256);
        append(result,golden.rotary_table_sha256); append(result,golden.context_capacity,8);
        append(result,golden.calibration_begin,8); append(result,golden.calibration_end,8);
        append(result,golden.prompt_prefill_profile_identity,8);
    }
    for(const auto& row:golden.rows) {
        append(result,row.source_position,8); append(result,row.source_input_token,4); append(result,row.next_input_token,4);
        append(result,row.selected_token,4); append(result,0,4);
        for(const auto value:row.target_hidden_bf16) append(result,value,2);
        for(const auto value:row.next_hidden_bf16) append(result,value,2);
        for(const auto value:row.full_logits) append(result,std::bit_cast<std::uint32_t>(value),4);
    }
    return result;
}
void require_mtp_forward_golden_geometry(const std::filesystem::path& path,std::uint64_t rows,
    std::uint64_t width,std::uint64_t vocabulary,std::uint64_t capacity) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    require(file.good(),"cannot open MTP golden geometry header");
    const auto end=file.tellg();
    require(end>=static_cast<std::streamoff>(header_bytes_v3),"MTP YaRN golden header is truncated");
    std::array<std::byte,header_bytes_v3> header{};file.seekg(0);
    require(static_cast<bool>(file.read(reinterpret_cast<char*>(header.data()),header.size())),"cannot read MTP golden geometry header");
    Reader reader{header};
    for(const auto value:magic) require(reader.integer(1)==value,"MTP golden file magic differs");
    require(reader.integer(4)==3U && reader.integer(4)==header_bytes_v3,"YaRN requires golden v3 geometry");
    const auto bytes=reader.integer(8);
    require(bytes==static_cast<std::uint64_t>(end) && bytes==geometry(header_bytes_v3,rows,width,vocabulary),"MTP golden file extent differs from calibration geometry");
    reader.offset=152U;
    require(reader.integer(4)==rows && reader.integer(4)==width && reader.integer(4)==vocabulary,"MTP golden calibration rows/width/vocabulary differ");
    reader.offset=256U;
    require(reader.integer(8)==capacity && reader.integer(8)==0U && reader.integer(8)==rows,"MTP golden retained capacity/calibration differs");
}
HipMtpForwardGolden read_mtp_forward_golden_file(const std::filesystem::path& path,
    const ModelContentIdentity& expected,std::uint64_t maximum) {
    require(expected.valid() && maximum,"MTP golden file requires a full-file SHA and explicit byte budget");
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file) throw std::runtime_error("cannot open MTP independent golden file");
    const auto end=file.tellg();
    if(end<0) throw std::runtime_error("cannot size MTP independent golden file");
    const auto bytes=static_cast<std::uint64_t>(static_cast<std::streamoff>(end));
    budget(bytes,maximum);
    require(bytes>=header_bytes_v1,"MTP golden file is shorter than its header");
    std::vector<std::byte> contents(static_cast<std::size_t>(bytes));
    file.seekg(0,std::ios::beg);
    if(!file.read(reinterpret_cast<char*>(contents.data()),static_cast<std::streamsize>(bytes)) ||
        file.peek()!=std::char_traits<char>::eof() || file.bad())
        throw std::runtime_error("MTP golden file changed size or could not be read completely");
    require(sha256_bytes(contents)==expected,"MTP golden full-file SHA differs");
    return decode(contents);
}
ModelContentIdentity write_mtp_forward_golden_file(const std::filesystem::path& path,
    const HipMtpForwardGolden& golden,std::uint64_t maximum) {
    const auto bytes=encode_mtp_forward_golden_file(golden,maximum);
    require(!std::filesystem::exists(path),"MTP golden output file already exists");
    {
        std::ofstream file(path,std::ios::binary|std::ios::trunc);
        if(!file.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size())) || !file.flush())
            throw std::runtime_error("cannot write MTP golden file");
    }
    return sha256_bytes(bytes);
}
void validate_mtp_forward_golden_rotary(const HipMtpForwardGolden& g,const RotaryIdentity& rotary,
    std::uint64_t capacity,std::uint64_t prompt,std::uint64_t calibration,std::uint64_t decode_profile) {
    if(rotary.descriptor().mode()==RotaryMode::native) {
        require(!g.rotary_factor,"YaRN golden cannot replay as native");
        return;
    }
    pins(g);
    require(g.rotary_factor==rotary.descriptor().factor() && g.context_capacity==capacity &&
        g.rotary_descriptor_sha256==rotary.descriptor().sha256() && g.rotary_table_sha256==rotary.table_sha256() &&
        (!prompt || g.prompt_prefill_profile_identity==prompt) &&
        (!calibration || g.target_prefill_profile_identity==calibration) &&
        (!decode_profile || g.target_decode_profile_identity==decode_profile),
        "MTP golden v3 descriptor/table/capacity/profile mismatch (legacy goldens are native only)");
}
ModelContentIdentity mtp_self_generated_golden_source() {
    constexpr std::string_view marker="strix-mtp-golden:self-generated-native-forward:not-independent:v1";
    return sha256_bytes(std::as_bytes(std::span<const char>(marker.data(),marker.size())));
}
} // namespace strix
