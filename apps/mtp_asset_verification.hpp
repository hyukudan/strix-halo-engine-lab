#pragma once
#include "hip_decode_options.hpp"
#include "strix/model_content_hash.hpp"
namespace strix::decode_cli {
inline ModelContentIdentity mtp_cli_digest(std::string_view text) {
    if(!mtp_sha256_text(text))throw std::invalid_argument("MTP SHA256 pin is invalid");
    const auto digit=[](char ch)->unsigned {if(ch>='0'&&ch<='9')return static_cast<unsigned>(ch-'0');
        if(ch>='a'&&ch<='f')return static_cast<unsigned>(ch-'a')+10U;
        return static_cast<unsigned>(ch-'A')+10U;};
    ModelContentIdentity out;
    for(std::size_t i=0;i<out.bytes.size();++i)out.bytes[i]=static_cast<std::uint8_t>((digit(text[i*2U])<<4U)|digit(text[i*2U+1U]));
    return out;
}
// Accepted hexadecimal pins denote binary SHA-256 digests in either case.
inline void verify_mtp_serving_assets(const MtpCliOptions& m,
    const ModelContentIdentity& sidecar, std::uint64_t sidecar_bytes,
    const ModelContentIdentity& golden, std::uint64_t golden_bytes) {
    if(sidecar_bytes!=m.sidecar_bytes || sidecar!=mtp_cli_digest(m.sidecar_sha256) ||
        golden!=mtp_cli_digest(m.golden_sha256) || golden_bytes>m.golden_max_bytes)
        throw std::invalid_argument("one-slot:asset size or digest differs");
}
} // namespace strix::decode_cli
