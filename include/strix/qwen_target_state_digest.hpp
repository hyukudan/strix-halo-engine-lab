#pragma once
#include "strix/model_content_hash.hpp"
#include <optional>
#include <string_view>
#include <vector>
#include <stdexcept>
namespace strix {
// Common numerical frame, independent of executor/window provenance and of the
// predictor-only registry extension. Component order and occupied byte ranges
// are part of this schema; empty target components are included too.
class QwenTargetStateDigestFrame {
public:
    QwenTargetStateDigestFrame(std::uint64_t position,std::uint64_t rows,std::uint64_t fixed)
        :position_(position),expected_(rows+fixed) {
        if(fixed>UINT64_MAX-rows)throw std::overflow_error("target digest component count overflow");
        word(UINT64_C(0x5441524745545631));word(position);word(rows);word(fixed);
    }
    void component(std::string_view id,std::uint64_t bytes,const ModelContentIdentity& hash) {
        if(!hash.valid())throw std::invalid_argument("target digest missing component hash");
        word(id.size());for(const auto c:id)frame_.push_back(static_cast<std::byte>(c));word(bytes);
        for(const auto c:hash.bytes)frame_.push_back(static_cast<std::byte>(c));
        ++components_;
    }
    ModelContentIdentity finish(std::span<const std::uint32_t> history,std::optional<std::uint32_t> pending) {
        if(history.size()!=position_ || components_!=expected_) throw std::invalid_argument("target digest history/components differ");
        word(history.size());for(const auto token:history)word(token);
        word(pending?*pending:UINT64_C(0x100000000));return sha256_bytes(frame_);
    }
private:
    void word(std::uint64_t n) {for(unsigned j=0;j<8;++j)frame_.push_back(static_cast<std::byte>((n>>(8U*j))&255U));}
    std::uint64_t position_,expected_,components_=0;
    std::vector<std::byte> frame_;
};
} // namespace strix
