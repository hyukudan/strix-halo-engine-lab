#pragma once
// Arm labels of the prefill quality tools (deepseek41_prefill_quality_main.cpp, deepseek41_prefill_continue_main.cpp).
#include "strix/deepseek41_prefill.hpp"
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::deepseek41::quality {

// Arm labels: exact, dense (FP16 all projections), attention (BF16 attention), or a diagnostic spec of '_' tokens:
// kind d (FP16 dense), a (BF16 attention), da (both), n (null: one-ulp flip of the stream after layer l<L>), then
// qa kv qb oa ob (projections; none = all), l<first>t<end> (layers), a8 (FP16 operands from the A8 codes), r<rows>
// (FP16 only at calls of >= rows rows), f (FP16 output form: STRIX_DS41_PREFILL_FP16_FAST), s<seed> (null: +-1 ulp by a
// hash of the seed instead of the last-bit flip, so seeds sample the noise floor).
inline PrefillOptions arm_precision(const std::string& label) {
    PrefillOptions o;
    if (label == "exact") return o;
    if (label == "dense") { o.fp16_dense = true; return o; }
    if (label == "attention") { o.bf16_attention = true; return o; }
    std::vector<std::string> tok;
    std::stringstream ss(label);
    for (std::string t; std::getline(ss, t, '_');) tok.push_back(t);
    if (tok.empty()) throw std::runtime_error("arm label");
    const std::string kind = tok[0];
    if (kind == "d" || kind == "da") o.fp16_dense = true;
    if (kind == "a" || kind == "da") o.bf16_attention = true;
    if (kind != "d" && kind != "a" && kind != "da" && kind != "n") throw std::runtime_error("arm kind " + label);
    std::uint32_t proj = 0;
    for (std::size_t i = 1; i < tok.size(); ++i) {
        const auto& t = tok[i];
        if (t == "qa") proj |= 1U;
        else if (t == "kv") proj |= 2U;
        else if (t == "qb") proj |= 4U;
        else if (t == "oa") proj |= 8U;
        else if (t == "ob") proj |= 16U;
        else if (t == "a8") o.fp16_keep_a8 = true;
        else if (t == "f") o.fp16_fast = true;
        else if (t == "w") o.fp16_wsplit = true;  // with f: weights as hi + lo FP16 halves
        else if (t[0] == 's' && kind == "n") o.null_seed = static_cast<std::uint32_t>(std::stoul(t.substr(1)));
        else if (t[0] == 'r') o.fp16_min_rows = static_cast<std::uint32_t>(std::stoul(t.substr(1)));
        else if (t[0] == 'l') {
            const auto at = t.find('t');
            if (kind == "n") o.null_ulp_layer = std::stoi(t.substr(1));
            else if (at == std::string::npos) throw std::runtime_error("layer token " + label);
            else {
                o.approx_layer_first = static_cast<std::uint32_t>(std::stoul(t.substr(1, at - 1)));
                o.approx_layer_end = static_cast<std::uint32_t>(std::stoul(t.substr(at + 1)));
            }
        } else throw std::runtime_error("arm token " + label);
    }
    if (proj) o.fp16_projections = proj;
    if (kind == "n" && o.null_ulp_layer < 0) throw std::runtime_error("null arm needs l<L>");
    return o;
}

}  // namespace strix::deepseek41::quality
