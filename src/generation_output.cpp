#include "strix/generation_output.hpp"

#include "strix/serial_decode_replay.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strix {
namespace {
constexpr auto bits(GenerationOutputNeed value) noexcept { return static_cast<std::uint32_t>(value); }
constexpr bool has(GenerationOutputNeed value, GenerationOutputNeed need) noexcept { return (bits(value)&bits(need))!=0; }
constexpr auto known_needs = GenerationOutputNeed::token_only|GenerationOutputNeed::requested_logprobs|
    GenerationOutputNeed::full_logits|GenerationOutputNeed::verify_decisions|
    GenerationOutputNeed::state_only|GenerationOutputNeed::mtp_hidden;

void validate(const GenerationOutputRequest& request) {
    if (!bits(request.needs) || (bits(request.needs)&~bits(known_needs)))
        throw std::invalid_argument("generation output needs are empty or unknown");
    if (has(request.needs,GenerationOutputNeed::mtp_hidden))
        throw std::invalid_argument("MTP hidden output requires its unresolved forward contract");
    if (has(request.needs,GenerationOutputNeed::state_only) && bits(request.needs)!=bits(GenerationOutputNeed::state_only))
        throw std::invalid_argument("state-only output cannot require a head result");
    if (!request.vocab_size || request.vocab_size>std::numeric_limits<std::uint32_t>::max() ||
        request.vocab_size>std::numeric_limits<std::size_t>::max()/sizeof(float) || request.input_token>=request.vocab_size)
        throw std::invalid_argument("generation output input/vocabulary is invalid");
    const bool logprobs=has(request.needs,GenerationOutputNeed::requested_logprobs);
    if (logprobs==request.logprob_token_ids.empty())
        throw std::invalid_argument("requested logprobs require an explicit nonempty ID list and matching need");
    for (const auto id:request.logprob_token_ids)
        if (id>=request.vocab_size) throw std::invalid_argument("requested logprob token is outside vocabulary");
}
} // namespace

void compute_requested_logprobs(std::span<const float> logits, std::span<const std::uint32_t> ids,
    std::span<RequestedLogprob> out) {
    if (logits.empty() || out.size()!=ids.size())
        throw std::invalid_argument("requested logprobs need logits and one output per ID");
    for (const auto id:ids) if (id>=logits.size()) throw std::invalid_argument("requested logprob token is outside vocabulary");
    double maximum=static_cast<double>(logits.front());
    for (const auto value:logits) if (static_cast<double>(value)>maximum) maximum=static_cast<double>(value);
    double denominator=0;
    for (const auto value:logits) denominator+=std::exp(static_cast<double>(value)-maximum);
    const double centered_log_normalizer=std::log(denominator);
    for (std::size_t i=0;i<ids.size();++i) {
        const auto id=ids[i];
        // Subtract in centered coordinates: max+log(sum) would lose the
        // normalizer for large finite F32 logits with equal maxima.
        out[i]={id,(static_cast<double>(logits[id])-maximum)-centered_log_normalizer};
    }
}

GenerationOutputResult execute_generation_output(const GenerationOutputRequest& request, SerialGreedyStepBackend& backend) {
    validate(request);
    GenerationOutputResult result;
    result.needs=request.needs;
    result.position=request.position;
    // Allocate presentation storage before the backend can consume model state.
    result.requested_logprobs.resize(request.logprob_token_ids.size());
    if (has(request.needs,GenerationOutputNeed::state_only)) {
        backend.execute_state(request.input_token,request.position);
        return result;
    }
    const bool token=has(request.needs,GenerationOutputNeed::token_only);
    const bool verify=has(request.needs,GenerationOutputNeed::verify_decisions);
    const bool full=has(request.needs,GenerationOutputNeed::full_logits);
    const bool logprobs=has(request.needs,GenerationOutputNeed::requested_logprobs);
    if (!full && !logprobs) {
        const auto selected=backend.execute_greedy(request.input_token,request.position,request.vocab_size);
        if (selected>=request.vocab_size) throw std::invalid_argument("backend greedy selection is outside vocabulary");
        if (token) result.token=selected;
        if (verify) result.verification_selection=selected;
        return result;
    }
    auto logits=backend.execute(request.input_token,request.position);
    validate_serial_decode_engine_logits(logits,request.vocab_size);
    if (token || verify) {
        const auto selected=select_greedy_token(logits,request.vocab_size);
        if (token) result.token=selected;
        if (verify) result.verification_selection=selected;
    }
    if (logprobs) compute_requested_logprobs(logits,request.logprob_token_ids,result.requested_logprobs);
    if (full) result.logits=std::move(logits);
    return result;
}
} // namespace strix
