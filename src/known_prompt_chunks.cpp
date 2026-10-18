#include "strix/known_prompt_chunks.hpp"
#include "strix/known_prefill_limits.hpp"

#include "strix/serial_decode_replay.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strix {
KnownPromptChunkCursor::KnownPromptChunkCursor(KnownPromptChunkRequest request,
    KnownPromptWindowBackend& windows, SerialGreedyStepBackend& serial)
    : request_(request), windows_(windows), serial_(serial) {
    // Reuse the existing complete-request validation without invoking a model.
    const SerialGreedyTokenCursor validated({request.prompt_tokens,request.vocab_size,
        request.context_capacity,request.max_new_tokens,request.eos_token_id});
    (void)validated;
    if(request.tiled_prefill) {
        if(request.logical_rows<16U || request.logical_rows>known_prefill_max_rows ||
            !windows.supports_tiled_prefill())
            throw std::invalid_argument("tiled prompt prefill requires an explicitly capable T"+known_prefill_rows_range()+" backend");
        if(request.output_policy==KnownWindowOutputPolicy::state_only && request.max_new_tokens)
            throw std::invalid_argument("state-only tiled prefill supports zero-emission requests only");
    } else if((request.logical_rows!=2U && request.logical_rows!=4U && request.logical_rows!=8U) ||
        windows.supports_tiled_prefill())
        throw std::invalid_argument("known prompt chunk rows must be closed2/4/8 or explicit tiled prefill");
    if(request.all_row_heads&&(!request.tiled_prefill||request.output_policy!=KnownWindowOutputPolicy::all_rows))
        throw std::invalid_argument("all-row heads require the tiled prefill with the all-rows output policy");
    if(windows.logical_rows()!=request.logical_rows)
        throw std::invalid_argument("known prompt chunk backend row width differs from the request");
    validate_known_window_output_policy(request.output_policy);
    if(windows.output_policy()!=request.output_policy)
        throw std::invalid_argument("known prompt chunk output policy differs from the backend");
    if(request.start_position>request.context_capacity ||
        request.prompt_tokens.size()>request.context_capacity-request.start_position)
        throw std::invalid_argument("known prompt chunks exceed the remaining context capacity");
    if(request.vocab_size>std::numeric_limits<std::size_t>::max()/sizeof(float))
        throw std::length_error("known prompt frontier exceeds host storage capacity");
    // New cursor-owned presentation storage exists before the first pair can
    // consume state. Returned backend vectors follow their own native contract.
    if(request.output_policy!=KnownWindowOutputPolicy::state_only || request.max_new_tokens)
        logits_.resize(static_cast<std::size_t>(request.vocab_size));
    progress_.next_position=request.start_position;
}

bool KnownPromptChunkCursor::advance() {
    if(progress_.failed) throw std::logic_error("failed known-prompt cursor cannot resume");
    if(progress_.cancelled || progress_.prefill_complete) return false;
    const auto consumed=static_cast<std::size_t>(progress_.consumed_prompt_tokens);
    const auto remaining=request_.prompt_tokens.size()-consumed;
    const bool state_only=request_.output_policy==KnownWindowOutputPolicy::state_only;
    const bool no_output=request_.max_new_tokens==0U;
    const bool window=remaining>=request_.logical_rows&&(!state_only||no_output||remaining>request_.logical_rows);
    // Tiled prefill: a window with more prompt after it is never read, so it runs without its head.
    // STRIX_PREFILL_HEADLESS=0 keeps the head on every block.
    static const bool allow_headless=[]{const char* v=std::getenv("STRIX_PREFILL_HEADLESS");return !v||v[0]!='0';}();
    const bool headless=allow_headless&&!request_.all_row_heads&&request_.tiled_prefill&&!state_only&&window&&remaining>request_.logical_rows;
    try {
        if(headless) {
            windows_.execute_state_rows(request_.prompt_tokens.subspan(consumed,request_.logical_rows),progress_.next_position);
            has_logits_=false;selection_.reset();
        } else if(state_only&&(no_output||window||remaining>1U)) {
            if(window) {
                windows_.execute_state_rows(request_.prompt_tokens.subspan(consumed,request_.logical_rows),progress_.next_position);
                ++progress_.state_only_windows;
            } else serial_.execute_state(request_.prompt_tokens[consumed],progress_.next_position);
            has_logits_=false;selection_.reset();
        } else {
            auto values=window ? windows_.execute_rows(request_.prompt_tokens.subspan(consumed, request_.logical_rows),
                    progress_.next_position) : serial_.execute(request_.prompt_tokens[consumed],progress_.next_position);
            validate_serial_decode_engine_logits(values,request_.vocab_size);
            std::copy(values.begin(),values.end(),logits_.begin());
            has_logits_=true;
        }
        const auto rows=window?request_.logical_rows:1U;
        progress_.consumed_prompt_tokens+=rows;
        progress_.next_position+=rows;
        if(window) ++progress_.windows_executed;
        else ++progress_.serial_tail_steps;
        progress_.prefill_complete=progress_.consumed_prompt_tokens==request_.prompt_tokens.size();
        if(progress_.prefill_complete && has_logits_) selection_=select_greedy_token(logits_,request_.vocab_size);
        return true;
    } catch(...) {
        progress_.failed=true;has_logits_=false;selection_.reset();
        throw;
    }
}

void KnownPromptChunkCursor::cancel() noexcept {
    if(!progress_.generation_complete && !progress_.failed) progress_.cancelled=true;
}

const SerialGreedyRequestResult& KnownPromptChunkCursor::finish_greedy() {
    if(progress_.failed || progress_.cancelled || !progress_.prefill_complete ||
        (!has_logits_ && request_.max_new_tokens) || generation_)
        throw std::logic_error("known-prompt generation requires a completed uncancelled fresh frontier");
    if(!has_logits_) {
        // Explicit state-only/maxnew0 consumed all known inputs without a head.
        // There is no selection or pending bonus to manufacture or replay.
        generation_.emplace();
        generation_->final_position=progress_.next_position;
        progress_.generation_complete=true;
        return *generation_;
    }
    const SerialGreedyContinuationRequest continuation{logits_,progress_.next_position,request_.vocab_size,
        request_.context_capacity,request_.max_new_tokens,request_.eos_token_id};
    try {
        auto result=execute_serial_greedy_continuation(continuation,serial_);
        progress_.consumed_generation_tokens=result.steps_executed;
        progress_.next_position=result.final_position;
        progress_.emitted_tokens=result.generated_tokens.size();
        if(!result.generated_tokens.empty()) {
            progress_.bonus_pending=result.generated_tokens.back();
            selection_=progress_.bonus_pending;
        }
        if(result.steps_executed) has_logits_=false;
        generation_=std::move(result);
        progress_.generation_complete=true;
        return *generation_;
    } catch(...) {
        progress_.failed=true;has_logits_=false;selection_.reset();
        throw;
    }
}

KnownPromptChunkFrontier KnownPromptChunkCursor::frontier() const noexcept {
    KnownPromptChunkFrontier value;
    value.valid=!progress_.failed && (progress_.consumed_prompt_tokens || has_logits_ || selection_.has_value());
    value.state_position=progress_.next_position;
    if(progress_.consumed_prompt_tokens || progress_.consumed_generation_tokens)
        value.last_input_position=progress_.next_position-1U;
    if(has_logits_) value.logits=logits_;
    value.unconsumed_selection=selection_;
    value.selection_emitted=progress_.bonus_pending.has_value();
    return value;
}
} // namespace strix
