#pragma once

#include "coordinator.hpp"
#include "gemma4/serve/adapter.hpp"

namespace strix::gemma4::engine {

struct ChatRequest {
    gemma4::serve::ChatInput input;
    Request generation;
    std::uint32_t thinking_budget = 0;
};
struct ChatResult {
    gemma4::serve::Output output;
    std::vector<std::uint32_t> ids;
    std::vector<strix::serve::Json> receipts;
    std::string finish;
    std::size_t prompt_tokens = 0;
    bool budget_continued = false;
};
// Callback returns false on cancellation/disconnect, including during prefill.
using ChatSink = std::function<bool(const gemma4::serve::OutputDelta&)>;
using Cancelled = std::function<bool()>;
ChatResult run_chat(Backend& backend, const gemma4::serve::Adapter& adapter,
    ChatRequest request, DraftFactory drafts = {}, const ChatSink& sink = {}, const Cancelled& cancelled = {},
    std::shared_ptr<CostModel> costs = {});

} // namespace strix::gemma4::engine
