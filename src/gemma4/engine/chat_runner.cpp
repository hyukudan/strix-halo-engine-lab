#include "chat_runner.hpp"

#include <stdexcept>

namespace strix::gemma4::engine {
ChatResult run_chat(Backend& backend, const gemma4::serve::Adapter& adapter, ChatRequest request,
    DraftFactory drafts, const ChatSink& sink, const Cancelled& cancelled, std::shared_ptr<CostModel> costs) {
    if (request.generation.max_tokens > UINT32_MAX) throw std::invalid_argument("max_tokens exceeds chat limit");
    if (request.thinking_budget && (!request.input.enable_thinking || request.thinking_budget >= request.generation.max_tokens))
        throw std::invalid_argument("thinking_budget requires thinking and must be below max_tokens");
    const auto prepared = adapter.prepare(request.input);
    ChatResult result;
    auto prompt = prepared.ids;
    bool starts_in_reasoning = prepared.starts_in_reasoning;
    const auto maximum = request.generation.max_tokens;
    std::uint64_t draws = request.generation.sampling.draws;
    for (unsigned phase = 0; phase < 2; ++phase) {
        auto job = request.generation;
        job.prompt = prompt; job.stop_ids = prepared.stop_ids; job.sampling.draws = draws;
        job.stop_ids.push_back(adapter.tokenizer().find("<|tool_response>"));
        job.max_tokens = phase == 0 && request.thinking_budget ? request.thinking_budget : maximum-result.ids.size();
        result.prompt_tokens += prompt.size();
        Coordinator coordinator(backend,drafts,costs);
        const auto slot = coordinator.submit(std::move(job));
        gemma4::serve::OutputParser parser(starts_in_reasoning);
        std::size_t sent = 0;
        bool disconnected = false;
        while (coordinator.result(slot).status == Status::running) {
            if ((cancelled && cancelled()) || disconnected) { coordinator.cancel(slot); break; }
            coordinator.step();
            const auto& ids = coordinator.result(slot).ids;
            while (sent < ids.size()) {
                const auto delta = parser.push(adapter.tokenizer().token_bytes(ids[sent++]));
                if (sink && !sink(delta)) { disconnected = true; coordinator.cancel(slot); break; }
            }
        }
        const auto tail = parser.finish();
        if (!disconnected && sink && !sink(tail)) disconnected = true;
        const auto& out = coordinator.result(slot);
        if (out.status == Status::failed) throw std::runtime_error(out.error);
        draws = out.rng_draws;
        auto receipt = coordinator.receipt(slot);
        receipt.set("chat_policy",strix::serve::Json(strix::serve::Json::Object{
            {"enable_thinking",request.input.enable_thinking},
            {"conversational_mode",request.input.conversational_mode},
            {"thinking_budget",std::uint64_t(request.thinking_budget)},
            {"phase",std::uint64_t(phase)},
            {"selection",phase ? "budget-answer-continuation" : request.thinking_budget ? "within-thinking-budget" : "ordinary-chat"}}));
        result.receipts.push_back(std::move(receipt));
        result.ids.insert(result.ids.end(),out.ids.begin(),out.ids.end());
        const auto& parsed = parser.output();
        result.output.reasoning += parsed.reasoning; result.output.content += parsed.content;
        for (const auto& call : parsed.tool_calls.as_array()) result.output.tool_calls.push(call);
        result.output.reasoning_complete = result.output.reasoning_complete || parsed.reasoning_complete;
        result.output.unfinished_call = parsed.unfinished_call; result.output.unfinished_control = parsed.unfinished_control;
        result.output.stopped = parsed.stopped;
        result.finish = disconnected ? "cancelled" : status_name(out.status);
        if (phase || !request.thinking_budget || out.status != Status::length || disconnected) break;
        const auto continuation = gemma4::serve::continue_after_budget(request.input,parsed,
            std::uint32_t(out.ids.size()),request.thinking_budget,std::uint32_t(maximum));
        prompt = adapter.tokenizer().encode(continuation.prompt);
        starts_in_reasoning = continuation.prompt.ends_with("<|channel>thought\n");
        result.budget_continued = true;
    }
    if (result.output.tool_calls.size() && result.finish == "stop") result.finish = "tool_calls";
    return result;
}
} // namespace strix::gemma4::engine
