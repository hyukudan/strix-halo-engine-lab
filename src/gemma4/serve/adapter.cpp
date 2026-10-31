#include "adapter.hpp"

#include <stdexcept>

namespace strix::gemma4::serve {

strix::serve::GenerationJob PreparedPrompt::job(std::uint32_t maximum) const {
    if (!maximum) throw std::invalid_argument("gemma4 adapter: max_tokens must be positive");
    strix::serve::GenerationJob out;
    out.prompt = ids;
    out.stop_tokens = stop_ids;
    out.max_tokens = maximum;
    return out;
}
Adapter::Adapter(std::shared_ptr<const Tokenizer> tokenizer) : tokenizer_(std::move(tokenizer)) {
    if (!tokenizer_ || tokenizer_->find("<bos>") != 2 || tokenizer_->find("<eos>") != 1 || tokenizer_->find("<turn|>") != 106)
        throw std::invalid_argument("gemma4 adapter: incompatible tokenizer boundaries");
}
PreparedPrompt Adapter::prepare(const ChatInput& input) const {
    PreparedPrompt out;
    out.rendered = render_chat(input);
    out.ids = tokenizer_->encode(out.rendered);
    out.stop_ids = {tokenizer_->find("<eos>"), tokenizer_->find("<turn|>")};
    out.starts_in_reasoning = out.rendered.ends_with("<|channel>thought\n");
    return out;
}
strix::serve::AssistantOutput shared_output(const Output& output) {
    strix::serve::AssistantOutput out;
    out.reasoning = output.reasoning; out.content = output.content;
    out.reasoning_complete = output.reasoning_complete;
    out.unfinished_call = output.unfinished_call;
    out.tool_calls = Json::array();
    for (const auto& call : output.tool_calls.as_array()) {
        const auto* fn = call.find("function");
        if (!fn || !fn->find("name") || !fn->find("arguments")) throw std::invalid_argument("gemma4 adapter: invalid parsed call");
        out.tool_calls.push(Json(Json::Object{{"name", *fn->find("name")}, {"arguments", *fn->find("arguments")}}));
        out.call_complete.push_back(true);
    }
    return out;
}

} // namespace strix::gemma4::serve
