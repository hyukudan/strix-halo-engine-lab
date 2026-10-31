#pragma once

#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/json.hpp"

namespace strix::gemma4::serve {

struct Output {
    std::string reasoning, content;
    // OpenAI shape: [{id, type:"function", function:{name, arguments:object}}].
    strix::serve::Json tool_calls = strix::serve::Json::array();
    bool reasoning_complete = false;
    bool unfinished_call = false;
    bool unfinished_control = false;
    bool stopped = false;
};
struct OutputDelta {
    std::string reasoning, content;
    // Complete calls only. Partial calls are retained until their frame closes.
    strix::serve::Json tool_calls = strix::serve::Json::array();
};

// Accepts arbitrary byte chunks, including chunks inside UTF-8 and control markers.
// starts_in_reasoning is true when the prompt ends in <|channel>thought\n.
// A length cutoff may leave an incomplete thought/call; malformed closed frames throw.
class OutputParser {
public:
    explicit OutputParser(bool starts_in_reasoning = false, std::size_t max_frame_bytes = 1U << 20U);
    OutputDelta push(std::string_view bytes);
    OutputDelta finish();
    [[nodiscard]] const Output& output() const noexcept { return output_; }
private:
    enum class Mode { content, reasoning, tool };
    void drain(bool final, OutputDelta& delta);
    void emit(std::string_view bytes, OutputDelta& delta);
    Mode mode_;
    std::string pending_;
    std::size_t max_frame_bytes_;
    bool finished_ = false;
    bool tool_quoted_ = false;
    std::string frame_;
    strix::serve::Utf8StreamDecoder content_decoder_, reasoning_decoder_;
    Output output_;
};

[[nodiscard]] strix::serve::Json parse_arguments(std::string_view text);

} // namespace strix::gemma4::serve
