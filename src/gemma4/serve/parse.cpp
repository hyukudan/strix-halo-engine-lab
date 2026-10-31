#include "parse.hpp"

#include <array>
#include <stdexcept>

namespace strix::gemma4::serve {
namespace {
using strix::serve::Json;
constexpr std::string_view quote = "<|\"|>";
[[noreturn]] void fail(const char* what) { throw std::invalid_argument(std::string("gemma4 output: ") + what); }
class ArgumentParser {
public:
    explicit ArgumentParser(std::string_view text) : text_(text) {}
    Json document() {
        auto j = value(0); skip();
        if (at_ != text_.size() || !j.is_object()) fail("tool arguments must be one object");
        (void)strix::serve::parse_json_unique(j.dump());
        return j;
    }
private:
    void skip() { while (at_ < text_.size() && (text_[at_] == ' ' || text_[at_] == '\n' || text_[at_] == '\t' || text_[at_] == '\r')) ++at_; }
    bool take(char c) { skip(); if (at_ < text_.size() && text_[at_] == c) { ++at_; return true; } return false; }
    std::string string() {
        if (text_.substr(at_, quote.size()) != quote) fail("expected tool string delimiter");
        at_ += quote.size(); const auto end = text_.find(quote, at_);
        if (end == std::string_view::npos) fail("unterminated tool string");
        auto s = std::string(text_.substr(at_, end - at_)); at_ = end + quote.size();
        // Reuse strict JSON Unicode validation without treating tool backslashes as escapes.
        (void)strix::serve::parse_json_unique(Json(s).dump());
        return s;
    }
    Json value(unsigned depth) {
        if (depth > 64) fail("tool nesting exceeds 64");
        skip();
        if (at_ >= text_.size()) fail("missing tool value");
        if (text_.substr(at_, quote.size()) == quote) return Json(string());
        if (take('{')) {
            auto out = Json::object();
            if (take('}')) return out;
            do {
                skip(); std::string key;
                if (text_.substr(at_, quote.size()) == quote) key = string();
                else {
                    const auto start = at_;
                    while (at_ < text_.size() && text_[at_] != ':' && text_[at_] != ' ' && text_[at_] != '\n' && text_[at_] != '\r' && text_[at_] != '\t') {
                        if (std::string_view("{}[],").find(text_[at_]) != std::string_view::npos) fail("invalid tool key");
                        ++at_;
                    }
                    key = text_.substr(start, at_ - start);
                    if (key.empty()) fail("empty unquoted tool key");
                }
                if (!take(':')) fail("missing colon after tool key");
                if (out.find(key)) fail("duplicate tool key");
                out.set(std::move(key), value(depth + 1));
            } while (take(','));
            if (!take('}')) fail("missing closing tool brace");
            return out;
        }
        if (take('[')) {
            auto out = Json::array();
            if (take(']')) return out;
            do { out.push(value(depth + 1)); } while (take(','));
            if (!take(']')) fail("missing closing tool bracket");
            return out;
        }
        const auto start = at_;
        while (at_ < text_.size() && std::string_view(",]} \r\n\t").find(text_[at_]) == std::string_view::npos) ++at_;
        const auto j = strix::serve::parse_json_unique(text_.substr(start, at_ - start));
        if (j.is_string() || j.is_array() || j.is_object()) fail("invalid tool scalar");
        return j;
    }
    std::string_view text_;
    std::size_t at_ = 0;
};
} // namespace

Json parse_arguments(std::string_view text) { return ArgumentParser(text).document(); }

OutputParser::OutputParser(bool starts_in_reasoning, std::size_t maximum)
    : mode_(starts_in_reasoning ? Mode::reasoning : Mode::content), max_frame_bytes_(maximum) {
    if (!maximum) fail("frame limit must be positive");
}
void OutputParser::emit(std::string_view bytes, OutputDelta& delta) {
    if (mode_ == Mode::reasoning) {
        const auto s = reasoning_decoder_.push(bytes); output_.reasoning += s; delta.reasoning += s;
    } else {
        const auto s = content_decoder_.push(bytes); output_.content += s; delta.content += s;
    }
}
void OutputParser::drain(bool final, OutputDelta& delta) {
    constexpr std::string_view thought = "<|channel>thought\n", close = "<channel|>", call = "<|tool_call>", end_call = "<tool_call|>";
    constexpr std::array<std::string_view, 7> markers{thought, close, call, "<turn|>", "<eos>", "<|tool_response>", "<|channel>"};
    std::size_t at = 0;
    while (at < pending_.size() && !output_.stopped) {
        const std::string_view rest(pending_.data() + at, pending_.size() - at);
        if (mode_ == Mode::tool) {
            if (rest.starts_with(quote)) {
                frame_ += quote;
                if (frame_.size() > max_frame_bytes_) fail("tool frame exceeds byte limit");
                tool_quoted_ = !tool_quoted_; at += quote.size(); continue;
            }
            if (!tool_quoted_ && rest.starts_with(end_call)) {
                if (!frame_.starts_with("call:")) fail("tool frame must start with call:");
                const auto brace = frame_.find('{', 5);
                if (brace == std::string::npos || brace == 5) fail("missing tool function name or arguments");
                const auto name = frame_.substr(5, brace - 5);
                if (name.find_first_of(" \t\r\n[]}:,") != std::string::npos) fail("invalid function name");
                auto arguments = parse_arguments(std::string_view(frame_).substr(brace));
                Json tc(Json::Object{{"id", "call_" + std::to_string(output_.tool_calls.size())}, {"type", "function"},
                    {"function", Json(Json::Object{{"name", name}, {"arguments", std::move(arguments)}})}});
                output_.tool_calls.push(tc); delta.tool_calls.push(tc);
                frame_.clear(); mode_ = Mode::content; at += end_call.size(); continue;
            }
            if (!final && (quote.starts_with(rest) || (!tool_quoted_ && end_call.starts_with(rest)))) break;
            frame_ += pending_[at++];
            if (frame_.size() > max_frame_bytes_) fail("tool frame exceeds byte limit");
            continue;
        }
        bool matched = false, prefix = false;
        if (final && rest.starts_with("<|channel>") && thought.starts_with(rest)) {
            output_.unfinished_control = true; at = pending_.size(); break;
        }
        for (const auto m : markers) {
            // The generic channel prefix must wait for the full thought header.
            if (!final && m == "<|channel>" && thought.starts_with(rest)) { prefix = true; break; }
            if (rest.starts_with(m)) {
                matched = true;
                if (m == thought) {
                    if (mode_ != Mode::content) fail("nested reasoning channel");
                    const auto tail = content_decoder_.flush(); output_.content += tail; delta.content += tail;
                    mode_ = Mode::reasoning;
                } else if (m == close) {
                    if (mode_ != Mode::reasoning) fail("reasoning close without open");
                    const auto tail = reasoning_decoder_.flush(); output_.reasoning += tail; delta.reasoning += tail;
                    output_.reasoning_complete = true; mode_ = Mode::content;
                } else if (m == call) {
                    if (mode_ == Mode::reasoning) fail("tool call inside reasoning channel");
                    mode_ = Mode::tool;
                } else if (m == "<|channel>") fail("unsupported output channel");
                else output_.stopped = true;
                at += m.size(); break;
            }
            if (!final && m.starts_with(rest)) prefix = true;
        }
        if (matched) continue;
        if (prefix && !final) break;
        emit(rest.substr(0, 1), delta); ++at;
    }
    pending_.erase(0, at);
    if (output_.stopped) pending_.clear();
}
OutputDelta OutputParser::push(std::string_view bytes) {
    if (finished_) fail("push after finish");
    OutputDelta delta;
    if (output_.stopped) return delta;
    pending_.append(bytes); drain(false, delta); return delta;
}
OutputDelta OutputParser::finish() {
    if (finished_) fail("finish called twice");
    finished_ = true;
    OutputDelta delta; drain(true, delta);
    output_.unfinished_call = mode_ == Mode::tool;
    const auto r = reasoning_decoder_.flush(), c = content_decoder_.flush();
    output_.reasoning += r; delta.reasoning += r; output_.content += c; delta.content += c;
    return delta;
}

} // namespace strix::gemma4::serve
