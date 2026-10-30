#pragma once
// Host-only frozen admission prior. Structural cells are pending native qualification.
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::minimax::dspark {
enum class Mode { off, automatic, on };
enum class Prior { structural_v1, unqualified_v1 };
inline constexpr const char* auto_prior_version = "structural-v1";
inline Prior parse_prior(const char* value) {
    if (value == nullptr || std::string_view(value) == "structural-v1") return Prior::structural_v1;
    if (std::string_view(value) == "unqualified-v1") return Prior::unqualified_v1;
    throw std::invalid_argument("STRIX_MM_DSPARK_PRIOR / --dspark-prior: structural-v1 | unqualified-v1");
}
inline const char* prior_name(Prior p) { return p == Prior::structural_v1 ? auto_prior_version : "unqualified-v1"; }
// Frozen structural-v1 constants. ASCII byte features; no language/prompt-ID lookup.
inline constexpr std::uint64_t long_prompt_tokens = 6551;
inline constexpr std::size_t json_min_pairs = 2, json_min_keys = 2, json_density_percent = 2;
inline constexpr std::size_t code_min_marks = 8, code_density_percent = 6;
inline constexpr std::size_t code_min_indented = 3, code_indent_percent = 40, code_min_keywords = 2;
struct Features {
    std::size_t bytes = 0, pairs = 0, keys = 0, marks = 0, lines = 0, indented = 0, keywords = 0;
    bool balanced = true, fence = false;
};
inline Features features(std::string_view text) {
    Features f; f.bytes = text.size(); f.fence = text.find("```") != std::string_view::npos;
    // A typed stack prevents crossed delimiters; strings/escapes do not count as structure.
    std::string stack;
    bool quoted = false, escaped = false;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (c == ';' || c == '{' || c == '}' || c == '(' || c == ')' || c == '=') ++f.marks;
        if (quoted) {
            if (escaped) { escaped = false; continue; }
            if (c == '\\') { escaped = true; continue; }
            if (c == '"') {
                quoted = false;
                auto j = i + 1;
                while (j < text.size() && (text[j] == ' ' || text[j] == '\t' || text[j] == '\r' || text[j] == '\n')) ++j;
                if (j < text.size() && text[j] == ':') ++f.keys;
            }
        } else if (c == '"') quoted = true;
        else if (c == '{' || c == '[') stack.push_back(c);
        else if (c == '}' || c == ']') {
            if (stack.empty() || stack.back() != (c == '}' ? '{' : '[')) f.balanced = false;
            else { stack.pop_back(); ++f.pairs; }
        }
    }
    f.balanced = f.balanced && stack.empty() && !quoted;
    for (std::size_t start = 0; start < text.size();) {
        auto end = text.find('\n', start); if (end == std::string_view::npos) end = text.size();
        auto line = text.substr(start, end - start);
        if (line.find_first_not_of(" \t\r") != std::string_view::npos) {
            ++f.lines;
            if (line.starts_with("    ") || line.starts_with("\t")) ++f.indented;
        }
        start = end + 1;
    }
    const auto word = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || (c >= '0' && c <= '9'); };
    for (std::size_t i = 0; i < text.size();) {
        if (!word(text[i])) { ++i; continue; }
        auto end = i + 1; while (end < text.size() && word(text[end])) ++end;
        const auto w = text.substr(i, end - i);
        for (const auto k : {"def", "return", "class", "import", "for", "while", "if", "else", "function", "const", "let", "var", "fn", "struct", "public", "void", "int", "package", "func"})
            if (w == k) { ++f.keywords; break; }
        i = end;
    }
    return f;
}
inline const char* structural_reason(const Features& f, std::uint64_t tokens) {
    if (tokens >= long_prompt_tokens) return "auto:structural_v1_long";
    if (f.bytes && f.balanced && f.pairs >= json_min_pairs && f.keys >= json_min_keys &&
        200 * f.pairs >= json_density_percent * f.bytes) return "auto:structural_v1_json";
    if (f.fence || (f.bytes && f.marks >= code_min_marks && 100 * f.marks >= code_density_percent * f.bytes) ||
        (f.lines && f.indented >= code_min_indented && 100 * f.indented >= code_indent_percent * f.lines && f.keywords >= code_min_keywords))
        return "auto:structural_v1_code";
    return "plain:structural_v1_prose";
}
inline Mode parse_mode(std::string_view value) {
    if (value == "0" || value == "off") return Mode::off;
    if (value == "auto") return Mode::automatic;
    if (value == "1" || value == "on") return Mode::on;
    throw std::invalid_argument("STRIX_MM_SERVE_DSPARK / dspark_mode: off | auto | on (0 | 1)");
}
inline const char* mode_name(Mode mode) {
    return mode == Mode::off ? "off" : mode == Mode::on ? "on" : "auto";
}
inline std::uint32_t parse_inflight(const char* value) {
    if (value == nullptr) return 1;
    const std::string_view v(value);
    if (v.size() != 1 || v[0] < '0' || v[0] > '8')
        throw std::invalid_argument("STRIX_MM_SERVE_DSPARK_INFLIGHT: 0..8");
    return static_cast<std::uint32_t>(v[0] - '0');
}
struct AutoAdmission {
    Mode mode = Mode::automatic;
    bool assets = false;
    std::uint32_t live = 0;
    bool competing = false;
    std::uint64_t prompt = 0, max_rows = 0, remaining = 0;
    bool room = false;
    Prior prior = Prior::structural_v1;
    Features structure;
};
struct AutoRequest {
    enum class State { plain, active, plain_final };
    State state = State::plain;
    const char* reason = "plain:prior_unqualified";
    void admit(const AutoAdmission& a) {
        if (state != State::plain) return;
        reason = a.mode == Mode::off ? "plain:off" : !a.assets ? "plain:assets_missing" :
                 a.live != 0 || a.competing ? "plain:concurrency" :
                 a.prompt > a.max_rows ? "plain:grouped_taps_unqualified" :
                 !a.room ? "plain:capacity" : a.remaining < 2 ? "plain:short_request" : a.prior == Prior::unqualified_v1 ? "plain:prior_unqualified" : structural_reason(a.structure, a.prompt);
        state = std::string_view(reason).starts_with("auto:") ? State::active : State::plain_final;
    }
    void retire(const char* why) { state = State::plain_final; reason = why; }
};
}  // namespace strix::minimax::dspark
