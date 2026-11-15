#pragma once
// Tool-result source for prompt lookup (host only, every backend).
//
// A coding agent reads a file through a tool and then writes it back (whole-file rewrite, edit call). The tool
// result in the prompt is often not the file's own text: it is line-numbered ("     12\t", "    12→", "12: ", "12 | ")
// or a JSON string with escapes ("\n", "\"", "é"). The model writes the original text, so the context lookup
// of the committed tokens follows the file only until the end of a line: there the context continues with the next
// number or an escape and the model with the next line.
//
// restore_tool_text() finds those blocks in a tool result and returns their original text: a run of lines numbered
// consecutively (one separator form, at least `min_lines` lines) loses only the recognised numbering, a JSON string
// literal with valid escapes is decoded (a literal with an invalid escape or a raw control character is left out).
// Nothing else is changed and the prompt itself is never touched: the restored text only feeds an auxiliary lookup
// corpus (lookup::Corpus), so a drafter can propose its continuation. The verifier decides every token, so the
// output is the greedy output whatever the corpus holds.
//
// CPU bound: one linear scan of each tool result (at most `max_scan_bytes` in all) and one tokenizer call per
// restored block (at most `max_restored_bytes` in all), once per request.
#include "strix/lever.hpp"
#include "strix/lookup/index.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strix::session {

// Server switch (request field "tool_lookup" overrides it per request). Drafts only: the output is the greedy output
// either way. Standalone experimental declaration (off until its ABAB receipt; then default on).
inline constexpr lever::Decl tool_source_lever{
    "tool-source-lookup", "STRIX_TOOL_LOOKUP", lever::Exactness::bitwise, false, {}, {},
    "context lookup also over tool results restored to their original text (numbering removed, JSON escapes decoded)"};
[[nodiscard]] inline bool tool_source_default() {
    static const auto choice = lever::select(tool_source_lever, lever::Inputs{});
    return choice.on;
}

struct ToolSourceConfig {
    std::uint32_t min_lines = 3;                 // a numbered run is recognised from this many consecutive lines
    std::size_t min_block_bytes = 24;            // shorter restored blocks are not worth a corpus entry
    std::size_t max_scan_bytes = 4U << 20;       // tool-result bytes scanned per request
    std::size_t max_restored_bytes = 256U << 10; // restored bytes tokenized per request (corpus index ~16 B/slot)
};

enum class RestoreKind : std::uint8_t { numbered = 0, json_escaped = 1, json_numbered = 2 };

struct RestoredBlock {
    RestoreKind kind = RestoreKind::numbered;
    std::string text;
};

// Restored blocks of one tool result, in order (empty: nothing recognised).
[[nodiscard]] std::vector<RestoredBlock> restore_tool_text(std::string_view text, const ToolSourceConfig& config = {});

// The first recognised numbered run of `text` with its numbering removed; empty when there is none.
[[nodiscard]] std::string strip_line_numbers(std::string_view text, std::uint32_t min_lines = 3);

// Decodes one JSON string body (between the quotes). False: an invalid escape, a lone surrogate or a raw control
// character (then `out` is unspecified).
[[nodiscard]] bool decode_json_string(std::string_view body, std::string& out);

struct ToolSourceStats {
    std::uint32_t results = 0, blocks = 0;
    std::size_t scanned_bytes = 0, restored_bytes = 0, tokens = 0;
    double build_ms = 0.0;
};

// The corpus of one request: every restored block of every tool result, tokenized by `encode` (the serving
// tokenizer, no special tokens). Null when no block was restored (the drafter then behaves as without it).
using EncodeText = std::function<std::vector<std::uint32_t>(std::string_view)>;
[[nodiscard]] std::shared_ptr<const lookup::Corpus> build_tool_source(std::span<const std::string_view> tool_results,
    const EncodeText& encode, const ToolSourceConfig& config = {}, ToolSourceStats* stats = nullptr);

} // namespace strix::session
