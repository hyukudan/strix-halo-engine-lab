#include "strix/session/tool_source.hpp"

#include <algorithm>
#include <array>
#include <chrono>

namespace strix::session {
namespace {

// Numbering separators, preferred first on a tie (": " before ":"): cat -n, an arrow (U+2192), table and colon forms.
constexpr std::array<std::string_view, 6> separators{"\t", "\xE2\x86\x92", " | ", ": ", ":", "| "};

struct Line {
    std::string_view body;  // without its '\n'
    bool terminated = false;
};

std::vector<Line> split_lines(std::string_view text) {
    std::vector<Line> lines;
    std::size_t at = 0;
    while (at < text.size()) {
        const auto nl = text.find('\n', at);
        if (nl == std::string_view::npos) { lines.push_back({text.substr(at), false}); break; }
        lines.push_back({text.substr(at, nl - at), true});
        at = nl + 1;
    }
    return lines;
}

// "   12<rest>": the number and what follows its digits; false when the line does not start that way.
bool numbered(std::string_view line, std::uint64_t& number, std::string_view& rest) {
    std::size_t i = 0;
    while (i < line.size() && i < 12 && line[i] == ' ') ++i;
    const auto digits = i;
    number = 0;
    while (i < line.size() && line[i] >= '0' && line[i] <= '9' && i - digits < 9) number = number * 10 + std::uint64_t(line[i++] - '0');
    if (i == digits || (i < line.size() && line[i] >= '0' && line[i] <= '9')) return false;
    rest = line.substr(i);
    return true;
}

// Content of `rest` under separator `sep`; a numbered empty line may carry the separator without trailing blanks.
bool content(std::string_view rest, std::string_view sep, std::string_view& out) {
    if (rest.substr(0, sep.size()) == sep) { out = rest.substr(sep.size()); return true; }
    auto bare = sep;
    while (!bare.empty() && bare.back() == ' ') bare.remove_suffix(1);
    if (bare.size() < sep.size() && rest == bare) { out = {}; return true; }
    return false;
}

// Every recognised run of `lines`, appended to `out` with its numbering removed.
void numbered_runs(const std::vector<Line>& lines, std::uint32_t min_lines, std::vector<std::string>& out) {
    const auto n = lines.size();
    std::size_t i = 0;
    while (i + std::max<std::uint32_t>(min_lines, 1) <= n) {
        std::uint64_t first = 0;
        std::string_view rest;
        if (!numbered(lines[i].body, first, rest)) { ++i; continue; }
        std::size_t best = 0, best_sep = 0;
        for (std::size_t s = 0; s < separators.size(); ++s) {
            std::size_t k = 0;
            for (; i + k < n; ++k) {
                std::uint64_t number = 0;
                std::string_view r, c;
                if (!numbered(lines[i + k].body, number, r) || number != first + k || !content(r, separators[s], c)) break;
                if (!lines[i + k].terminated && i + k + 1 < n) break;
            }
            if (k > best) { best = k; best_sep = s; }
        }
        if (best < min_lines || best < 2) { ++i; continue; }
        std::string text;
        for (std::size_t k = 0; k < best; ++k) {
            std::uint64_t number = 0;
            std::string_view r, c;
            (void)numbered(lines[i + k].body, number, r);
            (void)content(r, separators[best_sep], c);
            text.append(c);
            if (lines[i + k].terminated) text.push_back('\n');
        }
        out.push_back(std::move(text));
        i += best;
    }
}

int hex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

bool hex4(std::string_view s, std::size_t at, std::uint32_t& v) {
    if (at + 4 > s.size()) return false;
    v = 0;
    for (std::size_t k = 0; k < 4; ++k) {
        const int h = hex(s[at + k]);
        if (h < 0) return false;
        v = v * 16 + std::uint32_t(h);
    }
    return true;
}

void utf8(std::uint32_t cp, std::string& out) {
    if (cp < 0x80) out.push_back(char(cp));
    else if (cp < 0x800) { out.push_back(char(0xC0 | (cp >> 6))); out.push_back(char(0x80 | (cp & 0x3F))); }
    else if (cp < 0x10000) {
        out.push_back(char(0xE0 | (cp >> 12))); out.push_back(char(0x80 | ((cp >> 6) & 0x3F))); out.push_back(char(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(char(0xF0 | (cp >> 18))); out.push_back(char(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(char(0x80 | ((cp >> 6) & 0x3F))); out.push_back(char(0x80 | (cp & 0x3F)));
    }
}

} // namespace

bool decode_json_string(std::string_view body, std::string& out) {
    out.clear();
    out.reserve(body.size());
    for (std::size_t i = 0; i < body.size(); ++i) {
        const auto c = body[i];
        if (static_cast<unsigned char>(c) < 0x20 || c == '"') return false;
        if (c != '\\') { out.push_back(c); continue; }
        if (++i >= body.size()) return false;
        switch (body[i]) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
            std::uint32_t cp = 0;
            if (!hex4(body, i + 1, cp)) return false;
            i += 4;
            if (cp >= 0xDC00 && cp <= 0xDFFF) return false;  // lone low surrogate
            if (cp >= 0xD800 && cp <= 0xDBFF) {
                std::uint32_t lo = 0;
                if (i + 2 >= body.size() || body[i + 1] != '\\' || body[i + 2] != 'u' || !hex4(body, i + 3, lo) ||
                    lo < 0xDC00 || lo > 0xDFFF) return false;
                i += 6;
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            }
            utf8(cp, out);
            break;
        }
        default: return false;
        }
    }
    return true;
}

std::string strip_line_numbers(std::string_view text, std::uint32_t min_lines) {
    std::vector<std::string> runs;
    numbered_runs(split_lines(text), min_lines, runs);
    return runs.empty() ? std::string() : std::move(runs.front());
}

std::vector<RestoredBlock> restore_tool_text(std::string_view text, const ToolSourceConfig& config) {
    std::vector<RestoredBlock> blocks;
    std::vector<std::string> runs;
    numbered_runs(split_lines(text), config.min_lines, runs);
    for (auto& r : runs) if (r.size() >= config.min_block_bytes) blocks.push_back({RestoreKind::numbered, std::move(r)});
    // JSON string literals holding escapes: the decoded text (numbering removed when the decoded text is numbered).
    std::string decoded;
    std::size_t at = 0;
    while ((at = text.find('"', at)) != std::string_view::npos) {
        std::size_t j = at + 1;
        bool escaped = false, raw_control = false;
        while (j < text.size() && text[j] != '"') {
            if (static_cast<unsigned char>(text[j]) < 0x20) { raw_control = true; break; }
            if (text[j] == '\\') { escaped = true; ++j; }
            ++j;
        }
        if (raw_control) { at += 1; continue; }  // not a JSON string: resynchronise on the next quote
        if (j >= text.size()) break;
        const auto body = text.substr(at + 1, j - at - 1);
        at = j + 1;
        if (!escaped || !decode_json_string(body, decoded) || decoded.size() < config.min_block_bytes) continue;
        runs.clear();
        numbered_runs(split_lines(decoded), config.min_lines, runs);
        bool any = false;
        for (auto& r : runs) if (r.size() >= config.min_block_bytes) { blocks.push_back({RestoreKind::json_numbered, std::move(r)}); any = true; }
        if (!any) blocks.push_back({RestoreKind::json_escaped, decoded});
    }
    return blocks;
}

std::shared_ptr<const lookup::Corpus> build_tool_source(std::span<const std::string_view> tool_results,
    const EncodeText& encode, const ToolSourceConfig& config, ToolSourceStats* stats) {
    const auto t0 = std::chrono::steady_clock::now();
    ToolSourceStats s;
    std::vector<std::uint32_t> tokens;
    for (const auto result : tool_results) {
        if (s.scanned_bytes >= config.max_scan_bytes || s.restored_bytes >= config.max_restored_bytes) break;
        const auto text = result.substr(0, config.max_scan_bytes - s.scanned_bytes);
        s.scanned_bytes += text.size();
        ++s.results;
        for (const auto& block : restore_tool_text(text, config)) {
            if (s.restored_bytes + block.text.size() > config.max_restored_bytes) break;
            const auto ids = encode(block.text);
            if (ids.empty()) continue;
            s.restored_bytes += block.text.size();
            ++s.blocks;
            for (const auto id : ids) if (id != lookup::corpus_break) tokens.push_back(id);
            tokens.push_back(lookup::corpus_break);
        }
    }
    s.tokens = tokens.size() - s.blocks;
    std::shared_ptr<const lookup::Corpus> corpus;
    if (s.blocks) corpus = std::make_shared<const lookup::Corpus>(tokens);
    s.build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (stats) *stats = s;
    return corpus;
}

} // namespace strix::session
