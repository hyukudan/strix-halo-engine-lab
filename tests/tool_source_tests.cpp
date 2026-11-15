// Host tests: tool-result source (strix/session/tool_source.hpp), lookup::Corpus and the MtpNgramDraft binding.
#include "strix/lookup/index.hpp"
#include "strix/mtp_ngram_draft.hpp"
#include "strix/session/tool_source.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what.c_str()); }
}
using V = std::vector<std::uint32_t>;
V bytes(std::string_view s) { return V(s.begin(), s.end()); }  // byte "tokenizer" for the tests

const std::string file =
    "import sys\n"
    "\n"
    "def mean(values):\n"
    "    total = sum(values)\n"
    "    return total / len(values)\n"
    "\n"
    "def main(argv):\n"
    "    data = [float(x) for x in argv[1:]]\n"
    "    print(mean(data))\n"
    "    return 0\n";

std::string numbered_cat(const std::string& text) {  // cat -n form
    std::string out;
    std::size_t at = 0;
    int n = 1;
    while (at < text.size()) {
        const auto nl = text.find('\n', at);
        char head[16];
        std::snprintf(head, sizeof head, "%6d\t", n++);
        out += head;
        out += text.substr(at, nl - at + 1);
        at = nl + 1;
    }
    return out;
}

std::string json_escape(const std::string& text) {
    std::string out;
    for (const char c : text) {
        if (c == '\n') out += "\\n";
        else if (c == '\t') out += "\\t";
        else if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else out.push_back(c);
    }
    return out;
}

// Greedy copy of `truth` with a one-draft MTP chain: returns verify cycles; every emitted token equals truth.
std::size_t simulate(strix::MtpNgramDraft& d, const V& truth, bool& exact) {
    std::vector<double> acc;
    std::size_t pos = 0, cycles = 0;
    V out{truth[0]};
    while (pos + 2 < truth.size()) {
        const V drafts{truth[pos + 1]};
        auto ext = d.begin_cycle(truth[pos], drafts, 1, 8, acc);
        std::size_t take = 0;
        while (take < ext.size() && pos + 2 + take < truth.size() && ext[take] == truth[pos + 2 + take]) ++take;
        if (!ext.empty()) d.used(ext.size());
        V emitted(truth.begin() + std::ptrdiff_t(pos + 1), truth.begin() + std::ptrdiff_t(std::min(truth.size(), pos + 3 + take)));
        d.finish_cycle(emitted);
        out.insert(out.end(), emitted.begin(), emitted.end());
        pos += emitted.size();
        ++cycles;
    }
    exact = V(out.begin(), out.end()) == V(truth.begin(), truth.begin() + std::ptrdiff_t(out.size()));
    return cycles;
}
} // namespace

int main() {
    using namespace strix;
    {
        const auto numbered = numbered_cat(file);
        check(session::strip_line_numbers(numbered) == file, "cat -n numbering removed, text kept");
        check(session::strip_line_numbers("     1\xE2\x86\x92" "a = 1\n     2\xE2\x86\x92" "b = 2\n     3\xE2\x86\x92\n     4\xE2\x86\x92" "c\n") ==
                  "a = 1\nb = 2\n\nc\n", "arrow numbering, numbered empty line");
        check(session::strip_line_numbers("10: x = 1\n11: y = 2\n12:\n13: z\n") == "x = 1\ny = 2\n\nz\n", "colon numbering");
        check(session::strip_line_numbers("1\ta\n2\tb\n4\tc\n").empty(), "non-consecutive numbers are not a run");
        check(session::strip_line_numbers("1\ta\n2\tb\n").empty(), "two lines are below min_lines");
        check(session::strip_line_numbers("Results:\n1. apples\n2. pears\n3. plums\n").empty(), "list items keep their numbers");
        const auto head = session::strip_line_numbers("header line\n  7\tq = 7\n  8\tr = 8\n  9\ts = 9\ntrailer\n");
        check(head == "q = 7\nr = 8\ns = 9\n", "run inside a tool result, header and trailer left out");
    }
    {
        std::string out;
        check(session::decode_json_string("a\\nb\\t\\\"c\\\\\\/", out) && out == "a\nb\t\"c\\/", "simple escapes");
        check(session::decode_json_string("\\u00e9\\ud83d\\ude00", out) && out == "\xC3\xA9\xF0\x9F\x98\x80", "unicode and surrogate pair");
        check(!session::decode_json_string("bad \\x41", out), "invalid escape rejected");
        check(!session::decode_json_string("\\ud83d alone", out), "lone high surrogate rejected");
        check(!session::decode_json_string("\\ude00", out), "lone low surrogate rejected");
        check(!session::decode_json_string("raw\nnewline", out), "raw control character rejected");
    }
    {
        const std::string escaped = "{\"path\":\"m.py\",\"content\":\"" + json_escape(file) + "\"}";
        auto blocks = session::restore_tool_text(escaped);
        check(blocks.size() == 1 && blocks[0].kind == session::RestoreKind::json_escaped && blocks[0].text == file, "JSON content decoded");
        const std::string both = "{\"output\":\"" + json_escape(numbered_cat(file)) + "\"}";
        blocks = session::restore_tool_text(both);
        check(blocks.size() == 1 && blocks[0].kind == session::RestoreKind::json_numbered && blocks[0].text == file,
              "JSON-escaped numbered output decoded then de-numbered");
        check(session::restore_tool_text("plain prose with \"a quote\" and no numbering at all\n").empty(), "prose restores nothing");
    }
    {
        const V tokens{1, 2, 3, 4, lookup::corpus_break, 5, 6, 7};
        lookup::Corpus c(tokens);
        check(c.blocks() == 2, "two blocks");
        auto m = c.match(V{9, 2, 3}, 2, 7);
        check(m.length == 2 && m.available == 1 && m.next[0] == 4, "continuation stops at the block's end");
        m = c.match(V{5, 6}, 2, 7);
        check(m.length == 2 && m.available == 1 && m.next[0] == 7, "last block ends at the final break");
        m = c.match(V{4, 5}, 2, 7);
        check(m.length == 0, "no match across a break");
        m = c.match(V{8, 9}, 2, 7);
        check(m.length == 0, "no match");
        std::uint32_t out[8];
        check(c.continuation(0, out) == 3 && out[2] == 4, "continuation from a position");
    }
    {
        const std::string tool = numbered_cat(file);
        std::vector<std::string_view> results{tool};
        session::ToolSourceStats stats;
        const auto corpus = session::build_tool_source(results, [](std::string_view s) { return bytes(s); }, {}, &stats);
        check(corpus && stats.blocks == 1 && stats.tokens == file.size(), "corpus built from the numbered tool result");
        std::vector<std::string_view> none{std::string_view("no tool text here")};
        check(!session::build_tool_source(none, [](std::string_view s) { return bytes(s); }), "nothing restored: null corpus");

        // The model rewrites the file it read through a numbered tool result.
        const V prompt = bytes("Rewrite the file.\n" + tool + "\n<parameter=content>\n");
        const V truth = bytes(file + file.substr(0, 40));
        MtpNgramDraft plain, again, with_tool;
        plain.reset(prompt);
        again.reset(prompt);
        again.set_tool_source(nullptr);
        with_tool.reset(prompt);
        with_tool.set_tool_source(corpus);
        bool exact_plain = false, exact_again = false, exact_tool = false;
        const auto c_plain = simulate(plain, truth, exact_plain);
        const auto c_again = simulate(again, truth, exact_again);
        const auto c_tool = simulate(with_tool, truth, exact_tool);
        check(exact_plain && exact_again && exact_tool, "emitted tokens equal the greedy truth in every arm");
        check(c_again == c_plain && again.stats().accepted_rows == plain.stats().accepted_rows &&
                  again.stats().shadow_right_rows == plain.stats().shadow_right_rows, "null tool source: the route without it");
        check(c_tool < c_plain, "tool source cuts verify cycles");
        check(with_tool.stats().tool_accepted_rows > 0 && with_tool.stats().tool_cycles > 0, "tool rows counted");
        std::fprintf(stderr, "copy simulation: cycles plain %zu with tool source %zu (accepted ext rows %llu -> %llu)\n", c_plain, c_tool,
                     static_cast<unsigned long long>(plain.stats().accepted_rows),
                     static_cast<unsigned long long>(with_tool.stats().accepted_rows));
        // reset() drops the source.
        with_tool.reset(prompt);
        bool exact_reset = false;
        check(simulate(with_tool, truth, exact_reset) == c_plain && exact_reset, "reset clears the tool source");
    }
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::puts("tool source tests passed");
    return 0;
}
