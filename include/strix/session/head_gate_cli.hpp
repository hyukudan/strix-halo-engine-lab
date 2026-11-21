#pragma once

#include <charconv>
#include <filesystem>
#include <fstream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace strix::session {

enum class HeadGateMode { prefill, decode };

struct HeadGateWindow {
    std::string name;
    std::vector<unsigned> ids;
};

inline bool valid_head_gate_name(std::string_view name) {
    return !name.empty() &&
        name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") == name.npos;
}

inline unsigned head_gate_u32(std::string_view value) {
    unsigned number{};
    const auto result = std::from_chars(value.data(), value.data() + value.size(), number);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size())
        throw std::invalid_argument("expected an unsigned integer: " + std::string(value));
    return number;
}

inline std::vector<HeadGateWindow> read_head_gate_manifest(
    const std::filesystem::path& path, HeadGateMode mode, unsigned rows, unsigned max_context, unsigned vocabulary) {
    if (rows < 2U || rows > max_context)
        throw std::invalid_argument("--gate-rows must be at least 2 and fit the context");
    std::ifstream manifest(path);
    if (!manifest) throw std::runtime_error("could not open gate manifest: " + path.string());
    std::vector<HeadGateWindow> windows;
    std::unordered_set<std::string> seen;
    unsigned line_number = 0;
    for (std::string line; std::getline(manifest, line);) {
        ++line_number;
        std::istringstream fields(line);
        std::string name, count, ids_path, extra;
        if (!(fields >> name) || name.starts_with('#')) continue;
        const auto where = path.string() + ":" + std::to_string(line_number);
        if (!(fields >> count >> ids_path) || (fields >> extra) ||
            !valid_head_gate_name(name) || !seen.insert(name).second)
            throw std::invalid_argument(where + ": expected unique NAME TOKENS IDS_PATH");
        const auto size = head_gate_u32(count);
        if (size < rows || size > max_context || (mode == HeadGateMode::decode && size == rows))
            throw std::invalid_argument(where + ": invalid window size for --gate-rows and --gate-mode");
        const auto input = path.parent_path() / ids_path;
        std::ifstream tokens(input);
        if (!tokens) throw std::runtime_error("could not open gate IDs: " + input.string());
        auto& window = windows.emplace_back(HeadGateWindow{std::move(name), {}});
        window.ids.reserve(size);
        for (std::string field; tokens >> field;) {
            if (window.ids.size() == size)
                throw std::invalid_argument(where + ": more IDs than declared");
            const auto id = head_gate_u32(field);
            if (id >= vocabulary) throw std::invalid_argument(where + ": ID outside output vocabulary");
            window.ids.push_back(id);
        }
        if (!tokens.eof() || tokens.bad() || window.ids.size() != size)
            throw std::invalid_argument(where + ": ID count differs from manifest");
    }
    if (!manifest.eof() || manifest.bad() || windows.empty())
        throw std::invalid_argument("gate manifest is empty or unreadable: " + path.string());
    return windows;
}

// Each output row follows consumption of its teacher token. The reducer labels
// it with the next prompt ID and discards the unlabeled final row.
template <class Decoder>
void run_head_gate_window(Decoder& decoder, std::span<const unsigned> ids, HeadGateMode mode,
                          unsigned rows, unsigned vocabulary, std::span<float> logits) {
    if (rows < 2U || vocabulary == 0U || ids.size() < rows ||
        logits.size() % vocabulary != 0 || logits.size() / vocabulary != rows ||
        (mode == HeadGateMode::decode && ids.size() == rows))
        throw std::invalid_argument("invalid head gate window shape");
    decoder.reset_gate_session();
    if (mode == HeadGateMode::prefill) {
        decoder.prefill_gate_rows(ids, logits);
        return;
    }
    const auto split = ids.size() - rows;
    (void)decoder.prefill(ids.first(split), {});
    for (std::size_t row = 0; row < rows; ++row)
        (void)decoder.decode(ids[split + row], logits.subspan(row * vocabulary, vocabulary));
}

}  // namespace strix::session
