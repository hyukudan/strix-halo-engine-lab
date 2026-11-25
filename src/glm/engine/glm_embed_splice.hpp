#pragma once
// Host-only: which provided embedding rows of a prefill chunk a rank writes into its residual streams.
// The embed stage looks every token up, then overwrites the rows a splice names (session/embed_splice.hpp). With
// sequence-parallel prefill a rank embeds only the rows it owns (glm_seqpar.hpp), so it splices only those; the two
// ranks together write every provided row exactly once.
#include "glm/xfer/glm_seqpar.hpp"
#include "strix/session/embed_rows_file.hpp"
#include "strix/session/embed_splice.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strix::glm::engine {

struct EmbedPiece {
    std::uint32_t run_row = 0, rows = 0;  // chunk rows [run_row, run_row + rows)
    std::size_t staged_row = 0;           // their first row in the staging, where segments lie back to back
};

// chunk: the splice of one chunk of R rows (run rows relative to the chunk, validated by the caller).
[[nodiscard]] inline std::vector<EmbedPiece> embed_pieces(const session::EmbedSplice& chunk, unsigned R, bool seqpar,
                                                          unsigned pipeline, unsigned rank) {
    std::vector<xfer::SeqparRows> owned;
    if (seqpar) {
        for (unsigned i = 0; i < xfer::seqpar_tiles(R, pipeline); ++i)
            owned.push_back(xfer::seqpar_owner(xfer::seqpar_tile(R, pipeline, i), rank));
    } else {
        owned.push_back({0U, R});
    }
    std::vector<EmbedPiece> out;
    std::size_t staged = 0;
    for (const auto& seg : chunk.segments) {
        for (const auto& own : owned) {
            const std::uint32_t lo = std::max<std::uint32_t>(seg.run_row, own.first);
            const std::uint32_t hi = std::min<std::uint32_t>(seg.run_row + seg.rows, own.first + own.count);
            if (lo < hi) out.push_back({lo, hi - lo, staged + (lo - seg.run_row)});
        }
        staged += seg.rows;
    }
    return out;
}

// MTP teacher forcing over prompt rows [c0, c0 + pairs): drafter pair j joins the trunk hidden of row c0 + j with
// the embedding of row c0 + j + 1. The provided rows of those embeddings, with pair j as run row j.
[[nodiscard]] inline session::EmbedSplice mtp_pair_splice(const session::EmbedSplice& run, std::size_t c0, std::size_t pairs) {
    if (run.empty()) return {};
    return run.slice(c0 + 1U, c0 + 1U + pairs);
}

// Cap on the summed payload of all --embed-rows files: the rows stay in host memory for the whole run. 1 GiB is
// 131072 BF16 rows of the 4096-wide model, more than any prompt this engine prefills; a larger sum is a wrong file.
inline constexpr std::uint64_t kEmbedRowsByteCap = std::uint64_t{1} << 30U;

// Budget of the rows files from their headers alone, before any payload is read: the files must be exactly the
// prompt's runs of the spans' image token (count and lengths), so their rows never exceed the prompt's virtual
// positions, and the summed payload must fit byte_cap. Throws std::invalid_argument naming the mismatch.
inline void check_embed_row_headers(std::span<const std::uint32_t> prompt, std::span<const session::EmbedRowsFileInfo> infos,
                                    std::uint32_t width, std::uint64_t byte_cap = kEmbedRowsByteCap) {
    const auto fail = [](const std::string& what) { throw std::invalid_argument("embed rows: " + what); };
    if (infos.empty()) fail("no rows given");
    std::uint64_t rows = 0, bytes = 0;
    for (const auto& info : infos) {
        if (info.width != width) fail("a span does not have rows of the model's width");
        if (info.real_token_id != infos.front().real_token_id) fail("the spans were made for different image tokens");
        rows += info.rows;
        bytes += info.payload_bytes;
        if (rows > prompt.size()) fail("the rows files hold more rows than the prompt has positions (" + std::to_string(prompt.size()) + ")");
        if (bytes > byte_cap) fail("the rows files hold more than " + std::to_string(byte_cap) + " bytes");
    }
    const std::uint32_t token = infos.front().real_token_id;
    std::size_t next = 0;
    for (std::size_t i = 0; i < prompt.size();) {
        if (prompt[i] != token) {
            ++i;
            continue;
        }
        std::size_t end = i;
        while (end < prompt.size() && prompt[end] == token) ++end;
        if (next == infos.size()) fail("the prompt holds more image spans than rows were given (" + std::to_string(infos.size()) + ")");
        if (end - i != infos[next].rows)
            fail("image span " + std::to_string(next) + " of the prompt has " + std::to_string(end - i) + " tokens, its rows file has " +
                 std::to_string(infos[next].rows) + " rows");
        ++next;
        i = end;
    }
    if (next != infos.size())
        fail("the prompt holds " + std::to_string(next) + " image spans, " + std::to_string(infos.size()) + " rows files were given");
}

// The splice of a tokenized prompt whose image spans are runs of the spans' image token: the n-th run takes spans[n]
// and must be exactly as long as it has rows (two images are separated by their end and begin markers, so each is
// its own run). The splice owns the spans. Throws std::invalid_argument naming the mismatch.
[[nodiscard]] inline session::EmbedSplice prompt_embed_splice(std::span<const std::uint32_t> prompt,
                                                             std::vector<session::EmbedRows> spans, std::uint32_t width) {
    const auto fail = [](const std::string& what) { throw std::invalid_argument("embed rows: " + what); };
    if (spans.empty()) fail("no rows given");
    const std::uint32_t token = spans.front().real_token_id;
    for (const auto& span : spans) {
        if (span.width != width || span.rows == 0 || span.data.size() != std::size_t{span.rows} * span.width)
            fail("a span does not have rows of the model's width");
        if (span.real_token_id != token) fail("the spans were made for different image tokens");
        if (span.axes != 0 || !span.deep.empty()) fail("the engine takes plain content rows only");
    }
    const auto owned = std::make_shared<const std::vector<session::EmbedRows>>(std::move(spans));
    session::EmbedSplice out;
    out.width = width;
    std::size_t next = 0;
    for (std::size_t i = 0; i < prompt.size();) {
        if (prompt[i] != token) {
            ++i;
            continue;
        }
        std::size_t end = i;
        while (end < prompt.size() && prompt[end] == token) ++end;
        if (next == owned->size()) fail("the prompt holds more image spans than rows were given (" + std::to_string(owned->size()) + ")");
        const session::EmbedRows& span = (*owned)[next];
        if (end - i != span.rows)
            fail("image span " + std::to_string(next) + " of the prompt has " + std::to_string(end - i) + " tokens, its rows file has " +
                 std::to_string(span.rows) + " rows");
        session::EmbedSegment seg;
        seg.run_row = static_cast<std::uint32_t>(i);
        seg.rows = span.rows;
        seg.data = span.data.data();
        seg.extras = false;  // an image row is treated as the token row it replaces: the engine has no per-row extras
        out.segments.push_back(seg);
        ++next;
        i = end;
    }
    if (next != owned->size())
        fail("the prompt holds " + std::to_string(next) + " image spans, " + std::to_string(owned->size()) + " rows files were given");
    out.keep = owned;
    out.validate(prompt.size(), width);
    return out;
}

}  // namespace strix::glm::engine
