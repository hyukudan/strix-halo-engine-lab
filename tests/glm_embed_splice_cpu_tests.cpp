// Host test of the GLM prefill embed splice ownership (src/glm/engine/glm_embed_splice.hpp): a model of the embed
// stage of both ranks. Every chunk row must end with the right source (table or provided row), written by the rank
// that embeds it, for plain and sequence-parallel chunks, both pipelines, chunked prompts and cut segments.
#include "glm/engine/glm_embed_splice.hpp"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace strix;
using glm::engine::embed_pieces;

int failures = 0;
#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            ++failures;                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                               \
    } while (false)

constexpr std::uint32_t kWidth = 4;
constexpr std::int64_t kTable = -1, kUnwritten = -2;

// A prompt of `total` rows whose splice names `spans` (first row, rows); provided row j of a span holds the value
// base + j in its first element, so a row's source is readable from the data.
struct Prompt {
    std::vector<std::vector<std::uint16_t>> data;
    session::EmbedSplice splice;
    std::vector<std::int64_t> want;  // per prompt row: kTable or the provided value
    Prompt(const Prompt&) = delete;  // the segments point into `data`
    Prompt& operator=(const Prompt&) = delete;
    Prompt(std::uint32_t total, const std::vector<std::pair<std::uint32_t, std::uint32_t>>& spans) : want(total, kTable) {
        splice.width = kWidth;
        std::uint16_t base = 100;
        for (const auto& [first, rows] : spans) {
            auto& d = data.emplace_back(std::size_t{rows} * kWidth);
            for (std::uint32_t j = 0; j < rows; ++j) {
                d[std::size_t{j} * kWidth] = static_cast<std::uint16_t>(base + j);
                want[first + j] = base + j;
            }
            base = static_cast<std::uint16_t>(base + 1000);
        }
        for (std::size_t i = 0; i < spans.size(); ++i) {
            session::EmbedSegment s;
            s.run_row = spans[i].first;
            s.rows = spans[i].second;
            s.data = data[i].data();
            splice.segments.push_back(s);
        }
        splice.validate(total, kWidth);
    }
};

// The embed stage of one rank on one chunk: table rows on the rows it embeds, then its pieces from the staging.
void embed_rank(const session::EmbedSplice& chunk, unsigned R, bool seqpar, unsigned pipeline, unsigned rank,
                std::vector<std::int64_t>& x) {
    if (seqpar) {
        for (unsigned i = 0; i < glm::xfer::seqpar_tiles(R, pipeline); ++i) {
            const auto own = glm::xfer::seqpar_owner(glm::xfer::seqpar_tile(R, pipeline, i), rank);
            for (unsigned r = own.first; r < own.first + own.count; ++r) x[r] = kTable;
        }
    } else {
        for (unsigned r = 0; r < R; ++r) x[r] = kTable;
    }
    std::vector<std::uint16_t> staging;
    for (const auto& seg : chunk.segments) staging.insert(staging.end(), seg.data, seg.data + std::size_t{seg.rows} * kWidth);
    for (const auto& piece : embed_pieces(chunk, R, seqpar, pipeline, rank))
        for (std::uint32_t j = 0; j < piece.rows; ++j) {
            CHECK(piece.run_row + j < R);
            CHECK(x[piece.run_row + j] == kTable);  // only a row this rank embedded, and only once
            x[piece.run_row + j] = staging[(piece.staged_row + j) * kWidth];
        }
}

void run(const Prompt& p, std::uint32_t max_rows, unsigned pipeline, bool seqpar_on) {
    const auto total = static_cast<std::uint32_t>(p.want.size());
    for (std::uint32_t c0 = 0; c0 < total; c0 += max_rows) {
        const std::uint32_t R = std::min(max_rows, total - c0);
        const session::EmbedSplice chunk = p.splice.slice(c0, c0 + R);
        chunk.validate(R, kWidth);
        const bool seqpar = seqpar_on && glm::xfer::seqpar_admits(R);
        // What each rank holds after its embed stage; with sequence parallelism a row is defined on its owner only.
        std::vector<std::int64_t> x0(R, kUnwritten), x1(R, kUnwritten);
        embed_rank(chunk, R, seqpar, pipeline, 0, x0);
        embed_rank(chunk, R, seqpar, pipeline, 1, x1);
        for (std::uint32_t r = 0; r < R; ++r) {
            const std::int64_t want = p.want[c0 + r];
            if (seqpar) {
                CHECK((x0[r] == kUnwritten) != (x1[r] == kUnwritten));  // exactly one owner
                CHECK((x0[r] == kUnwritten ? x1[r] : x0[r]) == want);
            } else {
                CHECK(x0[r] == want && x1[r] == want);  // replicated: both ranks hold every row
            }
        }
    }
}

session::EmbedRows span_of(std::uint32_t rows, std::uint32_t token, std::uint16_t base) {
    session::EmbedRows s;
    s.rows = rows;
    s.width = kWidth;
    s.real_token_id = token;
    s.data.resize(std::size_t{rows} * kWidth);
    for (std::uint32_t j = 0; j < rows; ++j) s.data[std::size_t{j} * kWidth] = static_cast<std::uint16_t>(base + j);
    return s;
}
bool refused(const std::vector<std::uint32_t>& prompt, std::vector<session::EmbedRows> spans, const char* needle) {
    try {
        (void)glm::engine::prompt_embed_splice(prompt, std::move(spans), kWidth);
    } catch (const std::invalid_argument& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

// A prompt's image spans are runs of the image token; the n-th run reads the n-th rows file.
void test_prompt_splice() {
    constexpr std::uint32_t I = 9, B = 5, E = 6;
    const std::vector<std::uint32_t> prompt{1, 2, B, I, I, I, E, 3, B, I, I, E, 4};
    const auto splice = glm::engine::prompt_embed_splice(prompt, {span_of(3, I, 100), span_of(2, I, 200)}, kWidth);
    CHECK(splice.width == kWidth && splice.segments.size() == 2 && splice.rows() == 5 && splice.keep != nullptr);
    CHECK(splice.segments[0].run_row == 3 && splice.segments[0].rows == 3 && splice.segments[0].data[0] == 100 &&
          splice.segments[0].data[2 * kWidth] == 102 && !splice.segments[0].extras);
    CHECK(splice.segments[1].run_row == 9 && splice.segments[1].rows == 2 && splice.segments[1].data[kWidth] == 201);
    // The splice owns its rows: it outlives the vector it was built from.
    const session::EmbedSplice cut = splice.slice(4, 10);
    CHECK(cut.segments.size() == 2 && cut.segments[0].run_row == 0 && cut.segments[0].rows == 2 && cut.segments[0].data[0] == 101 &&
          cut.segments[1].run_row == 5 && cut.segments[1].rows == 1 && cut.segments[1].data[0] == 200);

    CHECK(refused(prompt, {}, "no rows given"));
    CHECK(refused(prompt, {span_of(3, I, 100)}, "more image spans than rows"));
    CHECK(refused(prompt, {span_of(3, I, 100), span_of(2, I, 200), span_of(2, I, 300)}, "2 image spans, 3 rows files"));
    CHECK(refused(prompt, {span_of(2, I, 100), span_of(3, I, 200)}, "has 3 tokens, its rows file has 2 rows"));
    CHECK(refused(prompt, {span_of(3, I, 100), span_of(2, I + 1, 200)}, "different image tokens"));
    CHECK(refused({1, 2, 3}, {span_of(3, I, 100)}, "0 image spans, 1 rows files"));
    session::EmbedRows wide = span_of(3, I, 100);
    wide.width = kWidth + 1;
    CHECK(refused(prompt, {wide, span_of(2, I, 200)}, "model's width"));
    session::EmbedRows axes = span_of(3, I, 100);
    axes.axes = 1;
    CHECK(refused(prompt, {axes, span_of(2, I, 200)}, "plain content rows"));
}

bool headers_refused(const std::vector<std::uint32_t>& prompt, const std::vector<session::EmbedRowsFileInfo>& infos, const char* needle,
                     std::uint64_t cap = glm::engine::kEmbedRowsByteCap) {
    try {
        glm::engine::check_embed_row_headers(prompt, infos, kWidth, cap);
    } catch (const std::invalid_argument& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

session::EmbedRowsFileInfo info_of(std::uint32_t rows, std::uint32_t token, std::uint32_t width = kWidth) {
    return {rows, width, token, std::uint64_t{rows} * width * 2U};
}

// --embed-rows budget from the headers alone, and a header read that never touches the payload.
void test_header_budget() {
    constexpr std::uint32_t I = 9, B = 5, E = 6;
    const std::vector<std::uint32_t> prompt{1, 2, B, I, I, I, E, 3, B, I, I, E, 4};
    glm::engine::check_embed_row_headers(prompt, std::vector{info_of(3, I), info_of(2, I)}, kWidth);  // fits: no throw
    CHECK(headers_refused(prompt, {}, "no rows given"));
    CHECK(headers_refused(prompt, {info_of(3, I, kWidth + 1), info_of(2, I)}, "model's width"));
    CHECK(headers_refused(prompt, {info_of(3, I), info_of(2, I + 1)}, "different image tokens"));
    CHECK(headers_refused(prompt, {info_of(3, I), info_of(2, I), info_of(9, I)}, "more rows than the prompt has positions"));
    CHECK(headers_refused(prompt, {info_of(3, I), info_of(2, I)}, "more than 30 bytes", 30));
    CHECK(headers_refused(prompt, {info_of(3, I)}, "more image spans than rows"));
    CHECK(headers_refused(prompt, {info_of(2, I), info_of(3, I)}, "has 3 tokens, its rows file has 2 rows"));
    CHECK(headers_refused({1, 2, 3}, {info_of(3, I)}, "0 image spans, 1 rows files"));

    // A file whose length does not match its header is refused without a payload read; a good header yields its facts.
    const auto path = std::filesystem::temp_directory_path() / "strix_glm_embed_header_budget.bin";
    session::EmbedRows rows;
    rows.rows = 3;
    rows.width = kWidth;
    rows.real_token_id = 9;
    rows.data.assign(std::size_t{3} * kWidth, 7);
    session::write_embed_rows_file(path, rows);
    const auto info = session::read_embed_rows_file_info(path, kWidth);
    CHECK(info.rows == 3 && info.width == kWidth && info.real_token_id == 9 && info.payload_bytes == 3U * kWidth * 2U);
    std::filesystem::resize_file(path, session::kEmbedRowsFileHeader + 4);
    bool length_refused = false;
    try {
        (void)session::read_embed_rows_file_info(path, kWidth);
    } catch (const std::invalid_argument& e) {
        length_refused = std::string(e.what()).find("length") != std::string::npos;
    }
    CHECK(length_refused);
    std::filesystem::remove(path);
}

// MTP teacher forcing: drafter pair j of the piece at c0 embeds prompt row c0 + j + 1. Over every piece of a chunked
// prompt, each prompt row but the first is embedded once, with the provided row where the run has one; the last
// piece has one pair fewer (its last row pairs with the generated token).
void test_mtp_pairs() {
    using Spans = std::vector<std::pair<std::uint32_t, std::uint32_t>>;
    for (const auto& [total, spans] : std::vector<std::pair<std::uint32_t, Spans>>{
             {64, {{5, 20}}}, {64, {{0, 64}}}, {64, {{0, 1}, {63, 1}}}, {61, {{7, 30}, {50, 11}}}, {200, {{10, 150}}}, {40, {}}}) {
        const Prompt p(total, spans);
        for (const std::uint32_t max_rows : {8U, 16U, 20U, 64U, 256U}) {
            std::vector<std::int64_t> embedded(total, kUnwritten);
            for (std::uint32_t c0 = 0; c0 < total; c0 += max_rows) {
                const std::uint32_t R = std::min(max_rows, total - c0), pairs = std::min(R, total - c0 - 1U);
                if (pairs == 0) continue;
                const session::EmbedSplice next = glm::engine::mtp_pair_splice(p.splice, c0, pairs);
                next.validate(pairs, kWidth);
                std::vector<std::int64_t> e(pairs, kTable);
                std::vector<std::uint16_t> staging;
                for (const auto& seg : next.segments) staging.insert(staging.end(), seg.data, seg.data + std::size_t{seg.rows} * kWidth);
                for (const auto& piece : embed_pieces(next, pairs, false, 1, 0))
                    for (std::uint32_t j = 0; j < piece.rows; ++j) e[piece.run_row + j] = staging[(piece.staged_row + j) * kWidth];
                for (std::uint32_t j = 0; j < pairs; ++j) {
                    CHECK(embedded[c0 + 1U + j] == kUnwritten);
                    embedded[c0 + 1U + j] = e[j];
                }
            }
            CHECK(embedded[0] == kUnwritten);
            for (std::uint32_t r = 1; r < total; ++r) CHECK(embedded[r] == p.want[r]);
        }
    }
}

}  // namespace

int main() {
    try {
        using Spans = std::vector<std::pair<std::uint32_t, std::uint32_t>>;
        const std::vector<std::pair<std::uint32_t, Spans>> cases{
            {64, {}},                              // text only
            {64, {{5, 20}}},                       // one image inside a chunk
            {64, {{0, 64}}},                       // every row provided
            {64, {{0, 1}, {63, 1}}},               // first and last row
            {64, {{3, 10}, {13, 7}, {40, 24}}},    // adjacent spans, one to the end
            {61, {{7, 30}, {50, 11}}},             // odd length: the last chunk takes the legacy path
            {200, {{10, 150}}},                    // a span cut by several chunks
        };
        for (const auto& [total, spans] : cases) {
            const Prompt p(total, spans);
            for (const std::uint32_t max_rows : {8U, 16U, 20U, 64U, 256U})
                for (const unsigned pipeline : {1U, 2U})
                    for (const bool seqpar : {false, true}) run(p, max_rows, pipeline, seqpar);
        }

        // Pieces of a plain chunk are the segments themselves, staged back to back.
        const Prompt p(32, {{2, 5}, {20, 3}});
        const auto pieces = embed_pieces(p.splice, 32, false, 1, 0);
        CHECK(pieces.size() == 2 && pieces[0].run_row == 2 && pieces[0].rows == 5 && pieces[0].staged_row == 0 &&
              pieces[1].run_row == 20 && pieces[1].rows == 3 && pieces[1].staged_row == 5);
        // Sequence parallelism, one tile: rank 0 owns rows [0, 16), rank 1 rows [16, 32).
        const auto r0 = embed_pieces(p.splice, 32, true, 1, 0), r1 = embed_pieces(p.splice, 32, true, 1, 1);
        CHECK(r0.size() == 1 && r0[0].run_row == 2 && r0[0].rows == 5 && r0[0].staged_row == 0);
        CHECK(r1.size() == 1 && r1[0].run_row == 20 && r1[0].rows == 3 && r1[0].staged_row == 5);
        CHECK(embed_pieces(session::EmbedSplice{}, 32, true, 2, 1).empty());
        test_prompt_splice();
        test_header_budget();
        test_mtp_pairs();
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    if (failures) return 1;
    std::puts("glm5-embed-splice: PASS");
    return 0;
}
