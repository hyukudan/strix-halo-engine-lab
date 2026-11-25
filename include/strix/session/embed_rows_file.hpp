#pragma once

// One span of embedding rows (embed_rows.hpp) as a file: what an offline encoder hands to a command-line run, and
// what both ranks of a pair read so that they splice the same bytes.
//
//   "STRIXEMB" | u32 version (1) | u32 rows | u32 width | u32 real_token_id | 32-byte digest | u64 FNV-1a of the rows
//   | rows * width BF16 values (little-endian, row-major)
//
// Content rows only: no roles, no position axes, no deep rows. Header only.

#include "strix/session/embed_rows.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <limits>
#include <stdexcept>
#include <string>

namespace strix::session {

inline constexpr char kEmbedRowsFileMagic[8] = {'S', 'T', 'R', 'I', 'X', 'E', 'M', 'B'};
inline constexpr std::uint32_t kEmbedRowsFileVersion = 1;
inline constexpr std::size_t kEmbedRowsFileHeader = 8 + 4 * 4 + 32 + 8;

[[nodiscard]] inline std::uint64_t embed_rows_fnv(const std::uint16_t* data, std::size_t count) noexcept {
    std::uint64_t h = 0xCBF29CE484222325ULL;
    for (std::size_t i = 0; i < count; ++i) {
        h = (h ^ (data[i] & 0xFFU)) * 0x100000001B3ULL;
        h = (h ^ (data[i] >> 8U)) * 0x100000001B3ULL;
    }
    return h;
}

// Throws std::invalid_argument for a span the format cannot hold, std::runtime_error on I/O failure.
inline void write_embed_rows_file(const std::filesystem::path& path, const EmbedRows& rows) {
    if (rows.rows == 0 || rows.rows > kEmbedMaxRows || rows.width == 0 ||
        rows.data.size() != std::size_t{rows.rows} * rows.width || rows.axes != 0 || !rows.deep.empty())
        throw std::invalid_argument("embed rows file: the span has no plain content rows");
    for (const EmbedRole role : rows.role)
        if (role != EmbedRole::content) throw std::invalid_argument("embed rows file: only content rows are stored");
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const auto put = [&out](const void* p, std::size_t n) { out.write(static_cast<const char*>(p), static_cast<std::streamsize>(n)); };
    const std::uint32_t head[4] = {kEmbedRowsFileVersion, rows.rows, rows.width, rows.real_token_id};
    const std::uint64_t fnv = embed_rows_fnv(rows.data.data(), rows.data.size());
    put(kEmbedRowsFileMagic, 8);
    put(head, sizeof head);
    put(rows.digest.data(), rows.digest.size());
    put(&fnv, 8);
    put(rows.data.data(), rows.data.size() * 2U);
    out.flush();
    if (!out) throw std::runtime_error("embed rows file: cannot write " + path.filename().string());
}

// Header facts of a rows file, read without touching the payload (budget a set of files before materializing any).
struct EmbedRowsFileInfo {
    std::uint32_t rows = 0;
    std::uint32_t width = 0;
    std::uint32_t real_token_id = 0;
    std::uint64_t payload_bytes = 0;  // rows * width * 2, equal to the file length minus the header
};

namespace detail {
// Opens the file, parses and validates the header and the file length; leaves `in` at the first payload byte.
inline EmbedRowsFileInfo open_embed_rows_file(std::ifstream& in, const std::filesystem::path& path, std::uint32_t width, EmbedRows& rows,
                                              std::uint64_t& fnv) {
    const std::string name = path.filename().string();
    in.open(path, std::ios::binary);
    if (!in) throw std::runtime_error("embed rows file: cannot open " + name);
    in.seekg(0, std::ios::end);
    const auto end = in.tellg();
    in.seekg(0, std::ios::beg);
    char magic[8];
    std::uint32_t head[4];
    fnv = 0;
    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(head), sizeof head);
    in.read(reinterpret_cast<char*>(rows.digest.data()), static_cast<std::streamsize>(rows.digest.size()));
    in.read(reinterpret_cast<char*>(&fnv), 8);
    if (!in || end < 0 || std::memcmp(magic, kEmbedRowsFileMagic, 8) != 0 || head[0] != kEmbedRowsFileVersion)
        throw std::invalid_argument("embed rows file: " + name + " is not a version 1 embed rows file");
    rows.rows = head[1];
    rows.width = head[2];
    rows.real_token_id = head[3];
    if (rows.rows == 0 || rows.rows > kEmbedMaxRows || rows.width == 0 || rows.width > 65536U)
        throw std::invalid_argument("embed rows file: " + name + " has an invalid shape");
    if (width != 0 && rows.width != width)
        throw std::invalid_argument("embed rows file: " + name + " has rows of width " + std::to_string(rows.width) +
                                    ", the model takes " + std::to_string(width));
    const std::uint64_t payload = std::uint64_t{rows.rows} * rows.width * 2U;
    if (payload > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()) ||
        payload / 2U > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t)))
        throw std::invalid_argument("embed rows file: " + name + " is too large for this platform");
    if (static_cast<std::uint64_t>(end) != kEmbedRowsFileHeader + payload)
        throw std::invalid_argument("embed rows file: " + name + " has a length other than its header states");
    return {rows.rows, rows.width, rows.real_token_id, payload};
}
}  // namespace detail

// Validates the header and the file length only; reads and allocates no rows.
[[nodiscard]] inline EmbedRowsFileInfo read_embed_rows_file_info(const std::filesystem::path& path, std::uint32_t width = 0) {
    std::ifstream in;
    EmbedRows scratch;
    std::uint64_t fnv = 0;
    return detail::open_embed_rows_file(in, path, width, scratch, fnv);
}

namespace detail {
inline EmbedRows read_embed_rows_payload(const std::filesystem::path& path, std::uint32_t width, const EmbedRowsFileInfo* admitted) {
    const std::string name = path.filename().string();
    std::ifstream in;
    EmbedRows rows;
    std::uint64_t fnv = 0;
    const auto info = detail::open_embed_rows_file(in, path, width, rows, fnv);
    if (admitted != nullptr && (info.rows != admitted->rows || info.width != admitted->width ||
                                info.real_token_id != admitted->real_token_id || info.payload_bytes != admitted->payload_bytes))
        throw std::invalid_argument("embed rows file: " + name + " changed after its header was admitted");
    const std::uint64_t payload = info.payload_bytes;
    rows.data.resize(std::size_t{rows.rows} * rows.width);
    in.read(reinterpret_cast<char*>(rows.data.data()), static_cast<std::streamsize>(payload));
    if (!in) throw std::runtime_error("embed rows file: short read of " + name);
    if (embed_rows_fnv(rows.data.data(), rows.data.size()) != fnv)
        throw std::invalid_argument("embed rows file: " + name + " fails its checksum");
    return rows;
}
}  // namespace detail

// width: the reader's LM hidden width (0: any). The file length must be exactly the header plus the rows. Throws
// std::invalid_argument for a malformed or foreign file, std::runtime_error on I/O failure.
[[nodiscard]] inline EmbedRows read_embed_rows_file(const std::filesystem::path& path, std::uint32_t width = 0) {
    return detail::read_embed_rows_payload(path, width, nullptr);
}

// The payload pass of a file whose header was admitted before (read_embed_rows_file_info + a budget): the reopened
// header must state exactly the admitted shape before any row is allocated, so a file replaced between the two opens
// is refused instead of growing the allocation past the admitted budget.
[[nodiscard]] inline EmbedRows read_embed_rows_file(const std::filesystem::path& path, const EmbedRowsFileInfo& admitted) {
    return detail::read_embed_rows_payload(path, admitted.width, &admitted);
}

} // namespace strix::session
