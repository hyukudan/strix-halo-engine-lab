#pragma once

// deepseek41 Engram n-gram memory, host side: the hash of every position (from the admitted
// EngramContract) and a concurrent reader of 264-byte rows from the model file on disk.
//
// Hash of position t at Engram layer e: the compressed ids of tokens t, t-1, ..., t-(N-1) (N =
// max_ngram_size; a position before the start of the sequence, and every position after it once one
// is blocked, reads the compressed pad id), products p_s = id_s * multiplier[e][s] in int64, the
// running XOR r_1 = p_0 ^ p_1, r_2 = r_1 ^ p_2, ...; the (s + 1)-gram's row for head h is
// r_s % prime[e][s - 1][h] + offset[e][(s - 1) * heads + h], offsets being the running sum of the
// layer's primes in that order. Columns are ordered n-gram size major, head minor.
//
// The hash, the reader and the row decode are the shared Engram component (strix/engram/*); this header is the
// DS41 adapter: the Spec of an admitted EngramContract, the 264-byte row format and the DS41 reader levers.

#include "strix/deepseek41_model.hpp"
#include "strix/engram/hasher.hpp"
#include "strix/engram/row_reader.hpp"

#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace strix::ds41 {

// Marks a token id of a sequence history (EngramHasher::push, AttentionState::tokens) as a dead n-gram position.
inline constexpr std::uint32_t kEngramDeadBit = engram::kDeadBit;
[[nodiscard]] constexpr std::uint32_t engram_dead_token(std::uint32_t token) noexcept { return engram::dead_token(token); }

// Shared Spec of an admitted contract: 264-byte rows (256 E4M3 + 8 E8M0), unmapped token ids rejected; checks that
// each layer's primes sum to its table rows.
[[nodiscard]] engram::Spec engram_spec(const deepseek41::EngramContract& contract);

class EngramHasher : public engram::Hasher {
public:
    explicit EngramHasher(const deepseek41::EngramContract& contract) : engram::Hasher(engram_spec(contract)) {}
};

using EngramTable = engram::Table;
using EngramRowRequest = engram::RowRequest;
using EngramReaderStats = engram::ReaderStats;

// STRIX_DS41_ENGRAM_PAYLOAD_CACHE=1 enables a per-reader CLOCK cache and batch offset deduplication.
// Default off. Budget is at most 1/64 of available host RAM and 256 MiB, including the fixed index;
// STRIX_DS41_ENGRAM_PAYLOAD_CACHE_MIB optionally lowers that budget. Unknown RAM disables caching.
[[nodiscard]] std::uint64_t engram_payload_cache_budget();

// The shared reader on 264-byte rows. STRIX_DS41_ENGRAM_URING=0: worker threads instead of the read queue;
// STRIX_DS41_ENGRAM_URING_DEPTH: reads in flight (default 256), both resolved before the file is opened.
class EngramRowReader : public engram::RowReader {
public:
    EngramRowReader(const std::filesystem::path& file, std::vector<EngramTable> tables, unsigned threads = 16,
                    bool direct_io = true, std::uint64_t payload_cache_bytes = engram_payload_cache_budget());
};

// Worker threads of the decode runners' Engram reader: STRIX_DS41_ENGRAM_THREADS=N (1..256), else `fallback`.
// Only the number of reads in flight changes; every row is the same bytes.
[[nodiscard]] inline unsigned engram_reader_threads(unsigned fallback) {
    const char* v = std::getenv("STRIX_DS41_ENGRAM_THREADS");
    if (v == nullptr || *v == 0) return fallback;
    const unsigned long n = std::strtoul(v, nullptr, 10);
    return n >= 1UL && n <= 256UL ? static_cast<unsigned>(n) : fallback;
}

// Engram table locations of an admitted target from its manifest.
[[nodiscard]] std::vector<EngramTable> engram_tables(const deepseek41::TargetModel& model);

} // namespace strix::ds41
