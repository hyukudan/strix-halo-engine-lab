// DS41 adapter of the shared Engram component (strix/engram/*): the Spec of an admitted contract, the reader levers
// and the table locations. Hash, reader and row decode live in src/engram.
#include "strix/ds41_engram.hpp"

#include "strix/ds41_formats.hpp"
#include "strix/env_flags.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace strix::ds41 {
namespace {

void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}

engram::ReaderOptions reader_options(unsigned threads, bool direct_io, std::uint64_t payload_cache_bytes) {
    engram::ReaderOptions o;
    o.row_bytes = static_cast<std::uint32_t>(engram_row_bytes);
    o.threads = threads;
    o.direct_io = direct_io;
    o.payload_cache_bytes = payload_cache_bytes;
    // Resolved before the shared reader opens the descriptor: invalid governed settings must not leak it.
    o.ring_io = envcfg::flag("STRIX_DS41_ENGRAM_URING", true, EnvFlagMatch::exact);
    o.ring_depth = envcfg::number<unsigned>("STRIX_DS41_ENGRAM_URING_DEPTH", 256U, 0U, 4096U);
    return o;
}

} // namespace

std::uint64_t engram_payload_cache_budget() {
    const char* flag = std::getenv("STRIX_DS41_ENGRAM_PAYLOAD_CACHE");
    if (flag != nullptr && std::strcmp(flag, "0") == 0) return 0;  // default on; =0: off
    std::ifstream info("/proc/meminfo");
    std::string key, unit;
    std::uint64_t kib = 0, available = 0;
    while (info >> key >> kib >> unit) if (key == "MemAvailable:") { available = kib * 1024U; break; }
    std::uint64_t budget = std::min<std::uint64_t>(available / 64U, 256ULL << 20U);
    if (const char* v = std::getenv("STRIX_DS41_ENGRAM_PAYLOAD_CACHE_MIB")) {
        char* end = nullptr;
        errno = 0;
        const auto mib = std::strtoull(v, &end, 10);
        if (errno != 0 || end == v || *end != 0 || *v == '-' || mib > 256U) return 0;
        budget = std::min<std::uint64_t>(budget, mib << 20U);
    }
    return budget;
}


engram::Spec engram_spec(const deepseek41::EngramContract& c) {
    engram::Spec s;
    s.layer_ids = c.layer_ids;
    s.max_ngram_size = c.max_ngram_size;
    s.heads_per_order = c.n_heads;
    s.moduli = c.primes;
    s.multipliers = c.multipliers;
    s.compress = c.token_map;
    s.unmapped = engram::Unmapped::reject;
    s.pad_token = c.pad_id;
    s.row = {static_cast<std::uint32_t>(engram_row_elements), engram::RowScale::e8m0_per32};
    s.residency = engram::Residency::file_direct;
    require(c.max_ngram_size >= 2 && c.n_heads >= 1 && !c.layer_ids.empty() &&
                c.primes.size() == c.layer_ids.size() * c.rows_per_token() &&
                c.multipliers.size() == c.layer_ids.size() * c.max_ngram_size && c.pad_id < c.token_map.size() &&
                c.rows.size() == c.layer_ids.size(),
            "Engram contract geometry");
    for (std::uint32_t e = 0; e < s.layers(); ++e)
        require(engram::layer_rows(s, e) == c.rows[e], "Engram primes do not sum to the table rows");
    return s;
}

EngramRowReader::EngramRowReader(const std::filesystem::path& file, std::vector<EngramTable> tables, unsigned threads,
                                 bool direct_io, std::uint64_t payload_cache_bytes)
    : engram::RowReader(file, std::move(tables), reader_options(threads, direct_io, payload_cache_bytes)) {}

std::vector<EngramTable> engram_tables(const deepseek41::TargetModel& model) {
    std::vector<EngramTable> out;
    for (std::size_t e = 0; e < model.engram.layer_ids.size(); ++e) {
        const std::string name = "blk." + std::to_string(model.engram.layer_ids[e]) + ".engram_embd.weight";
        bool found = false;
        for (const auto& t : model.manifest.tensors)
            if (t.name == name) {
                require(t.byte_size == model.engram.rows[e] * engram_row_bytes, "Engram table size differs from its rows");
                out.push_back({t.absolute_offset, model.engram.rows[e]});
                found = true;
            }
        require(found, "Engram table missing from the manifest");
    }
    return out;
}

} // namespace strix::ds41
