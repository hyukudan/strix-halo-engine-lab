// CPU-only byte-identity replay. One independent request per line of comma-separated token ids.
// strix-ds41-engram-trace-bench MODEL.gguf TRACE.ids [batch_rows=8]
#include "strix/ds41_engram.hpp"
#include "strix/ds41_formats.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

int main(int argc, char** argv) try {
    using namespace strix::ds41;
    if (argc < 3 || argc > 4) throw std::invalid_argument("usage: MODEL.gguf TRACE.ids [batch_rows=8]");
    const auto batch_rows = argc == 4 ? std::stoul(argv[3]) : 8UL;
    if (batch_rows == 0 || batch_rows > 1024) throw std::invalid_argument("batch rows 1..1024");
    const auto model = strix::deepseek41::validate_target(strix::deepseek41::read_file(argv[1]));
    const auto tables = engram_tables(model);
    const auto budget = engram_payload_cache_budget();
    if (budget == 0) throw std::invalid_argument("enable STRIX_DS41_ENGRAM_PAYLOAD_CACHE=1 for replay");
    EngramRowReader cached(argv[1], tables, 64, true, budget), plain(argv[1], tables, 64, true, 0);
    EngramHasher hasher(model.engram);
    std::ifstream trace(argv[2]);
    if (!trace) throw std::runtime_error("cannot open trace");
    std::size_t sessions = 0, tokens = 0;
    for (std::string line; std::getline(trace, line);) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream input(line);
        std::vector<std::uint32_t> ids;
        for (std::uint32_t id; input >> id;) ids.push_back(id);
        if (ids.empty()) continue;
        const auto rows = hasher.hash_sequence(ids);
        const auto per = std::size_t{hasher.layers()} * hasher.columns();
        for (std::size_t first = 0; first < ids.size(); first += batch_rows) {
            std::vector<EngramRowRequest> requests;
            for (auto r = first; r < std::min(ids.size(), first + batch_rows); ++r)
                for (std::uint32_t layer = 0; layer < hasher.layers(); ++layer)
                    for (std::uint32_t col = 0; col < hasher.columns(); ++col)
                        requests.push_back({layer, rows[r * per + std::size_t{layer} * hasher.columns() + col]});
            std::vector<std::uint8_t> a(requests.size() * engram_row_bytes), b(a.size());
            cached.read(requests, a); plain.read(requests, b);
            if (a != b) throw std::runtime_error("payload replay bytes differ");
        }
        ++sessions; tokens += ids.size();
    }
    if (tokens == 0) throw std::runtime_error("empty trace");
    const auto s = cached.stats(), off = plain.stats();
    std::printf("sessions=%zu tokens=%zu batch_rows=%lu requests=%llu hits=%llu misses=%llu duplicates=%llu read_bytes=%llu off_read_bytes=%llu storage_bytes=%llu capacity=%llu bytes_identical=1\n",
                sessions, tokens, batch_rows, static_cast<unsigned long long>(s.requests),
                static_cast<unsigned long long>(s.payload_hits), static_cast<unsigned long long>(s.payload_misses),
                static_cast<unsigned long long>(s.batch_duplicates), static_cast<unsigned long long>(s.page_bytes),
                static_cast<unsigned long long>(off.page_bytes), static_cast<unsigned long long>(s.cache_storage_bytes),
                static_cast<unsigned long long>(s.cache_capacity));
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
