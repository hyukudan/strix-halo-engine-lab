// Host test of strix/residency/copy_source.hpp: the measured runtime rule and the piece plan of a registered-source
// resident copy (pieces cover the range exactly, in order; registrations are whole pages, never share a page, cover
// their piece and stay inside the mapping).
#include "strix/residency/copy_source.hpp"

#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
int failures = 0;
void require(bool value, const std::string& what) {
    if (!value) {
        std::fprintf(stderr, "FAIL %s\n", what.c_str());
        ++failures;
    }
}

void check_plan(std::uint64_t begin, std::uint64_t end, std::uint64_t piece, std::uint64_t page, std::uint64_t mapped) {
    const auto pieces = strix::model_copy::source_pieces(begin, end, piece, page, mapped);
    const std::string where = "[" + std::to_string(begin) + "," + std::to_string(end) + ") piece " + std::to_string(piece);
    require(begin == end ? pieces.empty() : !pieces.empty(), where + ": empty iff the range is empty");
    std::uint64_t at = begin, last_page_end = 0;
    for (const auto& p : pieces) {
        require(p.begin == at && p.begin < p.end, where + ": pieces are contiguous and non-empty");
        require(!piece || p.end - p.begin <= piece + page, where + ": piece bounded by the registration piece");
        require(p.register_begin % page == 0 && p.register_end % page == 0, where + ": registrations are whole pages");
        require(p.register_begin <= p.begin && p.end <= p.register_end, where + ": registration covers its piece");
        require(p.register_end - p.register_begin < (p.end - p.begin) + 2 * page, where + ": registration is the piece's pages");
        require(p.register_begin >= last_page_end, where + ": consecutive registrations share no page");
        require(p.register_end <= (mapped + page - 1) / page * page, where + ": registration inside the mapping");
        last_page_end = p.register_end;
        at = p.end;
    }
    require(at == end || pieces.empty(), where + ": pieces end at the range end");
}
}  // namespace

int main() {
    using strix::model_copy::register_source;
    // Measured table (gfx1151, 3 GiB fresh GGUF regions): HIP 7.16 direct 74-79 GB/s > registered 50-57;
    // HIP 7.17 direct 24-27 GB/s < registered 50-57. Discrete GPUs keep the direct copy (not measured there).
    require(!register_source(71626385, true), "HIP 7.16 integrated: direct copy");
    require(register_source(71726434, true), "HIP 7.17 integrated: registered source");
    require(register_source(80000000, true), "later runtimes integrated: registered source");
    require(!register_source(71726434, false), "discrete GPU: direct copy");
    require(!register_source(0, true), "unknown runtime: direct copy");

    using Range = std::pair<std::uint64_t, std::uint64_t>;
    constexpr std::uint64_t page = 4096, mib = 1ULL << 20U;
    const std::uint64_t mapped = 15ULL * 1024 * mib + 1234;
    for (const std::uint64_t piece : std::initializer_list<std::uint64_t>{0, page, 3 * page + 17, 64 * mib, 256 * mib})
        for (const auto& range : std::initializer_list<Range>{{0, 0}, {0, 1}, {0, page}, {1, page + 1},
                 {4095, 4097}, {123, 700 * mib + 5}, {64 * mib, 320 * mib}, {mapped - 77, mapped}, {7, mapped}})
            check_plan(range.first, range.second, piece, page, mapped);
    for (const auto& bad : std::initializer_list<Range>{{5, 4}, {0, mapped + 1}}) {
        bool threw = false;
        try { (void)strix::model_copy::source_pieces(bad.first, bad.second, 64 * mib, page, mapped); }
        catch (const std::invalid_argument&) { threw = true; }
        require(threw, "invalid range refused");
    }
    bool threw = false;
    try { (void)strix::model_copy::source_pieces(0, 10, 64 * mib, 3000, mapped); } catch (const std::invalid_argument&) { threw = true; }
    require(threw, "non power-of-two page refused");
    if (failures) return 1;
    std::puts("model copy source: PASS");
    return 0;
}
