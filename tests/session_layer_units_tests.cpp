// Prefill pieces of a session: even pieces (every multi-row piece even, an odd piece gives its last row to the next
// piece) and the unit arithmetic (LayerUnits) that plans them equal the pieces an engine loop cuts.
#include "strix/session/prefill_session.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

void require(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "session layer units: %s\n", what);
        std::exit(1);
    }
}

using strix::session::even_piece_rows;
using strix::session::LayerUnits;

static_assert(even_piece_rows(2037, 8192, false) == 2037 && even_piece_rows(9000, 8192, false) == 8192);
static_assert(even_piece_rows(2037, 8192, true) == 2036 && even_piece_rows(1, 8192, true) == 1);
static_assert(even_piece_rows(2048, 8192, true) == 2048 && even_piece_rows(5000, 2049, true) == 2048);
static_assert(even_piece_rows(1, 1, true) == 1 && even_piece_rows(3, 1, true) == 1);

void pieces_equal_plan() {
    for (std::uint32_t max_rows : {1U, 2U, 3U, 7U, 64U, 2048U, 2049U, 8192U})
        for (std::uint32_t prompt : {1U, 2U, 3U, 63U, 64U, 65U, 2037U, 2048U, 2049U, 8191U, 8192U, 8193U, 20001U})
            for (bool even : {false, true}) {
                std::vector<std::uint32_t> sizes;
                for (std::uint64_t done = 0; done < prompt;) {
                    const auto n = even_piece_rows(prompt - done, max_rows, even);
                    require(n >= 1 && n <= max_rows && n <= prompt - done, "piece within bounds");
                    if (even && n > 1) require(n % 2 == 0, "multi-row piece is even");
                    if (even && max_rows >= 2 && n == 1) require(done + 1 == prompt, "only the final piece has 1 row");
                    sizes.push_back(n);
                    done += n;
                }
                const LayerUnits units{prompt, max_rows, 3, even};
                require(units.pieces() == sizes.size() && units.total() == 3U * sizes.size(), "plan counts the pieces");
                for (std::size_t k = 0; k < sizes.size(); ++k)
                    for (std::uint32_t layer = 0; layer < 3; ++layer)
                        require(units.piece_rows_at(k * 3 + layer) == sizes[k], "plan rows equal piece rows");
                require(units.all_pieces_pausable([=](std::uint32_t r) { return !even || r == 1 || r % 2 == 0; }),
                        "every piece admitted");
                if (!even && max_rows < prompt)
                    require(units.piece_rows_at(0) == max_rows, "uneven plan keeps full pieces first");
            }
    const LayerUnits off{2037, 8192, 45};
    require(off.pieces() == 1 && off.piece_rows_at(0) == 2037, "default pieces unchanged");
    const LayerUnits on{2037, 8192, 45, true};
    require(on.pieces() == 2 && on.total() == 90 && on.piece_rows_at(44) == 2036 && on.piece_rows_at(45) == 1,
            "odd prompt: even piece then a 1-row piece");
    require(!on.all_pieces_pausable([](std::uint32_t r) { return r % 2 == 0; }), "the 1-row tail is checked");
}

}  // namespace

int main() {
    pieces_equal_plan();
    std::puts("session layer units ok");
    return 0;
}
