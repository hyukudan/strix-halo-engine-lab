// DownFrame (moe_core/decode_schedule.hpp): the 2-way column split of the decode down projection.
// Invariants: Part 0 is the incumbent indexing (row * Hidden + column); Parts 1 and 2 together own every 256-column
// group exactly once, each half stores a dense [rows][Hidden / 2] image, and the concatenation of the head and tail
// images per row is the Part 0 image element for element (same values, no reordering of any sum).
#include "moe_core/decode_schedule.hpp"

#include <cstddef>
#include <cstdio>
#include <vector>

namespace {
using strix::moe_core::DownFrame;

int failures = 0;
void check(bool ok, const char* what, unsigned hidden, unsigned row, unsigned column) {
    if (ok) return;
    if (failures < 16) std::printf("FAIL %s hidden=%u row=%u column=%u\n", what, hidden, row, column);
    ++failures;
}

// The private form the frame replaces: out[r * (H / 2) + (Part == 2 ? o - H / 2 : o)] for Part 1 / 2, out[r * H + o].
template <unsigned Part, unsigned Hidden> std::size_t private_index(unsigned row, unsigned column) {
    if constexpr (Part == 0U) return static_cast<std::size_t>(row) * Hidden + column;
    else return static_cast<std::size_t>(row) * (Hidden / 2U) + (Part == 2U ? column - Hidden / 2U : column);
}

template <unsigned Hidden> void run(unsigned rows) {
    using Whole = DownFrame<0U, Hidden>;
    using Head = DownFrame<1U, Hidden>;
    using Tail = DownFrame<2U, Hidden>;
    static_assert(Whole::kLaunchGroups == Hidden / 256U && Head::kLaunchGroups + Tail::kLaunchGroups == Whole::kLaunchGroups);
    static_assert(Whole::kFirstGroup == 0U && Head::kFirstGroup == 0U && Tail::kFirstGroup == Hidden / 512U);
    std::vector<unsigned> owner(Whole::kGroups, 0U);
    for (unsigned g = 0; g < Head::kLaunchGroups; ++g) ++owner[Head::group(g)];
    for (unsigned g = 0; g < Tail::kLaunchGroups; ++g) ++owner[Tail::group(g)];
    for (unsigned g = 0; g < Whole::kGroups; ++g) {
        check(owner[g] == 1U, "group owned once", Hidden, 0U, g * 256U);
        check(Whole::group(g) == g, "part 0 group identity", Hidden, 0U, g * 256U);
    }
    // Values: a distinct float per (row, column) stands for the R16 result of that element (computed identically by
    // every part, the frame moves only the store).
    auto value = [](unsigned r, unsigned o) { return static_cast<float>(r) * 65536.0F + static_cast<float>(o); };
    std::vector<float> whole(static_cast<std::size_t>(rows) * Hidden, -1.0F), head(whole.size() / 2U, -1.0F), tail(whole.size() / 2U, -1.0F);
    std::vector<unsigned> head_hits(head.size(), 0U), tail_hits(tail.size(), 0U);
    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned g = 0; g < Whole::kLaunchGroups; ++g)
            for (unsigned c = 0; c < 256U; ++c) {
                const unsigned o = Whole::group(g) * 256U + c;
                check(Whole::out_index(r, o) == private_index<0U, Hidden>(r, o), "part 0 index", Hidden, r, o);
                whole[Whole::out_index(r, o)] = value(r, o);
            }
        for (unsigned g = 0; g < Head::kLaunchGroups; ++g)
            for (unsigned c = 0; c < 256U; ++c) {
                const unsigned o = Head::group(g) * 256U + c;
                const std::size_t i = Head::out_index(r, o);
                check(i == private_index<1U, Hidden>(r, o) && i < head.size(), "head index", Hidden, r, o);
                if (i < head.size()) head[i] = value(r, o), ++head_hits[i];
            }
        for (unsigned g = 0; g < Tail::kLaunchGroups; ++g)
            for (unsigned c = 0; c < 256U; ++c) {
                const unsigned o = Tail::group(g) * 256U + c;
                const std::size_t i = Tail::out_index(r, o);
                check(i == private_index<2U, Hidden>(r, o) && i < tail.size(), "tail index", Hidden, r, o);
                if (i < tail.size()) tail[i] = value(r, o), ++tail_hits[i];
            }
    }
    for (std::size_t i = 0; i < head.size(); ++i) {
        check(head_hits[i] == 1U, "head dense", Hidden, 0U, static_cast<unsigned>(i));
        check(tail_hits[i] == 1U, "tail dense", Hidden, 0U, static_cast<unsigned>(i));
    }
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned o = 0; o < Hidden; ++o) {
            const float split = o < Hidden / 2U ? head[static_cast<std::size_t>(r) * (Hidden / 2U) + o]
                                                : tail[static_cast<std::size_t>(r) * (Hidden / 2U) + (o - Hidden / 2U)];
            check(split == whole[static_cast<std::size_t>(r) * Hidden + o] && split == value(r, o), "halves == whole", Hidden, r, o);
        }
}
}  // namespace

int main() {
    for (unsigned rows = 1; rows <= 8; ++rows) {
        run<4096U>(rows);
        run<1024U>(rows);
        run<6144U>(rows);
    }
    static_assert(DownFrame<0U, 3328U>::kLaunchGroups == 13U);  // Part 0 admits an odd group count
    if (failures != 0) {
        std::printf("decode down frame: %d failures\n", failures);
        return 1;
    }
    std::printf("decode down frame: PASS\n");
    return 0;
}
