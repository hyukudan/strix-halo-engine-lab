#include "strix/ple_staged_gather.hpp"

#include <array>
#include <iostream>

namespace {

int failures = 0;
void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

constexpr std::size_t rows = 4;
constexpr std::uint32_t heads = 2;
using Hashes = std::array<strix::PleHashResult, rows>;
using Ids = std::array<std::uint32_t, rows * heads>;

// CPU copy oracle for the caller's deferred gather: one distinct byte per head.
using Bytes = std::array<std::uint8_t, rows * heads>;
void gather(const Hashes& hashes, std::span<const std::size_t> deferred, Bytes& bytes) {
    for (const auto r : deferred)
        for (std::uint32_t h = 0; h < heads; ++h)
            bytes[r * heads + h] = static_cast<std::uint8_t>(hashes[r].row_ids[h]);
}

}  // namespace

int main() {
    Hashes hashes{};
    Ids ids{};
    for (std::size_t r = 0; r < rows; ++r) {
        hashes[r].row_count = heads;
        for (std::uint32_t h = 0; h < heads; ++h) {
            ids[r * heads + h] = static_cast<std::uint32_t>(r * heads + h + 1U);
            hashes[r].row_ids[h] = ids[r * heads + h];
        }
    }
    const std::vector<std::size_t> all{0, 1, 2, 3}, tail{1, 2, 3};
    using strix::ple_detail::accept_staged_gather;
    for (const bool consumed : {false, true}) {
        auto deferred = consumed ? tail : all;
        Bytes bytes{};
        gather(hashes, all, bytes);  // completed staging worker
        if (consumed) gather(hashes, std::array<std::size_t, 1>{0}, bytes);
        expect(accept_staged_gather(hashes, heads, ids, consumed, deferred) == rows,
            "accepted receipt counts the entire staged block, including consumed row 0");
        expect(deferred.empty(), "accepted block clears all remaining deferred entries");
        gather(hashes, deferred, bytes);
        Bytes expected{};
        gather(hashes, all, expected);
        expect(bytes == expected, "staged hit preserves inline gather bytes");

        // Every head of every row must match, including independently consumed row 0.
        for (std::size_t i = 0; i < ids.size(); ++i) {
            auto wrong = ids;
            ++wrong[i];
            deferred = consumed ? tail : all;
            const auto original = deferred;
            expect(accept_staged_gather(hashes, heads, wrong, consumed, deferred) == 0,
                "any mismatching row/head rejects staging");
            expect(deferred == original, "hash mismatch retains the inline fallback work list");
            Bytes fallback{};
            if (consumed) gather(hashes, std::array<std::size_t, 1>{0}, fallback);
            gather(hashes, deferred, fallback);
            expect(fallback == expected, "rejected stage falls back to identical copied bytes");
        }
        for (std::size_t r = 0; r < rows; ++r) {
            auto wrong = hashes;
            --wrong[r].row_count;
            deferred = consumed ? tail : all;
            const auto original = deferred;
            expect(accept_staged_gather(wrong, heads, ids, consumed, deferred) == 0 && deferred == original,
                "any row head-count mismatch rejects without changing fallback");
        }
    }

    auto deferred = tail;
    expect(accept_staged_gather(hashes, heads, ids, false, deferred) == 0 && deferred == tail,
        "row 0 cannot be omitted without explicit successful consumption");
    for (const auto& invalid : {std::vector<std::size_t>{0, 2, 3},
             std::vector<std::size_t>{1, 1, 3}, std::vector<std::size_t>{1, 2}, all}) {
        deferred = invalid;
        expect(accept_staged_gather(hashes, heads, ids, true, deferred) == 0 && deferred == invalid,
            "consumption permits only the ordered tail, not arbitrary missing or duplicate rows");
    }
    deferred = tail;
    expect(accept_staged_gather(hashes, heads, std::span(ids).first(ids.size() - 1U), true, deferred) == 0 &&
        deferred == tail, "short staged extent rejects without changing fallback");
    expect(accept_staged_gather(hashes, 0, ids, true, deferred) == 0 && deferred == tail,
        "invalid head geometry rejects without changing fallback");
    expect(accept_staged_gather(hashes, strix::PleHashContract::maximum_heads + 1U, ids, true, deferred) == 0 &&
        deferred == tail, "oversized head geometry rejects without changing fallback");
    deferred = {1};
    expect(accept_staged_gather(std::span(hashes).first(2), heads, ids, true, deferred) == 2 && deferred.empty(),
        "a smaller launch may accept a matching prefix of staged rows");
    deferred.clear();
    expect(accept_staged_gather({}, heads, ids, true, deferred) == 0,
        "empty launch cannot be accepted");

    if (!failures) std::cout << "PLE staged gather acceptance and fallback tests passed\n";
    return failures ? 1 : 0;
}
