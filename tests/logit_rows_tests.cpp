// Shared multi-row logits pitch: every row of a multi-row logits buffer starts on the dense head's output
// alignment, for odd vocabularies too, and the strided greedy pick admits exactly the buffers that hold its rows.
#include "strix/head_argmax.hpp"
#include "strix/logit_rows.hpp"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::cerr << "logit_rows FAIL: " << what << '\n';
        ++failures;
    }
}

void check_vocabulary(std::size_t vocabulary) {
    using namespace strix::head;
    const auto pitch = logit_row_pitch(vocabulary);
    check(pitch >= vocabulary && pitch - vocabulary < kLogitRowAlignmentBytes / sizeof(float), "pitch is the smallest padding");
    for (std::size_t row = 0; row < 9U; ++row)
        check(logit_row_offset(row, pitch) * sizeof(float) % kLogitRowAlignmentBytes == 0U, "every row starts aligned");
    check(logit_rows_elements(1U, vocabulary, pitch) == vocabulary, "one row needs only the vocabulary");
    check(logit_rows_elements(8U, vocabulary, pitch) == 7U * pitch + vocabulary, "eight rows");
    check(strix::head_argmax::strided_rows_fit(logit_rows_elements(8U, vocabulary, pitch), 8U, pitch, vocabulary),
          "the pitched buffer holds its rows");
    check(!strix::head_argmax::strided_rows_fit(logit_rows_elements(8U, vocabulary, pitch) - 1U, 8U, pitch, vocabulary),
          "one element short is refused");
    check(!strix::head_argmax::strided_rows_fit(logit_rows_elements(8U, vocabulary, pitch), 9U, pitch, vocabulary),
          "one row too many is refused");
}

// Pitched rows head (core::enqueue_bf16_softcap_head_rows): simulate every launch's stores (launch row j, output o at
// first x pitch + j x pitch + o) and check that each row is projected once, by a launch of at most max_rows rows,
// that its live outputs land exactly at logit_row_offset(row, pitch), and that no pad store falls in a live span.
void check_pitched_rows(std::size_t rows, std::size_t max_rows, std::size_t live, std::size_t pitch) {
    using namespace strix::head;
    std::vector<int> stores(rows * pitch, 0);
    std::vector<int> projected(rows, 0);
    std::size_t launches = 0;
    for (std::size_t first = 0; first < rows;) {
        const auto launch = logit_rows_launch(rows, first, max_rows);
        check(launch.first == first && launch.rows != 0U && launch.rows <= max_rows, "launch within the row cap");
        if (launch.rows == 0U) return;
        for (std::size_t j = 0; j < launch.rows; ++j) {
            ++projected[first + j];
            for (std::size_t o = 0; o < pitch; ++o) {
                const std::size_t at = logit_row_offset(first, pitch) + j * pitch + o;
                check(at < stores.size(), "stores stay inside rows x pitch");
                if (at >= stores.size()) return;
                const bool live_slot = at % pitch < live;
                check(live_slot == (o < live), "pad stores never fall in a live span");
                if (o < live) check(at == logit_row_offset(first + j, pitch) + o, "live output at its row offset");
                ++stores[at];
            }
        }
        first += launch.rows;
        ++launches;
    }
    for (const int n : projected) check(n == 1, "every row projected once");
    for (const int n : stores) check(n == 1, "every slot stored once");
    check(launches == (rows + max_rows - 1U) / max_rows, "fewest launches");
    check(pitched_rows_fit(rows, live, pitch, pitch + 3U, rows * pitch), "the rows x pitch buffer is admitted");
}

}  // namespace

int main() {
    using namespace strix::head;
    // 200058: the packed layout puts row 1 eight bytes off the 16-byte boundary (the refused case).
    check(200058U * sizeof(float) % kLogitRowAlignmentBytes == 8U, "200058 packed rows are misaligned");
    check(logit_row_pitch(200058U) == 200060U, "200058 pitch");
    check(logit_row_pitch(151552U) == 151552U, "an aligned vocabulary keeps the packed layout");
    check(logit_row_pitch(262145U) == 262148U, "262144 + 1 pitch");
    check(logit_row_pitch(3U) == 4U && logit_row_pitch(4U) == 4U, "small vocabularies");
    check(logit_row_pitch(200058U, 2U) == 200064U, "BF16 / FP16 rows round to eight elements");
    for (const std::size_t vocabulary : {std::size_t{200058}, std::size_t{151552}, std::size_t{262145}, std::size_t{3}})
        check_vocabulary(vocabulary);
    // Invalid requests return 0 instead of a pitch that would break the alignment.
    check(logit_row_pitch(0U) == 0U && logit_row_pitch(5U, 0U) == 0U && logit_row_pitch(5U, 4U, 0U) == 0U,
          "zero sizes");
    check(logit_row_pitch(5U, 4U, 24U) == 0U && logit_row_pitch(5U, 8U, 4U) == 0U, "bad alignment");
    // Contiguous rows (stride = vocabulary) remain admitted: the incumbent enqueue delegates with that stride.
    check(strix::head_argmax::strided_rows_fit(3U * 200058U, 3U, 200058U, 200058U), "contiguous rows");
    check(!strix::head_argmax::strided_rows_fit(10U, 1U, 4U, 5U), "stride below the row is refused");
    check(!strix::head_argmax::strided_rows_fit(10U, 0U, 5U, 5U), "zero rows are refused");
    // Pitched rows head: verify windows (2..8 rows) and cohort lanes (up to 40 rows, three launches of <= 16).
    for (std::size_t rows = 1; rows <= 40U; ++rows) {
        check_pitched_rows(rows, 16U, 200058U % 97U + 5U, logit_row_pitch(200058U % 97U + 5U));
        check_pitched_rows(rows, 16U, 7U, 8U);
        check_pitched_rows(rows, 3U, 4U, 4U);
    }
    check(logit_rows_launch(40U, 32U, 16U).rows == 8U && logit_rows_launch(40U, 16U, 16U).rows == 16U, "tail launch");
    check(logit_rows_launch(4U, 4U, 16U).rows == 0U && logit_rows_launch(4U, 0U, 0U).rows == 0U, "invalid launch");
    // The decoder case: 200058 live of 201024 stored head rows at pitch 200060.
    check(pitched_rows_fit(8U, 200058U, 200060U, 201024U, 8U * 200060U), "eight Mini verify rows");
    check(!pitched_rows_fit(8U, 200058U, 200060U, 201024U, 8U * 200060U - 1U), "the last row's pad is stored too");
    check(!pitched_rows_fit(1U, 200058U, 200056U, 201024U, 200060U), "pitch below the live outputs");
    check(!pitched_rows_fit(1U, 200058U, 200060U, 200058U, 200060U), "pitch above the stored head rows");
    check(!pitched_rows_fit(0U, 5U, 8U, 8U, 8U) && !pitched_rows_fit(1U, 0U, 8U, 8U, 8U), "zero rows or outputs");
    if (failures == 0) std::cout << "logit_rows ok\n";
    return failures == 0 ? 0 : 1;
}
