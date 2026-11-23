// Shared multi-row logits pitch: every row of a multi-row logits buffer starts on the dense head's output
// alignment, for odd vocabularies too, and the strided greedy pick admits exactly the buffers that hold its rows.
#include "strix/head_argmax.hpp"
#include "strix/logit_rows.hpp"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>

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
    if (failures == 0) std::cout << "logit_rows ok\n";
    return failures == 0 ? 0 : 1;
}
