// Several score rows on the device (cohort decode lanes, verify rows): every row the dense head writes
// must start on a 16-byte boundary. The CUDA and HIP dense row projections refuse any other output
// address with "invalid argument", which a cohort decode of two or more lanes hit at its second row
// while rows were packed at the live vocabulary width (200058 floats = 800232 bytes, 8 bytes off).
#include "inkling2/inkling2_mini_resident_route.hpp"
#include "strix/head_argmax.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>

namespace {

namespace ink = strix::inkling2;
namespace pick = strix::head_argmax;

// The output admission of the dense row projection (valid_bf16_dense_row_group on CUDA, the BF16 rows
// path on HIP): a 16-byte aligned address. The buffer base is a device allocation (256-byte aligned).
constexpr bool head_output_admitted(std::size_t float_offset) {
    return float_offset * sizeof(float) % 16U == 0U;
}

// The two cohort widths: the shared CUDA dense decode window and the verify rows.
constexpr unsigned kRows[] = {2U, 4U, 8U, ink::kMiniMaxVerifyRows, 32U};

constexpr bool packed_rows_refused() {
    // Reproduces the refusal: row 1 of the packed layout is not admitted.
    return !head_output_admitted(std::size_t{1U} * ink::kOutputVocabulary);
}

constexpr bool pitched_rows_admitted() {
    if (ink::kMiniLogitRowPitch < ink::kOutputVocabulary ||
        ink::kMiniLogitRowPitch - ink::kOutputVocabulary >= 16U / sizeof(float))
        return false;
    for (const auto rows : kRows) {
        for (unsigned row = 0; row < rows; ++row) {
            const auto offset = ink::mini_logit_row_offset(row);
            if (!head_output_admitted(offset)) return false;
            // Disjoint rows: the next row starts after this row's live scores.
            if (row + 1U < rows && ink::mini_logit_row_offset(row + 1U) < offset + ink::kOutputVocabulary) return false;
        }
        const auto elements = ink::mini_logit_rows_elements(rows);
        // The last row ends the span; a rows x pitch buffer holds it.
        if (elements != ink::mini_logit_row_offset(rows - 1U) + ink::kOutputVocabulary ||
            elements > std::size_t{rows} * ink::kMiniLogitRowPitch)
            return false;
        // The greedy pick accepts exactly this strided span, and refuses one row too many or a short span.
        if (!pick::strided_rows_fit(elements, rows, ink::kMiniLogitRowPitch, ink::kOutputVocabulary) ||
            pick::strided_rows_fit(elements, rows + 1U, ink::kMiniLogitRowPitch, ink::kOutputVocabulary) ||
            pick::strided_rows_fit(elements - 1U, rows, ink::kMiniLogitRowPitch, ink::kOutputVocabulary) ||
            pick::strided_rows_fit(elements, rows, ink::kOutputVocabulary - 1U, ink::kOutputVocabulary))
            return false;
    }
    // The contiguous form stays admitted by the same rule (stride == count).
    return pick::strided_rows_fit(std::size_t{4U} * ink::kOutputVocabulary, 4U, ink::kOutputVocabulary,
                                  ink::kOutputVocabulary) &&
           !pick::strided_rows_fit(0U, 1U, ink::kOutputVocabulary, ink::kOutputVocabulary) &&
           !pick::strided_rows_fit(ink::kOutputVocabulary, 0U, ink::kOutputVocabulary, ink::kOutputVocabulary);
}

static_assert(packed_rows_refused());
static_assert(ink::mini_logit_rows_elements(1U) == ink::kOutputVocabulary);

}  // namespace

int main() {
    if (!packed_rows_refused()) {
        std::cerr << "FAIL packed Inkling Mini score rows were expected to be refused by the head\n";
        return 1;
    }
    if (!pitched_rows_admitted()) {
        std::cerr << "FAIL Inkling Mini score row pitch\n";
        return 1;
    }
    std::cout << "PASS Inkling Mini logit rows (pitch " << ink::kMiniLogitRowPitch << " floats)\n";
    return 0;
}
