// Several score rows on the device (cohort decode lanes, verify rows): every row the dense head writes
// must start on the head's output alignment. The CUDA and HIP dense row projections refuse any other
// output address with "invalid argument", which a cohort decode of two or more lanes hit at its second
// row while rows were packed at the live vocabulary width (200058 floats = 800232 bytes, 8 bytes off).
// The resident decoder lays its multi-row score buffers out by the shared head rule
// (strix::head::logit_row_pitch / logit_row_offset / logit_rows_elements); this test binds that rule to
// the Inkling 2 Mini vocabulary and widths. It holds no Inkling pitch constant of its own.
#include "inkling2/inkling2_mini_config.hpp"
#include "inkling2/inkling2_mini_resident_route.hpp"
#include "strix/head_argmax.hpp"
#include "strix/logit_rows.hpp"

#include <cstddef>
#include <iostream>

namespace {

namespace ink = strix::inkling2;
namespace head = strix::head;
namespace pick = strix::head_argmax;

// The pitch the resident decoder uses (MiniResidentDecoder::kLogitPitch is this expression).
constexpr std::size_t kPitch = head::logit_row_pitch(ink::kOutputVocabulary);

// The output admission of the dense row projection (valid_bf16_dense_row_group on CUDA, the BF16 rows
// path on HIP): an address aligned to the shared head alignment. The buffer base is a device allocation
// (256-byte aligned).
constexpr bool head_output_admitted(std::size_t float_offset) {
    return float_offset * sizeof(float) % head::kLogitRowAlignmentBytes == 0U;
}

// Cohort widths (shared CUDA dense decode windows) and the verify rows.
constexpr unsigned kRows[] = {1U, 2U, 3U, 4U, 8U, ink::kMiniMaxVerifyRows, 32U};

constexpr bool packed_rows_refused() {
    // Reproduces the refusal: row 1 of the packed layout is not admitted.
    return !head_output_admitted(head::logit_row_offset(1U, ink::kOutputVocabulary));
}

// Minimal padding: fewer than one alignment unit of extra floats per row.
static_assert(kPitch >= ink::kOutputVocabulary &&
              kPitch - ink::kOutputVocabulary < head::kLogitRowAlignmentBytes / sizeof(float));

constexpr bool pitched_rows_admitted() {
    for (const auto rows : kRows) {
        for (unsigned row = 0; row < rows; ++row) {
            const auto offset = head::logit_row_offset(row, kPitch);
            if (!head_output_admitted(offset)) return false;
            // Disjoint rows: the next row starts after this row's live scores.
            if (row + 1U < rows && head::logit_row_offset(row + 1U, kPitch) < offset + ink::kOutputVocabulary)
                return false;
        }
        const auto elements = head::logit_rows_elements(rows, ink::kOutputVocabulary, kPitch);
        // The last row ends the span; the decoder's rows x pitch buffer holds it.
        if (elements != head::logit_row_offset(rows - 1U, kPitch) + ink::kOutputVocabulary ||
            elements > std::size_t{rows} * kPitch)
            return false;
        // The greedy pick accepts exactly this strided span, and refuses one row too many, a short span
        // or a stride shorter than a row.
        if (!pick::strided_rows_fit(elements, rows, kPitch, ink::kOutputVocabulary) ||
            pick::strided_rows_fit(elements, rows + 1U, kPitch, ink::kOutputVocabulary) ||
            pick::strided_rows_fit(elements - 1U, rows, kPitch, ink::kOutputVocabulary) ||
            pick::strided_rows_fit(elements, rows, ink::kOutputVocabulary - 1U, ink::kOutputVocabulary))
            return false;
    }
    // The contiguous form stays admitted by the same rule (stride == count).
    return pick::strided_rows_fit(std::size_t{4U} * ink::kOutputVocabulary, 4U, ink::kOutputVocabulary,
                                  ink::kOutputVocabulary) &&
           !pick::strided_rows_fit(0U, 1U, ink::kOutputVocabulary, ink::kOutputVocabulary) &&
           !pick::strided_rows_fit(ink::kOutputVocabulary, 0U, ink::kOutputVocabulary, ink::kOutputVocabulary);
}

// Verify rows / cohort lanes head (HIP BF16, MiniResidentDecoder::project_head_rows) through the shared pitched rows head
// (core::enqueue_bf16_softcap_head_rows; its store layout is simulated by the shared logit-rows test). Mini binding:
// every verify window is ONE launch under the HIP tight-row cap (gemv_rows::max_rows = 16), its kPitch rows fit
// the stored head rows and the verify buffer of max verify rows x kPitch, and the greedy strided pick admits them.
bool verify_rows_head_layout() {
    constexpr std::size_t kTightRows = 16U;
    for (unsigned rows = 2U; rows <= ink::kMiniMaxVerifyRows; ++rows) {
        if (head::logit_rows_launch(rows, 0U, kTightRows).rows != rows) return false;
        if (!head::pitched_rows_fit(rows, ink::kOutputVocabulary, kPitch, ink::kVocabulary,
                                    std::size_t{ink::kMiniMaxVerifyRows} * kPitch))
            return false;
        if (!pick::strided_rows_fit(head::logit_rows_elements(rows, ink::kOutputVocabulary, kPitch), rows, kPitch,
                                    ink::kOutputVocabulary))
            return false;
    }
    return true;
}

static_assert(kPitch == 200060U, "the live 200058-float vocabulary rounds to 200060 floats");
static_assert(packed_rows_refused());
static_assert(head::logit_rows_elements(1U, ink::kOutputVocabulary, kPitch) == ink::kOutputVocabulary);

}  // namespace

int main() {
    if (!packed_rows_refused()) {
        std::cerr << "FAIL packed Inkling Mini score rows were expected to be refused by the head\n";
        return 1;
    }
    if (!pitched_rows_admitted()) {
        std::cerr << "FAIL Inkling Mini score rows on the shared head pitch\n";
        return 1;
    }
    if (!verify_rows_head_layout()) {
        std::cerr << "FAIL Inkling Mini verify rows head stores do not match the pitched score layout\n";
        return 1;
    }
    std::cout << "PASS Inkling Mini logit rows (shared head pitch " << kPitch << " floats)\n";
    return 0;
}
