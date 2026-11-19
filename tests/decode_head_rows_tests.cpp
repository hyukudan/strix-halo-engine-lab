// Head rows plan (strix/decode_head_rows.hpp): every output reads the input row of its own head, groups of `outs`
// outputs never straddle two heads when admitted, and the named MLA absorb geometries (model-B K_b / V_b, 32 and
// 64 heads) are admitted for every group width the q8_rows direct kernel uses (1, 2, 4, 8).
#include "strix/decode_head_rows.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

namespace {
int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// Incumbent form: H separate launches, launch h reads x + h * stride for its head_outputs outputs.
void check_mapping(const strix::decode::HeadRows& h) {
    for (std::uint32_t head = 0; head < h.heads; ++head)
        for (std::uint32_t o = 0; o < h.head_outputs; ++o) {
            const std::uint32_t out = head * h.head_outputs + o;
            if (strix::decode::head_input_offset(out, h.head_outputs, h.input_stride) !=
                static_cast<std::uint64_t>(head) * h.input_stride) {
                expect(false, "output reads the input row of its head");
                return;
            }
        }
}

// An admitted plan: every group of `outs` outputs has one head (so one input row per group is exact).
void check_groups(const strix::decode::HeadRows& h, unsigned outs) {
    const std::uint32_t outputs = h.heads * h.head_outputs;
    for (std::uint32_t first = 0; first < outputs; first += outs)
        for (unsigned k = 1; k < outs && first + k < outputs; ++k)
            if (strix::decode::head_input_offset(first + k, h.head_outputs, h.input_stride) !=
                strix::decode::head_input_offset(first, h.head_outputs, h.input_stride)) {
                expect(false, "admitted group stays inside one head");
                return;
            }
}
}  // namespace

int main() {
    using strix::decode::HeadRows;
    using strix::decode::head_rows_refusal;
    // model-B MLA: K_b absorb (q_nope 192 of a 256-wide head -> latent 512), V_b (latent 512 -> v 256).
    const HeadRows named[] = {
        {64U, 512U, 256U}, {32U, 512U, 256U},  // K_b: width 192, input stride 256 (q rows), 512 outputs per head
        {64U, 256U, 512U}, {32U, 256U, 512U},  // V_b: width 512, input stride 512 (latent rows), 256 outputs per head
    };
    const std::uint32_t widths[] = {192U, 192U, 512U, 512U};
    for (unsigned i = 0; i < 4U; ++i) {
        check_mapping(named[i]);
        for (const unsigned outs : {1U, 2U, 4U, 8U}) {
            expect(head_rows_refusal(named[i], widths[i], outs) == nullptr, "named MLA geometry admitted");
            check_groups(named[i], outs);
        }
    }
    // Refusals.
    expect(head_rows_refusal({4U, 6U, 64U}, 64U, 4U) != nullptr, "6 outputs per head with groups of 4 straddle");
    expect(head_rows_refusal({4U, 8U, 32U}, 64U, 4U) != nullptr, "input rows of two heads overlap");
    expect(head_rows_refusal({1U, 8U, 0U}, 64U, 4U) == nullptr, "one head needs no stride");
    expect(head_rows_refusal({0U, 8U, 64U}, 64U, 4U) != nullptr, "no heads");
    expect(head_rows_refusal({2U, 0U, 64U}, 64U, 4U) != nullptr, "no outputs per head");
    expect(head_rows_refusal({0x10000U, 0x10000U, 64U}, 64U, 1U) != nullptr, "outputs overflow 32 bits");
    // Random small plans: mapping and group property on every admitted one.
    std::uint32_t state = 12345U;
    const auto next = [&state](std::uint32_t mod) {
        state = state * 1664525U + 1013904223U;
        return 1U + (state >> 8) % mod;
    };
    unsigned admitted = 0;
    for (unsigned t = 0; t < 2000U; ++t) {
        const HeadRows h{next(16U), next(40U), next(300U)};
        const std::uint32_t width = 32U * next(8U);
        const unsigned outs = 1U << (next(4U) - 1U);
        if (head_rows_refusal(h, width, outs) != nullptr) continue;
        ++admitted;
        check_mapping(h);
        check_groups(h, outs);
    }
    expect(admitted > 100U, "random plans exercise the admitted path");
    if (failures) return EXIT_FAILURE;
    std::printf("decode head rows: named MLA geometries + %u random plans OK\n", admitted);
    return EXIT_SUCCESS;
}
