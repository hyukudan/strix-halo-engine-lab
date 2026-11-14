// Host tests of the shared one-completion-per-cycle pack (include/strix/session/cycle_pack.hpp): record layout,
// limits, and the lever row (bitwise, default OFF, explicit switch honoured).
#include "strix/session/cycle_pack.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#if defined(_MSC_VER)
#include <crtdbg.h>
#endif

using namespace strix;

namespace {
int failures = 0;
void check(bool value, const char* label) {
    if (!value) {
        ++failures;
        std::cerr << "FAIL: " << label << '\n';
    }
}
template <class E, class F> bool throws(F&& f) {
    try { f(); } catch (const E&) { return true; } catch (...) { return false; }
    return false;
}

void layout_places_records_back_to_back() {
    session::CyclePackLayout l;
    check(l.add(8) == 0, "first record at word 0");
    check(l.add(128) == 2, "second record after the 8-byte status");
    check(l.add(8) == 34, "third record after the 128-byte summary");
    check(l.records == 3 && l.words == 36, "three records, 36 words");
    check(l.size[1] == 32 && l.offset[2] == 34, "sizes and offsets kept per record");
    l.reset();
    check(l.records == 0 && l.words == 0 && l.add(4) == 0, "reset starts a new cycle at word 0");
}

void layout_rejects_bad_records() {
    session::CyclePackLayout l;
    check(throws<std::invalid_argument>([&] { l.add(0); }), "empty record rejected");
    check(throws<std::invalid_argument>([&] { l.add(6); }), "non 4-byte multiple rejected");
    check(throws<std::length_error>([&] { l.add(4U * session::CyclePackLayout::max_words + 4U); }), "oversized record rejected");
    for (std::uint32_t i = 0; i < session::CyclePackLayout::max_records; ++i) l.add(4);
    check(throws<std::length_error>([&] { l.add(4); }), "65th record rejected");
    session::CyclePackLayout full;
    full.add(4U * (session::CyclePackLayout::max_words - 1U));
    check(full.add(4) == session::CyclePackLayout::max_words - 1U, "last word usable");
    check(throws<std::length_error>([&] { full.add(4); }), "words beyond the block rejected");
}

void lever_row_is_bitwise_and_off_by_default() {
    const auto& d = session::serve_cycle_pack_lever;
    check(d.env == "STRIX_SERVE_CYCLE_PACK", "lever switch name");
    check(d.exactness == lever::Exactness::bitwise, "lever is bitwise (scheduling only)");
    check(!d.default_on && d.regime.axis == lever::Axis::none, "lever default OFF until measured");
    check(!lever::select(d, lever::Inputs{0, 0, 4}, nullptr).on, "unset switch: OFF");
    check(lever::select(d, lever::Inputs{0, 0, 4}, "1").on, "STRIX_SERVE_CYCLE_PACK=1: ON");
    check(!lever::select(d, lever::Inputs{0, 0, 4}, "0").on, "STRIX_SERVE_CYCLE_PACK=0: OFF");
}
}  // namespace

int main() {
#if defined(_MSC_VER)
    // A failed assertion or abort must fail the test, never open a dialog.
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif
    try {
        layout_places_records_back_to_back();
        layout_rejects_bad_records();
        lever_row_is_bitwise_and_off_by_default();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: unexpected exception: " << e.what() << '\n';
        return 1;
    }
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "cycle pack tests passed\n";
    return 0;
}
