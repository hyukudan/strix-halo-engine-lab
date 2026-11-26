#pragma once

// Verify window program vs the eager verify pass, bit for bit, on the model checkpoint (CUDA).
//
// Two decoders built alike step in lock step through the same windows: A runs every staged verify window eagerly
// (set_window_programs(false), the incumbent route the speculation identity was proven on), B through the captured
// window program (set_window_programs(true): captured at the first window of each width, replayed afterwards at any
// position). After every window the fixture compares, in forward order, per visit: the convolution histories of
// each stream and the K/V rows the window wrote at its positions; then the final hidden rows, the DSpark tap rows and
// the picks. The FIRST differing item names the stage (visit = layer, stream or K/V, window row). Each window is
// compared three times: as staged (B's first run), replayed at the same position after a discard (commit 0), and
// after the partial / full commit both decoders then make (a rollback rebuilds histories from the checkpoint and the
// window's convolution tape, so a tape the program recorded wrongly shows there). Widths 1..8 each appear several
// times at different positions (capture, then replays), with kept rows 0..K. Exit 0 PASS, 1 FIRST DIFF, 2 error.
//
// With a drafter (--dspark FILE, else the MTP block beside the checkpoint) each decoder also carries the command
// line's resident DSpark drafter, run between windows as in a drafting decode: after every commit (and after the
// prefill) both drafters start at the new anchor and propose one block, and the fixture compares what the drafter
// reads and returns: every committed DSpark tap row [0, committed) (a row written at another position than the
// window's shows here, not in the per-window rows above) and the proposals.

#include "inkling2/inkling2_mini_resident_decoder.hpp"
#include "inkling2/inkling2_mini_resident_dspark.hpp"
#include "strix/model_data/inkling2_mini.hpp"
#include "strix/spec/drafter_source.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix::inkling2::test {

struct WindowItem {
    std::string what;
    std::vector<unsigned char> bytes;
};

template <class T>
inline std::vector<unsigned char> window_device_bytes(const T* device, std::size_t elements) {
    std::vector<unsigned char> bytes(elements * sizeof(T));
    if (!bytes.empty()) {
        mini_gpu::copy(bytes.data(), device, bytes.size(), mini_gpu::device_to_host, nullptr);
        mini_gpu::synchronize(nullptr);
    }
    return bytes;
}

// The state a window of `rows` rows at `first` leaves on slot 0 (staged or committed), in forward order.
inline std::vector<WindowItem> window_state(MiniResidentDecoder& decoder, unsigned first, unsigned rows,
                                            std::span<const unsigned> picks, bool with_hidden) {
    std::vector<WindowItem> items;
    const auto view = decoder.slot_view(0U);
    const auto history = window_device_bytes(view.history.data(), view.history.size());
    for (const auto visit : MiniSessionStatePlan::visits()) {
        const auto& layout = view.plan.visit(visit);
        const auto layer = std::to_string(visit.logical_layer);
        for (std::size_t stream = 0; stream < layout.convolution_offsets.size(); ++stream) {
            const auto elements = static_cast<std::size_t>(layout.convolution_elements(stream));
            if (elements == 0U) continue;
            const auto begin = history.begin() + static_cast<std::ptrdiff_t>(layout.convolution_offsets[stream] * sizeof(float));
            items.push_back({"visit " + layer + " convolution history stream " + std::to_string(stream),
                             std::vector<unsigned char>(begin, begin + static_cast<std::ptrdiff_t>(elements * sizeof(float)))});
        }
        if (layout.kv_rows == 0U) continue;
        for (unsigned row = 0; row < rows; ++row) {
            const auto offset = static_cast<std::size_t>(layout.kv_offset +
                                                         attn::cache_rows::row_offset(layout.kv, first + row));
            items.push_back({"visit " + layer + " K row " + std::to_string(row),
                             window_device_bytes(view.kv.keys.data() + offset, layout.kv.width)});
            items.push_back({"visit " + layer + " V row " + std::to_string(row),
                             window_device_bytes(view.kv.values.data() + offset, layout.kv.width)});
        }
    }
    if (with_hidden) {
        const auto hidden = decoder.last_hidden_rows(rows);
        for (unsigned row = 0; row < rows; ++row)
            items.push_back({"final hidden row " + std::to_string(row),
                             window_device_bytes(hidden.data() + std::size_t{row} * kHiddenWidth, kHiddenWidth)});
    }
    const auto taps = decoder.draft_tap_plane();
    const auto tap_row = static_cast<std::size_t>(MiniDSparkTapPlan::elements(1U));
    for (unsigned row = 0; row < rows && !taps.empty(); ++row) {
        const auto at = static_cast<std::size_t>(MiniDSparkTapPlan::elements(first + row));
        if (at + tap_row > taps.size()) break;
        items.push_back({"DSpark tap row " + std::to_string(row), window_device_bytes(taps.data() + at, tap_row)});
    }
    for (std::size_t row = 0; row < picks.size(); ++row) {
        WindowItem pick{"pick row " + std::to_string(row), std::vector<unsigned char>(sizeof(unsigned))};
        std::memcpy(pick.bytes.data(), &picks[row], sizeof(unsigned));
        items.push_back(std::move(pick));
    }
    return items;
}

struct WindowCompare {
    bool failed = false;
    std::string first;
    unsigned items = 0U, differing = 0U;
};

// Compares item by item; the first difference (forward order) is the reported stage.
inline void compare_window(const std::vector<WindowItem>& eager, const std::vector<WindowItem>& program,
                           const std::string& where, WindowCompare& result) {
    if (eager.size() != program.size()) throw std::logic_error("window state shapes differ: " + where);
    unsigned differing = 0U;
    std::string first, names;
    for (std::size_t i = 0; i < eager.size(); ++i) {
        ++result.items;
        if (eager[i].what != program[i].what || eager[i].bytes.size() != program[i].bytes.size())
            throw std::logic_error("window state layouts differ: " + where);
        if (eager[i].bytes == program[i].bytes) continue;
        if (differing++ == 0U) {
            const auto& a = eager[i].bytes;
            const auto& b = program[i].bytes;
            std::size_t byte = 0;
            while (byte < a.size() && a[byte] == b[byte]) ++byte;
            char detail[160];
            std::snprintf(detail, sizeof(detail), " (byte %zu of %zu: eager 0x%02x program 0x%02x)", byte, a.size(),
                          static_cast<unsigned>(a[byte]), static_cast<unsigned>(b[byte]));
            first = eager[i].what + detail;
        }
        if (differing <= 6U) names += (names.empty() ? "" : "; ") + eager[i].what;
    }
    if (differing == 0U) return;
    result.differing += differing;
    std::cout << "WINFIX DIFF " << where << ": " << differing << " of " << eager.size() << " items; first: " << first
              << "; items: " << names << (differing > 6U ? "; ..." : "") << '\n';
    if (!result.failed) {
        result.failed = true;
        result.first = where + ": " + first;
    }
}

inline unsigned window_token(unsigned trial, unsigned index) {
    return 1000U + (trial * 7919U + index * 173U + index / 13U) % 20000U;
}

// The windows: every width 1..8 several times at advancing positions, kept rows from discard (0) to full (K).
struct WindowTrial {
    unsigned width, keep;
};
inline constexpr std::array<WindowTrial, 22> kWindowTrials{{
    {2, 1}, {3, 2}, {4, 4}, {5, 1}, {6, 3}, {7, 7}, {8, 2}, {8, 8}, {2, 2}, {5, 0}, {5, 5}, {8, 1},
    {3, 3}, {1, 1}, {8, 5}, {4, 2}, {6, 6}, {7, 3}, {2, 0}, {8, 8}, {8, 4}, {6, 1},
}};
inline constexpr unsigned kWindowPromptRows = 658U;  // the q51 code prompt length
inline constexpr unsigned kWindowContext = 2048U;
inline constexpr unsigned kWindowChunk = 512U;

inline std::unique_ptr<MiniResidentDecoder> window_decoder(const char* checkpoint, bool programs, bool screened) {
    // The command line's single-session drafting decoder: packed decode, step graphs, verify windows up to 8 rows,
    // the DSpark taps captured; the EXACT profile's levers (projected bias, screened head) as configure_mini_decoder.
    auto decoder = std::make_unique<MiniResidentDecoder>(checkpoint, kWindowContext, kWindowChunk, true,
        MiniResidentBackend::default_step_graphs, weight::RoleStorage{}, MiniResidentDecoder::kMaxVerifyRows, true);
    decoder->set_attention_projected_bias(true);
    (void)decoder->set_screened_head(screened);
    decoder->set_window_programs(programs);
    return decoder;
}

inline constexpr std::uint32_t kWindowDrafts = 7U;  // the command line's deepest model draft (8-row window)

// What a drafting decode's drafter reads after a commit and what it proposes: every committed tap row, then the block
// it drafts from `anchor` at the committed frontier (start + one draft, the command line's model pass).
inline std::vector<WindowItem> drafter_state(MiniResidentDecoder& decoder, MiniResidentDSpark& drafter,
                                             unsigned anchor, std::vector<std::uint32_t>& proposals) {
    std::vector<WindowItem> items;
    const auto taps = decoder.committed_taps();
    const auto tap_row = static_cast<std::size_t>(MiniDSparkTapPlan::elements(1U));
    const auto bytes = window_device_bytes(taps.data(), taps.size());
    const auto row_bytes = tap_row * sizeof(std::uint16_t);
    for (std::size_t row = 0; row * row_bytes < bytes.size(); ++row) {
        const auto begin = bytes.begin() + static_cast<std::ptrdiff_t>(row * row_bytes);
        items.push_back({"committed DSpark tap row at position " + std::to_string(row),
                         std::vector<unsigned char>(begin, begin + static_cast<std::ptrdiff_t>(row_bytes))});
    }
    const auto committed = static_cast<std::uint32_t>(taps.size() / tap_row);
    drafter.start(anchor, committed);
    drafter.draft({}, kWindowDrafts, proposals);
    // Fixed size (a shorter block pads with ~0) so the two sides always compare item by item.
    std::array<std::uint32_t, kWindowDrafts> padded{};
    padded.fill(~0U);
    std::copy_n(proposals.begin(), std::min<std::size_t>(proposals.size(), padded.size()), padded.begin());
    WindowItem block{"drafter proposals", std::vector<unsigned char>(sizeof(padded))};
    std::memcpy(block.bytes.data(), padded.data(), block.bytes.size());
    items.push_back(std::move(block));
    return items;
}

inline std::string window_ids(const std::vector<std::uint32_t>& ids) {
    std::string text;
    for (const auto id : ids) text += (text.empty() ? "" : ",") + std::to_string(id);
    return "[" + text + "]";
}

inline int window_program_identity(const char* checkpoint, bool screened, std::string_view dspark = {}) {
    auto eager_owner = window_decoder(checkpoint, false, screened);
    auto program_owner = window_decoder(checkpoint, true, screened);
    auto& eager = *eager_owner;
    auto& program = *program_owner;
    // The command line's drafter: an explicit file, else the MTP block beside the checkpoint when present.
    const auto drafter_file = spec::resolve_drafter_file({.file = dspark, .off = false, .speculating = true},
                                                         checkpoint, model_data::inkling2_mini::speculation::drafter);
    std::optional<MiniResidentDSpark> eager_drafter, program_drafter;
    if (!drafter_file.path.empty()) {
        eager_drafter.emplace(eager, kWindowContext, drafter_file.path);
        program_drafter.emplace(program, kWindowContext, drafter_file.path);
    }
    std::cout << "WINFIX drafter " << (drafter_file.path.empty() ? std::string("none") : drafter_file.path) << '\n';
    unsigned drafts_compared = 0U, drafts_differing = 0U;
    std::vector<std::uint32_t> proposals_eager, proposals_program;
    std::vector<unsigned> prompt(kWindowPromptRows);
    for (unsigned i = 0; i < prompt.size(); ++i) prompt[i] = window_token(0U, i);
    unsigned token = eager.prefill(prompt);
    if (program.prefill(prompt) != token) {
        std::cout << "WINFIX FIRST DIFF prefill pick (both eager: the decoders differ before any window)\n";
        return 1;
    }
    WindowCompare result;
    compare_window(window_state(eager, 0U, 0U, {}, false), window_state(program, 0U, 0U, {}, false), "after prefill",
                   result);
    const auto compare_drafters = [&](unsigned anchor, const std::string& where) {
        if (!eager_drafter || !program_drafter) return;
        const auto drafted_eager = drafter_state(eager, *eager_drafter, anchor, proposals_eager);
        const auto drafted_program = drafter_state(program, *program_drafter, anchor, proposals_program);
        compare_window(drafted_eager, drafted_program, where + " drafter", result);
        ++drafts_compared;
        const bool same = proposals_eager == proposals_program;
        drafts_differing += same ? 0U : 1U;
        std::cout << "WINFIX DRAFT " << where << " anchor " << anchor << " eager " << window_ids(proposals_eager)
                  << (same ? " == " : " != ") << "program " << window_ids(proposals_program) << '\n';
    };
    compare_drafters(token, "after prefill");
    auto& graphs = program.step_program_backend().window_graphs();
    std::array<unsigned, MiniResidentDecoder::kMaxVerifyRows + 1U> replays{};
    unsigned position = kWindowPromptRows;
    std::vector<unsigned> ids;
    for (std::size_t trial = 0; trial < kWindowTrials.size(); ++trial) {
        const unsigned width = kWindowTrials[trial].width;
        const unsigned keep = kWindowTrials[trial].keep;
        ids.assign(width, 0U);
        ids[0] = token;
        for (unsigned row = 1; row < width; ++row) ids[row] = window_token(static_cast<unsigned>(trial) + 1U, row);
        const auto tag = "trial " + std::to_string(trial) + " K=" + std::to_string(width) + " pos=" +
                         std::to_string(position);
        std::array<unsigned, MiniResidentDecoder::kMaxVerifyRows> top_eager{}, top_program{};
        const auto run = [&](MiniResidentDecoder& decoder, std::array<unsigned, MiniResidentDecoder::kMaxVerifyRows>& top) {
            const auto rows = decoder.forward_rows(ids.data(), width, MiniResidentDecoder::LogitRows::all);
            decoder.read_top(rows, width, top.data());
            return window_state(decoder, position, width, std::span<const unsigned>(top.data(), width), true);
        };
        const auto staged = run(eager, top_eager);
        const auto before = graphs.instantiations();
        // window_replayed() is cleared when the append ends: count runs through the monotonic counter.
        const auto runs_before = program.step_program_backend().window_runs();
        const auto first_run = run(program, top_program);
        const bool replayed = program.step_program_backend().window_runs() != runs_before;
        compare_window(staged, first_run, tag + (graphs.instantiations() != before ? " captured" : " staged"), result);
        // The same window again at the same position after a discard: the second replay of this width's program.
        program.commit(0U);
        const auto runs_second = program.step_program_backend().window_runs();
        const auto second_run = run(program, top_program);
        compare_window(staged, second_run, tag + " replay at the same position", result);
        replays[width] += (replayed ? 1U : 0U) + (program.step_program_backend().window_runs() != runs_second ? 1U : 0U);
        eager.commit(keep);
        program.commit(keep);
        compare_window(window_state(eager, position, keep, {}, false), window_state(program, position, keep, {}, false),
                       tag + " after commit keep=" + std::to_string(keep), result);
        compare_drafters(keep != 0U ? top_eager[keep - 1U] : token, tag + " after commit keep=" + std::to_string(keep));
        std::cout << "WINFIX " << tag << " keep=" << keep << " program " << (replayed ? "ran" : "declined")
                  << " picks " << top_eager[0] << (top_eager[0] == top_program[0] ? " == " : " != ") << top_program[0]
                  << '\n';
        position += keep;
        if (keep != 0U) token = top_eager[keep - 1U];
    }
    std::vector<float> logits_eager(kOutputVocabulary), logits_program(kOutputVocabulary);
    const auto next_eager = eager.decode(token, logits_eager);
    const auto next_program = program.decode(token, logits_program);
    if (next_eager != next_program ||
        std::memcmp(logits_eager.data(), logits_program.data(), logits_eager.size() * sizeof(float)) != 0) {
        std::cout << "WINFIX DIFF continuation decode at pos=" << position << ": picks " << next_eager << " / "
                  << next_program << '\n';
        if (!result.failed) {
            result.failed = true;
            result.first = "continuation decode logits";
        }
    }
    unsigned programs_ran = 0U;
    for (unsigned width = 2U; width < replays.size(); ++width) {
        std::cout << "WINFIX width " << width << " program runs " << replays[width] << " nodes " << graphs.nodes(width)
                  << '\n';
        programs_ran += replays[width];
    }
    std::cout << "WINFIX instantiations " << graphs.instantiations() << " compared items " << result.items
              << " differing " << result.differing << '\n';
    std::cout << "WINFIX drafts compared " << drafts_compared << " proposals differing " << drafts_differing << '\n';
    if (programs_ran == 0U) {
        std::cout << "WINFIX RESULT INVALID: no window program ran (every width declined)\n";
        return 2;
    }
    if (result.failed) {
        std::cout << "WINFIX RESULT FAIL FIRST DIFF " << result.first << '\n';
        return 1;
    }
    std::cout << "WINFIX RESULT PASS: every window (captured, replayed at the same and at other positions, after "
                 "commits 0..K) equals the eager verify pass bit for bit\n";
    return 0;
}

}  // namespace strix::inkling2::test
