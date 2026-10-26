#pragma once
#include "hip_mtp_model_qualification.hpp"
#include "strix/model_content_hash.hpp"
#include <hip/hip_runtime.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// Diagnostic only (STRIX_MTP_WIDE_GATE=1 after the prefix gate, in a run without STRIX_MTP_COPY_GATE): the wide
// copy verify (apps/hip_mtp_wide_copy.hpp, the decoder's own MtpWideCopyCycle) against an independent serial oracle.
// From the restored prompt frontier the oracle runs 176 serial target rows (tokens, full-vocabulary logits digests,
// committed state hashes and sessions at every frontier). Then, with real greedy cycles over the same frontier:
//  A. a wide cycle whose proposal is the oracle (full acceptance: every row's logits, all rows committed, the last
//     row adopted as the pending ticket);
//  B. a wide cycle with a wrong input at row 10: abandoned (nullopt) with the committed state, session, pending
//     ticket and predictor frontier exactly the pre-cycle ones; an immediate retry is refused by the cooldown;
//  C. ordinary cycles until the frontier passes the rejected row (they continue from the restored frontier);
//  D. a fully accepted wide cycle again; E. abandoned at row 1, then ordinary cycles past it; G. full again;
//  H/I. with a second configured size (--mtp-copy-wide-rows 32,16): a full and an abandoned (row 5) cycle on the
//  smaller owner, then ordinary cycles past it; F. a last ordinary cycle.
// Every emitted token must equal the oracle, every downloaded wide logits row its oracle digest (also the rows of
// abandoned cycles), and the committed state (all seven components, KV and index through the frontier), frontier
// and lineage validity the oracle's. The predictor's cache rows have no serial oracle here; the ordinary cycles
// only show that MTP cycles continue exactly.
namespace strix::tests::wide_gate {
using Hashes = std::vector<ModelContentIdentity>;
inline void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
inline void check(hipError_t value, const char* message) {
    if (value != hipSuccess) throw std::runtime_error(std::string(message) + ": " + hipGetErrorString(value));
}
inline void run(decode_cli::MtpDecodeQualificationContext& c, const std::filesystem::path& destination,
    const std::function<Hashes(std::uint64_t)>& hashes, const std::function<void()>& reset) {
    constexpr unsigned oracle_rows = 176U;
    const auto sizes = c.options.mtp.copy_wide_rows.empty() ? std::vector<std::uint32_t>{32U} : c.options.mtp.copy_wide_rows;
    const std::uint32_t rows = sizes.front();
    require(c.window && c.initial.valid() && !c.target->token_sampler(), "wide gate requires the greedy verifier and the prompt frontier");
    auto wide = decode_cli::make_mtp_wide_copy(*c.parents, sizes, *c.target);
    reset();
    const auto base = c.session;
    require(base.next_position + oracle_rows <= c.parents->resources.session_state_layout->context_tokens,
        "wide gate requires 176 free context rows");
    const auto& compiled = c.target->compiled();
    const auto use = compiled.io.steps.back().writes.front();
    const auto serial_logits = resolve_execution_buffer(compiled.execution_identity, compiled.arena, *c.parents->resources.scratch, use);
    const auto vocab = c.target->model().vocab_size;
    require(serial_logits.bytes == vocab * sizeof(float), "wide gate full vocabulary extent differs");
    std::vector<float> row(static_cast<std::size_t>(vocab));
    const auto stream = static_cast<hipStream_t>(c.target->stream());
    std::vector<std::uint32_t> tokens{c.initial.predicted_token()};
    std::vector<ModelContentIdentity> logits;
    std::vector<Hashes> states;
    std::vector<SerialSessionContext> sessions;
    for (unsigned i = 0; i < oracle_rows; ++i) {
        const auto ticket = c.target->execute_serial(c.session, c.parents->resources, c.dispatcher, tokens.back());
        tokens.push_back(ticket.predicted_token());
        check(hipMemcpyAsync(row.data(), reinterpret_cast<const void*>(serial_logits.address), row.size() * sizeof(float),
            hipMemcpyDeviceToHost, stream), "wide gate serial logits");
        check(hipStreamSynchronize(stream), "wide gate serial logits completion");
        logits.push_back(sha256_bytes(std::as_bytes(std::span<const float>(row))));
        states.push_back(hashes(c.session.next_position));
        sessions.push_back(c.session);
    }
    reset();
    c.window->begin_session();
    // The qualification route leaves the greedy runtime without the verifier; real cycles need it.
    c.greedy.use_window_verifier(c.window);
    MtpContextDraftConfig config;
    config.copy_edit = true;
    c.greedy.enable_context_draft(c.options.tokens, config);
    std::vector<ModelContentIdentity> observed;
    wide->set_row_logits([&](std::uint32_t, std::span<const float> values) { observed.push_back(sha256_bytes(std::as_bytes(values))); });
    std::ofstream out(destination / "wide-gate.pending.json", std::ios::binary);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << "{\"schema\":\"mtp-wide-copy-gate-v1\",\"performance_qualified\":false,\"wide_rows\":" << rows
        << ",\"initial_position\":" << base.next_position << ",\"cases\":[";
    auto frontier = c.initial;
    unsigned offset = 0, cases = 0, logit_rows = 0;
    const auto committed = [&](std::size_t emitted, const HipMtpTargetHiddenTicket& last) {
        c.window->flush_pending_window();
        check(hipDeviceSynchronize(), "wide gate materialized completion");
        const auto index = offset + emitted - 1U;
        require(hashes(c.session.next_position) == states[index], "wide gate committed state differs from serial");
        const auto& expected = sessions[index];
        require(c.session.next_position == expected.next_position && c.session.committed_tokens == expected.committed_tokens &&
            c.session.health == SerialSessionHealth::ready && c.session.state.valid == expected.state.valid &&
            c.session.execution_identity == expected.execution_identity, "wide gate committed frontier differs from serial");
        require(!c.session.arithmetic_lineage.tainted && arithmetic_lineage_valid(c.session.arithmetic_lineage, c.session.next_position),
            "wide gate lineage is invalid");
        require(last.valid() && last.source_position() + 1U == c.session.next_position && last.predicted_token() == tokens[index + 1U] &&
            c.runtime.next_source_position() == last.source_position(), "wide gate pending ticket or predictor frontier differs");
    };
    // w: the wide size; wrong: proposal index replaced (input row wrong + 1), or w - 1 for the oracle proposal.
    const auto forced_at = [&](std::uint32_t w, std::uint32_t wrong) {
        std::vector<std::uint32_t> forced(tokens.begin() + offset + 1, tokens.begin() + offset + w);
        if (wrong + 1U < w) forced[wrong] = (forced[wrong] + 1U) % static_cast<std::uint32_t>(vocab);
        return forced;
    };
    const auto wide_full = [&](const char* name, std::uint32_t w) {
        require(offset + w <= oracle_rows, "wide gate case exceeds the oracle");
        observed.clear();
        const auto result = wide->cycle(c.greedy, frontier, oracle_rows - offset, {}, forced_at(w, w - 1U));
        require(result.has_value(), "wide gate full cycle did not commit");
        require(result->emitted_tokens.size() == w && result->accepted_candidates == w - 1U && result->consumed_target_rows == w &&
            result->target_window_rows == w && !result->target_checkpoint_restored && result->accepted_target_window,
            "wide gate cycle accepted another prefix than the oracle");
        require(observed.size() == w, "wide gate did not download every committed row's logits");
        for (std::uint32_t r = 0; r < w; ++r) {
            require(result->emitted_tokens[r] == tokens[offset + r + 1U], "wide gate token differs from serial");
            require(observed[r] == logits[offset + r], "wide gate full logits differ from serial");
            ++logit_rows;
        }
        committed(w, result->final_target);
        out << (cases++ ? "," : "") << "{\"case\":\"" << name << "\",\"wide_rows\":" << w << ",\"offset\":" << offset
            << ",\"emitted\":" << w << '}';
        offset += w;
        frontier = result->final_target;
    };
    // The pre-cycle frontier exactly: the oracle's state after `offset` rows (offset > 0), the same session and predictor.
    const auto unchanged = [&]() {
        require(offset > 0U, "wide gate abandon case needs an earlier oracle frontier");
        c.window->flush_pending_window();
        check(hipDeviceSynchronize(), "wide gate abandon completion");
        const auto index = offset - 1U;
        require(hashes(c.session.next_position) == states[index], "wide gate abandoned cycle changed the committed state");
        const auto& expected = sessions[index];
        const auto& before = frontier.committed_session();
        require(c.session.next_position == expected.next_position && c.session.committed_tokens == expected.committed_tokens &&
            c.session.health == SerialSessionHealth::ready && c.session.state.valid == expected.state.valid &&
            c.session.arithmetic_lineage.intervals == before.arithmetic_lineage.intervals &&
            c.session.state.generations == before.state.generations, "wide gate abandoned cycle changed the committed session");
        require(c.runtime.next_source_position() == frontier.source_position(), "wide gate abandoned cycle moved the predictor");
    };
    // Returns the offset the committed frontier must pass before the next wide cycle (the rejected row).
    const auto wide_abandon = [&](const char* name, std::uint32_t w, std::uint32_t wrong) {
        require(offset + w <= oracle_rows && wrong + 1U < w, "wide gate abandon case exceeds the oracle");
        const auto n = wrong + 1U;
        observed.clear();
        require(!wide->cycle(c.greedy, frontier, oracle_rows - offset, {}, forced_at(w, wrong)).has_value(),
            "wide gate rejected cycle was not abandoned");
        require(observed.size() == n, "wide gate abandoned cycle did not stop at the rejected row");
        for (std::uint32_t r = 0; r < n; ++r) {
            require(observed[r] == logits[offset + r], "wide gate abandoned full logits differ from serial");
            ++logit_rows;
        }
        unchanged();
        // Cooldown (every size): even the oracle proposal is refused until the frontier passes the rejected row.
        for (const auto size : sizes) {
            observed.clear();
            require(!wide->cycle(c.greedy, frontier, oracle_rows - offset, {}, forced_at(size, size - 1U)).has_value() && observed.empty(),
                "wide gate cooldown admitted a wide cycle at the abandoned frontier");
        }
        unchanged();
        out << (cases++ ? "," : "") << "{\"case\":\"" << name << "\",\"wide_rows\":" << w << ",\"offset\":" << offset
            << ",\"abandoned_rows\":" << n << '}';
        return offset + n;
    };
    const auto ordinary = [&](const char* name) {
        const auto result = c.greedy.cycle(frontier, 7U, oracle_rows - offset, {});
        const auto n = static_cast<unsigned>(result.emitted_tokens.size());
        require(n && offset + n <= oracle_rows, "wide gate ordinary cycle exceeds the oracle");
        for (unsigned r = 0; r < n; ++r) require(result.emitted_tokens[r] == tokens[offset + r + 1U], "wide gate ordinary token differs from serial");
        committed(n, result.final_target);
        out << (cases++ ? "," : "") << "{\"case\":\"" << name << "\",\"offset\":" << offset << ",\"emitted\":" << n
            << ",\"rows\":" << result.target_window_rows << '}';
        offset += n;
        frontier = result.final_target;
    };
    wide_full("A-full", rows);
    for (const auto resume = wide_abandon("B-abandon-row-10", rows, 9U); offset <= resume;) ordinary("C-ordinary");
    wide_full("D-full", rows);
    for (const auto resume = wide_abandon("E-abandon-row-1", rows, 0U); offset <= resume;) ordinary("E-ordinary");
    wide_full("G-full", rows);
    // The smaller configured size (--mtp-copy-wide-rows 32,16): one full and one abandoned cycle on its own owner.
    const bool second = sizes.size() > 1U;
    if (second) {
        const auto small = sizes[1];
        wide_full("H-full-small", small);
        for (const auto resume = wide_abandon("I-abandon-small-row-5", small, 4U); offset <= resume;) ordinary("I-ordinary");
    }
    ordinary("F-ordinary");
    // Forced proposals bypass the context drafter, so only the executor counts these cycles.
    const auto& large = wide->sizes().front().stats;
    require(large.cycles == 3U && large.rows_accepted == 3U * (rows - 1U) && large.abandoned == 2U && large.abandoned_rows_wasted == 2U * rows,
        "wide gate counters of the largest size differ");
    if (second) {
        const auto& small = wide->sizes()[1].stats;
        require(small.cycles == 1U && small.rows_accepted == sizes[1] - 1U && small.abandoned == 1U && small.abandoned_rows_wasted == sizes[1],
            "wide gate counters of the smaller size differ");
    }
    const auto st = wide->stats();
    out << "],\"tokens_checked\":" << offset << ",\"wide_logit_rows_checked\":" << logit_rows
        << ",\"wide_ms_committed\":" << st.ms_committed << ",\"wide_ms_abandoned\":" << st.ms_abandoned << ",\"passed\":true}\n";
    out.close();
    std::filesystem::rename(destination / "wide-gate.pending.json", destination / "wide-gate-result.json");
}
} // namespace strix::tests::wide_gate
