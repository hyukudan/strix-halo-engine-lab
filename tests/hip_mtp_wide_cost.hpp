#pragma once
#include "hip_mtp_model_qualification.hpp"
#include "strix/model_content_hash.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

// Diagnostic only (stage S0 of the wide-verify plan), never read by the decoder: the GPU cost of
// W = 8/16/32 exact known rows at the actual decode frontier against the actual R8 verify window.
// Runs after the prefix gate (STRIX_MTP_WIDE_COST=1) on its serial oracle and state snapshot; every
// timed forward starts from the restored frontier and the session is restored after the last one.
//
// Which forward is timed. A T-row known forward with every row's head exists only as a tiled prefill
// owner (HipKnownPrefixWindow over CompiledKnownPrefixPrefill, T >= 16). Building one here is not
// possible: compile_known_prefix_prefill needs the manifest and tensor access plan, and the owner
// needs a KnownPromptMemory arena, D1 record, union scratch and the catalog config, all locals of
// the CLI setup (with their UMA reservation) that the qualification context does not carry. So W =
// 16/32 time the tiled owners the CLI built for the tiled MTP prompt (c.parents->tiled: the T-row
// block owner and the tail owners), when one has exactly W rows, the all-rows output policy and the
// decode catalog (no split prefill profile). Otherwise W serial target rows are timed instead. W = 8
// times the closed R8 verify owner executing its 8 known rows without capture (tiled owners start at
// 16), and r8_verify the real R8 verify (verifier, capture, 7 accepted candidates).
namespace strix::tests::wide_cost {
inline void check(hipError_t value, const char* message) {
    if (value != hipSuccess) throw std::runtime_error(std::string(message) + ": " + hipGetErrorString(value));
}
struct Timing { double min_ms = 0.0, median_ms = 0.0; };
// One warm-up (also the correctness probe), then five timed repeats, each from the restored frontier.
inline Timing measure(const std::function<void()>& reset, const std::function<void(bool)>& forward) {
    constexpr unsigned repeats = 5U;
    std::vector<double> ms;
    for (unsigned i = 0; i <= repeats; ++i) {
        reset();
        check(hipDeviceSynchronize(), "wide cost start fence");
        const auto start = std::chrono::steady_clock::now();
        forward(i == 0U);
        check(hipDeviceSynchronize(), "wide cost end fence");
        const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        if (i) ms.push_back(elapsed);
    }
    std::sort(ms.begin(), ms.end());
    return {ms.front(), ms[ms.size() / 2U]};
}
struct RowResult {
    RowResult(unsigned w, std::string p) : width(w), path(std::move(p)) {}
    unsigned width = 0;
    std::string path;
    Timing timing;
    std::optional<unsigned> argmax_matches, logits_bits_equal, rows_checked;
    std::optional<std::uint32_t> lm_head_rows_dispatches;
};
inline void run(decode_cli::MtpDecodeQualificationContext& c, const std::filesystem::path& destination,
    const std::vector<std::shared_ptr<HipKnownPrefixWindow>>& owners, const std::vector<std::uint32_t>& tokens,
    const std::vector<ModelContentIdentity>& serial_logits, const std::function<void()>& reset) {
    const auto vocab = c.target->model().vocab_size;
    const auto row_bytes = static_cast<std::size_t>(vocab * sizeof(float));
    std::vector<float> host(static_cast<std::size_t>(vocab));
    std::vector<RowResult> rows;
    std::string notes;
    // Row r of a forward whose input row 0 is tokens[0] predicts tokens[r + 1]; bits are compared with the serial oracle.
    const auto probe = [&](RowResult& out, const KnownWindowResult& result) {
        unsigned matches = 0, equal = 0;
        for (std::size_t k = 0; k < result.logits.size(); ++k) {
            const auto r = result.logit_row_indices.at(k);
            const auto& view = result.logits[k];
            if (view.bytes < row_bytes || r + 1U >= tokens.size()) throw std::logic_error("wide cost logit row extent differs");
            check(hipMemcpy(host.data(), reinterpret_cast<const void*>(view.address), row_bytes, hipMemcpyDeviceToHost), "wide cost logits");
            std::uint32_t best = 0;
            for (std::uint32_t t = 0; t < host.size(); ++t)
                if (std::isfinite(host[t]) && (!std::isfinite(host[best]) || host[t] > host[best])) best = t;
            matches += best == tokens[r + 1U];
            equal += sha256_bytes(std::as_bytes(std::span<const float>(host))) == serial_logits.at(r);
        }
        out.argmax_matches = matches; out.logits_bits_equal = equal;
        out.rows_checked = static_cast<unsigned>(result.logits.size()); out.lm_head_rows_dispatches = result.lm_head_rows_dispatches;
        if (matches != result.logits.size())
            notes += "W=" + std::to_string(out.width) + " " + out.path + ": " + std::to_string(result.logits.size() - matches) + " argmax mismatches; ";
    };
    const auto serial_rows = [&](unsigned width) {
        RowResult out{width, "serial-target-rows"};
        out.timing = measure(reset, [&](bool first) {
            unsigned matches = 0;
            for (unsigned r = 0; r < width; ++r)
                matches += c.target->execute_serial(c.session, c.parents->resources, c.dispatcher, tokens[r]).predicted_token() == tokens[r + 1U];
            if (first) {
                out.argmax_matches = matches; out.rows_checked = width;
                if (matches != width) notes += "W=" + std::to_string(width) + " serial: " + std::to_string(width - matches) + " token mismatches; ";
            }
        });
        return out;
    };
    // Real R8 verify: full acceptance of 7 serial-oracle candidates, deferred GDN rows left pending (discarded by reset).
    std::optional<Timing> r8_verify;
    try {
        const std::vector<std::uint32_t> candidates(tokens.begin() + 1, tokens.begin() + 8);
        unsigned emitted = 0;
        r8_verify = measure(reset, [&](bool first) {
            const auto verified = c.window->verify(c.session, c.dispatcher, c.initial, candidates, 8U, {});
            if (verified.chosen_rows != 8U) throw std::logic_error("wide cost verify did not choose the R8 owner");
            if (first) for (unsigned r = 0; r < verified.emitted_tokens.size(); ++r) emitted += verified.emitted_tokens[r] == tokens[r + 1U];
        });
        if (emitted != 8U) notes += "r8 verify: " + std::to_string(8U - emitted) + " of 8 emitted tokens differ; ";
    } catch (const std::exception& e) { notes += std::string("r8 verify failed: ") + e.what() + "; "; }
    for (const unsigned width : {8U, 16U, 32U}) {
        std::shared_ptr<HipKnownPrefixWindow> owner;
        std::string path;
        const auto& tiled = c.parents->tiled;
        if (width == 8U) { owner = owners.back(); path = "closed-r8-verify-owner-known-rows-all-heads-no-capture"; }
        else if (tiled.prefill_execution_identity) notes += "W=" + std::to_string(width) + ": split prefill profile, tiled owners run the prefill catalog; ";
        else {
            std::vector<std::shared_ptr<HipKnownPrefixWindow>> tiled_owners{tiled.block_owner};
            for (const auto& tail : tiled.tails) tiled_owners.push_back(tail.owner);
            for (const auto& candidate : tiled_owners)
                if (candidate && candidate->supports_tiled_prefill() && candidate->logical_rows() == width) {
                    if (candidate->output_policy() == KnownWindowOutputPolicy::all_rows) { owner = candidate; path = "tiled-prefill-owner-all-rows"; }
                    else notes += "W=" + std::to_string(width) + ": tiled owner policy is not all-rows; ";
                }
            if (!owner && path.empty()) notes += "W=" + std::to_string(width) + ": no tiled owner of this width (--known-prefix-prefill-rows/--known-prefix-tail-rows); ";
        }
        if (owner) {
            RowResult out{width, path};
            const std::span<const std::uint32_t> input(tokens.data(), width);
            try {
                out.timing = measure(reset, [&](bool first) {
                    const auto result = owner->execute(c.session, input);
                    // Closed deferred owner: keep all rows as a known prompt does; reset discards the pending GDN rows.
                    if (!owner->supports_tiled_prefill()) owner->commit_prefix(width);
                    if (first) { check(hipDeviceSynchronize(), "wide cost probe fence"); probe(out, result); }
                });
                rows.push_back(std::move(out));
            } catch (const std::exception& e) { notes += "W=" + std::to_string(width) + " " + path + " failed: " + e.what() + "; "; }
        }
        rows.push_back(serial_rows(width));
    }
    reset();
    std::ofstream out(destination / "wide-cost.pending.json", std::ios::binary);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    const auto write_optional = [&](const auto& value) { if (value) out << *value; else out << "null"; };
    out << "{\"schema\":\"mtp-wide-cost-v1\",\"performance_qualified\":false,\"initial_position\":" << c.session.next_position << ",\"rows\":[";
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        out << (i ? "," : "") << "{\"width\":" << r.width << ",\"path\":\"" << r.path << "\",\"min_ms\":" << r.timing.min_ms
            << ",\"median_ms\":" << r.timing.median_ms << ",\"argmax_matches\":";
        write_optional(r.argmax_matches); out << ",\"rows_checked\":"; write_optional(r.rows_checked);
        out << ",\"logits_bits_equal\":"; write_optional(r.logits_bits_equal);
        out << ",\"lm_head_rows_dispatches\":"; write_optional(r.lm_head_rows_dispatches); out << '}';
    }
    out << "],\"r8_verify\":";
    if (r8_verify) out << "{\"min_ms\":" << r8_verify->min_ms << ",\"median_ms\":" << r8_verify->median_ms << '}';
    else out << "null";
    for (auto& ch : notes) if (ch == '"' || ch == '\\' || static_cast<unsigned char>(ch) < 0x20U) ch = ' ';
    out << ",\"notes\":\"host wall ms between device fences, 1 warm-up + 5 repeats from the restored frontier; "
        << "tiled owners skip the closed-window HC/inject/shared-detach fusions and use R1 row-loop heads unless "
        << "lm_head_rows_dispatches > 0; " << notes << "\"}\n";
    out.close();
    std::filesystem::rename(destination / "wide-cost.pending.json", destination / "wide-cost.json");
}
} // namespace strix::tests::wide_cost
