#pragma once
#include "strix/hip_dense_trellis.hpp"

#include "hip_mtp_model_qualification.hpp"
#include "hip_mtp_wide_cost.hpp"
#include "hip_mtp_wide_gate.hpp"
#include "strix/dispatch_resources.hpp"
#include "strix/model_content_hash.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#if defined(STRIX_PREFIX_GATE_OPENSSL_SHA256)
#include <openssl/sha.h>
#endif

// Diagnostic only: uses the actual CLI prompt/profile, target and verifier.
// No new publication path and no performance qualification. The runner exits
// after this destructive, exclusively owned validation session.
namespace strix::tests::prefix_gate {
inline void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
inline void check(hipError_t value, const char* message) {
    if (value != hipSuccess) throw std::runtime_error(std::string(message) + ": " + hipGetErrorString(value));
}
using Bytes = std::vector<std::byte>;
using Hashes = std::vector<ModelContentIdentity>;
inline ModelContentIdentity digest(std::span<const std::byte> bytes) {
#if defined(STRIX_PREFIX_GATE_OPENSSL_SHA256)
    ModelContentIdentity result;
    require(SHA256(reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size(), result.bytes.data()) != nullptr,
        "prefix gate system SHA-256 failed");
    return result;
#else
    return sha256_bytes(bytes);
#endif
}
struct State {
    std::vector<SessionStateAddress> views;
    std::vector<Bytes> initial;
    const SerialSessionStateLayout& layout;
    const ModelContract& model;
    hipStream_t stream;
    State(decode_cli::MtpDecodeQualificationContext& c)
        : layout(*c.parents->resources.session_state_layout), model(c.target->model()),
          stream(static_cast<hipStream_t>(c.target->stream())) {
        require(layout.allocations.size() == 7U, "prefix gate requires all seven state components");
        require(layout.device_total_bytes + layout.host_total_bytes <= (1ULL << 30U), "prefix gate state exceeds 1 GiB");
        for (const auto& a : layout.allocations) {
            views.push_back(resolve_session_state(layout, *c.parents->resources.session_state_memory, a.component));
            initial.push_back(read(views.back()));
        }
    }
    Bytes read(const SessionStateAddress& v) const {
        Bytes bytes(static_cast<std::size_t>(v.bytes));
        if (v.storage == SessionStateStorage::device) {
            check(hipMemcpyAsync(bytes.data(), reinterpret_cast<const void*>(v.address), bytes.size(), hipMemcpyDeviceToHost, stream), "prefix state read");
            check(hipStreamSynchronize(stream), "prefix state read completion");
        } else std::memcpy(bytes.data(), reinterpret_cast<const void*>(v.address), bytes.size());
        return bytes;
    }
    void restore() const {
        for (std::size_t i = 0; i < views.size(); ++i) {
            const auto& v = views[i];
            if (v.storage == SessionStateStorage::device)
                check(hipMemcpyAsync(reinterpret_cast<void*>(v.address), initial[i].data(), initial[i].size(), hipMemcpyHostToDevice, stream), "prefix state restore");
            else std::memcpy(reinterpret_cast<void*>(v.address), initial[i].data(), initial[i].size());
        }
        check(hipStreamSynchronize(stream), "prefix state restore completion");
    }
    Hashes hashes(std::uint64_t frontier, Hashes* physical = nullptr) const {
        require(frontier <= layout.context_tokens, "prefix gate frontier exceeds capacity");
        Hashes result;
        for (const auto& v : views) {
            auto bytes = read(v);
            if (physical) physical->push_back(digest(bytes));
            if (v.component == SessionStateComponent::attention_kv || v.component == SessionStateComponent::sparse_index) {
                // Layer-major, then token-major in each QSA layer. Only cache
                // cells at/after the committed frontier are non-authoritative.
                // Every committed byte, all recurrent/conv state and host history
                // remain in the digest; raw physical tails are never rewritten.
                require(model.full_attention_layers && v.bytes % model.full_attention_layers == 0U, "prefix cache layer extent differs");
                const auto layer = v.bytes / model.full_attention_layers;
                require(layer % layout.context_tokens == 0U, "prefix cache token extent differs");
                const auto used = frontier * (layer / layout.context_tokens);
                for (std::uint64_t l = 0; l < model.full_attention_layers; ++l)
                    std::fill(bytes.begin() + static_cast<std::ptrdiff_t>(l * layer + used),
                        bytes.begin() + static_cast<std::ptrdiff_t>((l + 1U) * layer), std::byte{});
            }
            result.push_back(digest(bytes));
        }
        return result;
    }
};
inline void write_hashes(std::ostream& out, const Hashes& hashes) {
    out << '[';
    for (std::size_t i = 0; i < hashes.size(); ++i)
        out << (i ? "," : "") << '"' << model_content_identity_hex(hashes[i]) << '"';
    out << ']';
}
inline void run(decode_cli::MtpDecodeQualificationContext& c, const std::filesystem::path& destination) {
    require(c.window && c.initial.valid() && !c.target->token_sampler() && !c.window->survival_controller(),
        "prefix gate requires a greedy min-p verifier and actual prompt frontier");
    const auto owners = c.window->window_owners();
    require(owners.size() == 3U && owners[0]->logical_rows() == 2U && owners[1]->logical_rows() == 4U && owners[2]->logical_rows() == 8U,
        "prefix gate requires owners R2/R4/R8");
    for (const auto& owner : owners) require(owner->defers_state(), "prefix gate requires deferred GDN/PLE owners");
    const auto base = c.session;
    constexpr unsigned reference_rows = 32U;
    require(base.next_position + reference_rows <= c.parents->resources.session_state_layout->context_tokens, "prefix gate requires 32 free context rows");
    require(std::filesystem::create_directory(destination), "prefix gate output directory must be new");
    const std::array<std::byte, 3> hash_check{std::byte{'a'}, std::byte{'b'}, std::byte{'c'}};
    require(digest(hash_check) == sha256_bytes(hash_check), "prefix gate SHA-256 providers differ");
    State state(c);
    Hashes initial_physical;
    const auto initial_state = state.hashes(base.next_position, &initial_physical);
    const auto reset = [&] {
        // Every prior case is materialized before reset. Discard the owners'
        // non-authoritative caches before replacing session storage, including
        // optional persistent QSA pools supplied by another specialization.
        for (const auto& owner : owners) owner->discard_deferred();
        require(!c.window->pending_window(), "prefix gate reset retained pending GDN rows");
        for (const auto& owner : owners)
            require(!owner->deferred_pending(), "prefix gate reset retained shared pending GDN rows");
        state.restore(); c.session = base;
    };
    std::vector<std::uint32_t> tokens{c.initial.predicted_token()};
    std::vector<ModelContentIdentity> logits;
    std::vector<Hashes> states;
    std::vector<SerialSessionContext> sessions;
    const auto& compiled = c.target->compiled();
    const auto use = compiled.io.steps.back().writes.front();
    const auto serial_logits = resolve_execution_buffer(compiled.execution_identity, compiled.arena, *c.parents->resources.scratch, use);
    const auto vocab = c.target->model().vocab_size;
    require(serial_logits.bytes == vocab * sizeof(float), "prefix gate full vocabulary extent differs");
    Bytes logit_bytes(static_cast<std::size_t>(serial_logits.bytes));
    for (unsigned i = 0; i < reference_rows; ++i) {
        const auto ticket = c.target->execute_serial(c.session, c.parents->resources, c.dispatcher, tokens.back());
        tokens.push_back(ticket.predicted_token());
        check(hipMemcpyAsync(logit_bytes.data(), reinterpret_cast<const void*>(serial_logits.address), logit_bytes.size(),
            hipMemcpyDeviceToHost, state.stream), "prefix gate serial logits");
        check(hipStreamSynchronize(state.stream), "prefix gate serial logits completion");
        logits.push_back(digest(logit_bytes));
        states.push_back(state.hashes(c.session.next_position));
        sessions.push_back(c.session);
    }
    reset();
    std::ofstream out(destination / "result.pending.json", std::ios::binary);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << "{\"schema\":\"mtp-prefix-gate-v1\",\"performance_qualified\":false,\"initial_position\":" << base.next_position
        << ",\"numerical_profile\":\"" << numerical_profile_name(c.options.numerical_profile) << '"'
        << ",\"target_content\":\"" << model_content_identity_hex(c.initial.target_content_identity()) << '"'
        << ",\"serial_execution_identity\":" << compiled.execution_identity << ",\"serial_kernel_plan_identity\":" << compiled.kernel_plan_identity
        << ",\"state_scope\":\"all-seven-components-kv-and-index-only-through-committed-frontier\""
        << ",\"physical_state_scope\":\"full-seven-components-including-discarded-cache-tails-cross-arm-only\",";
#if defined(STRIX_HAS_EXPERIMENTAL_DENSE_TRELLIS)
    if (hip_dense_trellis_active())
        require(hip_dense_trellis_used_layers() == ((1ULL << 48U) - 1U), "prefix gate did not consume every selected dense tensor");
#endif
    out << "\"initial_committed_state\":"; write_hashes(out, initial_state);
    out << ",\"initial_physical_state\":"; write_hashes(out, initial_physical);
    out << ",\"cases\":[";
    unsigned cases = 0;
    const auto exercise = [&](unsigned rows, unsigned keep, unsigned offset, const HipMtpTargetHiddenTicket& initial) {
        require(rows == 1U || rows == 2U || rows == 4U || rows == 8U, "prefix gate unsupported rows");
        require(keep && keep <= rows && offset + rows <= reference_rows, "prefix gate case exceeds reference");
        const auto owner = rows == 1U ? std::shared_ptr<HipKnownPrefixWindow>{} : owners[rows == 2U ? 0U : rows == 4U ? 1U : 2U];
        std::vector<Bytes> window_logits(rows, Bytes(logit_bytes.size()));
        unsigned taps = 0;
        if (owner) owner->set_window_step_observer([&](std::size_t, const std::function<ExecutionDispatchContext(std::uint32_t)>& row) {
            if (row(0).op != SerialOp::lm_head) return;
            const auto* scope = current_hip_launch_context();
            require(scope && scope->stream, "prefix gate observer requires actual window stream");
            for (unsigned r = 0; r < rows; ++r) {
                const auto view = resolve_dispatch_write(row(r), 0);
                require(view.bytes == logit_bytes.size(), "prefix gate window full vocabulary extent differs");
                check(hipMemcpyAsync(window_logits[r].data(), reinterpret_cast<const void*>(view.address), window_logits[r].size(),
                    hipMemcpyDeviceToHost, scope->stream), "prefix gate window logits");
                ++taps;
            }
        });
        struct ClearObserver { std::shared_ptr<HipKnownPrefixWindow> owner; ~ClearObserver() { if (owner) owner->set_window_step_observer({}); } } clear{owner};
        std::vector<std::uint32_t> candidates;
        for (unsigned r = 1; r < rows; ++r) candidates.push_back(tokens[offset + r]);
        if (keep < rows) candidates[keep - 1U] = (tokens[offset + keep] + 1U) % static_cast<std::uint32_t>(vocab);
        require(initial.valid() && initial.predicted_token() == tokens[offset] &&
            initial.source_position() + 1U == base.next_position + offset, "prefix gate requires actual prior continuation ticket");
        const auto before_generations = c.session.state.generations;
        const auto verified = c.window->verify(c.session, c.dispatcher, initial, candidates, rows, {});
        require(verified.chosen_rows == rows && verified.emitted_tokens.size() == keep &&
            verified.committed_rows.size() == keep && verified.accepted_candidates == keep - 1U,
            "prefix gate did not execute the requested owner/rejection depth");
        if (owner) {
            require(verified.attempted_window && verified.accepted_window == (keep == rows), "prefix gate acceptance receipt differs");
            require(c.window->pending_window(), "prefix gate committed window has no pending GDN rows");
            for (const auto& shared : owners)
                require(shared->deferred_pending(), "prefix gate owners disagree on pending GDN rows");
            owner->flush_deferred();
            require(!c.window->pending_window(), "prefix gate flush retained pending GDN rows");
            for (const auto& shared : owners)
                require(!shared->deferred_pending(), "prefix gate flush retained shared pending GDN rows");
            require(taps == rows, "prefix gate did not capture every physical window logit row");
        } else {
            require(!verified.attempted_window, "prefix gate serial fallback attempted a window");
            check(hipMemcpyAsync(window_logits[0].data(), reinterpret_cast<const void*>(serial_logits.address), logit_bytes.size(),
                hipMemcpyDeviceToHost, state.stream), "prefix gate fallback logits");
        }
        check(hipDeviceSynchronize(), "prefix gate materialized state completion");
        for (unsigned r = 0; r < keep; ++r) {
            require(verified.emitted_tokens[r] == tokens[offset + r + 1U], "prefix gate target token differs from serial reference");
            require(digest(window_logits[r]) == logits[offset + r], "prefix gate full logits differ from serial reference");
        }
        Hashes physical;
        const auto hashes = state.hashes(c.session.next_position, &physical);
        require(hashes == states[offset + keep - 1U], "prefix gate committed state differs from serial reference");
        const auto& expected = sessions[offset + keep - 1U];
        require(c.session.next_position == expected.next_position && c.session.committed_tokens == expected.committed_tokens,
            "prefix gate committed frontier differs");
        require(c.session.health == SerialSessionHealth::ready && c.session.state.valid == expected.state.valid,
            "prefix gate health or valid state differs");
        // Generations count physical writes, including discarded speculative
        // rows. Prefix acceptance rewinds frontier/lineage, not these versions.
        // Check the actual R-row write schedule, and retain the counters in the
        // receipt for baseline/OFF/ON comparison instead of normalizing them.
        for (std::size_t flag = 0; flag < before_generations.size(); ++flag) {
            const auto per_row = sessions.front().state.generations[flag] - base.state.generations[flag];
            require(c.session.state.generations[flag] == before_generations[flag] + rows * per_row,
                "prefix gate physical-write generation differs");
        }
        require(
            c.session.execution_identity == expected.execution_identity && c.session.state_layout_identity == expected.state_layout_identity &&
            c.session.runtime_compatibility_identity == expected.runtime_compatibility_identity,
            "prefix gate execution identities differ");
        require(!c.session.arithmetic_lineage.tainted && arithmetic_lineage_valid(c.session.arithmetic_lineage, c.session.next_position),
            "prefix gate lineage is invalid");
        out << (cases++ ? "," : "") << "{\"rows\":" << rows << ",\"keep\":" << keep << ",\"offset\":" << offset << ",\"tokens\":[";
        for (unsigned r = 0; r < keep; ++r) out << (r ? "," : "") << verified.emitted_tokens[r];
        out << "],\"all_physical_logits\":[";
        for (unsigned r = 0; r < rows; ++r)
            out << (r ? "," : "") << '"' << model_content_identity_hex(digest(window_logits[r])) << '"';
        out << "],\"committed_state\":"; write_hashes(out, hashes);
        out << ",\"full_physical_state\":"; write_hashes(out, physical);
        out << ",\"physical_write_generations\":[";
        for (std::size_t flag = 0; flag < c.session.state.generations.size(); ++flag)
            out << (flag ? "," : "") << c.session.state.generations[flag];
        out << "]}";
        return verified.committed_rows.back();
    };
    // Wrong draft at every possible index, plus complete acceptance. Restore
    // physical state only after the previous prefix was fully materialized.
    for (const unsigned rows : {2U, 4U, 8U})
        for (unsigned keep = 1; keep <= rows; ++keep) {
            reset();
            static_cast<void>(exercise(rows, keep, 0U, c.initial));
        }
    // Continue with actual accepted tickets across owner changes and a serial
    // fallback. Each committed state still has the independent serial oracle.
    reset();
    auto current = c.initial;
    unsigned offset = 0;
    for (const auto& [rows, keep] : {std::pair{8U, 1U}, std::pair{2U, 2U}, std::pair{1U, 1U}, std::pair{4U, 3U},
                                  std::pair{8U, 7U}, std::pair{4U, 1U}, std::pair{2U, 2U}, std::pair{8U, 8U}}) {
        current = exercise(rows, keep, offset, current);
        offset += keep;
    }
    out << "],\"case_count\":" << cases << ",\"passed\":true,\"full_logits_bits_equal\":true,\"all_seven_committed_state_components_equal\":true}\n";
    out.close();
    std::filesystem::rename(destination / "result.pending.json", destination / "result.json");
    // Optional wide copy verify gate (see hip_mtp_wide_gate.hpp): the decoder's wide cycle against its own serial
    // oracle from the restored frontier. It attaches the verifier to the greedy runtime, so it runs without the
    // copy cycle gate below (a separate process). The ordinary decoder never reads this switch.
    if (const char* wide_gate_mode = std::getenv("STRIX_MTP_WIDE_GATE"); wide_gate_mode && std::string_view(wide_gate_mode) == "1") {
        require(!std::getenv("STRIX_MTP_COPY_GATE"), "the wide gate and the copy cycle gate run in separate processes");
        wide_gate::run(c, destination, [&](std::uint64_t frontier) { return state.hashes(frontier); }, reset);
    }
    // Optional real-drafter qualification, separate from the fixed-schedule receipt above.
    // The ordinary decoder never reads this diagnostic switch. The serial oracle is
    // already captured at every frontier, so changing proposal/cycle lengths cannot
    // hide a token, committed-state or full-vocabulary logit difference.
    if (const char* mode = std::getenv("STRIX_MTP_COPY_GATE")) {
        const std::string_view selected(mode);
        require(selected == "ctx" || selected == "copy", "copy cycle gate is ctx or copy");
        reset();
        c.window->begin_session();
        // The qualification route leaves the greedy runtime without the verifier; real cycles need it.
        c.greedy.use_window_verifier(c.window);
        MtpContextDraftConfig config;
        config.copy_edit = selected == "copy";
        c.greedy.enable_context_draft(c.options.tokens, config);
        std::vector<Bytes> observed(8U, Bytes(logit_bytes.size()));
        unsigned observed_rows = 0;
        for (const auto& owner : owners) {
            const auto rows = owner->logical_rows();
            owner->set_window_step_observer([&, rows](std::size_t,
                const std::function<ExecutionDispatchContext(std::uint32_t)>& row) {
                if (row(0).op != SerialOp::lm_head) return;
                const auto* scope = current_hip_launch_context();
                require(scope && scope->stream, "copy cycle gate needs actual window stream");
                for (std::uint32_t r = 0; r < rows; ++r) {
                    const auto view = resolve_dispatch_write(row(r), 0);
                    require(view.bytes == logit_bytes.size(), "copy cycle logit extent differs");
                    check(hipMemcpyAsync(observed[r].data(), reinterpret_cast<const void*>(view.address),
                        observed[r].size(), hipMemcpyDeviceToHost, scope->stream), "copy cycle logits");
                }
                observed_rows = rows;
            });
        }
        struct ClearCopyObservers {
            const std::vector<std::shared_ptr<HipKnownPrefixWindow>>& owners;
            ~ClearCopyObservers() { for (const auto& owner : owners) owner->set_window_step_observer({}); }
        } clear_copy_observers{owners};
        std::ofstream copy_out(destination / "copy-cycle.pending.json", std::ios::binary);
        copy_out.exceptions(std::ios::failbit | std::ios::badbit);
        copy_out << "{\"schema\":\"mtp-copy-cycle-gate-v1\",\"mode\":\"" << selected
            << "\",\"performance_qualified\":false,\"cycles\":[";
        auto frontier = c.initial;
        unsigned copied = 0, cycle_count = 0, compared_logits = 0;
        std::uint64_t forward_rows = 0;
        while (copied < reference_rows) {
            observed_rows = 0;
            const auto before_native = c.runtime.execution_stats();
            const auto before_copy = c.greedy.context_stats()->copy_first_cycles;
            const auto actual = c.greedy.cycle(frontier, 7U, reference_rows - copied, {});
            const auto after_native = c.runtime.execution_stats();
            require(after_native.full_forward_rows >= before_native.full_forward_rows, "copy cycle forward counter regressed");
            forward_rows += after_native.full_forward_rows - before_native.full_forward_rows;
            if (c.greedy.context_stats()->copy_first_cycles > before_copy)
                require(after_native.full_forward_rows == before_native.full_forward_rows, "copy-first executed predictor forwards");
            const auto count = static_cast<unsigned>(actual.emitted_tokens.size());
            require(count && count <= reference_rows - copied, "copy cycle exceeds serial oracle");
            for (unsigned r = 0; r < count; ++r)
                require(actual.emitted_tokens[r] == tokens[copied + r + 1U], "copy cycle token differs from serial");
            for (const auto& owner : owners) owner->flush_deferred();
            check(hipDeviceSynchronize(), "copy cycle materialized completion");
            if (actual.attempted_target_window && !actual.target_checkpoint_restored) {
                require(observed_rows >= count, "copy cycle omitted physical window logits");
                for (unsigned r = 0; r < count; ++r) {
                    require(digest(observed[r]) == logits[copied + r], "copy cycle full logits differ from serial");
                    ++compared_logits;
                }
            } else {
                // Serial fallback may compare several candidates. Its scratch retains
                // the final row; the fixed schedule gate checks every serial frontier.
                check(hipMemcpy(logit_bytes.data(), reinterpret_cast<const void*>(serial_logits.address),
                    logit_bytes.size(), hipMemcpyDeviceToHost), "copy serial final logits");
                require(digest(logit_bytes) == logits[copied + count - 1U], "copy serial final logits differ");
                ++compared_logits;
            }
            const auto hashes = state.hashes(c.session.next_position);
            require(hashes == states[copied + count - 1U], "copy cycle committed state differs from serial");
            copy_out << (cycle_count++ ? "," : "") << "{\"offset\":" << copied << ",\"emitted\":" << count
                << ",\"rows\":" << actual.target_window_rows << ",\"accepted\":" << actual.accepted_candidates
                << ",\"committed_state\":";
            write_hashes(copy_out, hashes);
            copy_out << '}';
            copied += count;
            frontier = actual.final_target;
        }
        const auto* ng = c.greedy.context_stats();
        require(ng && ng->accepted_rows <= ng->extension_rows && ng->extension_rows <= ng->selected_rows,
            "copy cycle delivery counters differ");
        copy_out << "],\"tokens_checked\":" << copied << ",\"full_logit_rows_checked\":" << compared_logits
            << ",\"rows_selected\":" << ng->selected_rows << ",\"rows_delivered\":" << ng->extension_rows
            << ",\"rows_accepted\":" << ng->accepted_rows << ",\"copy_first_cycles\":" << ng->copy_first_cycles
            << ",\"predictor_forward_rows\":" << forward_rows << ",\"passed\":true}\n";
        copy_out.close();
        std::filesystem::rename(destination / "copy-cycle.pending.json", destination / "copy-cycle-result.json");
    }
    // Optional S0 cost diagnostic of wide verify windows (see hip_mtp_wide_cost.hpp). The ordinary decoder never
    // reads this switch; it times from the restored frontier on the oracle above and restores the session.
    if (const char* wide = std::getenv("STRIX_MTP_WIDE_COST"); wide && std::string_view(wide) == "1")
        wide_cost::run(c, destination, owners, tokens, logits, reset);
    std::cout << "{\"mode\":\"native-mtp-prefix-gate\",\"cases\":" << cases << ",\"passed\":true,\"performance_qualified\":false}\n";
}
} // namespace strix::tests::prefix_gate
