#include "inkling2/inkling2_mini_resident_forward.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using namespace strix::inkling2;

struct FixtureBackend {
    std::array<unsigned, kLogicalVisits> attention_visits{};
    std::array<unsigned, kLogicalVisits> moe_visits{};
    std::vector<std::array<float, 4>> captured_banks;
    unsigned fail_visit = kLogicalVisits;
    unsigned finish_calls = 0;
    bool fail_finish = false;

    template <class T>
    void copy(std::span<const T> source, std::span<T> output) {
        if (source.size() != output.size()) throw std::logic_error("fixture copy geometry");
        std::ranges::copy(source, output.begin());
    }

    void residual_read(std::span<const MiniConstRows> blocks, MiniConstRows current, std::string_view name,
                       std::span<const MiniMutableRows> outputs) {
        if (name == "attn_res.output_query" && blocks.size() != 4U) throw std::logic_error("missing residual blocks");
        for (std::size_t i = 0; i < outputs.size(); ++i) {
            const auto delta = 100.0F * static_cast<float>(i);
            std::ranges::transform(current, outputs[i].begin(), [delta](float value) { return value + delta; });
        }
    }

    void capture_message_bank(std::array<MiniConstRows, 2> bank, std::array<MiniMutableRows, 2> normalized) {
        const auto second_row = bank[0].size() == 2U * kHiddenWidth ? kHiddenWidth : 0U;
        captured_banks.push_back({bank[0][0], bank[1][0], bank[0][second_row], bank[1][second_row]});
        copy(bank[0], normalized[0]);
        copy(bank[1], normalized[1]);
    }

    void layer_input(LayerVisit visit, MessagePath path, MiniConstRows raw, std::span<const MiniConstRows> bank,
                     MiniMutableRows output) {
        if ((visit.pass == 0U) != bank.empty()) throw std::logic_error("unexpected previous-pass bank");
        copy(raw, output);
        if (visit.pass != 0 && visit.physical_layer == kCoreFirst && path == MessagePath::query) {
            for (std::size_t i = 0; i < output.size(); ++i) output[i] += bank[0][i] + bank[1][i];
        }
    }

    void attention(LayerVisit visit, MiniResidentChunk chunk, MiniConstRows q, MiniConstRows k, MiniConstRows v,
                   MiniResidentVisitState state, MiniMutableRows output) {
        if (q.size() != std::size_t{chunk.rows} * kHiddenWidth || ++attention_visits[visit.logical_layer] != 1U) {
            throw std::logic_error("unexpected attention visit");
        }
        if (is_full_attention(visit.physical_layer)) {
            if (k[0] != q[0] + 100.0F || v[0] != q[0] + 200.0F) throw std::logic_error("wrong full QKV lanes");
        } else if ((q.data() == k.data() && k.data() == v.data()) != (visit.pass == 0U)) {
            throw std::logic_error("wrong local QKV stream identity");
        }

        state.history[0][0] += 1.0F;
        state.history[1][0] += 2.0F;
        for (unsigned row = 0; row < chunk.rows; ++row) {
            const auto position = chunk.first_position + row;
            const auto slot = state.kv.wrapped ? position % state.kv.slots : position;
            state.kv.keys[std::size_t{slot} * kKvWidth] = static_cast<std::uint16_t>(1U + visit.logical_layer);
            state.kv.values[std::size_t{slot} * kKvWidth] = static_cast<std::uint16_t>(1U + visit.logical_layer * 2U);
        }
        if (starts_residual_block(visit)) copy(q, output);
        else std::ranges::fill(output, 0.0F);
    }

    void convolve(LayerVisit, MiniConvolutionStream stream, MiniConstRows input, MiniMutableRows history,
                  MiniMutableRows output) {
        if (stream != MiniConvolutionStream::attention && stream != MiniConvolutionStream::moe) {
            throw std::logic_error("wrong output convolution stream");
        }
        history[0] += static_cast<float>(static_cast<unsigned>(stream) + 1U);
        copy(input, output);
    }

    void add(MiniConstRows lhs, MiniConstRows rhs, MiniMutableRows output, bool) {
        for (std::size_t i = 0; i < output.size(); ++i) output[i] = lhs[i] + rhs[i];
    }

    void moe(LayerVisit visit, MiniConstRows, MiniMutableRows output) {
        if (++moe_visits[visit.logical_layer] != 1U) throw std::logic_error("unexpected expert visit");
        if (visit.logical_layer == fail_visit) throw std::runtime_error("fixture launch failure");
        std::ranges::fill(output, 1.0F);
    }

    void norm(MiniConstRows input, std::string_view name, MiniMutableRows output) {
        if (name != "norm.weight") throw std::logic_error("wrong final RMS parameter");
        copy(input, output);
    }

    void finish() {
        ++finish_calls;
        if (fail_finish) throw std::runtime_error("fixture asynchronous failure");
    }

    void reset_trace() {
        attention_visits.fill(0);
        moe_visits.fill(0);
        captured_banks.clear();
    }
};

}  // namespace

int main() {
    constexpr auto schedule = mini_schedule();
    constexpr MiniSessionStatePlan wrap_one(2049, 1);
    static_assert(!mini_append_overwrites_committed_row(wrap_one.visit(schedule[0]), {1023, 1}));
    static_assert(mini_append_overwrites_committed_row(wrap_one.visit(schedule[0]), {1024, 1}));
    static_assert(!mini_append_overwrites_committed_row(wrap_one.visit(schedule[5]), {1024, 1}));
    constexpr MiniSessionStatePlan wrap_chunk(4096, 1025);
    static_assert(mini_append_overwrites_committed_row(wrap_chunk.visit(schedule[6]), {1100, 1025}));
    static_assert(!mini_append_overwrites_committed_row(wrap_chunk.visit(schedule[6]), {1100, 1024}));
    static_assert(mini_resident_max_kv_checkpoint_elements() == 2U * 40U * kKvWidth);
    static_assert(mini_controller_path(MessagePath::moe) == "mlp");
    static_assert(mini_input_norm_suffix(schedule[5], MessagePath::key) == "k_input_norm.weight");
    static_assert(mini_input_norm_suffix(schedule[6], MessagePath::value) == "attn_norm.weight");

    constexpr MiniSessionStatePlan session_plan(4, 2);
    std::vector<std::uint16_t> keys(session_plan.kv_elements_per_plane()), values(keys.size());
    std::vector<float> histories(session_plan.convolution_elements()), backup(histories.size());
    std::vector<float> scratch(mini_resident_activation_elements(2));
    std::vector<float> inputs(2U * kHiddenWidth), output(inputs.size());
    std::ranges::fill(std::span(inputs).subspan(kHiddenWidth), 10.0F);
    MiniResidentSessionState state(session_plan, {keys, values}, histories);
    FixtureBackend backend;
    MiniResidentForward forward(backend, state, {scratch, backup, {}});

    bool finalized = false;
    forward.append({0, 2}, inputs, output, [&](MiniResidentChunk chunk, MiniConstRows hidden) {
        if (state.committed() != 0 || chunk.rows != 2 || hidden[0] != 114.0F) throw std::logic_error("early commit");
        finalized = true;
    });
    if (!finalized || state.committed() != 2 || state.poisoned() || backend.finish_calls != 1U ||
        output[0] != 114.0F || output[kHiddenWidth] != 184.0F || backend.captured_banks.size() != 2U ||
        backend.captured_banks[0] != std::array{12.0F, 18.0F, 22.0F, 28.0F} ||
        backend.captured_banks[1] != std::array{42.0F, 48.0F, 72.0F, 78.0F}) return 1;
    for (const auto visit : schedule) {
        const auto& layout = session_plan.visit(visit);
        if (backend.attention_visits[visit.logical_layer] != 1 || backend.moe_visits[visit.logical_layer] != 1 ||
            keys[layout.kv_offset] != visit.logical_layer + 1U || keys[layout.kv_offset + kKvWidth] != visit.logical_layer + 1U) return 2;
        for (std::size_t i = 0; i < layout.convolution_offsets.size(); ++i)
            if (histories[layout.convolution_offsets[i]] != static_cast<float>(i + 1U)) return 3;
    }

    const auto committed_histories = histories;
    const auto one_input = std::span(inputs).first(kHiddenWidth);
    auto one_out = std::span(output).first(kHiddenWidth);
    backend.reset_trace();
    backend.fail_visit = 18;
    try {
        forward.append({2, 1}, one_input, one_out);
        return 4;
    } catch (const std::runtime_error&) {
        if (state.committed() != 2 || state.poisoned() || histories != committed_histories) return 5;
    }

    backend.reset_trace();
    backend.fail_visit = kLogicalVisits;
    try {
        forward.append({2, 1}, one_input, one_out,
                       [](MiniResidentChunk, MiniConstRows) { throw std::runtime_error("fixture head failure"); });
        return 6;
    } catch (const std::runtime_error&) {
        if (state.committed() != 2 || state.poisoned() || histories != committed_histories) return 7;
    }

    backend.reset_trace();
    forward.append({2, 1}, one_input, one_out);
    if (state.committed() != 3 || one_out[0] != 114.0F || state.poisoned()) return 8;
    try {
        forward.append({2, 1}, one_input, one_out);
        return 9;
    } catch (const std::invalid_argument&) {
        if (state.committed() != 3 || state.poisoned()) return 10;
    }

    backend.reset_trace();
    backend.fail_finish = true;
    try {
        forward.append({3, 1}, one_input, one_out);
        return 11;
    } catch (const std::runtime_error&) {
        if (state.committed() != 3 || !state.poisoned()) return 12;
    }
    try {
        forward.append({3, 1}, one_input, one_out);
        return 13;
    } catch (const std::invalid_argument&) {
        return 0;
    }
}
