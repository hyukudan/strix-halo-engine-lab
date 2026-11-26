#include "inkling2/inkling2_mini_resident_forward.hpp"
#include "attn_core/prepared_gqa.hpp"

#include <algorithm>
#include <array>
#include <cstring>
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
    std::vector<std::pair<unsigned, float>> tap_inputs;

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

    void attention_batch(LayerVisit visit, std::span<const MiniResidentAttentionGroup> groups,
                         MiniConstRows q, MiniConstRows k, MiniConstRows v, MiniMutableRows output) {
        auto& seen = attention_visits[visit.logical_layer];
        if (seen != 0U) throw std::logic_error("unexpected packed attention visit");
        std::size_t offset = 0U;
        for (const auto& group : groups) {
            const auto count = std::size_t{group.chunk.rows} * kHiddenWidth;
            seen = 0;
            attention(visit, group.chunk, q.subspan(offset, count), k.subspan(offset, count),
                      v.subspan(offset, count), group.state, output.subspan(offset, count));
            offset += count;
        }
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

    void store_draft_tap(MiniConstRows input, std::span<std::uint16_t> output) {
        const auto visit = std::ranges::count(attention_visits, 1U);
        tap_inputs.emplace_back(static_cast<unsigned>(visit), input.front());
        for (std::size_t i = 0; i < input.size(); ++i)
            output[i / kHiddenWidth * MiniDSparkTapPlan::width + i % kHiddenWidth] =
                strix::attn::cache_rows::encode(input[i]);
    }

    void finish() {
        ++finish_calls;
        if (fail_finish) throw std::runtime_error("fixture asynchronous failure");
    }

    void reset_trace() {
        attention_visits.fill(0);
        moe_visits.fill(0);
        captured_banks.clear();
        tap_inputs.clear();
    }
};

struct RingBackend : FixtureBackend {
    MiniResidentChunk active{};
    bool trace_banks = false;
    std::array<std::vector<float>, 4> banks;
    unsigned bank_index = 0;

    void capture_message_bank(std::array<MiniConstRows, 2> input, std::array<MiniMutableRows, 2> normalized) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            if (trace_banks) {
                auto& trace = banks.at(bank_index++);
                trace.insert(trace.end(), input[i].begin(), input[i].end());
            }
            copy(input[i], normalized[i]);
        }
    }

    static float shift_history(MiniMutableRows history, unsigned width, float marker) {
        const float oldest = history.front();
        std::ranges::copy(history.subspan(width), history.begin());
        for (unsigned i = 0; i < width; ++i)
            history[history.size() - width + i] = marker + static_cast<float>(i) / 4096.0F;
        return oldest;
    }

    void attention(LayerVisit visit, MiniResidentChunk chunk, MiniConstRows query, MiniConstRows keys,
                   MiniConstRows values, MiniResidentVisitState state, MiniMutableRows output) {
        active = chunk;
        for (unsigned row = 0; row < chunk.rows; ++row) {
            const auto position = chunk.first_position + row;
            const auto slot = state.kv.wrapped ? position % state.kv.slots : position;
            const auto kv_offset = std::size_t{slot} * kKvWidth;
            const auto hidden_offset = std::size_t{row} * kHiddenWidth;
            const unsigned marker = 1U + position + visit.logical_layer * 257U;
            for (unsigned i = 0; i < kKvWidth; ++i) {
                state.kv.keys[kv_offset + i] = static_cast<std::uint16_t>(marker + i);
                state.kv.values[kv_offset + i] = static_cast<std::uint16_t>(3U * marker + i);
            }
            const auto past_k = shift_history(state.history[0], kKvWidth,
                                              static_cast<float>(marker) + keys[hidden_offset]);
            const auto past_v = shift_history(state.history[1], kKvWidth,
                                              static_cast<float>(2U * marker) + values[hidden_offset]);
            const auto out = output.subspan(hidden_offset, kHiddenWidth);
            if (starts_residual_block(visit)) copy(query.subspan(hidden_offset, kHiddenWidth), out);
            else std::ranges::fill(out, 0.0F);
            out.front() += (past_k + past_v) / 131072.0F;
        }
    }

    void convolve(LayerVisit visit, MiniConvolutionStream stream, MiniConstRows input, MiniMutableRows history,
                  MiniMutableRows output) {
        copy(input, output);
        for (unsigned row = 0; row < active.rows; ++row) {
            const auto offset = std::size_t{row} * kHiddenWidth;
            const auto marker = active.first_position + row + 257U * visit.logical_layer +
                                4096U * static_cast<unsigned>(stream);
            output[offset] += shift_history(history, kHiddenWidth, static_cast<float>(marker) + input[offset])
                              / 131072.0F;
        }
    }
};

constexpr unsigned kRingEnd = 2U * kSlidingWindow + 2U;

struct RingLane {
    explicit RingLane(unsigned max_rows)
        : plan(kRingEnd, max_rows), keys(plan.kv_elements_per_plane()), values(keys.size()),
          histories(plan.convolution_elements()), history_backup(histories.size()),
          scratch(mini_resident_activation_elements(max_rows)), kv_backup(mini_resident_max_kv_checkpoint_elements()),
          state(plan, {keys, values}, histories), forward(backend, state, {scratch, history_backup, kv_backup}) {}

    template <class Finalize = std::nullptr_t>
    void append(unsigned rows, MiniMutableRows output, Finalize finalize = nullptr) {
        backend.reset_trace();
        backend.bank_index = 0;
        const auto first = state.committed();
        const auto input = std::span(inputs).first(std::size_t{rows} * kHiddenWidth);
        for (unsigned row = 0; row < rows; ++row)
            for (unsigned i = 0; i < kHiddenWidth; ++i)
                input[std::size_t{row} * kHiddenWidth + i] = static_cast<float>(first + row) +
                                                           static_cast<float>(i) / 4096.0F;
        forward.append({first, rows}, input, output, finalize);
    }

    MiniSessionStatePlan plan;
    std::vector<std::uint16_t> keys, values;
    std::vector<float> histories, history_backup, scratch;
    std::vector<std::uint16_t> kv_backup;
    std::array<float, 2U * kHiddenWidth> inputs{};
    RingBackend backend;
    MiniResidentSessionState state;
    MiniResidentForward<RingBackend> forward;
};

bool equal_rows(MiniConstRows lhs, MiniConstRows rhs) {
    return lhs.size() == rhs.size() && (lhs.empty() || std::memcmp(lhs.data(), rhs.data(), lhs.size_bytes()) == 0);
}

template <class T>
struct HostBuffer {
    explicit HostBuffer(std::size_t size) : storage(size) {}
    std::span<T> span() { return storage; }
    std::vector<T> storage;
};

bool equal_committed_state(RingLane& lhs, RingLane& rhs) {
    const auto frontier = lhs.state.committed();
    if (rhs.state.committed() != frontier || lhs.state.poisoned() || rhs.state.poisoned() ||
        !equal_rows(lhs.histories, rhs.histories)) return false;
    for (const auto visit : mini_schedule()) {
        const auto& layout = lhs.plan.visit(visit);
        const auto left = mini_visit_kv_cache(lhs.plan, visit, {lhs.keys, lhs.values});
        const auto right = mini_visit_kv_cache(rhs.plan, visit, {rhs.keys, rhs.values});
        const auto first = layout.window == 0U ? 0U : frontier - std::min(frontier, layout.window);
        for (unsigned pos = first; pos < frontier; ++pos) {
            const auto i = std::size_t{left.wrapped ? pos % left.slots : pos} * kKvWidth;
            const auto j = std::size_t{right.wrapped ? pos % right.slots : pos} * kKvWidth;
            if (!std::equal(left.keys + i, left.keys + i + kKvWidth, right.keys + j) ||
                !std::equal(left.values + i, left.values + i + kKvWidth, right.values + j)) return false;
        }
    }
    return true;
}

bool check_ring_append() {
    RingLane multi(2);
    RingLane single(1);
    const auto visit = mini_schedule().front();
    const auto cache = mini_visit_kv_cache(single.plan, visit, {single.keys, single.values});
    if (!cache.wrapped || cache.slots != kSlidingWindow || multi.plan.visit(visit).kv.ring != 2U * kSlidingWindow)
        return false;
    constexpr unsigned boundary = 2U * kSlidingWindow - 1U;
    std::array<float, 2U * kHiddenWidth> multi_out{}, single_out{};
    const auto first_out = std::span(single_out).first(kHiddenWidth);
    while (multi.state.committed() < boundary) {
        const unsigned rows = std::min(2U, boundary - multi.state.committed());
        multi.append(rows, std::span(multi_out).first(std::size_t{rows} * kHiddenWidth));
        for (unsigned row = 0; row < rows; ++row) single.append(1, first_out);
    }

    multi.backend.trace_banks = single.backend.trace_banks = true;
    multi.append(2, multi_out);
    single.append(1, first_out);
    single.append(1, std::span(single_out).last(kHiddenWidth));
    if (multi.state.committed() != boundary + 2U || !equal_rows(multi_out, single_out) ||
        !equal_committed_state(multi, single)) return false;
    for (std::size_t i = 0; i < multi.backend.banks.size(); ++i) {
        const auto& bank = multi.backend.banks[i];
        if (bank.size() != multi_out.size() || !equal_rows(bank, single.backend.banks[i])) return false;
    }

    const auto slot = std::size_t{single.state.committed() % cache.slots} * kKvWidth;
    const auto old_key = cache.keys[slot];
    const auto old_value = cache.values[slot];
    bool overwrote = false;
    bool rejected = false;
    const auto finish_calls = single.backend.finish_calls;
    single.backend.trace_banks = false;
    try {
        single.append(1, first_out, [&](MiniResidentChunk, MiniConstRows) {
            overwrote = cache.keys[slot] != old_key && cache.values[slot] != old_value;
            throw std::runtime_error("fixture failure after wrapped append");
        });
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    if (!rejected || !overwrote || single.backend.finish_calls != finish_calls + 2U ||
        !equal_committed_state(multi, single)) return false;

    // Retry the rejected token and compare against a lane that never saw the
    // failed write. Check all visit state and both repeat banks on continuation.
    multi.backend.banks = {};
    single.backend.banks = {};
    single.backend.trace_banks = true;
    const auto control_out = std::span(multi_out).first(kHiddenWidth);
    multi.append(1, control_out);
    single.append(1, first_out);
    return equal_rows(control_out, first_out) && equal_committed_state(multi, single) &&
           std::ranges::equal(multi.backend.banks, single.backend.banks, equal_rows);
}

bool check_cohort_append() {
    struct Lane {
        MiniSessionStatePlan plan{5U, 3U};
        strix::session::ResidentRowSlot<MiniSessionStatePlan, HostBuffer> slot{plan};
        Lane() { reset(); }
        void reset() { slot.reset([](std::span<float> history) { std::ranges::fill(history, 0.0F); }); }
        MiniResidentSessionState& state() { return slot.state(); }
    };
    std::array<Lane, 3> batched, serial;
    std::vector<float> arena(mini_resident_activation_elements(3U));
    std::vector<float> backup(serial.front().slot.history().size());
    std::vector<float> inputs(3U * kHiddenWidth), output(inputs.size()), solo(inputs.size());
    FixtureBackend backend;
    const MiniResidentForwardWorkspace work{arena, backup, {}};
    const auto one_row = [&](Lane& lane, std::size_t i) {
        backend.reset_trace();
        MiniResidentForward forward(backend, lane.state(), work);
        forward.append({lane.state().committed(), 1U}, std::span(inputs).subspan(i * kHiddenWidth, kHiddenWidth),
                       std::span(solo).subspan(i * kHiddenWidth, kHiddenWidth));
    };
    for (unsigned i = 0; i < batched.size(); ++i) {
        std::ranges::fill(std::span(inputs).subspan(i * kHiddenWidth, kHiddenWidth), static_cast<float>(i));
        for (unsigned p = 0; p < i; ++p) {
            one_row(batched[i], i);
            one_row(serial[i], i);
        }
        one_row(serial[i], i);
    }

    const std::array groups{MiniResidentAppendGroup{&batched[0].state(), {0U, 1U}},
                            MiniResidentAppendGroup{&batched[1].state(), {1U, 1U}},
                            MiniResidentAppendGroup{&batched[2].state(), {2U, 1U}}};
    MiniResidentForward forward(backend, work);
    backend.reset_trace();
    const auto before = backend.finish_calls;
    const std::array duplicate{groups[0], groups[0], groups[2]};
    try {
        forward.append_batch(duplicate, 3U, inputs, output, [] {}, [](MiniConstRows) {}, [] {});
        return false;
    } catch (const std::invalid_argument&) {
        if (backend.finish_calls != before || batched[0].state().committed() != 0U) return false;
    }
    forward.append_batch(groups, 3U, inputs, output, [] {}, [](MiniConstRows) {}, [&] {
        if (batched[1].state().committed() != 1U) throw std::logic_error("frontier published before validation");
    });
    if (backend.finish_calls != before + 1U || !equal_rows(output, solo)) return false;
    for (std::size_t i = 0; i < batched.size(); ++i) {
        const auto left = batched[i].slot.planes();
        const auto right = serial[i].slot.planes();
        if (batched[i].state().committed() != serial[i].state().committed() ||
            !std::ranges::equal(left.keys, right.keys) || !std::ranges::equal(left.values, right.values) ||
            !equal_rows(batched[i].slot.history(), serial[i].slot.history())) return false;
    }

    auto next = groups;
    for (auto& group : next) ++group.chunk.first_position;
    backend.reset_trace();
    try {
        forward.append_batch(next, 3U, inputs, output, [] {}, [](MiniConstRows) {}, [] {
            throw std::runtime_error("batch head rejected");
        });
    } catch (const std::runtime_error&) {
        if (!std::ranges::all_of(next, [](const auto& group) {
            return group.state->poisoned() && group.state->committed() == group.chunk.first_position;
        })) return false;
        batched[2].reset();
        return batched[2].state().committed() == 0U && !batched[2].state().poisoned() &&
            std::ranges::all_of(batched[2].slot.history(), [](float value) { return value == 0.0F; });
    }
    return false;
}

struct PreparedShape {
    static constexpr unsigned kMaxContext = 8, kHiddenWidth = 4, kQueryWidth = 4, kKvWidth = 4;
    static constexpr unsigned kKvHeads = 1, kQueryHeads = 1, kConvTaps = 4, kHeadWidth = 4;
    static constexpr unsigned kRelativeWidth = 4, kRelativeExtent = 4;
    static constexpr float kQkClip = 1, kRelativeClip = 1, kFullConvDivisor = 1, kLocalConvDivisor = 1;
    static constexpr unsigned projection_width(strix::attn_core::PreparedGqaProjection projection) {
        return projection == strix::attn_core::PreparedGqaProjection::sink ? 1U : 4U;
    }
};

struct PreparedFixture {
    using Status = int;
    static constexpr Status success = 0, invalid = 1;
    mutable std::vector<unsigned> dense_rows, scaled_positions;

    Status linear(const float* input, unsigned width, unsigned rows,
                  std::span<const strix::attn_core::PreparedGqaLinear> projections) const {
        dense_rows.push_back(rows);
        for (const auto& p : projections)
            for (unsigned row = 0; row < rows; ++row)
                std::fill_n(p.output + row * p.outputs, p.outputs, input[row * width]);
        return success;
    }
    Status round(float*, std::size_t, unsigned* = nullptr) const { return success; }
    Status rms_values(float*, unsigned) const { return success; }
    Status relative_logits(float*, const std::uint16_t*, float*, unsigned, unsigned*) const { return success; }
    Status clip(float*, const std::uint16_t*, unsigned, unsigned, unsigned, float,
                bool scaled, unsigned position) const {
        if (scaled) scaled_positions.push_back(position);
        return success;
    }
    Status convolve(const strix::core::ShortConvolution&, unsigned rows, const float* input, const float*,
                    const float*, float* history, float* output) const {
        if (rows != 1U) throw std::logic_error("packed convolution crosses lanes");
        output[0] = input[0] + history[0];
        history[0] = input[0];
        return success;
    }
};

bool check_prepared_groups() {
    namespace ac = strix::attn_core;
    alignas(16) const std::array<float, 8> input{1, 0, 0, 0, 2, 0, 0, 0};
    alignas(16) std::array<std::uint16_t, 8> matrix{};
    alignas(16) std::array<std::array<float, 8>, 11> scratch{};
    alignas(16) std::array<std::array<float, 4>, 4> history{{{10}, {20}, {30}, {40}}};
    ac::PreparedGqaWeights weights;
    weights.matrix.fill(matrix.data());
    weights.query_gain = weights.relative_gain = matrix.data();
    weights.key_convolution = weights.value_convolution = input.data();
    const ac::PreparedGqaWorkspace work{scratch[0].data(), scratch[1].data(), scratch[2].data(),
                                        scratch[3].data(), scratch[4].data(), nullptr, nullptr};
    const ac::PreparedGqaRows output{scratch[5].data(), scratch[6].data(), scratch[7].data(),
                                    scratch[8].data(), scratch[9].data(), scratch[10].data()};
    const std::array groups{ac::PreparedGqaGroup{2U, 1U, history[0].data(), history[1].data()},
                            ac::PreparedGqaGroup{5U, 1U, history[2].data(), history[3].data()}};
    const ac::PreparedGqaInputs inputs{input.data(), input.data(), input.data()};
    PreparedFixture backend;
    if (ac::enqueue_prepared_gqa<PreparedShape>(backend, true, 0U, 2U, inputs, weights,
                                                work, output, nullptr, nullptr, groups) != PreparedFixture::success)
        return false;
    if (backend.dense_rows != std::vector{2U, 2U, 2U} || backend.scaled_positions != std::vector{2U, 5U, 2U, 5U} ||
        output.key[0] != 11.0F || output.key[4] != 32.0F || output.value[0] != 21.0F || output.value[4] != 42.0F)
        return false;
    backend.dense_rows.clear();
    return ac::enqueue_prepared_gqa<PreparedShape>(backend, true, 0U, 2U, inputs, weights,
               work, output, nullptr, nullptr, std::span(groups).first(1U)) == PreparedFixture::invalid && backend.dense_rows.empty();
}

struct CaptureBackend {
    bool capturing = true;
    unsigned copies = 0;
    unsigned submits = 0;
    unsigned drains = 0;

    template <class T>
    void copy(std::span<const T> input, std::span<T> output) {
        ++copies;
        if (!capturing) std::ranges::copy(input, output.begin());
    }

    bool abandon_capture() noexcept {
        const bool discarded = capturing;
        capturing = false;
        return discarded;
    }

    void submit_capture() {
        ++submits;
        capturing = false;
    }

    void finish() {
        if (capturing) throw std::logic_error("capture must end before a drain");
        ++drains;
    }
};

bool check_discarded_capture() {
    constexpr MiniSessionStatePlan plan(4, 1);
    std::vector<std::uint16_t> keys(plan.kv_elements_per_plane()), values(keys.size());
    std::vector<float> histories(plan.convolution_elements(), 1.0F), backup(histories.size(), 2.0F);
    MiniResidentSessionState state(plan, {keys, values}, histories);
    CaptureBackend backend;
    try {
        state.append(backend, {0, 1}, backup, {},
                     [] { throw std::runtime_error("fixture failed while recording the head"); },
                     [] { throw std::logic_error("abandoned capture reached validation"); });
        return false;
    } catch (const std::runtime_error&) {
        return state.committed() == 0U && !state.poisoned() && !backend.capturing &&
               backend.copies == 1U && backend.submits == 0U && backend.drains == 1U &&
               histories.front() == 1.0F && backup.front() == 2.0F;
    }
}

// History contract of the convolution tape (session::RowAppendState): every history holds the last input rows of
// its stream. Key / value inputs are recorded by the attention (as attn_core does for the prepared GQA
// convolutions); attention / MoE inputs by the forward. A partial commit by copies must equal the replay.
struct TapeBackend : RingBackend {
    // Copies into the bound tape: each (visit, stream) region is recorded once per staged append (a duplicate
    // recording wrote the same bytes twice: one more launch per stream and visit in every verify window).
    std::span<const float> tape_range;
    struct TapeCopy {
        MiniConstRows source;
        MiniMutableRows destination;
        void replay() const { std::ranges::copy(source, destination.begin()); }
    };
    std::vector<TapeCopy> copies;
    std::size_t next_copy = 0;
    bool replaying = false;
    unsigned windows = 0;

    // The fake kernels run the production walk to generate new inputs at each leaf. Keep the
    // D2D commands from the first capture, including both physical endpoints and their extent.
    template <class Work>
    bool try_window_graph(MiniResidentChunk, Work& work, bool taped) {
        if (!taped || tape_range.empty()) return false;
        next_copy = 0;
        replaying = windows != 0U;
        work();
        if (next_copy != copies.size() || copies.empty()) throw std::runtime_error("incomplete window tape capture");
        ++windows;
        return true;
    }
    template <class T>
    void copy(std::span<const T> source, std::span<T> output) {
        if constexpr (std::is_same_v<T, float>) {
            const auto* at = output.data();
            if (!output.empty() && !tape_range.empty() && at >= tape_range.data() &&
                at < tape_range.data() + tape_range.size()) {
                if (!replaying) {
                    if (std::ranges::any_of(copies, [at](const auto& cmd) { return cmd.destination.data() == at; }))
                        throw std::runtime_error("duplicate window tape capture");
                    copies.push_back({source, output});
                }
                const auto& captured = copies.at(next_copy++);
                if (source.data() != captured.source.data() || output.data() != captured.destination.data() ||
                    source.size() != captured.source.size() || output.size() != captured.destination.size() ||
                    source.size() != output.size()) throw std::runtime_error("changed window tape binding");
                captured.replay();
                return;
            }
        }
        RingBackend::copy(source, output);
    }
    static void push(MiniMutableRows history, MiniConstRows row) {
        const auto width = row.size();
        std::ranges::copy(history.subspan(width), history.begin());
        std::ranges::copy(row, history.end() - static_cast<std::ptrdiff_t>(width));
    }

    void attention(LayerVisit visit, MiniResidentChunk chunk, MiniConstRows query, MiniConstRows keys,
                   MiniConstRows values, MiniResidentVisitState state, MiniMutableRows output) {
        active = chunk;
        std::array<float, kKvWidth> k{}, v{};
        for (unsigned row = 0; row < chunk.rows; ++row) {
            const auto hidden_offset = std::size_t{row} * kHiddenWidth;
            const float past = state.history[0].front() + state.history[1].front();
            for (unsigned i = 0; i < kKvWidth; ++i) {
                k[i] = keys[hidden_offset + i] + static_cast<float>(visit.logical_layer);
                v[i] = 0.5F * values[hidden_offset + i];
            }
            for (unsigned stream = 0; stream < 2U; ++stream) {
                const auto& in = stream == 0U ? k : v;
                if (!state.tape[stream].empty())
                    std::ranges::copy(in, state.tape[stream].begin() + std::ptrdiff_t{row} * kKvWidth);
                push(state.history[stream], in);
            }
            const auto out = output.subspan(hidden_offset, kHiddenWidth);
            copy(query.subspan(hidden_offset, kHiddenWidth), out);
            out.front() += past / 64.0F;
        }
    }

    void convolve(LayerVisit, MiniConvolutionStream, MiniConstRows input, MiniMutableRows history,
                  MiniMutableRows output) {
        copy(input, output);
        for (unsigned row = 0; row < active.rows; ++row) {
            const auto offset = std::size_t{row} * kHiddenWidth;
            output[offset] += history.front() / 64.0F;
            push(history, input.subspan(offset, kHiddenWidth));
        }
    }
};

struct TapeLane {
    static constexpr unsigned kRows = 4U;
    explicit TapeLane(bool taped)
        : plan(64U, kRows), keys(plan.kv_elements_per_plane()), values(keys.size()),
          histories(plan.convolution_elements()), history_backup(histories.size()),
          scratch(mini_resident_activation_elements(kRows)), kv_backup(mini_resident_max_kv_checkpoint_elements()),
          tape(std::size_t{kRows} * plan.convolution_row_elements()), inputs(std::size_t{kRows} * kHiddenWidth),
          outputs(inputs.size()), state(plan, {keys, values}, histories),
          forward(backend, state, {scratch, history_backup, kv_backup}) {
        if (taped) state.bind_convolution_tape(tape, kRows);
    }
    void append(unsigned rows, bool stage) {
        backend.reset_trace();
        const auto first = state.committed();
        for (std::size_t i = 0; i < std::size_t{rows} * kHiddenWidth; ++i)
            inputs[i] = static_cast<float>((first + i / kHiddenWidth) % 7U) + static_cast<float>(i % 97U) / 128.0F;
        const auto count = std::size_t{rows} * kHiddenWidth;
        forward.append({first, rows}, std::span(inputs).first(count), std::span(outputs).first(count),
                       nullptr, nullptr, nullptr, nullptr, stage);
    }

    MiniSessionStatePlan plan;
    std::vector<std::uint16_t> keys, values;
    std::vector<float> histories, history_backup, scratch;
    std::vector<std::uint16_t> kv_backup;
    std::vector<float> tape, inputs, outputs;
    TapeBackend backend;
    MiniResidentSessionState state;
    MiniResidentForward<TapeBackend> forward;
};

bool check_convolution_tape_commit() {
    TapeLane taped(true), replayed(false);
    taped.append(3, false);
    replayed.append(3, false);
    taped.backend.tape_range = taped.tape;
    unsigned replays = 0;
    for (const unsigned keep : {1U, 0U, 3U, 2U, 4U, 1U}) {
        std::ranges::fill(taped.tape, std::numeric_limits<float>::quiet_NaN());
        taped.append(TapeLane::kRows, true);
        replayed.append(TapeLane::kRows, true);
        taped.state.commit_staged(taped.backend, keep, [&](unsigned rows) { ++replays; taped.append(rows, false); });
        replayed.state.commit_staged(replayed.backend, keep, [&](unsigned rows) { replayed.append(rows, false); });
        if (taped.state.committed() != replayed.state.committed() || taped.state.poisoned() ||
            !equal_rows(taped.histories, replayed.histories)) return false;
    }
    // One more row continues identically from the rebuilt histories.
    taped.append(1, false);
    replayed.append(1, false);
    return taped.backend.windows == 6U && replays == 0U && equal_rows(taped.histories, replayed.histories) &&
           equal_rows(std::span(taped.outputs).first(kHiddenWidth), std::span(replayed.outputs).first(kHiddenWidth));
}

bool check_draft_tap_frontier() {
    constexpr MiniSessionStatePlan plan(5, 2);
    constexpr auto width = MiniDSparkTapPlan::width;
    std::vector<std::uint16_t> keys(plan.kv_elements_per_plane()), values(keys.size());
    std::vector<std::uint16_t> taps(MiniDSparkTapPlan::elements(5));
    std::vector<float> histories(plan.convolution_elements()), backup(histories.size());
    std::vector<float> workspace(mini_resident_activation_elements(2));
    std::array<float, 2U * kHiddenWidth> input{}, output{};
    std::ranges::fill(std::span(input).last(kHiddenWidth), 10.0F);
    MiniResidentSessionState state(plan, {keys, values}, histories);
    FixtureBackend backend;
    MiniResidentForward forward(backend, state, {workspace, backup, {}, taps});
    const auto append = [&](unsigned rows, bool stage) {
        backend.reset_trace();
        const auto count = rows * kHiddenWidth;
        forward.append({state.committed(), rows}, std::span(input).first(count), std::span(output).first(count),
                       nullptr, nullptr, nullptr, nullptr, stage);
    };
    const auto replay = [&](unsigned rows) { append(rows, false); };

    append(1, false);
    const std::vector<std::pair<unsigned, float>> expected{{5, 5.0F}, {23, 41.0F}, {41, 107.0F}, {48, 114.0F}};
    if (backend.tap_inputs != expected || forward.committed_taps().size() != width) return false;
    const std::vector<std::uint16_t> first(taps.begin(), taps.begin() + width);
    for (unsigned i = 0; i < MiniDSparkTapPlan::count; ++i) {
        if (first[i * kHiddenWidth] != strix::attn::cache_rows::encode(expected[i].second)) return false;
    }

    append(2, true);
    if (!std::ranges::equal(forward.committed_taps(), first)) return false;
    std::ranges::fill(std::span(taps).subspan(2U * width, width), std::uint16_t{0xffff});
    state.commit_staged(backend, 1, replay);
    const auto after_partial = forward.committed_taps();
    if (after_partial.size() != 2U * width || !std::ranges::equal(after_partial.first(width), first) ||
        !std::ranges::equal(after_partial.last(width), first)) return false;

    append(2, true);
    state.commit_staged(backend, 0, replay);
    if (forward.committed_taps().size() != 2U * width) return false;
    append(2, true);
    state.commit_staged(backend, 2, replay);
    const auto committed = forward.committed_taps();
    return committed.size() == 4U * width &&
           std::ranges::equal(committed.subspan(2U * width, width), first) &&
           committed.back() == strix::attn::cache_rows::encode(184.0F);
}

}  // namespace

int main() {
    constexpr auto schedule = mini_schedule();
    static_assert(MiniDSparkTapPlan::index(5) == 0U && schedule[5].physical_layer == 5U);
    static_assert(MiniDSparkTapPlan::index(23) == 1U && schedule[23].physical_layer == 11U && schedule[23].pass == 1U);
    static_assert(MiniDSparkTapPlan::index(41) == 2U && schedule[41].physical_layer == 17U && schedule[41].pass == 2U);
    static_assert(MiniDSparkTapPlan::index(kLogicalVisits) == 3U);
    static_assert(MiniDSparkTapPlan::index(40) == MiniDSparkTapPlan::count);
    constexpr MiniSessionStatePlan wrap_one(2049, 1);
    static_assert(!strix::session::overwrites_committed_row(wrap_one.visit(schedule[0]), {1023, 1}));
    static_assert(strix::session::overwrites_committed_row(wrap_one.visit(schedule[0]), {1024, 1}));
    static_assert(!strix::session::overwrites_committed_row(wrap_one.visit(schedule[5]), {1024, 1}));
    constexpr MiniSessionStatePlan wrap_chunk(4096, 1025);
    static_assert(strix::session::overwrites_committed_row(wrap_chunk.visit(schedule[6]), {1100, 1025}));
    static_assert(!strix::session::overwrites_committed_row(wrap_chunk.visit(schedule[6]), {1100, 1024}));
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
    } catch (const std::invalid_argument&) {}

    // Reject a completed head after the one wait and before advancing the frontier.
    FixtureBackend head_backend;
    MiniResidentSessionState head_state(session_plan, {keys, values}, histories);
    MiniResidentForward head_forward(head_backend, head_state, {scratch, backup, {}});
    head_forward.append({0, 1}, one_input, one_out);
    head_backend.reset_trace();
    bool checked_head = false;
    try {
        head_forward.append({1, 1}, one_input, one_out, nullptr, [&] {
            checked_head = head_backend.finish_calls == 2U && head_state.committed() == 1U;
            throw std::runtime_error("fixture rejected completed head");
        });
        return 14;
    } catch (const std::runtime_error&) {
        if (!checked_head || !head_state.poisoned() || head_state.committed() != 1U ||
            head_backend.finish_calls != 2U) return 15;
    }
    if (!check_ring_append()) return 16;
    if (!check_discarded_capture()) return 19;
    if (!check_draft_tap_frontier()) return 20;
    if (!check_cohort_append()) return 23;
    if (!check_prepared_groups()) return 24;
    try {
        if (!check_convolution_tape_commit()) return 25;
    } catch (const std::logic_error&) {
        return 25;
    } catch (const std::runtime_error&) {
        return 25;
    }

    // A failed later prefill chunk must retire the partially committed lane.
    FixtureBackend sequence_backend;
    MiniResidentSessionState sequence_state(session_plan, {keys, values}, histories);
    MiniResidentForward sequence_forward(sequence_backend, sequence_state, {scratch, backup, {}});
    constexpr std::array sequence{1U, 2U, 3U};
    bool saw_last = false;
    try {
        (void)sequence_state.append_sequence(std::span<const unsigned>(sequence),
            [&](std::span<const unsigned> tokens, bool last) -> unsigned {
                if (last) {
                    saw_last = tokens.size() == 1U && tokens.front() == 3U;
                    throw std::runtime_error("fixture failed after first prefill chunk");
                }
                sequence_forward.append({0U, static_cast<unsigned>(tokens.size())}, inputs, output);
                return 0U;
            });
    } catch (const std::runtime_error&) {
        return saw_last && sequence_state.poisoned() && sequence_state.committed() == 2U ? 0 : 17;
    }
    return 18;
}
