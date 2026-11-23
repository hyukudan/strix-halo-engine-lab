#pragma once

#include "inkling2_mini_dspark_taps.hpp"
#include "inkling2_mini_messages.hpp"
#include "inkling2_mini_state.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace strix::inkling2 {

using MiniConstRows = std::span<const float>;
using MiniMutableRows = std::span<float>;

using MiniResidentChunk = session::KvAppendChunk;
using MiniResidentSessionState = session::KvStore::RowAppendState<MiniSessionStatePlan>;
using MiniResidentVisitState = MiniResidentSessionState::VisitRows;
using MiniResidentAppendGroup = MiniResidentSessionState::AppendGroup;

struct MiniResidentAttentionGroup {
    MiniResidentChunk chunk;
    MiniResidentVisitState state;
};

enum class MiniConvolutionStream : unsigned { key, value, attention, moe };

struct MiniResidentForwardWorkspace {
    std::span<float> activations;
    std::span<float> convolution_checkpoint;
    std::span<std::uint16_t> kv_checkpoint;
    std::span<std::uint16_t> draft_taps{};
};

inline constexpr unsigned kMiniResidentImages = 16;

inline constexpr std::size_t mini_resident_activation_elements(unsigned rows) {
    constexpr auto per_row = std::size_t{kMiniResidentImages} * kHiddenWidth;
    if (rows == 0 || rows > kMaxContext || per_row > std::numeric_limits<std::size_t>::max() / sizeof(float) / rows) {
        throw std::invalid_argument("invalid Mini resident activation workspace geometry");
    }
    return rows * per_row;
}

inline constexpr std::size_t mini_resident_max_kv_checkpoint_elements() {
    return MiniSessionStatePlan(kMaxContext, 1U).kv_checkpoint_elements();
}

// Short checkpoint names used by the backend of layer_input(). On a repeated
// pass, gate the FP32 residual with the message before applying this norm.
inline constexpr std::string_view mini_input_norm_suffix(LayerVisit visit, MessagePath path) {
    if (path == MessagePath::moe) return "mlp_norm.weight";
    if (!is_full_attention(visit.physical_layer)) return "attn_norm.weight";
    switch (path) {
        case MessagePath::query: return "q_input_norm.weight";
        case MessagePath::key: return "k_input_norm.weight";
        case MessagePath::value: return "v_input_norm.weight";
        case MessagePath::moe: break;
    }
    throw std::invalid_argument("invalid Mini input path");
}

inline constexpr std::string_view mini_controller_path(MessagePath path) {
    switch (path) {
        case MessagePath::query: return "attn";
        case MessagePath::key: return "k";
        case MessagePath::value: return "v";
        case MessagePath::moe: return "mlp";
    }
    throw std::invalid_argument("invalid Mini controller path");
}

// One device chunk, including decode (rows=1). All callbacks enqueue on the
// same stream, throw on admission/enqueue failure, and retain no host span
// descriptors or names. Arithmetic lives in backend kernels. finish() waits for prior
// work and reports device errors. The returned hidden rows are usable only on
// success. Input/output, persistent state and scratch allocations are disjoint.
template <class Backend>
class MiniResidentForward {
public:
    MiniResidentForward(Backend& backend, MiniResidentSessionState& session, MiniResidentForwardWorkspace workspace)
        : backend_(backend), session_(&session), workspace_(workspace) {}
    MiniResidentForward(Backend& backend, MiniResidentForwardWorkspace workspace)
        : backend_(backend), workspace_(workspace) {}

    // Dense slots beyond the target frontier are private to append/verify.
    // A partial verify commit replays the kept positions into the same slots.
    [[nodiscard]] std::span<const std::uint16_t> committed_taps() const {
        if (workspace_.draft_taps.empty()) return {};
        if (!session_) throw std::logic_error("Mini resident tap frontier needs a session");
        if (session_->poisoned()) throw std::logic_error("invalid Mini resident tap frontier");
        return workspace_.draft_taps.first(MiniDSparkTapPlan::elements(session_->committed()));
    }

    // Inputs are already embed-normalized and BF16-rounded, widened to FP32.
    // An optional finalizer may enqueue a head projection before the state
    // commits. It receives the enqueue-ordered final hidden rows. The optional
    // validator consumes the completed reply after finish() and before commit.
    template <class Finalize = std::nullptr_t, class Validate = std::nullptr_t,
              class Prepare = std::nullptr_t, class AfterSubmit = std::nullptr_t>
    void append(MiniResidentChunk chunk, MiniConstRows embeddings, MiniMutableRows output,
                Finalize finalize = nullptr, Validate validate = nullptr,
                Prepare prepare = nullptr, AfterSubmit after_submit = nullptr, bool stage = false) {
        if (!session_) throw std::logic_error("Mini contiguous append needs a session");
        session_->require_append(chunk);
        const auto activation_elements = mini_resident_activation_elements(chunk.rows);
        const auto image_elements = activation_elements / kMiniResidentImages;
        if (embeddings.size() != image_elements || output.size() != image_elements ||
            workspace_.activations.size() < activation_elements) {
            throw std::invalid_argument("incomplete Mini resident forward workspace or output");
        }
        auto taps = workspace_.draft_taps;
        if (!taps.empty()) {
            const auto offset = MiniDSparkTapPlan::elements(chunk.first_position);
            const auto length = MiniDSparkTapPlan::elements(chunk.rows);
            if (offset > taps.size() || length > taps.size() - offset)
                throw std::invalid_argument("incomplete Mini resident tap workspace");
            taps = taps.subspan(offset, length);
        }
        session_->append(backend_, chunk, workspace_.convolution_checkpoint, workspace_.kv_checkpoint, [&] {
            if constexpr (!std::is_same_v<Prepare, std::nullptr_t>) prepare();
            ChunkForward<false> forward(*this, chunk, image_elements, taps);
            forward.run(embeddings, output);
            if constexpr (!std::is_same_v<Finalize, std::nullptr_t>) finalize(chunk, MiniConstRows(output));
        }, [&] {
            if constexpr (!std::is_same_v<Validate, std::nullptr_t>) validate();
        }, after_submit, stage);
    }

    // Packed independent lanes reuse the exact same model stage walk and
    // row-wise projections. Only the stateful operator bindings split by lane.
    template <class Start, class Finalize, class Validate>
    void append_batch(std::span<const MiniResidentAppendGroup> groups, unsigned rows,
                      MiniConstRows embeddings, MiniMutableRows output, Start start,
                      Finalize finalize, Validate validate) {
        const auto activation_elements = mini_resident_activation_elements(rows);
        const auto image_elements = activation_elements / kMiniResidentImages;
        if (embeddings.size() != image_elements || output.size() != image_elements ||
            workspace_.activations.size() < activation_elements)
            throw std::invalid_argument("incomplete Mini resident batch workspace or output");
        unsigned remaining = rows;
        for (const auto& group : groups) {
            if (group.chunk.rows > remaining) throw std::invalid_argument("invalid Mini batch rows");
            remaining -= group.chunk.rows;
        }
        if (remaining) throw std::invalid_argument("incomplete Mini batch rows");
        MiniResidentSessionState::append_batch(groups, [&] { backend_.finish(); }, [&] {
            run_batch(groups, rows, embeddings, output, start);
            finalize(MiniConstRows(output));
        }, validate);
    }

    // The device work of one packed multi-lane forward, without the session transaction (the caller owns
    // it). With the backend in cohort lane mode (a width-class program capture), `groups` may repeat lane
    // 0 as padding: lane-stateful stages then read every lane from the device step block, not from groups.
    template <class Start>
    void run_batch(std::span<const MiniResidentAppendGroup> groups, unsigned rows, MiniConstRows embeddings,
                   MiniMutableRows output, Start start) {
        const auto activation_elements = mini_resident_activation_elements(rows);
        const auto image_elements = activation_elements / kMiniResidentImages;
        if (groups.empty() || embeddings.size() != image_elements || output.size() != image_elements ||
            workspace_.activations.size() < activation_elements)
            throw std::invalid_argument("incomplete Mini resident batch workspace or output");
        start();
        ChunkForward<true> forward(*this, {0U, rows}, image_elements, {}, groups);
        forward.run(embeddings, output);
    }

private:
    template <bool Batched>
    class ChunkForward {
    public:
        ChunkForward(MiniResidentForward& owner, MiniResidentChunk chunk, std::size_t elements,
                     std::span<std::uint16_t> taps, std::span<const MiniResidentAppendGroup> groups = {})
            : backend_(owner.backend_), session_(owner.session_), chunk_(chunk), taps_(taps), groups_(groups) {
            if constexpr (Batched) visit_groups_.reserve(groups.size());
            auto arena = owner.workspace_.activations;
            const auto take = [&] {
                const auto image = arena.first(elements);
                arena = arena.subspan(elements);
                return image;
            };
            hidden_ = take();
            entry_hidden_ = take();
            for (auto& block : block_storage_) block = take();
            for (auto& message : message_bank_) message = take();
            for (auto& read : readout_) read = take();
            for (auto& input : attention_input_) input = take();
            operator_output_ = take();
            residual_ = take();
        }

        void run(MiniConstRows embeddings, MiniMutableRows output) {
            backend_.copy(embeddings, hidden_);
            MiniLayers::run(*this);
            residual_read(hidden_, "attn_res.output_query", 1);
            backend_.norm(MiniConstRows(readout_[0]), "norm.weight", output);
            capture_draft_tap(kLogicalVisits, output);
        }

        void save_core_entry() {
            backend_.copy(MiniConstRows(hidden_), entry_hidden_);
            entry_blocks_ = block_count_;
        }

        void capture_previous_pass(unsigned) {
            if (block_count_ != entry_blocks_ + 2U) throw std::logic_error("incomplete Mini previous-pass bank");
            // The incoming layer-12 residual is the newest committed block of
            // this pass. The layer-6 boundary is the shared prelude input.
            backend_.capture_message_bank(std::array<MiniConstRows, 2>{blocks_[block_count_ - 1U], hidden_}, message_bank_);
        }

        void restore_core_entry() {
            backend_.copy(MiniConstRows(entry_hidden_), hidden_);
            // Those prefix block images were never overwritten by the pass.
            block_count_ = entry_blocks_;
        }

        void run_layer(LayerVisit visit) {
            capture_draft_tap(visit.logical_layer, hidden_);
            const auto physical = visit.physical_layer;
            const bool full = is_full_attention(physical);
            const std::array<MiniConstRows, 2> bank{message_bank_[0], message_bank_[1]};
            const auto previous = visit.pass == 0U ? std::span<const MiniConstRows>{} : std::span<const MiniConstRows>(bank);
            if constexpr (Batched) {
                visit_groups_.clear();
                for (const auto& group : groups_) visit_groups_.push_back({group.chunk, group.state->visit(visit)});
            } else {
                visit_state_ = session_->visit(visit);
            }

            if (physical != 0U) {
                residual_read(hidden_, "attn_res.attn_queries." + std::to_string(physical - 1U), full ? 3U : 1U);
            }
            const MiniConstRows q_raw = physical == 0U ? hidden_ : readout_[0];
            backend_.layer_input(visit, MessagePath::query, q_raw, previous, attention_input_[0]);
            const bool separate = full || visit.pass != 0U;
            if (separate) {
                backend_.layer_input(visit, MessagePath::key, full ? readout_[1] : q_raw, previous, attention_input_[1]);
                backend_.layer_input(visit, MessagePath::value, full ? readout_[2] : q_raw, previous, attention_input_[2]);
            }
            const MiniConstRows query = attention_input_[0], key = attention_input_[separate ? 1U : 0U],
                                value = attention_input_[separate ? 2U : 0U];
            if constexpr (Batched) backend_.attention_batch(visit, visit_groups_, query, key, value, operator_output_);
            else backend_.attention(visit, chunk_, query, key, value, visit_state_, operator_output_);
            convolve(visit, MiniConvolutionStream::attention, operator_output_, residual_);
            if (starts_residual_block(visit)) {
                const auto block = block_storage_[block_count_];
                backend_.copy(MiniConstRows(hidden_), block);
                blocks_[block_count_++] = block;
            } else {
                backend_.add(MiniConstRows(residual_), MiniConstRows(hidden_), residual_, false);
            }

            residual_read(residual_, "attn_res.mlp_queries." + std::to_string(physical), 1);
            // Attention has retired its reads before these scratch images are reused.
            backend_.layer_input(visit, MessagePath::moe, MiniConstRows(readout_[0]), previous, attention_input_[0]);
            backend_.moe(visit, MiniConstRows(attention_input_[0]), operator_output_);
            convolve(visit, MiniConvolutionStream::moe, operator_output_, readout_[0]);
            backend_.add(MiniConstRows(residual_), MiniConstRows(readout_[0]), hidden_, true);
        }

    private:
        void capture_draft_tap(unsigned visit, MiniConstRows source) {
            if constexpr (requires { backend_.store_draft_tap(source, taps_); }) {
                if (taps_.empty()) return;
                const auto tap = MiniDSparkTapPlan::index(visit);
                if (tap != MiniDSparkTapPlan::count)
                    backend_.store_draft_tap(source, taps_.subspan(tap * kHiddenWidth));
            }
        }

        void residual_read(MiniConstRows current, std::string_view name, unsigned queries) {
            backend_.residual_read(std::span<const MiniConstRows>(blocks_).first(block_count_), current, name,
                                   std::span<const MiniMutableRows>(readout_).first(queries));
        }

        void convolve(LayerVisit visit, MiniConvolutionStream stream, MiniConstRows input,
                      MiniMutableRows output) {
            const auto path = static_cast<unsigned>(stream);
            if constexpr (Batched) {
                // Cohort lane mode: one launch over every lane, histories resolved on the device.
                if constexpr (requires { backend_.convolve_lanes(visit, stream, input, visit_groups_.front().state.history[path], output); }) {
                    if (backend_.cohort_lanes_active()) {
                        backend_.convolve_lanes(visit, stream, input, visit_groups_.front().state.history[path], output);
                        return;
                    }
                }
                std::size_t base = 0;
                for (const auto& group : visit_groups_) {
                    const auto elements = std::size_t{group.chunk.rows} * kHiddenWidth;
                    backend_.convolve(visit, stream, input.subspan(base, elements), group.state.history[path],
                                      output.subspan(base, elements));
                    base += elements;
                }
            } else {
                backend_.convolve(visit, stream, input, visit_state_.history[path], output);
                // A taped (verify) chunk keeps every input row of the stream for its partial commit.
                if (const auto tape = visit_state_.tape[path]; !tape.empty()) backend_.copy(input, tape);
            }
        }

        Backend& backend_;
        MiniResidentSessionState* session_;
        MiniResidentChunk chunk_;
        std::span<std::uint16_t> taps_;
        std::span<const MiniResidentAppendGroup> groups_;
        std::vector<MiniResidentAttentionGroup> visit_groups_;
        MiniResidentVisitState visit_state_;
        MiniMutableRows hidden_, entry_hidden_, operator_output_, residual_;
        std::array<MiniMutableRows, kPhysicalLayers / kResidualBlockLayers> block_storage_;
        std::array<MiniMutableRows, 2> message_bank_;
        std::array<MiniMutableRows, 3> readout_, attention_input_;
        std::array<MiniConstRows, kPhysicalLayers / kResidualBlockLayers> blocks_{};
        unsigned block_count_ = 0;
        unsigned entry_blocks_ = 0;
    };

    Backend& backend_;
    MiniResidentSessionState* session_ = nullptr;
    MiniResidentForwardWorkspace workspace_;
};

}  // namespace strix::inkling2
