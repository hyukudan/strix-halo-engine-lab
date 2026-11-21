#pragma once

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

namespace strix::inkling2 {

using MiniConstRows = std::span<const float>;
using MiniMutableRows = std::span<float>;

struct MiniResidentChunk {
    unsigned first_position;
    unsigned rows;
};

enum class MiniConvolutionStream : unsigned { key, value, attention, moe };

struct MiniResidentVisitState {
    MiniVisitKvCache kv;
    std::array<std::span<float>, 4> history;
};

struct MiniResidentForwardWorkspace {
    std::span<float> activations;
    std::span<float> convolution_checkpoint;
    std::span<std::uint16_t> kv_checkpoint;
};

inline constexpr unsigned kMiniResidentImages = 16;

inline constexpr std::size_t mini_resident_activation_elements(unsigned rows) {
    constexpr auto per_row = std::size_t{kMiniResidentImages} * kHiddenWidth;
    if (rows == 0 || rows > kMaxContext || per_row > std::numeric_limits<std::size_t>::max() / sizeof(float) / rows) {
        throw std::invalid_argument("invalid Mini resident activation workspace geometry");
    }
    return rows * per_row;
}

// A wrapped launch may overwrite at most one row from the window at the old
// frontier. The layout reserves window-1 prior rows alongside the new chunk.
// Keeping the additional oldest row in the rollback scratch also preserves
// the entire old last-position window, not just the next query's prefix.
inline constexpr bool mini_append_overwrites_committed_row(const MiniVisitStatePlan& visit, MiniResidentChunk chunk) {
    return visit.kv.ring != 0U && chunk.rows > visit.kv.ring - std::min(visit.window, chunk.first_position);
}

inline constexpr std::size_t mini_resident_max_kv_checkpoint_elements() {
    std::size_t result = 0;
    for (const auto visit : mini_schedule()) {
        if (uses_sliding_attention(visit)) result += 2U * kKvWidth;
    }
    return result;
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

template <class Backend> class MiniResidentForward;

// A single externally serialized lane. The caller owns disjoint device
// allocations, zeros the histories at session creation and keeps all memory
// alive through the backend's finish(). No unread KV slot requires clearing.
class MiniResidentSessionState {
public:
    MiniResidentSessionState(MiniSessionStatePlan plan, MiniSessionKvPlanes kv, std::span<float> convolution)
        : plan_(plan), kv_(kv), convolution_(convolution) {
        if (!mini_visit_kv_cache(plan_, mini_schedule().front(), kv_).keys ||
            convolution_.size() < plan_.convolution_elements()) {
            throw std::invalid_argument("incomplete Mini resident session storage");
        }
        convolution_ = convolution_.first(static_cast<std::size_t>(plan_.convolution_elements()));
    }

    MiniResidentSessionState(const MiniResidentSessionState&) = delete;
    MiniResidentSessionState& operator=(const MiniResidentSessionState&) = delete;

    const MiniSessionStatePlan& plan() const { return plan_; }
    unsigned committed() const { return committed_; }
    bool poisoned() const { return poisoned_; }

private:
    template <class Backend> friend class MiniResidentForward;

    MiniResidentVisitState visit(LayerVisit visit) const {
        const auto& layout = plan_.visit(visit);
        MiniResidentVisitState state{mini_visit_kv_cache(plan_, visit, kv_), {}};
        for (unsigned stream = 0; stream < state.history.size(); ++stream) {
            state.history[stream] = convolution_.subspan(static_cast<std::size_t>(layout.convolution_offsets[stream]),
                                                         static_cast<std::size_t>(layout.convolution_elements(stream)));
        }
        return state;
    }

    const MiniSessionStatePlan plan_;
    MiniSessionKvPlanes kv_;
    std::span<float> convolution_;
    unsigned committed_ = 0;
    bool poisoned_ = false;
};

// One device chunk, including decode (rows=1). All callbacks enqueue on the
// same stream, throw on admission/enqueue failure, and retain no host span
// descriptors or names. Arithmetic lives in backend kernels. finish() waits for prior
// work and reports device errors. The returned hidden rows are usable only on
// success. Input/output, persistent state and scratch allocations are disjoint.
template <class Backend>
class MiniResidentForward {
public:
    MiniResidentForward(Backend& backend, MiniResidentSessionState& session, MiniResidentForwardWorkspace workspace)
        : backend_(backend), session_(session), workspace_(workspace) {}

    // Inputs are already embed-normalized and BF16-rounded, widened to FP32.
    // An optional finalizer may enqueue a head projection before the state
    // commits. It receives the enqueue-ordered final hidden rows. The optional
    // validator consumes the completed reply after finish() and before commit.
    template <class Finalize = std::nullptr_t, class Validate = std::nullptr_t>
    void append(MiniResidentChunk chunk, MiniConstRows embeddings, MiniMutableRows output,
                Finalize finalize = nullptr, Validate validate = nullptr) {
        if (session_.poisoned() || chunk.first_position != session_.committed() ||
            !session_.plan().admits(chunk.first_position, chunk.rows)) {
            throw std::invalid_argument("invalid Mini resident append frontier");
        }
        const auto activation_elements = mini_resident_activation_elements(chunk.rows);
        const auto image_elements = activation_elements / kMiniResidentImages;
        if (embeddings.size() != image_elements || output.size() != image_elements ||
            workspace_.activations.size() < activation_elements ||
            workspace_.convolution_checkpoint.size() < session_.convolution_.size()) {
            throw std::invalid_argument("incomplete Mini resident forward workspace or output");
        }

        const auto backups = make_kv_backups(chunk);
        const auto histories = workspace_.convolution_checkpoint.first(session_.convolution_.size());
        try {
            backend_.copy(MiniConstRows(session_.convolution_), histories);
            for (const auto& backup : backups) {
                if (backup.saved.empty()) break;
                backend_.copy(std::span<const std::uint16_t>(backup.target), backup.saved);
            }
        } catch (...) {
            // Only checkpoint scratch has been written. Drain any borrowed
            // buffers before returning or quarantine the lane if not possible.
            drain_or_poison();
            throw;
        }

        try {
            ChunkForward forward(*this, chunk, image_elements);
            forward.run(embeddings, output);
            if constexpr (!std::is_same_v<Finalize, std::nullptr_t>) finalize(chunk, MiniConstRows(output));
        } catch (...) {
            restore(backups, histories);
            throw;
        }

        try {
            backend_.finish();
            if constexpr (!std::is_same_v<Validate, std::nullptr_t>) validate();
        } catch (...) {
            // A failed completion or rejected result retires the lane
            // without publishing the append.
            session_.poisoned_ = true;
            throw;
        }
        session_.committed_ += chunk.rows;
    }

private:
    struct KvBackup {
        std::span<std::uint16_t> target;
        std::span<std::uint16_t> saved;
    };
    using KvBackups = std::array<KvBackup, 2U * kLogicalVisits>;

    KvBackups make_kv_backups(MiniResidentChunk chunk) const {
        KvBackups backups{};
        std::size_t count = 0;
        for (const auto visit : mini_schedule()) {
            const auto& layout = session_.plan().visit(visit);
            if (!mini_append_overwrites_committed_row(layout, chunk)) continue;
            if (workspace_.kv_checkpoint.size() / kKvWidth < count + 2U) {
                throw std::invalid_argument("incomplete Mini resident KV checkpoint workspace");
            }
            const auto position = chunk.first_position - std::min(layout.window, chunk.first_position);
            const auto start = layout.kv_offset + attn::cache_rows::row_offset(layout.kv, position);
            const auto offset = static_cast<std::size_t>(start);
            for (auto plane : {session_.kv_.keys, session_.kv_.values}) {
                backups[count] = {plane.subspan(offset, kKvWidth), workspace_.kv_checkpoint.subspan(count * kKvWidth, kKvWidth)};
                ++count;
            }
        }
        return backups;
    }

    void drain_or_poison() noexcept {
        try {
            backend_.finish();
        } catch (...) {
            session_.poisoned_ = true;
        }
    }

    void restore(const KvBackups& backups, MiniConstRows histories) noexcept {
        try {
            backend_.finish();
            backend_.copy(histories, session_.convolution_);
            for (const auto& backup : backups) {
                if (backup.saved.empty()) break;
                backend_.copy(std::span<const std::uint16_t>(backup.saved), backup.target);
            }
            backend_.finish();
        } catch (...) {
            session_.poisoned_ = true;
            drain_or_poison();
        }
    }

    class ChunkForward {
    public:
        ChunkForward(MiniResidentForward& owner, MiniResidentChunk chunk, std::size_t elements)
            : backend_(owner.backend_), session_(owner.session_), chunk_(chunk) {
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
            const auto physical = visit.physical_layer;
            const bool full = is_full_attention(physical);
            const std::array<MiniConstRows, 2> bank{message_bank_[0], message_bank_[1]};
            const auto previous = visit.pass == 0U ? std::span<const MiniConstRows>{} : std::span<const MiniConstRows>(bank);
            auto state = session_.visit(visit);

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
            backend_.attention(visit, chunk_, MiniConstRows(attention_input_[0]), MiniConstRows(attention_input_[separate ? 1U : 0U]),
                               MiniConstRows(attention_input_[separate ? 2U : 0U]), state, operator_output_);
            convolve(visit, MiniConvolutionStream::attention, operator_output_, state, residual_);
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
            convolve(visit, MiniConvolutionStream::moe, operator_output_, state, readout_[0]);
            backend_.add(MiniConstRows(residual_), MiniConstRows(readout_[0]), hidden_, true);
        }

    private:
        void residual_read(MiniConstRows current, std::string_view name, unsigned queries) {
            backend_.residual_read(std::span<const MiniConstRows>(blocks_).first(block_count_), current, name,
                                   std::span<const MiniMutableRows>(readout_).first(queries));
        }

        void convolve(LayerVisit visit, MiniConvolutionStream stream, MiniConstRows input,
                      const MiniResidentVisitState& state, MiniMutableRows output) {
            backend_.convolve(visit, stream, input, state.history[static_cast<unsigned>(stream)], output);
        }

        Backend& backend_;
        MiniResidentSessionState& session_;
        MiniResidentChunk chunk_;
        MiniMutableRows hidden_, entry_hidden_, operator_output_, residual_;
        std::array<MiniMutableRows, kPhysicalLayers / kResidualBlockLayers> block_storage_;
        std::array<MiniMutableRows, 2> message_bank_;
        std::array<MiniMutableRows, 3> readout_, attention_input_;
        std::array<MiniConstRows, kPhysicalLayers / kResidualBlockLayers> blocks_{};
        unsigned block_count_ = 0;
        unsigned entry_blocks_ = 0;
    };

    Backend& backend_;
    MiniResidentSessionState& session_;
    MiniResidentForwardWorkspace workspace_;
};

}  // namespace strix::inkling2
