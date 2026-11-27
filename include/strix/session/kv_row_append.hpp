#pragma once

#include "strix/session/prompt_windows.hpp"

#include "strix/attn/cache_rows.hpp"
#include "strix/session/kv.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace strix::session {

struct KvAppendChunk {
    unsigned first_position;
    unsigned rows;
};

template <class T>
constexpr std::size_t cache_buffer_elements(std::uint64_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
        throw std::length_error("resident buffer exceeds host address space");
    return static_cast<std::size_t>(count);
}

template <std::size_t Streams>
struct VisitCacheLayout {
    static constexpr auto stream_count = Streams;
    attn::cache_rows::Rows kv{};
    unsigned kv_rows{};
    std::array<unsigned, Streams> convolution_widths{};
    unsigned window{};
    std::uint64_t kv_offset{};
    std::array<std::uint64_t, Streams> convolution_offsets{};
    unsigned history_rows{};

    constexpr std::uint64_t kv_elements() const { return std::uint64_t{kv_rows} * kv.width; }
    constexpr std::uint64_t kv_bytes() const { return kv_elements() * sizeof(std::uint16_t) * 2U; }
    constexpr std::uint64_t convolution_elements(std::size_t stream) const {
        return std::uint64_t{history_rows} * convolution_widths.at(stream);
    }
};

// A launch can write all its rows before the first query reads the old tail.
template <std::size_t Streams>
constexpr VisitCacheLayout<Streams> make_visit_cache_layout(
    unsigned context, unsigned max_chunk, unsigned width, unsigned window,
    std::array<unsigned, Streams> history_widths, unsigned history_rows) {
    const auto need = window != 0U ? std::uint64_t{window} - 1U + max_chunk : 0U;
    // If rounding can reach or exceed context the full cache is sufficient.
    const unsigned slots = need != 0U && need < context &&
                                   need <= std::numeric_limits<unsigned>::max() / 2U + 1U
                               ? attn::cache_rows::window_ring(window, max_chunk) : 0U;
    const unsigned ring = slots < context ? slots : 0U;
    return {{width, ring}, ring == 0U ? context : ring, history_widths, window, 0, {}, history_rows};
}

// Traits declares Visit, schedule(), index(Visit), valid(Visit) and
// layout(Visit, context, max_chunk). Each scheduled visit has its own slice.
template <class Traits>
class VisitCachePlan {
public:
    using Visit = typename Traits::Visit;
    static constexpr auto visits() { return Traits::schedule(); }
    using Layout = decltype(Traits::layout(Visit{}, 0U, 0U));

    constexpr VisitCachePlan(unsigned context, unsigned max_chunk)
        : context_(context), max_chunk_(max_chunk) {
        if (context == 0U || max_chunk == 0U || max_chunk > context)
            throw std::invalid_argument("invalid session cache context or chunk size");
        for (const auto visit : visits()) {
            auto layout = Traits::layout(visit, context_, max_chunk_);
            layout.kv_offset = kv_elements_;
            kv_elements_ += layout.kv_elements();
            for (std::size_t stream = 0; stream < layout.convolution_offsets.size(); ++stream) {
                layout.convolution_offsets[stream] = convolution_elements_;
                convolution_elements_ += layout.convolution_elements(stream);
            }
            layouts_[Traits::index(visit)] = layout;
        }
    }

    constexpr bool admits(unsigned first, unsigned count) const {
        return count != 0U && count <= max_chunk_ && admits_sequence(first, count);
    }

    constexpr bool admits_sequence(unsigned first, std::size_t count) const {
        return first < context_ && count != 0U && count <= context_ - first;
    }
    constexpr unsigned max_chunk() const { return max_chunk_; }

    static constexpr bool valid(Visit visit) { return Traits::valid(visit); }

    constexpr const Layout& visit(Visit visit) const {
        if (!valid(visit)) throw std::invalid_argument("invalid session cache visit");
        return layouts_[Traits::index(visit)];
    }

    constexpr std::uint64_t kv_elements_per_plane() const { return kv_elements_; }
    constexpr std::uint64_t kv_bytes() const { return kv_elements_ * sizeof(std::uint16_t) * 2U; }
    constexpr std::uint64_t convolution_elements() const { return convolution_elements_; }
    // Elements of one input row of every convolution stream of every visit. A convolution tape for staged
    // chunks of up to R rows holds R times this many floats.
    constexpr std::uint64_t convolution_row_elements() const {
        std::uint64_t total = 0;
        for (const auto entry : visits())
            for (const auto width : layouts_[Traits::index(entry)].convolution_widths) total += width;
        return total;
    }
    constexpr std::uint64_t convolution_bytes() const { return convolution_elements_ * sizeof(float); }
    constexpr std::size_t kv_checkpoint_elements() const {
        std::uint64_t count = 0;
        for (const auto& layout : layouts_)
            if (layout.kv.ring != 0U) count += 2U * std::uint64_t{layout.kv.width};
        return cache_buffer_elements<std::uint16_t>(count);
    }

private:
    unsigned context_;
    unsigned max_chunk_;
    std::uint64_t kv_elements_ = 0;
    std::uint64_t convolution_elements_ = 0;
    std::array<Layout, Traits::schedule().size()> layouts_{};
};

struct KvRowPlanes {
    std::span<std::uint16_t> keys;
    std::span<std::uint16_t> values;
};

struct VisitKvRows {
    std::uint16_t* keys{};
    std::uint16_t* values{};
    unsigned slots{};
    bool wrapped{};
};

template <class Plan>
VisitKvRows slice_kv_visit(const Plan& plan, typename Plan::Visit visit, KvRowPlanes planes) {
    if (!plan.valid(visit) || planes.keys.size() < plan.kv_elements_per_plane() ||
        planes.values.size() < plan.kv_elements_per_plane() || planes.keys.data() == planes.values.data())
        return {};
    const auto& layout = plan.visit(visit);
    return {planes.keys.data() + layout.kv_offset, planes.values.data() + layout.kv_offset,
            layout.kv_rows, layout.kv.ring != 0U};
}

template <std::size_t Streams>
struct VisitCacheRows {
    VisitKvRows kv;
    std::array<std::span<float>, Streams> history;
    // Convolution tape of a staged append, [rows][width] per stream: the work writes the input row r of each
    // convolution stream at tape[stream][r * width, (r + 1) * width) in addition to its usual history update.
    // Empty when the append records no tape (then the work writes nothing extra).
    std::array<std::span<float>, Streams> tape{};
};

// CPU reference storage grows dense visits by the written frontier. Windowed
// visits and their recurrent histories are allocated at the admitted size.
template <class Element, std::size_t Streams>
struct GrowingVisitCache {
    std::vector<Element> keys;
    std::vector<Element> values;
    std::array<std::vector<float>, Streams> history;

    void prepare(const VisitCacheLayout<Streams>& layout, unsigned position) {
        const auto rows = layout.kv.ring == 0U ? position + 1U : layout.kv.ring;
        const auto elements = cache_buffer_elements<Element>(std::uint64_t{rows} * layout.kv.width);
        keys.resize(elements);
        values.resize(elements);
        for (std::size_t stream = 0; stream < Streams; ++stream)
            history[stream].resize(cache_buffer_elements<float>(layout.convolution_elements(stream)));
    }
};

// The append ring holds window-1 prior rows alongside the new chunk. Preserve
// the additional oldest row to recover the entire old last-position window.
template <std::size_t Streams>
constexpr bool overwrites_committed_row(const VisitCacheLayout<Streams>& layout, KvAppendChunk chunk) {
    return layout.kv.ring != 0U && chunk.rows > layout.kv.ring - std::min(layout.window, chunk.first_position);
}

// True when two spans share at least one byte (empty spans share none). Used for the scratch-aliasing
// contracts of staged appends; the spans are compared as address ranges.
template <class A, class B>
inline bool spans_overlap(std::span<A> a, std::span<B> b) noexcept {
    if (a.empty() || b.empty()) return false;
    const auto a0 = reinterpret_cast<std::uintptr_t>(a.data());
    const auto b0 = reinterpret_cast<std::uintptr_t>(b.data());
    return a0 < b0 + b.size_bytes() && b0 < a0 + a.size_bytes();
}

// One externally serialized physical lane, borrowing disjoint caller-owned
// state and scratch. Backend copy/finish enqueue and drain one stream. Publish
// the frontier only after work, completion and reply validation succeed.
template <class Plan>
class KvStore::RowAppendState {
public:
    using Visit = typename Plan::Visit;
    using Layout = typename Plan::Layout;
    using VisitRows = VisitCacheRows<Layout::stream_count>;
    struct AppendGroup {
        RowAppendState* state;
        KvAppendChunk chunk;
    };

    // Every lane of a batch is present, distinct and admits its chunk; writes nothing. A cohort that runs as several
    // append_batch passes calls it over the whole cohort first, so a bad lane in a later pass is refused before any
    // pass writes.
    static void require_batch(std::span<const AppendGroup> groups) {
        if (groups.empty()) throw std::invalid_argument("empty resident append batch");
        for (std::size_t i = 0; i < groups.size(); ++i) {
            const auto& group = groups[i];
            if (!group.state || std::ranges::any_of(groups.first(i), [&](const auto& prior) {
                    return prior.state == group.state;
                }))
                throw std::invalid_argument("invalid or duplicate resident append lane");
            group.state->require_append(group.chunk);
        }
    }

    // Each contiguous group belongs to a different lane. Work enqueues all
    // groups on one stream; publish all frontiers after one completion and
    // reply validation. A failed batch retires all its lanes, so it needs no
    // history or ring rollback images.
    template <class Finish, class Work, class Validate>
    static void append_batch(std::span<const AppendGroup> groups, Finish finish, Work work, Validate validate) {
        require_batch(groups);
        bool completed = false;
        try {
            work();
            finish();
            completed = true;
            validate();
        } catch (...) {
            for (const auto& group : groups) group.state->poison();
            if (!completed) {
                try { finish(); } catch (...) {}
            }
            throw;
        }
        for (const auto& group : groups) group.state->committed_ += group.chunk.rows;
    }

    // One lane of a staged batch: its chunk and its OWN checkpoint workspaces (disjoint from every other lane's
    // workspaces, histories and tapes), which hold the lane's rollback images until its commit_staged.
    struct StagedGroup {
        RowAppendState* state;
        KvAppendChunk chunk;
        std::span<float> history_checkpoint;
        std::span<std::uint16_t> kv_checkpoint;
    };

    // Staged batch (the mixed verify pass of a cohort): ONE work over several lanes' windows, each lane left staged
    // exactly as its own append(..., stage = true) would leave it, so each lane then keeps its own prefix with
    // commit_staged (a taped lane by copies; otherwise restore and a replay of that lane alone). Every lane's
    // checkpoints are taken first, then the tape choice is made per lane by append's rule (the chunk fits the lane's
    // bound tape and saved no ring row); the work sees each lane's tape through visit(). A failure before the work
    // wrote only scratch (nothing published, no lane changed); a failed work, finish or validation retires every lane
    // of the batch, as append_batch does. No program capture: the work runs eagerly.
    template <class Backend, class Work, class Validate>
    static void stage_batch(Backend& backend, std::span<const StagedGroup> groups, Work work, Validate validate) {
        if (groups.empty()) throw std::invalid_argument("empty resident staged batch");
        for (std::size_t i = 0; i < groups.size(); ++i) {
            const auto& group = groups[i];
            if (!group.state || std::ranges::any_of(groups.first(i), [&](const auto& prior) {
                    return prior.state == group.state;
                }))
                throw std::invalid_argument("invalid or duplicate resident staged lane");
            group.state->require_append(group.chunk);
            if (group.history_checkpoint.size() < group.state->history_.size())
                throw std::invalid_argument("incomplete resident history checkpoint workspace");
        }
        for (const auto& group : groups) {
            for (const auto& other : groups) {
                // A lane's rollback images may not alias any lane's resident history or tape, nor another lane's images.
                const auto& o = *other.state;
                if (spans_overlap(group.history_checkpoint, o.history_) || spans_overlap(group.kv_checkpoint, o.history_) ||
                    spans_overlap(group.history_checkpoint, o.tape_) || spans_overlap(group.kv_checkpoint, o.tape_))
                    throw std::invalid_argument("staged batch checkpoint overlaps a lane's resident state");
                if (&other != &group &&
                    (spans_overlap(group.history_checkpoint, other.history_checkpoint) ||
                     spans_overlap(group.kv_checkpoint, other.kv_checkpoint) ||
                     spans_overlap(group.history_checkpoint, other.kv_checkpoint)))
                    throw std::invalid_argument("staged batch lanes share a checkpoint workspace");
            }
        }
        struct Taken {
            std::span<const float> history;
            KvBackups kv;
            bool tape;
        };
        std::vector<Taken> taken;
        taken.reserve(groups.size());
        try {
            for (const auto& group : groups) {
                auto& s = *group.state;
                const auto histories = group.history_checkpoint.first(s.history_.size());
                const auto backups = s.make_kv_backups(group.chunk, group.kv_checkpoint);
                backend.copy(std::span<const float>(s.history_), histories);
                for (const auto& backup : backups) {
                    if (backup.saved.empty()) break;
                    backend.copy(std::span<const std::uint16_t>(backup.target), backup.saved);
                }
                taken.push_back({histories, backups, group.chunk.rows <= s.tape_rows_ && backups.front().saved.empty()});
            }
        } catch (...) {
            // Only scratch has been written: drain; no lane changed (a failed drain retires the lanes).
            for (const auto& group : groups) group.state->drain_or_poison([&] { backend.finish(); });
            throw;
        }
        bool completed = false;
        try {
            for (std::size_t i = 0; i < groups.size(); ++i)
                groups[i].state->taping_ = taken[i].tape ? groups[i].chunk.rows : 0U;
            work();
            for (const auto& group : groups) group.state->taping_ = 0U;
            backend.finish();
            completed = true;
            validate();
        } catch (...) {
            for (const auto& group : groups) {
                group.state->taping_ = 0U;
                group.state->poison();
            }
            if (!completed) {
                try { backend.finish(); } catch (...) {}
            }
            throw;
        }
        for (std::size_t i = 0; i < groups.size(); ++i)
            groups[i].state->pending_.emplace(
                Pending{groups[i].chunk.rows, taken[i].history, taken[i].kv, false, taken[i].tape});
    }

    RowAppendState(Plan plan, KvRowPlanes kv, std::span<float> history)
        : plan_(std::move(plan)), kv_(kv), history_(history) {
        if (!slice_kv_visit(plan_, Plan::visits().front(), kv_).keys ||
            history_.size() < plan_.convolution_elements())
            throw std::invalid_argument("incomplete resident session storage");
        history_ = history_.first(static_cast<std::size_t>(plan_.convolution_elements()));
    }

    RowAppendState(const RowAppendState&) = delete;
    RowAppendState& operator=(const RowAppendState&) = delete;

    const Plan& plan() const { return plan_; }
    unsigned committed() const { return committed_; }
    bool staged() const { return pending_.has_value(); }
    bool poisoned() const { return poisoned_; }
    void poison() noexcept { poisoned_ = true; }

    // Mutable slices are for the transaction's work callback.
    VisitRows visit(Visit visit) const {
        const auto& layout = plan_.visit(visit);
        VisitRows rows{slice_kv_visit(plan_, visit, kv_), {}, {}};
        for (std::size_t stream = 0; stream < rows.history.size(); ++stream) {
            rows.history[stream] = history_.subspan(static_cast<std::size_t>(layout.convolution_offsets[stream]),
                                                   static_cast<std::size_t>(layout.convolution_elements(stream)));
            if (taping_ != 0U)
                rows.tape[stream] = tape_.subspan(static_cast<std::size_t>(tape_offset(visit, stream)),
                                                  std::size_t{taping_} * layout.convolution_widths[stream]);
        }
        return rows;
    }

    std::uint64_t convolution_row_elements() const { return plan_.convolution_row_elements(); }

    // Convolution tape (optional, caller-owned scratch). Contract of a convolution history: the last
    // history_rows input rows of the stream, oldest first. A staged append of up to `rows` rows then records
    // its input rows (VisitCacheRows::tape) and a partial commit rebuilds every history by copies:
    //   history <- last history_rows rows of (history before the append ++ tape rows [0, keep)),
    // which is the history a serial append of the kept rows leaves when the kept rows' inputs are those of the
    // staged rows (row-invariant forward). It is used only when the staged chunk saved no ring-KV row; otherwise
    // the commit keeps the replay. An empty span unbinds. The tape must not share memory with the resident
    // history (rejected here); a staged append whose checkpoint workspaces share memory with it keeps the replay.
    void bind_convolution_tape(std::span<float> tape, unsigned rows) {
        if (staged()) throw std::logic_error("bind the convolution tape outside a staged append");
        if (tape.empty() || rows == 0U) {
            tape_ = {};
            tape_rows_ = 0U;
            return;
        }
        if (tape.size() / rows < convolution_row_elements())
            throw std::invalid_argument("incomplete resident convolution tape");
        if (spans_overlap(tape, history_))
            throw std::invalid_argument("convolution tape overlaps the resident history");
        tape_ = tape;
        tape_rows_ = rows;
    }
    unsigned convolution_tape_rows() const { return tape_rows_; }
    const float* convolution_tape_data() const noexcept { return tape_.data(); }

    void require_append(KvAppendChunk chunk) const {
        if (poisoned_ || staged() || chunk.first_position != committed_ || !plan_.admits(chunk.first_position, chunk.rows))
            throw std::invalid_argument("invalid resident append frontier");
    }

    // If a later chunk fails after earlier publication, prevent a retry from
    // appending the prefix twice. tail_rows: see prompt_windows::append_chunks (the final chunk absorbs a remainder of
    // up to tail_rows); chunk_limit + tail_rows must fit the plan's chunk capacity.
    template <class Token, class Append>
    unsigned append_sequence(std::span<const Token> tokens, Append append, unsigned chunk_limit = 0U,
                             unsigned tail_rows = 0U) {
        const auto max_chunk = chunk_limit == 0U ? plan_.max_chunk() : chunk_limit;
        if (max_chunk > plan_.max_chunk() || tail_rows > plan_.max_chunk() - max_chunk)
            throw std::invalid_argument("resident append chunk exceeds the plan");
        const auto first = committed_;
        try {
            unsigned result{};
            prompt_windows::append_chunks(tokens, max_chunk, [&](auto chunk, bool last) {
                result = append(chunk, last);
            }, tail_rows);
            return result;
        } catch (...) {
            if (committed_ != first) poison();
            throw;
        }
    }

    template <class Finish>
    void drain_or_poison(Finish finish) noexcept {
        try {
            finish();
        } catch (...) {
            poison();
        }
    }

    template <class Backend, class Work, class Validate, class AfterSubmit = std::nullptr_t>
    void append(Backend& backend, KvAppendChunk chunk, std::span<float> history_checkpoint,
                std::span<std::uint16_t> kv_checkpoint, Work work, Validate validate, AfterSubmit after_submit = nullptr,
                bool stage = false) {
        require_append(chunk);
        if (history_checkpoint.size() < history_.size())
            throw std::invalid_argument("incomplete resident history checkpoint workspace");
        const auto finish = [&] { backend.finish(); };
        const auto abandon_capture = [&] {
            if constexpr (requires { backend.abandon_capture(); }) return backend.abandon_capture();
            else return false;
        };
        const auto complete = [&] {
            if constexpr (!std::is_same_v<AfterSubmit, std::nullptr_t>) after_submit();
            finish();
            validate();
        };
        if constexpr (requires { backend.try_step_graph(chunk, work); }) {
            // A discarded capture never wrote state. Once graph submission is
            // attempted, failure retires the session just like eager completion.
            // A staged (speculative) append keeps its checkpoints: eager only.
            try {
                if (!stage && backend.try_step_graph(chunk, work)) {
                    complete();
                    committed_ += chunk.rows;
                    return;
                }
            } catch (...) {
                abandon_capture();
                poison();
                throw;
            }
        }

        const auto histories = history_checkpoint.first(history_.size());
        const auto backups = make_kv_backups(chunk, kv_checkpoint);
        try {
            backend.copy(std::span<const float>(history_), histories);
            for (const auto& backup : backups) {
                if (backup.saved.empty()) break;
                backend.copy(std::span<const std::uint16_t>(backup.target), backup.saved);
            }
        } catch (...) {
            // Only scratch has been written. Drain the borrowed allocations.
            abandon_capture();
            drain_or_poison(finish);
            throw;
        }
        // A staged chunk records its convolution inputs when a tape covers it, no ring row needed saving and the
        // tape is disjoint from both checkpoint workspaces (recording would otherwise overwrite the base the
        // partial commit rebuilds from); any other staged chunk keeps the replay.
        const bool tape = stage && chunk.rows <= tape_rows_ && backups.front().saved.empty() &&
                          !spans_overlap(tape_, histories) && !spans_overlap(tape_, kv_checkpoint);
        try {
            taping_ = tape ? chunk.rows : 0U;
            // A taped staged window may replay a captured window program: the checkpoints above stay eager (their
            // addresses depend on the position), the work is position-independent once the backend reads the window's
            // positions and ids from its device descriptor. `tape` is passed so a program captured with its tape
            // copies replays only for a window that records them too. Declined: the work runs eagerly.
            bool graphed = false;
            if constexpr (requires { backend.try_window_graph(chunk, work, tape); }) {
                if (tape) graphed = backend.try_window_graph(chunk, work, tape);
            }
            if (!graphed) work();
            taping_ = 0U;
        } catch (...) {
            taping_ = 0U;
            if (abandon_capture()) {
                // Neither the checkpoints nor the mutations were submitted.
                drain_or_poison(finish);
                throw;
            }
            try {
                finish();
                restore(backend, histories, backups);
                finish();
            } catch (...) {
                abandon_capture();
                poison();
                drain_or_poison(finish);
            }
            throw;
        }
        try {
            if constexpr (requires { backend.submit_capture(); }) backend.submit_capture();
            complete();
        } catch (...) {
            abandon_capture();
            poison();
            throw;
        }
        if (stage) pending_.emplace(Pending{chunk.rows, histories, backups, false, tape});
        else committed_ += chunk.rows;
    }

    // Full acceptance publishes the existing append. Partial acceptance of a
    // taped chunk rebuilds the convolution histories from the checkpoint and the
    // tape (copies only, no forward). Otherwise it restores the base convolution
    // and endangered ring rows, then replays the requested prefix. Dense KV bytes
    // beyond the frontier are invisible. A zero prefix discards the entire staged
    // append without replay.
    template <class Backend, class Replay>
    void commit_staged(Backend& backend, unsigned keep, Replay replay) {
        if (poisoned_ || !pending_ || pending_->external || keep > pending_->rows)
            throw std::invalid_argument("invalid resident staged prefix");
        if (keep == pending_->rows) {
            committed_ += keep;
            pending_.reset();
            return;
        }
        const auto first = committed_;
        if (pending_->taped) {
            try {
                advance_histories(backend, pending_->history, keep);
                backend.finish();
                pending_.reset();
                committed_ = first + keep;
            } catch (...) {
                poison();
                drain_or_poison([&] { backend.finish(); });
                throw;
            }
            return;
        }
        try {
            restore(backend, pending_->history, pending_->kv);
            backend.finish();
            pending_.reset();
            if (keep != 0U) replay(keep);
            if (committed_ != first + keep || staged())
                throw std::logic_error("resident replay did not publish the requested prefix");
        } catch (...) {
            poison();
            drain_or_poison([&] { backend.finish(); });
            throw;
        }
    }

    // Pending window: the work writes caller-owned pending storage only (it never writes the resident
    // histories or KV rows), so no checkpoint is taken and a rejected suffix needs no restore and no
    // replay. The window stays staged until commit_pending. A failed work or validation drains the
    // backend and leaves the frontier and the resident state as they were (nothing was published).
    // A backend with a step graph receives the window through try_step_graph, exactly like append: its
    // lowering learns the window width there (a width-R graph), and the work runs eagerly only when the
    // backend declines the graph. No checkpoint is needed either way, so a pending window may be graphed.
    template <class Backend, class Work, class Validate>
    void append_pending(Backend& backend, KvAppendChunk chunk, Work work, Validate validate) {
        require_append(chunk);
        try {
            bool graphed = false;
            if constexpr (requires { backend.try_step_graph(chunk, work); }) {
                try {
                    graphed = backend.try_step_graph(chunk, work);
                } catch (...) {
                    if constexpr (requires { backend.abandon_capture(); }) (void)backend.abandon_capture();
                    throw;
                }
            }
            if (!graphed) work();
            backend.finish();
            validate();
        } catch (...) {
            drain_or_poison([&] { backend.finish(); });
            throw;
        }
        pending_.emplace(Pending{chunk.rows, {}, {}, true, false});
    }

    // Publish the kept prefix of a pending window: publish(keep) copies rows [0, keep) of the window
    // (KV rows and the convolution history after row keep - 1) into the resident state, then the
    // frontier moves by keep. keep = 0 discards the window without a copy. A failed publication may
    // have written part of the prefix, so it poisons the session.
    template <class Publish>
    void commit_pending(unsigned keep, Publish publish) {
        if (poisoned_ || !pending_ || !pending_->external || keep > pending_->rows)
            throw std::invalid_argument("invalid resident pending prefix");
        pending_.reset();
        if (keep == 0U) return;
        try {
            publish(keep);
        } catch (...) {
            poison();
            throw;
        }
        committed_ += keep;
    }

private:
    struct KvBackup {
        std::span<std::uint16_t> target;
        std::span<std::uint16_t> saved;
    };
    using KvBackups = std::array<KvBackup, 2U * Plan::visits().size()>;
    struct Pending {
        unsigned rows;
        std::span<const float> history;
        KvBackups kv;
        bool external;  // pending window: nothing to restore, commit by publication
        bool taped;
    };

    // Element offset of (visit, stream) in the tape: streams of visits in schedule order, tape_rows_ rows each.
    std::uint64_t tape_offset(Visit target, std::size_t target_stream) const {
        std::uint64_t offset = 0;
        const auto* wanted = &plan_.visit(target);  // Visit needs no operator==: layouts are unique per visit
        for (const auto visit : Plan::visits()) {
            const auto& layout = plan_.visit(visit);
            const auto& widths = layout.convolution_widths;
            for (std::size_t stream = 0; stream < widths.size(); ++stream) {
                if (&layout == wanted && stream == target_stream) return offset;
                offset += std::uint64_t{tape_rows_} * widths[stream];
            }
        }
        throw std::invalid_argument("convolution tape visit is not scheduled");
    }

    // history <- last h rows of (base ++ tape[0, keep)) for every stream; base is the pre-append checkpoint.
    template <class Backend>
    void advance_histories(Backend& backend, std::span<const float> base, unsigned keep) {
        for (const auto visit : Plan::visits()) {
            const auto& layout = plan_.visit(visit);
            const std::size_t h = layout.history_rows;
            for (std::size_t stream = 0; stream < layout.convolution_widths.size(); ++stream) {
                const std::size_t width = layout.convolution_widths[stream];
                if (h == 0U || width == 0U) continue;
                const auto at = static_cast<std::size_t>(layout.convolution_offsets[stream]);
                const auto history = history_.subspan(at, h * width);
                const auto tape = std::span<const float>(tape_).subspan(
                    static_cast<std::size_t>(tape_offset(visit, stream)), std::size_t{keep} * width);
                if (keep >= h) {
                    backend.copy(tape.subspan((keep - h) * width), history);
                } else {
                    // Older history rows shift down by `keep`; the kept rows fill the newest slots.
                    if (keep != 0U) {
                        backend.copy(base.subspan(at + keep * width, (h - keep) * width),
                                     history.first((h - keep) * width));
                        backend.copy(tape, history.subspan((h - keep) * width));
                    } else {
                        backend.copy(base.subspan(at, h * width), history);
                    }
                }
            }
        }
    }

    template <class Backend>
    void restore(Backend& backend, std::span<const float> histories, const KvBackups& backups) {
        backend.copy(histories, history_);
        for (const auto& backup : backups) {
            if (backup.saved.empty()) break;
            backend.copy(std::span<const std::uint16_t>(backup.saved), backup.target);
        }
    }

    KvBackups make_kv_backups(KvAppendChunk chunk, std::span<std::uint16_t> scratch) const {
        KvBackups backups{};
        std::size_t count = 0;
        for (const auto visit : Plan::visits()) {
            const auto& layout = plan_.visit(visit);
            if (!overwrites_committed_row(layout, chunk)) continue;
            const auto width = layout.kv.width;
            if (scratch.size() / width < 2U)
                throw std::invalid_argument("incomplete resident KV checkpoint workspace");
            const auto position = chunk.first_position - std::min(layout.window, chunk.first_position);
            const auto offset = static_cast<std::size_t>(layout.kv_offset + attn::cache_rows::row_offset(layout.kv, position));
            for (auto plane : {kv_.keys, kv_.values}) {
                backups[count++] = {plane.subspan(offset, width), scratch.first(width)};
                scratch = scratch.subspan(width);
            }
        }
        return backups;
    }

    const Plan plan_;
    KvRowPlanes kv_;
    std::span<float> history_;
    std::span<float> tape_;
    std::optional<Pending> pending_;
    unsigned committed_ = 0;
    unsigned tape_rows_ = 0;
    unsigned taping_ = 0;  // rows of the staged chunk whose work records the tape, else 0
    bool poisoned_ = false;
};

// Storage binding for a resident lane. Buffer<T> allocates a typed span. Reset
// clears recurrent histories and renews the state frontier; append only reads
// KV rows visible from that frontier, so the old KV planes need no clear.
template <class Plan, template <class> class Buffer>
class ResidentRowSlot {
public:
    explicit ResidentRowSlot(const Plan& plan)
        : plan_(plan), keys_(cache_buffer_elements<std::uint16_t>(plan.kv_elements_per_plane())),
          values_(keys_.span().size()),
          histories_(cache_buffer_elements<float>(plan.convolution_elements())) {}

    template <class Clear>
    void reset(Clear clear) {
        state_.reset();
        clear(histories_.span());
        state_.emplace(plan_, planes(), histories_.span());
        if (tape_rows_ != 0U) state_->bind_convolution_tape(tape_, tape_rows_);
    }

    // The slot keeps the convolution tape across resets (RowAppendState::bind_convolution_tape). It may be
    // bound before the first reset: the size check then uses the plan, and reset hands the tape to the state.
    void bind_convolution_tape(std::span<float> tape, unsigned rows) {
        if (state_) state_->bind_convolution_tape(tape, rows);
        else if (!tape.empty() && rows != 0U && tape.size() / rows < plan_.convolution_row_elements())
            throw std::invalid_argument("incomplete resident convolution tape");
        tape_ = tape.empty() ? std::span<float>{} : tape;
        tape_rows_ = tape.empty() ? 0U : rows;
    }

    [[nodiscard]] std::uint64_t convolution_row_elements() const { return plan_.convolution_row_elements(); }
    [[nodiscard]] KvRowPlanes planes() { return {keys_.span(), values_.span()}; }
    [[nodiscard]] std::span<float> history() { return histories_.span(); }
    [[nodiscard]] KvStore::RowAppendState<Plan>& state() {
        if (!state_) throw std::logic_error("resident slot has no active state");
        return *state_;
    }

private:
    Plan plan_;
    Buffer<std::uint16_t> keys_, values_;
    Buffer<float> histories_;
    std::optional<KvStore::RowAppendState<Plan>> state_;
    std::span<float> tape_;
    unsigned tape_rows_ = 0;
};

}  // namespace strix::session
