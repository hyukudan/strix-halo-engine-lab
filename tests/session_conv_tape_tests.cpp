// Convolution tape of RowAppendState: a partial commit of a staged chunk rebuilds the convolution histories by
// copies and leaves exactly the state that restoring and replaying the kept rows leaves (the incumbent commit).
#include "strix/session/kv_row_append.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

namespace session = strix::session;

void check(bool ok, const char* what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        std::exit(1);
    }
}

template <class T>
class VecBuffer {
public:
    explicit VecBuffer(std::size_t count) : values_(count) {}
    std::span<T> span() { return values_; }

private:
    std::vector<T> values_;
};

constexpr unsigned kContext = 256;
constexpr unsigned kHistoryRows = 3;  // a 4-tap short convolution keeps 3 input rows

// Visit 0: full cache, two streams (widths 3 and 2). Visit 1: sliding window 4 (ring), one stream of width 5.
template <unsigned MaxChunk>
struct Traits {
    using Visit = unsigned;
    static constexpr auto schedule() { return std::array{0U, 1U}; }
    static constexpr unsigned index(Visit visit) { return visit; }
    static constexpr bool valid(Visit visit) { return visit < 2U; }
    static constexpr session::VisitCacheLayout<2> layout(Visit visit, unsigned context, unsigned max_chunk) {
        return visit == 0U ? session::make_visit_cache_layout(context, max_chunk, 1U, 0U, std::array{3U, 2U}, kHistoryRows)
                           : session::make_visit_cache_layout(context, max_chunk, 1U, 4U, std::array{5U, 0U}, kHistoryRows);
    }
};

// Row-invariant inputs: a function of (position, visit, stream, element) only.
float input(unsigned position, unsigned visit, std::size_t stream, std::size_t element) {
    return static_cast<float>(position) * 0.5F + static_cast<float>(visit * 100U + stream * 10U + element) * 0.001F + 1.0F;
}

template <unsigned MaxChunk>
struct Lane {
    using Plan = session::VisitCachePlan<Traits<MaxChunk>>;
    static constexpr Plan kPlan{kContext, MaxChunk};
    std::vector<std::uint16_t> keys = std::vector<std::uint16_t>(kPlan.kv_elements_per_plane());
    std::vector<std::uint16_t> values = std::vector<std::uint16_t>(kPlan.kv_elements_per_plane());
    std::vector<float> history = std::vector<float>(kPlan.convolution_elements());
    std::vector<float> history_backup = std::vector<float>(kPlan.convolution_elements());
    std::vector<std::uint16_t> kv_backup = std::vector<std::uint16_t>(kPlan.kv_checkpoint_elements() + 1U);
    std::vector<float> tape;
    session::KvStore::RowAppendState<Plan> state{kPlan, {keys, values}, history};
    unsigned replays = 0;
    // Window program gate (RowAppendState::append offers a taped staged window to try_window_graph). 1: the backend
    // runs the window (a program replay, here the same work); otherwise it declines and the work runs eagerly.
    int window_mode = 0;
    unsigned window_offers = 0, window_runs = 0;

    template <class T>
    void copy(std::span<const T> from, std::span<T> to) {
        check(from.size() == to.size(), "copy extents");
        std::copy(from.begin(), from.end(), to.begin());
    }
    void finish() {}

    template <class Work>
    bool try_window_graph(session::KvAppendChunk, Work& work, bool taped) {
        ++window_offers;
        check(taped, "only a taped staged window is offered");
        check(std::equal(history.begin(), history.end(), history_backup.begin()), "checkpoint taken before the offer");
        if (window_mode != 1) return false;
        work();
        ++window_runs;
        return true;
    }

    // Host "forward": every stream's history shifts in the chunk's input rows (oldest first); KV cells take the
    // position; the tape (when the transaction asks) receives the same input rows.
    void append(unsigned rows, bool stage) {
        const auto first = state.committed();
        state.append(*this, {first, rows}, history_backup, kv_backup, [&] { forward(first, rows); }, [] {}, nullptr,
                     stage);
    }

    // The lane's rows of one work (an append, or its part of a staged batch).
    void forward(unsigned first, unsigned rows) {
        {
            for (const auto visit : Plan::visits()) {
                const auto view = state.visit(visit);
                const auto& layout = kPlan.visit(visit);
                for (unsigned r = 0; r < rows; ++r) {
                    const auto position = first + r;
                    const auto cell = view.kv.wrapped ? position % view.kv.slots : position;
                    view.kv.keys[cell] = static_cast<std::uint16_t>(position + 1U);
                }
                for (std::size_t stream = 0; stream < 2U; ++stream) {
                    const std::size_t width = layout.convolution_widths[stream];
                    auto h = view.history[stream];
                    for (unsigned r = 0; r < rows; ++r) {
                        for (std::size_t row = 0; row + 1U < kHistoryRows; ++row)
                            for (std::size_t e = 0; e < width; ++e) h[row * width + e] = h[(row + 1U) * width + e];
                        for (std::size_t e = 0; e < width; ++e) {
                            const auto x = input(first + r, visit, stream, e);
                            h[(kHistoryRows - 1U) * width + e] = x;
                            if (!view.tape[stream].empty()) view.tape[stream][r * width + e] = x;
                        }
                    }
                }
            }
        }
    }

    void commit(unsigned keep) {
        state.commit_staged(*this, keep, [&](unsigned rows) {
            ++replays;
            append(rows, false);
        });
    }
};

template <unsigned MaxChunk>
void tape_equals_replay(unsigned prefix, unsigned rows, unsigned keep, bool expect_tape) {
    Lane<MaxChunk> taped, replayed;
    taped.tape.resize(std::size_t{MaxChunk} * taped.state.convolution_row_elements());
    taped.state.bind_convolution_tape(taped.tape, MaxChunk);
    for (auto* lane : {&taped, &replayed}) {
        for (unsigned done = 0; done < prefix;) {
            const unsigned chunk = prefix - done < MaxChunk ? prefix - done : MaxChunk;
            lane->append(chunk, false);
            done += chunk;
        }
        lane->append(rows, true);
        check(lane->state.staged(), "staged append");
        lane->commit(keep);
        check(!lane->state.staged() && !lane->state.poisoned(), "commit published");
        check(lane->state.committed() == prefix + keep, "committed frontier");
    }
    const unsigned replay = keep != 0U && keep != rows ? 1U : 0U;  // full acceptance publishes, zero discards
    check(taped.replays == (expect_tape ? 0U : replay), "tape commit avoided the replay");
    check(replayed.replays == replay, "incumbent replay ran");
    check(taped.history == replayed.history, "histories equal the replayed commit");
    // KV cells below the frontier are equal; cells beyond it are invisible.
    for (const auto visit : Lane<MaxChunk>::Plan::visits()) {
        const auto a = taped.state.visit(visit).kv, b = replayed.state.visit(visit).kv;
        for (unsigned position = 0; position < prefix + keep; ++position) {
            if (a.wrapped && position + 4U <= prefix + keep - 1U) continue;  // outside the window: may be reused
            const auto cell = a.wrapped ? position % a.slots : position;
            check(a.keys[cell] == b.keys[cell], "visible KV cells equal");
        }
    }
    // The next append continues identically.
    taped.append(2, false);
    replayed.append(2, false);
    check(taped.history == replayed.history, "next append equal");
}

// A taped staged window is offered to the backend's window program after its checkpoints; a run window publishes
// exactly what the eager window publishes (taped partial commits, no replay); other appends are never offered.
void window_program_gate() {
    for (const int mode : {0, 1}) {
        Lane<8> lane, reference;
        lane.window_mode = mode;
        for (auto* l : {&lane, &reference}) {
            l->tape.resize(std::size_t{8} * l->state.convolution_row_elements());
            l->state.bind_convolution_tape(l->tape, 8);
            l->append(3, false);
        }
        check(lane.window_offers == 0U, "an unstaged append is not offered");
        for (const unsigned keep : {2U, 5U, 0U, 1U}) {
            lane.append(5, true);
            reference.append(5, true);
            lane.commit(keep);
            reference.commit(keep);
            check(lane.history == reference.history && lane.state.committed() == reference.state.committed(),
                  "window program publishes the eager window's state");
        }
        check(lane.window_offers == 4U && lane.window_runs == (mode == 1 ? 4U : 0U), "taped windows offered");
        check(lane.replays == 0U && reference.replays == 0U, "taped commits never replay");
    }
    Lane<8> untaped;
    untaped.window_mode = 1;
    untaped.append(2, false);
    untaped.append(3, true);
    check(untaped.window_offers == 0U, "an untaped staged window is not offered");
    untaped.commit(3);
}


// RowAppendState::stage_batch: one work over several lanes' windows leaves each lane staged as its own staged append
// would, and each lane's commit_staged then publishes exactly the lone lane's state (taped lanes by copies, a lane
// whose window saved a ring row by its own replay). Refusals: a duplicate lane, shared checkpoints; a failed work
// retires every lane of the batch.
void staged_batch_equals_lone() {
    using L = Lane<8>;
    const unsigned prefixes[3] = {0U, 5U, 37U};
    const unsigned rows[3] = {3U, 1U, 5U};
    const unsigned keeps[4][3] = {{2U, 1U, 0U}, {3U, 1U, 5U}, {1U, 0U, 3U}, {0U, 1U, 1U}};
    for (const bool taped : {false, true})
        for (const auto& keep : keeps) {
            L batch[3], lone[3];
            for (unsigned i = 0; i < 3U; ++i)
                for (auto* l : {&batch[i], &lone[i]}) {
                    if (taped) {
                        l->tape.resize(std::size_t{8} * l->state.convolution_row_elements());
                        l->state.bind_convolution_tape(l->tape, 8);
                    }
                    for (unsigned done = 0; done < prefixes[i];) {
                        const unsigned chunk = prefixes[i] - done < 8U ? prefixes[i] - done : 8U;
                        l->append(chunk, false);
                        done += chunk;
                    }
                }
            using State = decltype(batch[0].state);
            std::vector<State::StagedGroup> groups;
            for (unsigned i = 0; i < 3U; ++i)
                groups.push_back({&batch[i].state, {batch[i].state.committed(), rows[i]}, batch[i].history_backup,
                                  batch[i].kv_backup});
            unsigned works = 0;
            State::stage_batch(batch[0], std::span<const State::StagedGroup>(groups), [&] {
                ++works;
                for (unsigned i = 0; i < 3U; ++i) batch[i].forward(prefixes[i], rows[i]);
            }, [] {});
            check(works == 1U, "staged batch: one work for every lane");
            for (unsigned i = 0; i < 3U; ++i) {
                check(batch[i].state.staged() && batch[i].state.committed() == prefixes[i], "staged batch: lane staged");
                lone[i].append(rows[i], true);
                batch[i].commit(keep[i]);
                lone[i].commit(keep[i]);
                check(!batch[i].state.staged() && !batch[i].state.poisoned(), "staged batch: lane published");
                check(batch[i].state.committed() == lone[i].state.committed(), "staged batch: frontier == lone");
                check(batch[i].history == lone[i].history, "staged batch: histories == lone");
                check(batch[i].replays == lone[i].replays, "staged batch: same commit form as lone (tape or replay)");
                for (const auto visit : L::Plan::visits()) {
                    const auto a = batch[i].state.visit(visit).kv, b = lone[i].state.visit(visit).kv;
                    const unsigned end = prefixes[i] + keep[i];
                    for (unsigned position = 0; position < end; ++position) {
                        if (a.wrapped && position + 4U <= end - 1U) continue;
                        const auto cell = a.wrapped ? position % a.slots : position;
                        check(a.keys[cell] == b.keys[cell], "staged batch: visible KV cells == lone");
                    }
                }
                batch[i].append(2, false);
                lone[i].append(2, false);
                check(batch[i].history == lone[i].history, "staged batch: next append == lone");
            }
            if (taped) check(batch[0].replays == 0U, "staged batch: a taped lane without ring rows never replays");
        }
    // Refusals before any write, and a failed work retiring the batch.
    {
        L a, b;
        using State = decltype(a.state);
        a.append(2, false);
        b.append(3, false);
        const auto before = a.history;
        std::vector<State::StagedGroup> dup{{&a.state, {2U, 2U}, a.history_backup, a.kv_backup},
                                           {&a.state, {2U, 2U}, b.history_backup, b.kv_backup}};
        bool refused = false;
        try { State::stage_batch(a, std::span<const State::StagedGroup>(dup), [] {}, [] {}); }
        catch (const std::invalid_argument&) { refused = true; }
        check(refused && !a.state.staged() && a.history == before, "staged batch: duplicate lane refused");
        std::vector<State::StagedGroup> shared{{&a.state, {2U, 2U}, a.history_backup, a.kv_backup},
                                              {&b.state, {3U, 2U}, a.history_backup, b.kv_backup}};
        refused = false;
        try { State::stage_batch(a, std::span<const State::StagedGroup>(shared), [] {}, [] {}); }
        catch (const std::invalid_argument&) { refused = true; }
        check(refused && !a.state.staged() && !b.state.staged(), "staged batch: shared checkpoint refused");
        std::vector<State::StagedGroup> over{{&a.state, {2U, 2U}, b.history, a.kv_backup},
                                            {&b.state, {3U, 2U}, b.history_backup, b.kv_backup}};
        refused = false;
        try { State::stage_batch(a, std::span<const State::StagedGroup>(over), [] {}, [] {}); }
        catch (const std::invalid_argument&) { refused = true; }
        check(refused, "staged batch: checkpoint over another lane's history refused");
        std::vector<State::StagedGroup> ok{{&a.state, {2U, 2U}, a.history_backup, a.kv_backup},
                                          {&b.state, {3U, 2U}, b.history_backup, b.kv_backup}};
        bool failed = false;
        try {
            State::stage_batch(a, std::span<const State::StagedGroup>(ok), [] { throw std::runtime_error("device"); },
                               [] {});
        } catch (const std::runtime_error&) { failed = true; }
        check(failed && a.state.poisoned() && b.state.poisoned(), "staged batch: a failed work retires every lane");
    }
}
}  // namespace

int main() {
    // Verify windows: R = 1..8, every kept prefix, with and without committed history before the window.
    for (unsigned prefix : {0U, 1U, 2U, 5U, 37U})
        for (unsigned rows = 1; rows <= 8U; ++rows)
            for (unsigned keep = 0; keep <= rows; ++keep) tape_equals_replay<8>(prefix, rows, keep, true);
    // A chunk that saves ring rows (13 rows overwrite the window ring of 16 slots) keeps the replay.
    tape_equals_replay<13>(20, 13, 5, false);
    tape_equals_replay<13>(20, 12, 5, true);
    // A tape shorter than the chunk keeps the replay.
    {
        Lane<8> lane;
        lane.tape.resize(std::size_t{4} * lane.state.convolution_row_elements());
        lane.state.bind_convolution_tape(lane.tape, 4);
        lane.append(6, true);
        lane.commit(3);
        check(lane.replays == 1U, "short tape falls back to replay");
    }
    // Review regression: a tape that shares memory with the history checkpoint keeps the replay (recording would
    // overwrite the base the partial commit rebuilds from) and leaves the replayed commit's state.
    for (unsigned rows = 1; rows <= kHistoryRows; ++rows)
        for (unsigned keep = 0; keep <= rows; ++keep) {
            Lane<8> aliased, replayed;
            aliased.state.bind_convolution_tape(aliased.history_backup, kHistoryRows);
            for (auto* lane : {&aliased, &replayed}) {
                lane->append(5, false);
                lane->append(rows, true);
                lane->commit(keep);
                check(lane->state.committed() == 5U + keep, "aliased tape: committed frontier");
            }
            check(aliased.replays == (keep != 0U && keep != rows ? 1U : 0U), "aliased tape keeps the replay");
            check(aliased.history == replayed.history, "aliased tape: histories equal the replayed commit");
        }
    // A tape over the resident history is refused at binding.
    {
        Lane<8> lane;
        bool refused = false;
        try { lane.state.bind_convolution_tape(lane.history, kHistoryRows); } catch (const std::invalid_argument&) { refused = true; }
        check(refused, "tape over the resident history refused");
    }
    // An undersized tape is refused.
    {
        Lane<8> lane;
        std::vector<float> small(lane.state.convolution_row_elements() * 2U - 1U);
        bool refused = false;
        try { lane.state.bind_convolution_tape(small, 2); } catch (const std::invalid_argument&) { refused = true; }
        check(refused, "undersized tape refused");
    }
    // A resident slot accepts its tape before the first reset (decoders bind at construction, then reset):
    // the size comes from the plan, and every reset hands the tape to the new state.
    {
        using Plan = Lane<8>::Plan;
        session::ResidentRowSlot<Plan, VecBuffer> slot(Lane<8>::kPlan);
        check(slot.convolution_row_elements() == Lane<8>::kPlan.convolution_row_elements(), "slot row elements");
        std::vector<float> small(slot.convolution_row_elements() * 4U - 1U);
        bool refused = false;
        try { slot.bind_convolution_tape(small, 4); } catch (const std::invalid_argument&) { refused = true; }
        check(refused, "undersized slot tape refused before reset");
        std::vector<float> tape(slot.convolution_row_elements() * 4U);
        slot.bind_convolution_tape(tape, 4);
        const auto clear = [](std::span<float> values) { std::fill(values.begin(), values.end(), 0.0F); };
        slot.reset(clear);
        check(slot.state().convolution_tape_rows() == 4U, "tape bound before reset reaches the state");
        check(slot.state().convolution_row_elements() == slot.convolution_row_elements(), "state row elements");
        slot.reset(clear);
        check(slot.state().convolution_tape_rows() == 4U, "tape kept across resets");
    }
    window_program_gate();
    staged_batch_equals_lone();
    std::cout << "session conv tape tests passed\n";
    return 0;
}
