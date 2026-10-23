#include "strix/deepseek41_batch_dspark.hpp"

#include "strix/ds41_dspark_vocab.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

namespace strix::deepseek41 {
namespace {

namespace ds = ds41::dspark;
using clock_type = std::chrono::steady_clock;
double since(clock_type::time_point t) { return std::chrono::duration<double>(clock_type::now() - t).count(); }

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("deepseek41 batch dspark: ") + what + ": " + hipGetErrorString(e));
}

constexpr std::uint32_t stage_rows = 128;  // kept rows staged per slot before they go to its drafting context

// world 2: the draft round, rank 0 -> rank 1, and rank 1's drafts back (one 512-byte exchange each)
struct DraftRound {
    std::uint32_t magic = BatchSpeculator::round_magic, done = 0, commits = 0, requests = 0;
    // one entry per request (at most one commit and one draft per request and step), whatever the rows per step
    std::uint32_t commit[ds::BatchWindowController::max_requests]{};  // slot << 16 | kept
    std::uint32_t slot[ds::BatchWindowController::max_requests]{}, token[ds::BatchWindowController::max_requests]{};
};
struct DraftReply {
    std::uint32_t magic = 0x44524632U, count = 0;
    std::uint32_t slot[ds::BatchWindowController::max_requests]{};
    std::uint32_t tokens[ds::BatchWindowController::max_requests][ds::block_size]{};
    float confidence[ds::BatchWindowController::max_requests][ds::block_size]{};
};
constexpr std::size_t round_bytes = BatchSpeculator::round_bytes;
static_assert(sizeof(DraftRound) <= round_bytes && sizeof(DraftReply) <= round_bytes);

// sends `out`, returns what the peer sent as an In (each rank sends its half of the round in the same exchange)
template <class In, class Out>
In swap_packets(const std::function<void(std::span<const std::byte>, std::span<std::byte>)>& exchange, const Out& out) {
    std::array<std::byte, round_bytes> o{}, i{};
    std::memcpy(o.data(), &out, sizeof(Out));
    exchange(o, i);
    In in;
    std::memcpy(&in, i.data(), sizeof(In));
    return in;
}

// cost means: plain running mean for the first 16 samples, then an EMA (the costs drift with the context)
void accumulate(double& mean, std::uint32_t& n, double sample) {
    ++n;
    const double w = n <= 16U ? 1.0 / n : 1.0 / 16.0;
    mean += w * (sample - mean);
}

}  // namespace

struct BatchSpeculator::Slots {
    std::uint32_t count = 0;
    float* staging = nullptr;                          // [count][stage_rows][main_width]
    std::vector<std::uint32_t> staged;                 // rows staged per slot
    std::vector<std::vector<std::uint32_t>> context;   // committed tokens not yet in the slot's draft vocabulary
    ~Slots() {
        if (staging) (void)hipFree(staging);
    }
};

BatchSpeculator::BatchSpeculator(BatchForward& target, ds::DeviceDrafter& drafter, std::uint32_t slots, BatchSpecOptions options)
    : target_(target), drafter_(drafter), options_(std::move(options)), controller_(slots), slots_(std::make_unique<Slots>()) {
    if (options_.fixed_rows == 1U || options_.fixed_rows > ds::block_size + 1U)
        throw std::invalid_argument("deepseek41 batch dspark: a fixed window has 2..6 rows");
    if (options_.max_rows == 0U || options_.max_rows > BatchForward::max_rows || options_.prompt_rows == 0U)
        throw std::invalid_argument("deepseek41 batch dspark: rows per step");
    if (slots == 0U || slots > ds::BatchWindowController::max_requests) throw std::invalid_argument("deepseek41 batch dspark: 1..8 slots");
    if (options_.world == 0U || options_.world > 2U || options_.rank >= options_.world || (options_.world == 2U && !options_.exchange))
        throw std::invalid_argument("deepseek41 batch dspark: drafts on two ranks need rank, world 2 and an exchange");
    prompts_.resize(slots);
    positions_.assign(slots, 0U);
    kept_.fill(0xFFFFFFFFU);
    owners_.assign(slots, 0U);
    (void)target_.captured_streams();  // throws without capture_main_hidden
    drafter_.set_sequences(slots);
    auto& S = *slots_;
    S.count = slots;
    S.staged.assign(slots, 0U);
    S.context.resize(slots);
    check(hipMalloc(reinterpret_cast<void**>(&S.staging), std::size_t{slots} * stage_rows * ds::main_width * sizeof(float)), "staging");
}

BatchSpeculator::~BatchSpeculator() = default;

std::uint64_t BatchSpeculator::staging_bytes(std::uint32_t slots) noexcept {
    return std::uint64_t{slots} * stage_rows * ds::main_width * sizeof(float);
}

float* BatchSpeculator::staging(std::uint32_t slot) const {
    return slots_->staging + (std::size_t{slot} * stage_rows + slots_->staged[slot]) * ds::main_width;
}

void BatchSpeculator::flush(std::uint32_t slot) {
    auto& S = *slots_;
    if (S.staged[slot] == 0U && S.context[slot].empty()) return;
    drafter_.select(slot);
    if (S.staged[slot] != 0U)
        drafter_.append_hidden_device(S.staging + std::size_t{slot} * stage_rows * ds::main_width, S.staged[slot], target_.stream());
    S.staged[slot] = 0;
    if (!S.context[slot].empty()) (void)drafter_.add_vocabulary(S.context[slot]);
    S.context[slot].clear();
}

// rows [first, first + count) of the last step's captured inputs -> the slot's staging (then its drafting context)
void BatchSpeculator::stage(std::uint32_t slot, std::uint32_t first, std::uint32_t count) {
    stage_from(slot, target_.captured_streams(), first, count, target_.stream());
}

void BatchSpeculator::stage_from(std::uint32_t slot, std::array<const float*, 3> streams, std::uint32_t first, std::uint32_t count, void* stream) {
    auto& S = *slots_;
    if (count > stage_rows) throw std::invalid_argument("deepseek41 batch dspark: at most 128 rows staged at once");
    if (S.staged[slot] + count > stage_rows) flush(slot);
    const auto& c = target_.model().config;
    const std::size_t row = std::size_t{c.hc_mult} * c.hidden_size;
    ds::enqueue_main_hidden(streams[0] + first * row, streams[1] + first * row, streams[2] + first * row, count, staging(slot), stream);
    if (stream != target_.stream()) check(hipStreamSynchronize(static_cast<hipStream_t>(stream)), "seed rows");  // flush() orders on the target's
    S.staged[slot] += count;
}

void BatchSpeculator::start(std::uint32_t slot, std::span<const std::uint32_t> prompt) {
    auto& S = *slots_;
    if (slot >= S.count) throw std::invalid_argument("deepseek41 batch dspark: no such slot");
    drafter_.select(slot);
    drafter_.reset();
    drafter_.set_vocabulary(ds::choose_vocabulary(options_.vocabulary, prompt).vocabulary);
    S.staged[slot] = 0;
    S.context[slot].clear();
    controller_.reset(slot);
    prompts_[slot].assign(prompt.begin(), prompt.end());
    positions_[slot] = 0;
    owners_[slot] = options_.world == 2U && options_.split_started ? slot % 2U : 0U;
}

void BatchSpeculator::imported(std::uint32_t slot, std::span<const std::uint32_t> prompt, std::uint32_t position, std::uint32_t first) {
    imported(slot, prompt, position, first, {}, 0U, nullptr);
}

void BatchSpeculator::imported(std::uint32_t slot, std::span<const std::uint32_t> prompt, std::uint32_t position, std::uint32_t first,
                               std::array<const float*, 3> seed, std::uint32_t rows, void* stream) {
    start(slot, prompt);  // selects the slot's context, empty
    auto& ctx = slots_->context[slot];
    ctx.assign(prompt.begin(), prompt.end());
    ctx.push_back(first);
    positions_[slot] = position;
    owners_[slot] = options_.world == 2U ? slot % 2U : 0U;
    rows = owner(slot) == options_.rank && seed[0] != nullptr ? std::min({rows, position, stage_rows}) : 0U;
    drafter_.skip_to(position - rows);  // the seed rows (if any) come first
    if (rows != 0U) {
        stage_from(slot, seed, 0U, rows, stream);
        flush(slot);  // now, not at the first draft: a one-time append the per-draft costs must not carry
    }
}

bool BatchSpeculator::follow_round(std::span<const BatchRow> last_rows, std::span<const std::uint32_t> last_tokens, std::uint64_t step) {
    if (options_.world != 2U || options_.rank != 1U) throw std::logic_error("deepseek41 batch dspark: follow_round is rank 1's");
    std::array<std::byte, round_bytes> o{}, i{};
    const DraftReply none;  // rank 1 sends nothing in this half
    std::memcpy(o.data(), &none, sizeof none);
    options_.exchange(o, i);
    return follow_packet(i, last_rows, last_tokens, step);
}

std::vector<std::uint32_t> BatchSpeculator::take_commits() {
    std::vector<std::uint32_t> out;
    out.swap(pending_commits_);
    return out;
}

void BatchSpeculator::follow_commit(std::uint32_t slot, std::uint32_t kept) {
    target_.commit(slot, kept);
    kept_.at(slot) = kept;
}

void BatchSpeculator::follow_stage(std::span<const BatchRow> last_rows, std::span<const std::uint32_t> last_tokens, std::uint64_t step) {
    if (step == staged_step_) return;  // once per step: a draft round may have staged it already
    staged_step_ = step;
    auto& S = *slots_;
    // the last step's kept rows: positions, draft vocabulary context, and the hidden rows of this rank's requests
    for (std::uint32_t r0 = 0; r0 < last_rows.size();) {
        const auto slot = last_rows[r0].slot;
        std::uint32_t count = 1;
        while (r0 + count < last_rows.size() && last_rows[r0 + count].slot == slot) ++count;
        const std::uint32_t k = std::min(count, kept_.at(slot));
        const bool prompt = positions_[slot] < prompts_[slot].size();
        if (owner(slot) == 1U) {
            stage(slot, r0, k);
            if (prompt) {
                if (positions_[slot] + count >= prompts_[slot].size()) {
                    S.context[slot].insert(S.context[slot].end(), prompts_[slot].begin(), prompts_[slot].end());
                    S.context[slot].push_back(last_tokens[r0 + count - 1U]);
                }
            } else {
                S.context[slot].insert(S.context[slot].end(), last_tokens.begin() + r0, last_tokens.begin() + r0 + k);
            }
        }
        positions_[slot] += k;
        r0 += count;
    }
    kept_.fill(0xFFFFFFFFU);
}

bool BatchSpeculator::follow_packet(std::span<const std::byte> packet, std::span<const BatchRow> last_rows, std::span<const std::uint32_t> last_tokens,
                                    std::uint64_t step, std::vector<std::pair<std::uint32_t, std::uint32_t>>* applied) {
    if (options_.world != 2U || options_.rank != 1U) throw std::logic_error("deepseek41 batch dspark: follow_packet is rank 1's");
    DraftRound round;
    if (packet.size() < sizeof round) throw std::invalid_argument("deepseek41 batch dspark: short draft round");
    std::memcpy(&round, packet.data(), sizeof round);
    if (round.magic != round_magic || round.commits > ds::BatchWindowController::max_requests ||
        round.requests > ds::BatchWindowController::max_requests)
        throw std::runtime_error("deepseek41 batch dspark: bad draft round");
    for (std::uint32_t c = 0; c < round.commits; ++c) {
        follow_commit(round.commit[c] >> 16U, round.commit[c] & 0xFFFFU);
        if (applied) applied->emplace_back(round.commit[c] >> 16U, round.commit[c] & 0xFFFFU);
    }
    follow_stage(last_rows, last_tokens, step);
    if (round.done) return false;
    DraftReply reply;
    for (std::uint32_t q = 0; q < round.requests; ++q) {
        const auto slot = round.slot[q];
        flush(slot);
        drafter_.select(slot);
        const auto d = drafter_.draft(round.token[q], false);
        reply.slot[q] = slot;
        for (std::uint32_t j = 0; j < ds::block_size; ++j) reply.tokens[q][j] = d.tokens[j + 1U], reply.confidence[q][j] = d.confidence[j];
    }
    reply.count = round.requests;
    (void)swap_packets<DraftRound>(options_.exchange, reply);
    return true;
}

const PlannedStep& BatchSpeculator::plan(std::span<BatchSequence* const> seqs, std::span<const std::uint32_t> remaining) {
    const auto t0 = clock_type::now();
    const auto n = static_cast<std::uint32_t>(seqs.size());
    if (remaining.size() != n) throw std::invalid_argument("deepseek41 batch dspark: one remaining count per sequence");
    plan_ = {};
    drafts_.assign(n, ds::Draft{});
    window_rows_.assign(n, 0U);
    drafted_.assign(n, false);
    // decoding requests: one row each at least
    std::vector<std::uint32_t> dec;
    for (std::uint32_t i = 0; i < n; ++i)
        if (!seqs[i]->reading_prompt() && !seqs[i]->generated.empty() && remaining[i] != 0U) dec.push_back(i);
    if (dec.size() > options_.max_rows) throw std::invalid_argument("deepseek41 batch dspark: more decoding requests than rows per step");
    const auto nd = static_cast<std::uint32_t>(dec.size());
    // prompt chunks next (oldest first), as plan_step
    std::uint32_t used = nd;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> chunks;  // (sequence, rows)
    for (std::uint32_t i = 0; i < n && used < options_.max_rows; ++i) {
        const auto& s = *seqs[i];
        if (!s.reading_prompt()) continue;
        std::size_t take = std::min<std::size_t>({options_.prompt_rows, options_.max_rows - used, s.prompt.size() - s.fed});
        if (s.position == 0U) take = std::min<std::size_t>(take, 1U);
        if (take == 0U) continue;
        chunks.emplace_back(i, static_cast<std::uint32_t>(take));
        used += static_cast<std::uint32_t>(take);
    }
    const std::uint32_t budget = options_.max_rows - (used - nd);  // rows for the windows (>= nd)
    // per decoding request: at most 6 rows, its remaining tokens, its room, and one row for every other request
    std::vector<std::uint32_t> slots(nd), limit(nd);
    for (std::uint32_t k = 0; k < nd; ++k) {
        const auto& s = *seqs[dec[k]];
        const std::uint32_t room = target_.capacity(s.slot) > s.position ? target_.capacity(s.slot) - s.position : 1U;
        slots[k] = s.slot;
        limit[k] = std::max(1U, std::min({ds::block_size + 1U, remaining[dec[k]], room, budget - (nd - 1U)}));
        if (options_.align != 0U) limit[k] = std::max(1U, std::min(limit[k], options_.align - s.position % options_.align));
    }
    // stage 1: who drafts (the first decode step of every request count is plain: its cost is measured first)
    std::vector<bool> drafting(nd, false);
    if (nd != 0U && budget > nd && allow_drafts_) {
        if (options_.fixed_rows != 0U) {
            for (std::uint32_t k = 0; k < nd; ++k) drafting[k] = limit[k] > 1U;
        } else if (plain_n_[nd] != 0U || !chunks.empty()) {
            drafting = controller_.plan_drafts(slots, limit, budget);
        }
    }
    std::vector<ds::Draft> drafts(nd);
    std::uint32_t ndraft = 0;
    // world 2: the draft round -- rank 1's requests out (with the last step's commits), its drafts back after ours
    const auto t_round = clock_type::now();
    std::vector<std::uint32_t> remote;  // k drafted by rank 1
    for (std::uint32_t k = 0; k < nd; ++k)
        if (drafting[k] && owner(slots[k]) == 1U) remote.push_back(k);
    const bool split = options_.world == 2U && !remote.empty();  // otherwise no round: the commits go with the step's packet
    if (split) {
        DraftRound round;
        round.commits = static_cast<std::uint32_t>(pending_commits_.size());
        std::copy(pending_commits_.begin(), pending_commits_.end(), round.commit);
        pending_commits_.clear();
        for (const auto k : remote) {
            round.slot[round.requests] = slots[k];
            round.token[round.requests++] = seqs[dec[k]]->generated.back();
        }
        (void)swap_packets<DraftReply>(options_.exchange, round);
    }
    for (std::uint32_t k = 0; k < nd; ++k) {
        if (!drafting[k]) continue;
        if (options_.world == 2U && owner(slots[k]) == 1U) {
            drafted_[dec[k]] = true;
            ++ndraft;
            continue;
        }
        const auto td = clock_type::now();
        flush(slots[k]);
        drafter_.select(slots[k]);
        drafts[k] = drafter_.draft(seqs[dec[k]]->generated.back(), false);
        const double ds_ = since(td);
        stats_.draft_seconds += ds_;
        if (chunks.empty()) accumulate(draft_mean_, draft_n_, ds_);
        drafts_[dec[k]] = drafts[k];
        drafted_[dec[k]] = true;
        ++ndraft;
    }
    if (split) {
        const auto rp = swap_packets<DraftReply>(options_.exchange, DraftRound{});  // rank 0 sends nothing in this half
        if (rp.magic != DraftReply{}.magic || rp.count != remote.size()) throw std::runtime_error("deepseek41 batch dspark: bad draft reply");
        for (std::uint32_t q = 0; q < rp.count; ++q) {
            const auto k = remote[q];
            if (rp.slot[q] != slots[k]) throw std::runtime_error("deepseek41 batch dspark: draft reply for another slot");
            auto& d = drafts[k];
            d.tokens[0] = seqs[dec[k]]->generated.back();
            for (std::uint32_t j = 0; j < ds::block_size; ++j) d.tokens[j + 1U] = rp.tokens[q][j], d.confidence[j] = rp.confidence[q][j];
            drafts_[dec[k]] = d;
        }
        if (ndraft != 0U && chunks.empty() && draft_n_ != 0U)  // the round's cost beyond its parallel drafts
            accumulate(round_mean_, round_n_, since(t_round) - draft_mean_ * ((ndraft + 1U) / 2U));
    }
    // stage 2: rows per request
    std::vector<std::uint32_t> rows(nd, 1U);
    if (ndraft != 0U) {
        if (options_.fixed_rows != 0U) {
            for (std::uint32_t k = 0; k < nd; ++k) rows[k] = drafting[k] ? std::min(options_.fixed_rows, limit[k]) : 1U;
            std::uint32_t total = 0;
            for (const auto r : rows) total += r;
            while (total > budget) {  // the last of the largest windows gives a row back
                std::uint32_t big = 0;
                for (std::uint32_t k = 0; k < nd; ++k)
                    if (rows[k] >= rows[big]) big = k;
                --rows[big];
                --total;
            }
        } else {
            rows = controller_.choose_rows(slots, limit, budget, drafting, drafts);
        }
    }
    // the rows: every decoding request's window, then the prompt chunks
    std::uint32_t total = 0;
    for (std::uint32_t k = 0; k < nd; ++k) {
        const auto i = dec[k];
        const auto& s = *seqs[i];
        window_rows_[i] = rows[k];
        plan_.rows.push_back({s.slot, s.generated.back()});
        plan_.owner.push_back(i);
        plan_.emits.push_back(true);
        for (std::uint32_t j = 1; j < rows[k]; ++j) {
            plan_.rows.push_back({s.slot, drafts[k].tokens[j]});
            plan_.owner.push_back(i);
            plan_.emits.push_back(true);
        }
        total += rows[k];
    }
    for (const auto& [i, take] : chunks) {
        const auto& s = *seqs[i];
        for (std::uint32_t k = 0; k < take; ++k) {
            plan_.rows.push_back({s.slot, s.prompt[s.fed + k]});
            plan_.owner.push_back(i);
            plan_.emits.push_back(s.fed + k + 1U == s.prompt.size());
        }
    }
    planned_decoders_ = nd;
    planned_drafting_ = ndraft;
    planned_total_ = total;
    planned_prompt_rows_ = !chunks.empty();
    plan_seconds_ = since(t0);
    stats_.plan_seconds += plan_seconds_;
    return plan_;
}

std::vector<std::pair<std::uint32_t, std::uint32_t>> BatchSpeculator::finish(std::span<BatchSequence* const> seqs, std::span<const std::uint32_t> tokens) {
    const auto t0 = clock_type::now();
    if (tokens.size() != plan_.rows.size()) throw std::invalid_argument("deepseek41 batch dspark: one token per planned row");
    auto& S = *slots_;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> commits;
    std::uint64_t emitted = 0;
    for (std::uint32_t r0 = 0; r0 < plan_.rows.size();) {
        const auto i = plan_.owner[r0];
        std::uint32_t count = 1;
        while (r0 + count < plan_.rows.size() && plan_.owner[r0 + count] == i) ++count;
        auto& s = *seqs[i];
        const auto slot = s.slot;
        if (window_rows_[i] != 0U) {  // a window: row j + 1 is kept when draft j + 1 equalled row j's greedy token
            std::uint32_t accepted = 0;
            while (accepted + 1U < count && plan_.rows[r0 + accepted + 1U].token == tokens[r0 + accepted]) ++accepted;
            const std::uint32_t kept = accepted + 1U;
            if (kept < count) {
                target_.commit(slot, kept);
                commits.emplace_back(slot, kept);
                if (options_.world == 2U) pending_commits_.push_back(slot << 16U | kept);
            }
            if (owner(slot) == options_.rank) stage(slot, r0, kept);
            for (std::uint32_t j = 0; j < kept; ++j) {
                s.generated.push_back(tokens[r0 + j]);
                if (owner(slot) == options_.rank) S.context[slot].push_back(tokens[r0 + j]);
            }
            s.position += kept;
            emitted += kept;
            ++stats_.window_rows[count];
            if (count == 1U) ++stats_.plain_rows;
            else ++stats_.windows;
            if (drafted_[i]) {  // with one row the step's own token still checks draft 1
                const std::uint32_t verified = std::max(count - 1U, 1U);
                const std::uint32_t acc = count > 1U ? accepted : (drafts_[i].tokens[1] == tokens[r0] ? 1U : 0U);
                controller_.observe(slot, drafts_[i], verified, acc);
                stats_.drafts_verified += count - 1U;
                stats_.drafts_accepted += accepted;
                for (std::uint32_t j = 0; j + 1U < count && j <= accepted && j < ds::block_size; ++j) {
                    ++stats_.reached[j];
                    if (j < accepted) ++stats_.hit[j];
                }
            }
        } else {  // prompt rows: every one is kept
            const bool mine = owner(slot) == options_.rank;
            if (mine) stage(slot, r0, count);
            s.position += count;
            s.fed += count;
            if (!s.reading_prompt()) {
                if (mine) S.context[slot].insert(S.context[slot].end(), s.prompt.begin(), s.prompt.end());
                s.generated.push_back(tokens[r0 + count - 1U]);
                if (mine) S.context[slot].push_back(tokens[r0 + count - 1U]);
            }
        }
        r0 += count;
    }
    const double fs = since(t0);
    const double ss = target_.last_stats().seconds;
    stats_.finish_seconds += fs;
    stats_.step_seconds += ss;
    ++stats_.steps;
    stats_.rows += plan_.rows.size();
    if (planned_drafting_ != 0U) ++stats_.drafted_steps;
    if (!planned_prompt_rows_ && planned_decoders_ != 0U) {
        ++stats_.decode_steps;
        stats_.decode_tokens += emitted;
        stats_.decode_seconds += plan_seconds_ + ss + fs;
        // costs: plain[n] from steps without drafts, the row slope from steps with windows (n requests known)
        const double wall = ss + fs;
        const auto nd = planned_decoders_;
        if (planned_drafting_ == 0U && planned_total_ == nd) {
            accumulate(plain_mean_[nd], plain_n_[nd], wall);
        } else if (plain_n_[nd] != 0U && planned_total_ > nd) {
            accumulate(row_mean_, row_n_, (wall - plain_mean_[nd]) / (planned_total_ - nd));
        }
        ds::BatchWindowController::Costs c;
        for (std::uint32_t k = 1; k <= ds::BatchWindowController::max_requests; ++k)
            c.plain[k] = plain_n_[k] != 0U ? static_cast<float>(plain_mean_[k]) : 0.0F;
        c.row = row_n_ != 0U ? static_cast<float>(std::max(row_mean_, 1e-4)) : 0.0F;
        c.draft = draft_n_ != 0U ? static_cast<float>(draft_mean_) : 0.0F;
        c.extra = 0.0F;  // appends and commits are inside the measured step walls
        c.drafters = options_.world;
        c.round = round_n_ != 0U ? static_cast<float>(std::max(round_mean_, 0.0)) : 0.0F;
        controller_.set_costs(c);
    }
    return commits;
}

}  // namespace strix::deepseek41
