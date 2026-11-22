// Shared multi-request decoder (strix/serve/cohort_decode.hpp).
#include "strix/serve/cohort_decode.hpp"
#include "strix/serve/job_sampling.hpp"

#include "strix/serve/short_first.hpp"

#include <algorithm>
#include <chrono>
#include <optional>
#include <stdexcept>
#include <utility>

namespace strix::serve {

std::span<const float> CohortBackend::read_final_logits(std::uint32_t) {
    throw std::logic_error("temperature: this cohort backend cannot read final logits");
}

struct CohortDecoder::Lane {
    CohortRequest request;
    CohortResult result;
    std::uint64_t order = 0;
    std::uint64_t prompt_read = 0;  // prompt rows appended to the lane (the last prompt token is the first decode row)
    std::uint64_t position = 0;     // committed positions
    std::uint32_t pending = 0;      // next decode input
    unsigned overtakes = 0;
    double waited_ms = 0;
    std::uint64_t bytes = 0;
    std::unique_ptr<MtpContextDraft> drafter;
    std::optional<TokenSampler> sampler;
    std::vector<double> tried, kept;  // kept histogram per draft row index: rows the target verified / kept
    [[nodiscard]] std::uint64_t prompt_left() const { return request.prompt.size() - 1U - prompt_read; }
};

CohortDecoder::CohortDecoder(CohortBackend& backend, CohortConfig config) : backend_(backend), config_(std::move(config)) {
    // Every running lane holds at least one row, so a step never parks a lane: lanes <= rows of one forward.
    const std::uint32_t n = std::min(backend_.lanes(), backend_.rows_per_forward());
    if (n == 0) throw std::invalid_argument("cohort decoder: backend has no lanes");
    lanes_.resize(n);
}

CohortDecoder::~CohortDecoder() = default;

double CohortDecoder::now_ms() {
    if (fake_clock_ms_ > 0) return fake_now_ += fake_clock_ms_;
    using Clock = std::chrono::steady_clock;
    static const auto origin = Clock::now();
    return std::chrono::duration<double, std::milli>(Clock::now() - origin).count();
}

std::uint64_t CohortDecoder::reserved_bytes(const CohortRequest& r) const {
    const std::uint64_t positions = r.prompt.size() + r.max_tokens + 1U +
        (token_sampling_enabled(r.sampling) ? 0U : config_.max_draft);
    return positions * config_.position_bytes + config_.lane_bytes;
}

void CohortDecoder::submit(CohortRequest request) {
    Waiting w;
    w.submit_ms = now_ms();
    w.order = order_++;
    std::string error;
    if (request.prompt.size() < 2) error = "prompt needs at least 2 tokens";
    else if (request.max_tokens == 0) error = "max_tokens must be at least 1";
    else if (request.max_tokens > backend_.max_positions() ||
             request.prompt.size() > backend_.max_positions() - request.max_tokens) error = "request longer than a lane";
    else if (request.sampling.temperature != 0.0F) {
        try {
            validate_token_sampling_policy(request.sampling);
            if (!backend_.supports_final_logits()) error = "temperature: this cohort backend cannot read final logits";
        } catch (const std::invalid_argument& ex) {
            error = ex.what();
        }
    }
    if (error.empty() && config_.memory_budget != 0 && reserved_bytes(request) > config_.memory_budget)
        error = "request exceeds the memory budget";
    if (!error.empty()) {
        CohortResult r;
        r.id = request.id;
        r.error = std::move(error);
        r.submit_ms = r.done_ms = w.submit_ms;
        finished_.push_back(std::move(r));
        return;
    }
    w.request = std::move(request);
    waiting_.push_back(std::move(w));
}

bool CohortDecoder::idle() const noexcept { return waiting_.empty() && live() == 0; }

std::uint32_t CohortDecoder::live() const noexcept {
    std::uint32_t n = 0;
    for (const auto& l : lanes_) n += l ? 1U : 0U;
    return n;
}

std::vector<CohortResult> CohortDecoder::take_finished() { return std::exchange(finished_, {}); }

// Join at the step boundary: shortest prompt first with the shared eight-overtake bound, while a lane is free and the
// reserved bytes fit. An aged request that does not fit blocks younger ones (no starvation by smaller requests).
void CohortDecoder::admit() {
    while (!waiting_.empty()) {
        const auto free = std::find_if(lanes_.begin(), lanes_.end(), [](const auto& l) { return !l; });
        if (free == lanes_.end()) return;
        const auto fits = [&](const Waiting& w) {
            return config_.memory_budget == 0 || used_bytes_ + reserved_bytes(w.request) <= config_.memory_budget;
        };
        const std::size_t pick = select_waiting(
            waiting_, true, [](const Waiting& w) { return w.request.prompt.size(); },
            [](Waiting& w) -> unsigned& { return w.skips; }, waiting_.size(), fits);
        if (pick >= waiting_.size()) return;
        auto lane = std::make_unique<Lane>();
        Waiting w = std::move(waiting_[pick]);
        waiting_.erase(waiting_.begin() + static_cast<std::ptrdiff_t>(pick));
        lane->request = std::move(w.request);
        lane->order = w.order;
        lane->result.id = lane->request.id;
        lane->result.submit_ms = w.submit_ms;
        lane->bytes = reserved_bytes(lane->request);
        used_bytes_ += lane->bytes;
        if (token_sampling_enabled(lane->request.sampling)) {
            // A stable request identity makes an explicit seed replayable across lane assignments.
            lane->sampler.emplace(lane->request.sampling, TokenSamplerOptions{});
            lane->result.sampling = "sampling-v1";
        } else if (config_.context_draft) {
            // the drafter's context is what the lane holds; the pending (last prompt) token joins at the first step
            lane->drafter = std::make_unique<MtpContextDraft>(config_.draft);
            const auto& p = lane->request.prompt;
            lane->drafter->reset(std::span<const std::uint32_t>(p.data(), p.size() - 1U));
            lane->tried.assign(config_.max_draft, 0.0);
            lane->kept.assign(config_.max_draft, 0.0);
        }
        *free = std::move(lane);
        peak_live_ = std::max(peak_live_, live());
    }
}

void CohortDecoder::finish(Lane& lane, std::string error) {
    lane.result.error = std::move(error);
    lane.result.done_ms = now_ms();
    if (lane.sampler) lane.result.sampling_receipt = job_sampling_receipt(*lane.sampler);
    for (std::size_t i = 0; i < lanes_.size(); ++i) {
        if (lanes_[i].get() != &lane) continue;
        used_bytes_ -= lane.bytes;
        finished_.push_back(std::move(lane.result));
        backend_.release(static_cast<std::uint32_t>(i));
        lanes_[i].reset();
        return;
    }
}

CycleCost CohortDecoder::cost() const { return fitted_cost(sums_, config_, live()); }

CycleCost CohortDecoder::fitted_cost(const CycleLedger::FitSums& sums, const CohortConfig& config, std::uint32_t lanes) {
    if (!config.learn_cost || sums.n < static_cast<double>(std::max(config.fit_min_steps, 2U))) return config.prior;
    const auto f = CycleLedger::fit(sums);
    if (f.terms < 2 || !(f.row_ms > 0.0)) return config.prior;
    double fixed = f.fixed_ms, lane = f.lane_ms;
    if (lane < 0.0) fixed += lane * static_cast<double>(lanes), lane = 0.0;
    if (!(fixed > 0.0)) return config.prior;
    return {fixed, f.row_ms, lane, config.prior.draft_row_ms};
}

void CohortDecoder::observe_cost(std::uint32_t rows, std::uint32_t lanes, double ms) {
    const double r = rows, l = lanes;
    sums_.n += 1.0;
    sums_.r += r, sums_.l += l, sums_.rr += r * r, sums_.rl += r * l, sums_.ll += l * l;
    sums_.y += ms, sums_.ry += r * ms, sums_.ly += l * ms, sums_.yy += ms * ms;
}

void CohortDecoder::step() {
    admit();
    for (auto& lane : lanes_) {
        if (lane && lane->request.cancelled && lane->request.cancelled()) {
            lane->result.cancelled = true;
            finish(*lane);
        }
    }
    if (live() == 0) return;
    ++steps_;

    // 1. Prompt service: decode first, one bounded slice of the chosen prompt (short-first with aging, prompt debt).
    std::vector<StepLane> step_lanes;
    for (std::size_t i = 0; i < lanes_.size(); ++i) {
        const auto& l = lanes_[i];
        if (!l) continue;
        StepLane s;
        s.lane = static_cast<std::uint32_t>(i);
        s.prompt = l->prompt_left() > 0;
        s.prompt_remaining = l->prompt_left();
        s.order = l->order;
        s.overtakes = l->overtakes;
        step_lanes.push_back(s);
    }
    const std::uint32_t units[1] = {std::max(config_.prompt_slice, 1U)};
    StepLimits limits;
    limits.prompt_units = units;
    limits.prompt_share = config_.prompt_share;
    limits.short_first_rows = config_.short_first_rows;
    const StepPlan plan = planner_.plan(step_lanes, limits);
    for (const auto l : plan.overtaken) ++lanes_[l]->overtakes;
    double prompt_ms = 0;
    std::uint64_t prompt_rows = 0;
    if (!plan.prompt.empty()) {
        Lane& l = *lanes_[plan.prompt.front().lane];
        const std::uint64_t want = plan.prompt.front().rows ? plan.prompt.front().rows : units[0];
        prompt_rows = std::min<std::uint64_t>(want, l.prompt_left());
        const double t0 = now_ms();
        backend_.prefill(plan.prompt.front().lane,
                         std::span<const std::uint32_t>(l.request.prompt.data() + l.prompt_read, prompt_rows), l.prompt_read == 0);
        prompt_ms = now_ms() - t0;
        prompt_ms_ += prompt_ms;
        l.prompt_read += prompt_rows;
        l.position += prompt_rows;
        if (l.request.prefilled) l.request.prefilled(l.prompt_read);
        if (l.prompt_left() == 0) l.pending = l.request.prompt.back();
    }

    // 2. Decode round: every decoding lane holds its certain row; draft rows are bought by plan_rows.
    const double t0 = now_ms();
    std::vector<Lane*> run;
    for (const auto id : plan.decode_lanes) run.push_back(lanes_[id].get());
    if (run.empty()) {
        planner_.account(prompt_ms, 0.0, 0.0, prompt_rows);
        return;
    }
    const std::size_t n = run.size();
    std::vector<std::vector<std::uint32_t>> drafts(n);
    std::vector<std::vector<float>> accept(n);
    std::vector<LaneDemand> demands(n);
    for (std::size_t k = 0; k < n; ++k) {
        Lane& l = *run[k];
        const std::uint64_t left = l.request.max_tokens - l.result.ids.size();
        const std::uint64_t room_pos = backend_.max_positions() - l.position - 1U;
        const std::size_t room = static_cast<std::size_t>(std::min<std::uint64_t>({config_.max_draft, left - 1U, room_pos}));
        std::vector<double> acc;
        if (l.drafter) {
            // room 0 still observes the pending token once (the drafter's context follows the lane)
            drafts[k] = l.drafter->begin_cycle(l.pending, {}, 0, room, acc, config_.min_p);
            if (drafts[k].size() > room) drafts[k].resize(room);
        }
        accept[k].push_back(1.0F);
        const double w = config_.accept_weight;
        for (std::size_t d = 0; d < drafts[k].size(); ++d) {
            const double claim = d < acc.size() ? acc[d] : config_.accept_prior;
            const double p = w > 0.0 && d < l.tried.size() ? (l.kept[d] + w * claim) / (l.tried[d] + w) : claim;
            accept[k].push_back(static_cast<float>(p));
        }
        // drain rate: tokens per step the lane realized so far, this step's full offer counted as one more step
        // (a lane that is not paced realizes what it is offered, so the slowest lane's rate is its potential)
        double chain = 1.0, expected = 1.0;  // the pending row is certain
        for (std::size_t d = 1; d < accept[k].size(); ++d) expected += chain *= static_cast<double>(accept[k][d]);
        demands[k].rate = (static_cast<double>(l.result.ids.size()) + expected) / static_cast<double>(l.result.steps + 1U);
        demands[k].lane = plan.decode_lanes[k];
        demands[k].remaining_tokens = static_cast<std::uint32_t>(left);
        demands[k].min_rows = 1;
        demands[k].max_rows = 1U + static_cast<std::uint32_t>(drafts[k].size());
        demands[k].accept = accept[k];
        demands[k].waited_ms = l.waited_ms;
    }
    RowLimits row_limits;
    row_limits.rows_per_forward = backend_.rows_per_forward();
    row_limits.drain = waiting_.empty();  // nobody waits for a lane: a lane that ends early frees nothing
    const RowPlan rp =
        plan_rows(demands, fitted_cost(sums_, config_, static_cast<std::uint32_t>(n)), row_limits, config_.accept_prior);

    std::vector<CohortGroup> groups(n);
    std::vector<std::uint32_t> tokens;
    for (std::size_t k = 0; k < n; ++k) {
        const std::uint32_t rows = std::min(rp.rows.at(k), demands[k].max_rows);
        if (rows == 0) throw std::logic_error("cohort decoder: a running lane was parked (lanes > rows of a forward)");
        groups[k] = {demands[k].lane, rows};
        tokens.push_back(run[k]->pending);
        tokens.insert(tokens.end(), drafts[k].begin(), drafts[k].begin() + (rows - 1U));
    }
    std::vector<std::uint32_t> top(tokens.size());
    backend_.verify(groups, tokens, top);
    std::vector<std::uint32_t> keep(n);
    std::size_t base = 0;
    for (std::size_t k = 0; k < n; ++k) {
        if (auto& sampler = run[k]->sampler) {
            // Context drafts have no lossless sampled acceptance. This lane holds just its pending row.
            top[base] = sampler->decide(backend_.read_final_logits(groups[k].lane), run[k]->position + 1U, nullptr).token;
        }
        std::uint32_t a = 1;  // the pending row is always kept; each draft row equal to the target's choice is too
        while (a < groups[k].rows && tokens[base + a] == top[base + a - 1U]) ++a;
        keep[k] = a;
        base += groups[k].rows;
    }
    backend_.commit(keep);
    const double decode_ms = now_ms() - t0;
    decode_ms_ += decode_ms;
    std::uint32_t total_rows = 0;
    for (const auto& g : groups) total_rows += g.rows;
    rows_ += total_rows;
    observe_cost(total_rows, static_cast<std::uint32_t>(n), decode_ms);
    planner_.account(prompt_ms, decode_ms, 0.0, prompt_rows);

    base = 0;
    std::vector<Lane*> done;
    for (std::size_t k = 0; k < n; ++k) {
        Lane& l = *run[k];
        const std::span<const std::uint32_t> emitted(top.data() + base, keep[k]);
        base += groups[k].rows;
        if (l.drafter) l.drafter->finish_cycle(emitted, groups[k].rows - 1U, keep[k] - 1U);
        // kept histogram: verified draft rows up to the first rejection; an offer the plan did not buy still tells
        // whether its first row was the target's next token
        const std::uint32_t delivered = groups[k].rows - 1U, hits = keep[k] - 1U;
        if (!l.tried.empty()) {
            for (std::uint32_t r = 0; r < hits; ++r) l.tried[r] += 1.0, l.kept[r] += 1.0;
            if (hits < delivered) l.tried[hits] += 1.0;
            if (delivered == 0 && !drafts[k].empty()) {
                l.tried[0] += 1.0;
                if (drafts[k][0] == emitted[0]) l.kept[0] += 1.0;
            }
        }
        l.result.steps += 1;
        l.result.drafted += groups[k].rows - 1U;
        l.result.accepted += keep[k] - 1U;
        l.position += keep[k];
        l.pending = emitted.back();
        if (l.result.ids.empty()) l.result.first_ms = now_ms();
        const auto before = l.result.ids.size();
        bool end = false;
        for (const auto id : emitted) {
            l.result.ids.push_back(id);
            if (l.result.ids.size() >= l.request.max_tokens ||
                std::find(l.request.stop.begin(), l.request.stop.end(), id) != l.request.stop.end()) {
                end = true;
                break;
            }
        }
        if (l.request.publish &&
            !l.request.publish(emitted.first(l.result.ids.size() - before))) {
            l.result.cancelled = true;
            end = true;
        }
        if (end) done.push_back(&l);
    }
    for (auto* l : done) finish(*l);
}

void CohortDecoder::run() {
    while (!idle()) step();
}

}  // namespace strix::serve
