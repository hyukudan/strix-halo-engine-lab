// Shared multi-request decoder (strix/serve/cohort_decode.hpp).
#include "strix/serve/cohort_decode.hpp"
#include "strix/serve/job_sampling.hpp"

#include "strix/serve/short_first.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace strix::serve {

std::span<const float> CohortBackend::read_final_logits(std::uint32_t) {
    throw std::logic_error("temperature: this cohort backend cannot read final logits");
}

void CohortBackend::propose(std::uint32_t, std::uint32_t, std::vector<std::uint32_t>& tokens,
                            std::vector<double>& probability) {
    tokens.clear();
    probability.clear();
}

std::uint64_t CohortBackend::restore(std::uint32_t, std::span<const std::uint32_t>) { return 0; }

void CohortBackend::capture(std::uint32_t, std::span<const std::uint32_t>) {}

PrefillGroupCapacity CohortBackend::prompt_group_capacity() const {
    PrefillGroupCapacity capacity;
    capacity.members = 1;  // no grouping
    capacity.reason = "backend-has-no-group-pass";
    return capacity;
}

std::uint32_t CohortBackend::prompt_route_class(std::uint64_t) const { return 0; }

std::uint32_t CohortBackend::prefill_top(std::uint32_t) {
    throw std::logic_error("cohort backend: prefill_top without prefill_emits_first");
}

std::uint32_t CohortBackend::max_window(std::uint32_t) const { return std::numeric_limits<std::uint32_t>::max(); }

void CohortBackend::prefill_group(std::span<const CohortPrompt> members) {
    for (const auto& m : members) prefill(m.lane, m.tokens, m.first);
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
    std::uint64_t prompt_start = 0;   // prompt rows restored at admission (a group member starts from 0 only)
    bool whole = false;               // the backend's prefill reads the whole prompt and emits the first id
    [[nodiscard]] std::uint64_t prompt_left() const {
        return request.prompt.size() - (whole ? 0U : 1U) - prompt_read;
    }
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
    if (backend_.prefill_emits_first() ? request.prompt.empty() : request.prompt.size() < 2)
        error = backend_.prefill_emits_first() ? "prompt needs at least 1 token" : "prompt needs at least 2 tokens";
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
    if (error.empty() && request.selection) {
        if (token_sampling_enabled(request.sampling)) error = "selection: sampled requests with a selection policy are not served by the cohort";
        else if (request.selection_reads_logits && !backend_.supports_final_logits())
            error = "selection: this cohort backend cannot read final logits";
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
        lane->whole = backend_.prefill_emits_first();
        if (token_sampling_enabled(lane->request.sampling)) {
            // A stable request identity makes an explicit seed replayable across lane assignments (the request's
            // options carry it: the default identity, or the one the engine's own job sampler would have drawn).
            lane->sampler.emplace(lane->request.sampling, lane->request.sampler_options);
            lane->result.sampling = "sampling-v1";
        } else if (config_.context_draft || backend_.device_drafts()) {
            if (config_.context_draft) {
                // the drafter's context is what the lane holds; the pending token joins at the first step (the last
                // prompt token, or the prefill's first id when the prefill reads the whole prompt)
                lane->drafter = std::make_unique<MtpContextDraft>(config_.draft);
                const auto& p = lane->request.prompt;
                lane->drafter->reset(std::span<const std::uint32_t>(p.data(), p.size() - (lane->whole ? 0U : 1U)));
            }
            lane->tried.assign(config_.max_draft, 0.0);
            lane->kept.assign(config_.max_draft, 0.0);
        }
        Lane& l = *lane;
        const auto index = static_cast<std::uint32_t>(free - lanes_.begin());
        *free = std::move(lane);
        peak_live_ = std::max(peak_live_, live());
        if (l.request.reuse) {
            // A retained prefix replaces the first prompt rows; the last prompt token stays the first decode row.
            const auto& p = l.request.prompt;
            const std::uint64_t restored = backend_.restore(index, std::span<const std::uint32_t>(p.data(), p.size()));
            if (restored > p.size() - 1U) throw std::logic_error("cohort decoder: restore returned more than the prompt less one");
            l.prompt_read = l.prompt_start = l.position = l.result.restored = restored;
            if (restored != 0 && l.request.prefilled) l.request.prefilled(restored);
            if (l.prompt_left() == 0) l.pending = p.back();
        }
    }
}

void CohortDecoder::finish(Lane& lane, std::string error) {
    lane.result.error = std::move(error);
    lane.result.done_ms = now_ms();
    if (lane.sampler) lane.result.sampling_receipt = job_sampling_receipt(*lane.sampler);
    for (std::size_t i = 0; i < lanes_.size(); ++i) {
        if (lanes_[i].get() != &lane) continue;
        used_bytes_ -= lane.bytes;
        if (lane.request.retain && lane.result.error.empty() && !lane.result.cancelled && !lane.result.ids.empty()) {
            // committed inputs: the prompt (its last token was the first decode row), then every emitted id that a
            // later row consumed; ids cut at a stop are all consumed, so prompt ++ ids is cut at `position`.
            std::vector<std::uint32_t> committed = lane.request.prompt;
            committed.insert(committed.end(), lane.result.ids.begin(), lane.result.ids.end());
            committed.resize(static_cast<std::size_t>(std::min<std::uint64_t>(committed.size(), lane.position)));
            backend_.capture(static_cast<std::uint32_t>(i), committed);
        }
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

    // Appends a lane's emitted ids through max_tokens / stop and publishes them; true = the request ends here.
    const auto deliver = [&](Lane& l, std::span<const std::uint32_t> emitted) {
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
        if (l.request.publish && !l.request.publish(emitted.first(l.result.ids.size() - before))) {
            l.result.cancelled = true;
            end = true;
        }
        return end;
    };
    // A whole-prompt lane whose prefill just ended: its first id comes from the prefill's last row (the selection
    // policy, the sampler on its final logits, or its greedy id). The id is not committed: it is the lane's pending row.
    std::vector<std::pair<Lane*, std::string>> first_done;
    const auto emit_first = [&](std::uint32_t index) {
        Lane& l = *lanes_[index];
        if (!l.whole || l.prompt_left() != 0) {
            if (l.prompt_left() == 0) l.pending = l.request.prompt.back();
            return;
        }
        l.result.first_from_prefill = true;
        std::uint32_t id = 0;
        if (auto& selection = l.request.selection) {
            try {
                selection->reset();
                std::unique_ptr<MtpSelectionReceipt> receipt;
                if (l.request.selection_reads_logits) {
                    id = selection->select(backend_.read_final_logits(index));
                    receipt = selection->seal();
                } else {
                    const std::uint32_t top[1] = {backend_.prefill_top(index)};
                    selection->expect(std::span<const std::uint32_t>());
                    auto window = select_greedy_window(*selection, std::span<const std::uint32_t>(),
                                                       std::span<const std::uint32_t>(top));
                    if (window.ids.empty()) throw MtpSelectionError("the policy selected no first token");
                    id = window.ids.front();
                    receipt = std::move(window.receipt);
                }
                if (receipt) receipt->publish();  // the prompt is committed: the selection state follows it
            } catch (const MtpSelectionError& error) {
                selection->discard();
                first_done.emplace_back(&l, std::string("selection: ") + error.what());
                return;
            }
        } else if (l.sampler) {
            id = l.sampler->decide(backend_.read_final_logits(index), l.position, nullptr).token;
        } else {
            id = backend_.prefill_top(index);
        }
        const std::uint32_t one[1] = {id};
        if (deliver(l, std::span<const std::uint32_t>(one))) first_done.emplace_back(&l, std::string());
    };

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
    // Grouped prompts: when the planner's head is a whole short prompt, other whole short prompts that arrived after
    // it may share its pass (plan_prefill_group: arrival order, one route class, at most one slice of rows).
    std::vector<std::uint32_t> group;
    if (!plan.prompt.empty() && config_.prompt_group.groups()) {
        const std::uint32_t head = plan.prompt.front().lane;
        const Lane& h = *lanes_[head];
        // the planner runs the head's whole prompt this step (its entry = the rows it gives the head, 0 = one unit);
        // the group may fill the rest of one slice unit
        const std::uint64_t head_rows = plan.prompt.front().rows ? plan.prompt.front().rows : units[0];
        const std::uint64_t want = units[0];
        const PrefillGroupCapacity capacity = backend_.prompt_group_capacity();
        if (capacity.members >= 2U && h.prompt_start == 0 && h.prompt_read == 0 && h.prompt_left() <= head_rows) {
            std::vector<std::uint32_t> others;
            for (std::size_t i = 0; i < lanes_.size(); ++i) {
                const auto& c = lanes_[i];
                if (c && i != head && c->prompt_start == 0 && c->prompt_read == 0 && c->prompt_left() > 0 && c->order > h.order)
                    others.push_back(static_cast<std::uint32_t>(i));
            }
            std::sort(others.begin(), others.end(), [&](auto a, auto b) { return lanes_[a]->order < lanes_[b]->order; });
            std::vector<GroupCandidate> candidates;
            const auto offer = [&](std::uint32_t i) {
                const std::uint64_t rows = lanes_[i]->prompt_left();
                GroupCandidate c;
                c.id = i;
                c.rows = static_cast<std::uint32_t>(std::min<std::uint64_t>(rows, std::numeric_limits<std::uint32_t>::max()));
                c.arrival_seq = lanes_[i]->order;
                c.route_class = backend_.prompt_route_class(rows);
                candidates.push_back(c);
            };
            offer(head);
            for (const auto i : others) offer(i);
            PrefillGroupPolicy policy = config_.prompt_group;
            policy.max_rows = static_cast<std::uint32_t>(std::min<std::uint64_t>(policy.max_rows, want));
            for (const auto id : plan_prefill_group(candidates, policy, PrefillGroupCost{},
                                                    std::numeric_limits<double>::infinity(), capacity))
                group.push_back(static_cast<std::uint32_t>(id));
        }
    }
    if (group.size() > 1) {
        std::vector<CohortPrompt> members;
        for (const auto i : group) {
            const Lane& m = *lanes_[i];
            members.push_back({i, std::span<const std::uint32_t>(m.request.prompt.data(), m.prompt_left()), true});
        }
        const double t0 = now_ms();
        backend_.prefill_group(members);
        prompt_ms = now_ms() - t0;
        prompt_ms_ += prompt_ms;
        ++prompt_groups_;
        grouped_prompts_ += group.size();
        for (const auto i : group) {
            Lane& m = *lanes_[i];
            const std::uint64_t rows = m.prompt_left();
            prompt_rows += rows;
            m.prompt_read += rows;
            m.position += rows;
            if (m.request.prefilled) m.request.prefilled(m.prompt_read);
        }
        for (const auto i : group) emit_first(i);
    } else if (!plan.prompt.empty()) {
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
        emit_first(plan.prompt.front().lane);
    }
    for (auto& [l, error] : first_done) finish(*l, std::move(error));

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
    std::vector<std::size_t> device_rows(n, 0);  // leading device-draft rows of a window (context rows follow them)
    std::vector<std::uint32_t> device;
    std::vector<double> device_p;
    for (std::size_t k = 0; k < n; ++k) {
        Lane& l = *run[k];
        const std::uint64_t left = l.request.max_tokens - l.result.ids.size();
        const std::uint64_t room_pos = backend_.max_positions() - l.position - 1U;
        // a selection that reads logits decides one row per step from the lane's final logits
        const bool one_row = l.request.selection && l.request.selection_reads_logits;
        // the backend's own cap on the lane's window (pending row + drafts)
        const std::uint64_t window_cap = std::max<std::uint32_t>(backend_.max_window(plan.decode_lanes[k]), 1U) - 1U;
        const std::size_t room = one_row ? 0U : static_cast<std::size_t>(std::min<std::uint64_t>(
                                                     {config_.max_draft, left - 1U, room_pos, window_cap}));
        std::vector<double> acc;
        device.clear();
        device_p.clear();
        if (!l.sampler && room != 0 && backend_.device_drafts()) {
            backend_.propose(plan.decode_lanes[k], static_cast<std::uint32_t>(room), device, device_p);
            if (device.size() > room) device.resize(room);
        }
        std::vector<double> device_acc(device.size());
        double device_expected = 0.0, chain_p = 1.0;
        for (std::size_t d = 0; d < device.size(); ++d) {
            const double p = d < device_p.size() && std::isfinite(device_p[d]) ? std::clamp(device_p[d], 0.0, 1.0)
                                                                               : config_.accept_prior;
            device_acc[d] = p;
            device_expected += chain_p *= p;
        }
        if (l.drafter) {
            // room 0 still observes the pending token once (the drafter's context follows the lane). A device proposal
            // is the confident prefix: the context drafter continues it, or replaces it with a long exact copy span
            // expected to deliver more rows.
            const auto replaced = l.drafter->stats().replaced_cycles;
            auto context = l.drafter->begin_cycle(l.pending, device, device.size(), room, acc, config_.min_p,
                                                  device.empty() ? -1.0 : device_expected);
            if (device.empty() || l.drafter->stats().replaced_cycles != replaced) {
                drafts[k] = std::move(context);
            } else {
                drafts[k] = device;
                drafts[k].insert(drafts[k].end(), context.begin(), context.end());
                device_acc.insert(device_acc.end(), acc.begin(), acc.end());
                acc = std::move(device_acc);
                device_rows[k] = device.size();
            }
        } else if (!device.empty()) {
            drafts[k] = device;
            acc = std::move(device_acc);
            device_rows[k] = device.size();
        }
        if (drafts[k].size() > room) drafts[k].resize(room);
        device_rows[k] = std::min(device_rows[k], drafts[k].size());
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
    std::size_t base = 0;
    for (std::size_t k = 0; k < n; ++k) {
        if (auto& selection = run[k]->request.selection) {
            // one attempt per step; the hint lets the policy prepare the rows' masks while the device runs
            selection->reset();
            selection->expect(std::span<const std::uint32_t>(tokens.data() + base + 1U, groups[k].rows - 1U));
        }
        base += groups[k].rows;
    }
    std::vector<std::uint32_t> top(tokens.size());
    backend_.verify(groups, tokens, top);
    std::vector<std::uint32_t> keep(n);
    std::vector<std::vector<std::uint32_t>> chosen(n);  // the ids each lane emits this step
    std::vector<std::unique_ptr<MtpSelectionReceipt>> receipts(n);
    std::vector<std::string> refused(n);
    base = 0;
    for (std::size_t k = 0; k < n; ++k) {
        Lane& l = *run[k];
        const std::uint32_t rows = groups[k].rows;
        if (auto& selection = l.request.selection) {
            try {
                if (l.request.selection_reads_logits) {
                    chosen[k] = {selection->select(backend_.read_final_logits(groups[k].lane))};
                    receipts[k] = selection->seal();
                } else {
                    auto window = select_greedy_window(*selection,
                                                       std::span<const std::uint32_t>(tokens.data() + base + 1U, rows - 1U),
                                                       std::span<const std::uint32_t>(top.data() + base, rows));
                    chosen[k] = std::move(window.ids);
                    receipts[k] = std::move(window.receipt);
                }
            } catch (const MtpSelectionError& error) {
                selection->discard();
                refused[k] = std::string("selection: ") + error.what();
                chosen[k].clear();
            }
            keep[k] = std::max<std::uint32_t>(1U, static_cast<std::uint32_t>(chosen[k].size()));
        } else {
            if (auto& sampler = l.sampler) {
                // Context drafts have no lossless sampled acceptance. This lane holds just its pending row.
                top[base] = sampler->decide(backend_.read_final_logits(groups[k].lane), l.position + 1U, nullptr).token;
            }
            std::uint32_t a = 1;  // the pending row is always kept; each draft row equal to the target's choice is too
            while (a < rows && tokens[base + a] == top[base + a - 1U]) ++a;
            keep[k] = a;
            chosen[k].assign(top.begin() + static_cast<std::ptrdiff_t>(base),
                             top.begin() + static_cast<std::ptrdiff_t>(base + a));
        }
        base += rows;
    }
    backend_.commit(keep);
    for (auto& receipt : receipts)
        if (receipt) receipt->publish();  // the model prefix is committed: the selection state follows it
    const double decode_ms = now_ms() - t0;
    decode_ms_ += decode_ms;
    std::uint32_t total_rows = 0;
    for (const auto& g : groups) total_rows += g.rows;
    rows_ += total_rows;
    observe_cost(total_rows, static_cast<std::uint32_t>(n), decode_ms);
    planner_.account(prompt_ms, decode_ms, 0.0, prompt_rows);

    std::vector<std::pair<Lane*, std::string>> done;
    for (std::size_t k = 0; k < n; ++k) {
        Lane& l = *run[k];
        if (!refused[k].empty()) {
            done.emplace_back(&l, std::move(refused[k]));
            continue;
        }
        const std::span<const std::uint32_t> emitted(chosen[k]);
        if (l.drafter) {
            // the context drafter's own rows follow the device rows of the window
            const std::size_t off = device_rows[k], delivered_rows = groups[k].rows - 1U, hit_rows = keep[k] - 1U;
            l.drafter->finish_cycle(emitted, delivered_rows > off ? delivered_rows - off : 0U,
                                    hit_rows > off ? hit_rows - off : 0U);
        }
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
        if (deliver(l, emitted)) done.emplace_back(&l, std::string());
    }
    for (auto& [l, error] : done) finish(*l, std::move(error));
}

void CohortDecoder::run() {
    while (!idle()) step();
}

}  // namespace strix::serve
