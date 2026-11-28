// Shared multi-request decoder (strix/serve/cohort_decode.hpp).
#include "strix/serve/cohort_decode.hpp"
#include "strix/spec/draft_depth.hpp"
#include "strix/env_flags.hpp"
#include "strix/serve/job_sampling.hpp"
#include "strix/serve/lane_speculation.hpp"

#include "strix/serve/short_first.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace strix::serve {

std::span<const float> CohortBackend::read_final_logits(std::uint32_t) {
    throw std::logic_error("temperature: this cohort backend cannot read final logits");
}

std::uint32_t CohortBackend::sample_final(std::uint32_t, TokenSampler&, std::uint64_t) {
    throw std::logic_error("temperature: this cohort backend does not sample on the device");
}

std::span<const float> CohortBackend::read_row_logits(std::uint32_t, std::uint32_t) {
    throw std::logic_error("sampled drafts: this cohort backend cannot read window rows");
}

std::uint32_t CohortBackend::sample_row(std::uint32_t, std::uint32_t, TokenSampler&, std::uint64_t) {
    throw std::logic_error("sampled drafts: this cohort backend cannot sample window rows on the device");
}

void CohortBackend::sample_round(std::span<CohortSample> lanes) {
    for (auto& s : lanes) {
        if (s.sampler == nullptr) throw std::logic_error("cohort backend: a round sample without its sampler");
        s.token = sample_final(s.lane, *s.sampler, s.position);
    }
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

void CohortBackend::bind_lane(std::uint32_t, std::uint64_t, std::span<const std::uint32_t>) {}

void CohortBackend::control(std::span<const std::uint32_t>) {}

namespace {
// Cleanup while a contract violation (std::logic_error) is in flight: the lanes are retired either way, and a backend
// release failure inside the cleanup is dropped so that the contract error is the one rethrown.
template <class Cleanup>
void cleanup_keeping_contract_error(Cleanup&& cleanup) noexcept {
    try {
        cleanup();
    } catch (...) {
    }
}

// The most ids a request runs to: max_tokens, or the engine's context ceiling when that is lower.
std::uint64_t token_limit(const CohortRequest& r) {
    return r.context_tokens != 0 ? std::min(r.max_tokens, r.context_tokens) : r.max_tokens;
}
// The sampled id of a lane's final row: on the device when the backend samples there, else the host sampler over
// the row's final logits. Both record one decision in the sampler.
std::uint32_t sample_lane(CohortBackend& backend, std::uint32_t lane, TokenSampler& sampler, std::uint64_t position) {
    if (backend.samples_on_device()) return backend.sample_final(lane, sampler, position);
    return sampler.decide(backend.read_final_logits(lane), position, nullptr).token;
}
// The sampled id of row `row` of a lane's window (sampled drafts): decided as without a draft at its position.
std::uint32_t sample_window_row(CohortBackend& backend, std::uint32_t lane, std::uint32_t row, TokenSampler& sampler,
                                std::uint64_t position) {
    if (backend.samples_on_device()) return backend.sample_row(lane, row, sampler, position);
    return sampler.decide(backend.read_row_logits(lane, row), position, nullptr).token;
}
// The lane's counters of one request (CohortResult::lane_speculation): integers and booleans only.
struct LaneSpecCounts {
    std::uint64_t steps = 0, suspends = 0, resumes = 0;
};
std::string lane_speculation_receipt(const LaneSpeculator& sp, std::uint32_t lane, const LaneSpecCounts& counts) {
    const auto& drafter = sp.drafter();
    const auto& stats = drafter.stats();
    const auto& arbiter = drafter.arbiter();
    const auto u = [](std::uint64_t v) { return std::to_string(v); };
    const auto b = [](bool v) { return std::string(v ? "true" : "false"); };
    const auto source = [&](spec::Source s) {
        const auto k = static_cast<std::size_t>(s);
        return "{\"passes\":" + u(stats.source_passes[k]) + ",\"proposed\":" + u(stats.source_proposed[k]) +
               ",\"accepted\":" + u(stats.source_accepted[k]) + "}";
    };
    return "{\"lane\":" + u(lane) + ",\"steps\":" + u(counts.steps) + ",\"suspends\":" + u(counts.suspends) +
           ",\"resumes\":" + u(counts.resumes) + ",\"chained\":" + b(sp.chained()) + ",\"lookup_only\":" +
           b(sp.lookup_only()) + ",\"chain_starts\":" + u(sp.chain_starts()) + ",\"ready_false\":" +
           u(sp.ready_refusals()) + ",\"chain_null\":" + u(sp.chain_refusals()) + ",\"passes\":" + u(stats.passes) +
           ",\"emitted\":" + u(stats.emitted) + ",\"model\":" + source(spec::Source::model) + ",\"lookup\":" +
           source(spec::Source::lookup) + ",\"serial\":" + source(spec::Source::serial) + ",\"episode_open\":" +
           b(arbiter.episode_open()) + ",\"episode_closed_loss\":" + b(arbiter.episode_loss_ms() >= 0.0) + "}";
}
}  // namespace

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
    bool sampled_drafts = false;  // a sampled lane that takes device drafts (CohortConfig::sampled_drafts)
    // Lone-lane speculation (CohortConfig::lane_speculation): the lane's speculator, made at its first speculator step
    // and kept for the request (suspended while the lane takes the round), and the ids it emitted that the context
    // drafter has not seen yet (fed to it when the lane takes the round again), after the pending token spec_anchor.
    std::unique_ptr<LaneSpeculator> speculator;
    std::vector<std::uint32_t> spec_unseen;
    std::uint32_t spec_anchor = 0;
    bool spec_lookup_only = false;  // counted in lane_speculation_lookup_only
    std::uint64_t spec_steps = 0, spec_suspends = 0, spec_resumes = 0;  // this request's (CohortResult::lane_speculation)
    std::vector<double> tried, kept;  // kept histogram per draft row index: rows the target verified / kept
    std::uint64_t prompt_start = 0;   // prompt rows restored at admission (a group member starts from 0 only)
    bool decoded = false;             // a decode pass ran since the prompt finished (burst follow-up group rule)
    bool whole = false;               // the backend's prefill reads the whole prompt and emits the first id
    bool first_due = false;           // whole prompt restored: the first id is due from the restored last row
    // Bounded slices (prompt_slice_decoding): the rows [region_start, region_end) the unbounded schedule passes in one
    // call and the end of the backend chunk being read (prompt_read units).
    std::uint64_t region_start = 0, region_end = 0, chunk_end = 0;
    [[nodiscard]] std::uint64_t prompt_left() const {
        return request.prompt.size() - (whole ? 0U : 1U) - prompt_read;
    }
};

struct CohortDecoder::StepScratch {
    struct DecodeLane {
        std::vector<std::uint32_t> drafts, chosen;
        std::vector<float> accept;
        std::string refused;
        std::size_t device_rows = 0;  // leading device-draft rows (context rows follow them)
    };
    struct Receipts {
        std::vector<std::unique_ptr<MtpSelectionReceipt>>& storage;
        ~Receipts() { storage.clear(); }
    };

    explicit StepScratch(std::size_t lanes) : decode(lanes) {}

    std::vector<std::pair<Lane*, std::string>> done;
    std::vector<std::uint32_t> decode_ids;  // the lanes that take the round (not a speculator step)
    std::vector<StepLane> step_lanes;
    StepPlan plan;
    std::vector<std::uint64_t> group;
    std::vector<std::uint32_t> others;
    std::vector<GroupCandidate> candidates;
    std::vector<CohortPrompt> members;
    std::vector<Lane*> run;
    std::vector<DecodeLane> decode;
    std::vector<LaneDemand> demands;
    std::vector<std::uint32_t> device;
    std::vector<double> device_p, acc, device_acc;
    std::vector<std::uint32_t> rooms;       // each decoding lane's draft room this round
    std::vector<CohortProposal> proposals;  // the round's device proposal (CohortBackend::propose_round)
    std::vector<CohortSample> samples;      // the round's sampled lanes (CohortBackend::sample_round)
    std::vector<std::uint32_t> sampled;     // per decoding lane: index into samples + 1 (0 = not a round sample)
    RowPlanner row_planner;
    std::vector<std::unique_ptr<MtpSelectionReceipt>> receipts;
};

CohortDecoder::CohortDecoder(CohortBackend& backend, CohortConfig config) : backend_(backend), config_(std::move(config)) {
    // Every running lane holds at least one row, so a step never parks a lane: lanes <= rows of one forward.
    const std::uint32_t n = std::min(backend_.lanes(), backend_.rows_per_forward());
    if (n == 0) throw std::invalid_argument("cohort decoder: backend has no lanes");
    validate_prompt_pieces(config_, backend_.prompt_piece_rule(),
                           config_.prompt_slice_decoding != 0U ? backend_.prompt_call_chunk(config_.prompt_slice) : 0U);
    if (config_.lane_speculation.enabled) lane_spec_rows_ = validate_lane_speculation(backend_, config_.lane_speculation);
    lanes_.resize(n);
    step_scratch_ = std::make_unique<StepScratch>(n);
}

void validate_prompt_pieces(const CohortConfig& config, PromptPieceRule rule, std::uint64_t slice_first_chunk) {
    if (const std::uint32_t busy = config.prompt_slice_decoding; busy != 0U) {
        if (!rule.cuttable())
            throw std::invalid_argument("cohort decoder: bounded prompt slices need the backend's piece rule");
        if (busy >= config.prompt_slice)
            throw std::invalid_argument("cohort decoder: the decoding slice must be smaller than the prompt slice");
        if (std::uint64_t{busy} / rule.align * rule.align <= 2ULL * rule.short_rows + rule.align)
            throw std::invalid_argument("cohort decoder: the decoding slice is too short for the backend's piece rule");
        if (slice_first_chunk != config.prompt_slice)
            throw std::invalid_argument("cohort decoder: bounded prompt slices need the prompt slice to be one backend chunk");
    }
    if (!std::isfinite(config.prompt_gap_ms) || !std::isfinite(config.prompt_pass_fixed_ms) ||
        config.prompt_gap_ms < 0.0 || config.prompt_pass_fixed_ms < 0.0)
        throw std::invalid_argument("cohort decoder: the prompt gap and pass cost must be finite and nonnegative");
    if (!std::isfinite(config.prompt_share) || config.prompt_share < 0.0)
        throw std::invalid_argument("cohort decoder: the prompt share must be finite and nonnegative");
    if (config.prompt_gap_ms > 0.0 && config.prompt_slice_decoding == 0U)
        throw std::invalid_argument("cohort decoder: gap-bounded pieces need prompt_slice_decoding (the first piece)");
}

CohortDecoder::~CohortDecoder() = default;

void bind_mixed_service(CohortConfig& config, MixedServicePieces pieces, PromptPieceRule rule, std::uint64_t chunk) {
    config.prompt_slice_decoding = mixed_service_piece_rows(pieces, rule, chunk);
    if (config.prompt_slice_decoding != 0U) config.prompt_share = pieces.prompt_share;
    config.prompt_share = env_number<double>("STRIX_SERVE_PROMPT_SHARE", config.prompt_share, 0.0, 1.0e6);
    config.trace_steps = config.trace_steps || env_flag_default_off("STRIX_SERVE_STEP_TRACE");
}

std::uint64_t CohortDecoder::piece_rows(Lane& l, std::uint64_t want, std::uint32_t full_slice, bool bounded,
                                        bool priced, bool& over) {
    over = false;
    const std::uint64_t left = l.prompt_left();
    if (l.prompt_read >= l.region_end) {
        // A region starts: the rows the unbounded schedule passes in ONE call from here, and its first backend chunk.
        const std::uint64_t region = prompt_slice_rows(full_slice, full_slice, left, backend_.prompt_tail_rows());
        const std::uint64_t first = backend_.prompt_call_chunk(region);
        if (first == 0U || first > region) throw std::logic_error("cohort decoder: backend chunk outside its call");
        // piece_rows tracks at most two chunks per call (the slice and a remainder the backend runs on its own)
        if (const std::uint64_t second = region - first; second != 0U && backend_.prompt_call_chunk(second) != second)
            throw std::logic_error("cohort decoder: a prefill call of more than two backend chunks");
        l.region_start = l.prompt_read;
        l.region_end = l.prompt_read + region;
        l.chunk_end = l.prompt_read + first;
    } else if (l.prompt_read >= l.chunk_end) {
        l.chunk_end = l.region_end;  // the region's second chunk (a remainder the backend runs on its own)
    }
    if (priced) {
        // A priced grant (gap pieces beside a decoding lane) bounds the ACTUAL chunk: the largest bit-safe cut within
        // the grant, an absorbed tail included (64 + 8 under 64 -> 64, then 8); only when no cut fits, the smallest
        // bit-safe piece, recorded as an overshoot. The chunk's own end is a cut point, so a piece is ONE backend pass.
        const PromptPieceChoice c = prompt_piece_within(l.chunk_end - l.prompt_read, want, backend_.prompt_piece_rule());
        over = c.over;
        return c.rows;
    }
    if (l.prompt_read == l.region_start && want >= std::min<std::uint64_t>(left, full_slice))
        // the unbounded call itself; beside a decoding lane only its first chunk: the grant priced one backend pass,
        // and the remainder (a chunk the backend runs on its own) is the next call, after a decode round
        return (bounded ? l.chunk_end : l.region_end) - l.prompt_read;
    return prompt_piece_rows(l.chunk_end - l.prompt_read, want, full_slice, backend_.prompt_piece_rule());
}

double CohortDecoder::now_ms() {
    if (fake_clock_ms_ > 0) return fake_now_ += fake_clock_ms_;
    using Clock = std::chrono::steady_clock;
    static const auto origin = Clock::now();
    return std::chrono::duration<double, std::milli>(Clock::now() - origin).count();
}

std::uint64_t CohortDecoder::reserved_bytes(const CohortRequest& r) const {
    const std::uint64_t positions = r.prompt.size() + token_limit(r) + 1U +
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
    else if (request.context_tokens != 0 &&
             (request.prompt.size() > backend_.max_positions() ||  // the ceiling's last id is never a row
              request.context_tokens - 1U > backend_.max_positions() - request.prompt.size()))
        error = "context ceiling longer than a lane";
    else if (request.context_tokens == 0 &&
             (request.max_tokens > backend_.max_positions() ||
              request.prompt.size() > backend_.max_positions() - request.max_tokens)) error = "request longer than a lane";
    else if (request.sampling.temperature != 0.0F) {
        try {
            validate_token_sampling_policy(request.sampling);
            if (!backend_.supports_final_logits() && !backend_.samples_on_device())
                error = "temperature: this cohort backend cannot read final logits";
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

void CohortDecoder::control(std::span<const std::uint32_t> words) { backend_.control(words); }

std::uint32_t max_draft_for_live(const CohortConfig& config, std::uint32_t live) noexcept {
    return spec::draft_depth_for_lanes(config.max_draft_by_live, live, config.max_draft);
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
        lane->whole = backend_.prefill_emits_first();
        if (token_sampling_enabled(lane->request.sampling)) {
            // A stable request identity makes an explicit seed replayable across lane assignments (the request's
            // options carry it: the default identity, or the one the engine's own job sampler would have drawn).
            lane->sampler.emplace(lane->request.sampling, lane->request.sampler_options);
            lane->result.sampling = "sampling-v1";
            // coupled acceptance keeps the ids of the run without drafts; rejection acceptance keeps one row
            if (config_.sampled_drafts && lane->sampler->coupled() && backend_.reads_window_rows() &&
                backend_.device_drafts()) {
                lane->sampled_drafts = true;
                lane->tried.assign(config_.max_draft, 0.0);
                lane->kept.assign(config_.max_draft, 0.0);
            }
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
        used_bytes_ += lane->bytes;
        *free = std::move(lane);
        peak_live_ = std::max(peak_live_, live());
        // From here the lane is published: a backend or callback failure retires it as a failed request (finish:
        // budget returned, lane released) instead of leaving it half-initialised in its slot.
        try {
        // the engine's per-request words reach the backend (every rank under ReplicatedCohort) before any prompt row;
        // a request without traits makes no bind call, so plain requests make exactly the round-1 backend calls
        if (!l.request.traits.empty()) backend_.bind_lane(index, l.request.id, l.request.traits);
        if (l.request.reuse) {
            // A retained prefix replaces the first prompt rows; the last prompt token stays the first decode row.
            const auto& p = l.request.prompt;
            const std::uint64_t restored = backend_.restore(index, std::span<const std::uint32_t>(p.data(), p.size()));
            // a whole-prompt backend may restore the whole prompt: its first id then comes from the restored last row
            if (restored > p.size() - (l.whole ? 0U : 1U))
                throw std::logic_error("cohort decoder: restore returned more rows than the lane may restore");
            l.prompt_read = l.prompt_start = l.position = l.result.restored = restored;
            if (restored != 0 && l.request.prefilled) l.request.prefilled(restored);
            if (l.whole && l.prompt_left() == 0) l.first_due = true;
            else if (l.prompt_left() == 0) l.pending = p.back();
        }
        } catch (const std::logic_error& ex) {  // a contract violation stays fatal, after the lane is retired
            cleanup_keeping_contract_error([&] { finish(l, std::string("admit: ") + ex.what()); });
            throw;
        } catch (const std::exception& ex) {
            finish(l, std::string("admit: ") + ex.what());
        }
    }
}

void CohortDecoder::fail_lanes(std::span<Lane* const> lanes, const std::string& error) {
    std::exception_ptr release_error;
    for (Lane* lane : lanes) {
        try {
            finish(*lane, error);  // frees the lane and returns its budget before the backend release
        } catch (...) {
            if (!release_error) release_error = std::current_exception();
        }
    }
    if (release_error) std::rethrow_exception(release_error);
}

void CohortDecoder::finish(Lane& lane, std::string error) {
    lane.result.error = std::move(error);
    {
        const auto& r = lane.request;
        const auto& ids = lane.result.ids;
        if (!lane.result.error.empty()) lane.result.finish = "error";
        else if (lane.result.cancelled) lane.result.finish = "cancelled";
        else if (!ids.empty() && std::find(r.stop.begin(), r.stop.end(), ids.back()) != r.stop.end())
            lane.result.finish = "stop";
        else if (ids.size() >= r.max_tokens) lane.result.finish = "length";  // a tie with the ceiling is "length"
        else lane.result.finish = r.context_tokens != 0 && ids.size() >= r.context_tokens ? "context" : "length";
    }
    lane.result.done_ms = now_ms();
    if (lane.sampler) lane.result.sampling_receipt = job_sampling_receipt(*lane.sampler);
    for (std::size_t i = 0; i < lanes_.size(); ++i) {
        if (lanes_[i].get() != &lane) continue;
        if (lane.speculator && lane.speculator->started()) {
            lane.result.lane_speculation = lane_speculation_receipt(*lane.speculator, static_cast<std::uint32_t>(i),
                                                                    {lane.spec_steps, lane.spec_suspends, lane.spec_resumes});
            if (config_.lane_speculation.log) {
                std::string line = lane.result.lane_speculation;
                line.pop_back();  // the closing brace: the drafter's stats and the decoder's totals follow
                std::fprintf(stderr,
                             "strix-lane-spec %s,\"totals\":{\"steps\":%llu,\"suspends\":%llu,\"resumes\":%llu,"
                             "\"lookup_only_lanes\":%llu,\"ready_false\":%llu},\"stats\":%s}\n",
                             line.c_str(), static_cast<unsigned long long>(lane_spec_steps_),
                             static_cast<unsigned long long>(lane_spec_suspends_),
                             static_cast<unsigned long long>(lane_spec_resumes_),
                             static_cast<unsigned long long>(lane_spec_lookup_only_),
                             static_cast<unsigned long long>(lane_spec_ready_false_),
                             lane.speculator->drafter().stats().json().c_str());
                std::fflush(stderr);
            }
        }
        used_bytes_ -= lane.bytes;
        // prefix capture is optional: its failure must never keep the lane from retiring
        try {
        if (lane.request.retain && lane.result.error.empty() && !lane.result.cancelled && !lane.result.ids.empty()) {
            // committed inputs: the prompt (its last token was the first decode row), then every emitted id that a
            // later row consumed; ids cut at a stop are all consumed, so prompt ++ ids is cut at `position`.
            std::vector<std::uint32_t> committed = lane.request.prompt;
            committed.insert(committed.end(), lane.result.ids.begin(), lane.result.ids.end());
            committed.resize(static_cast<std::size_t>(std::min<std::uint64_t>(committed.size(), lane.position)));
            backend_.capture(static_cast<std::uint32_t>(i), committed);
        }
        } catch (const std::exception& ex) {
            std::cerr << "cohort decoder: prefix capture failed (" << ex.what() << "); lane retired without it\n";
        }
        finished_.push_back(std::move(lane.result));
        try {
            backend_.release(static_cast<std::uint32_t>(i));
        } catch (...) {
            lanes_[i].reset();
            throw;
        }
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
    auto& round = *step_scratch_;
    ++steps_;
    const double step_start_ms = config_.trace_steps ? now_ms() : 0.0;

    // Appends a lane's emitted ids through max_tokens / stop and publishes them; true = the request ends here.
    const auto deliver = [&](Lane& l, std::span<const std::uint32_t> emitted) {
        l.pending = emitted.back();
        if (l.result.ids.empty()) l.result.first_ms = now_ms();
        const auto before = l.result.ids.size();
        bool end = false;
        for (const auto id : emitted) {
            l.result.ids.push_back(id);
            if (l.result.ids.size() >= token_limit(l.request) ||
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
    auto& first_done = round.done;
    first_done.clear();
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
            id = sample_lane(backend_, index, *l.sampler, l.position);
        } else {
            id = backend_.prefill_top(index);
        }
        const std::uint32_t one[1] = {id};
        if (deliver(l, std::span<const std::uint32_t>(one))) first_done.emplace_back(&l, std::string());
    };
    // Whole prompts restored at admission: no prefill runs, the first id is due now (before the plan, so a lane that
    // ends on its first id never reaches the decode round).
    for (std::size_t i = 0; i < lanes_.size(); ++i) {
        if (!lanes_[i] || !lanes_[i]->first_due) continue;
        lanes_[i]->first_due = false;
        emit_first(static_cast<std::uint32_t>(i));
    }
    for (auto& [l, error] : first_done) finish(*l, std::move(error));
    first_done.clear();

    // 1. Prompt service: decode first, one bounded slice of the chosen prompt (short-first with aging, prompt debt).
    auto& step_lanes = round.step_lanes;
    step_lanes.clear();
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
    // Regime: with a decoding lane the prompt grant is the decoding slice (prompt_slice_decoding), else the full slice.
    const std::uint32_t full_slice = std::max(config_.prompt_slice, 1U);
    const bool bounded_regime = config_.prompt_slice_decoding != 0U &&
        std::any_of(step_lanes.begin(), step_lanes.end(), [](const StepLane& s) { return !s.prompt; });
    std::uint32_t busy = config_.prompt_slice_decoding;
    const bool gap_bounded = bounded_regime && config_.prompt_gap_ms > 0.0;
    // the rows one pass may hold under the target, unfloored: grouped company never takes a pass past it; a floored
    // grant above it is an overshoot, as is any dispatched pass priced above the target. Before a row cost is learned
    // the price of a pass is bounded below by its fixed cost and the last decode round (row cost 0): when that bound
    // alone exceeds the target no company joins and the head pass is recorded as an overshoot.
    const double row_ms_bound = last_row_ms_ > 0.0 ? last_row_ms_ : 0.0;
    std::uint64_t gap_room = std::numeric_limits<std::uint64_t>::max();
    if (gap_bounded) {
        busy = static_cast<std::uint32_t>(gap_bounded_rows(config_.prompt_gap_ms, config_.prompt_pass_fixed_ms,
                                                           last_row_ms_, last_decode_ms_, busy, full_slice));
        if (row_ms_bound > 0.0)
            gap_room = gap_bounded_rows(config_.prompt_gap_ms, config_.prompt_pass_fixed_ms, row_ms_bound,
                                        last_decode_ms_, 0U, full_slice);
        else if (gap_pass_ms(config_.prompt_pass_fixed_ms, 0.0, last_decode_ms_, 0U) > config_.prompt_gap_ms)
            gap_room = 0U;
    }
    const auto account_gap = [&](std::uint64_t rows, bool over) {  // the ONE overshoot record of both dispatch paths
        if (!gap_bounded) return;
        const double predicted = gap_pass_ms(config_.prompt_pass_fixed_ms, row_ms_bound, last_decode_ms_, rows);
        if (over || predicted > config_.prompt_gap_ms) {
            ++gap_overshoots_;
            gap_overshoot_ms_ = std::max(gap_overshoot_ms_, predicted - config_.prompt_gap_ms);
        }
    };
    const std::uint32_t units[1] = {bounded_regime ? busy : full_slice};
    StepLimits limits;
    limits.prompt_units = units;
    limits.prompt_share = config_.prompt_share;
    limits.short_first_rows = config_.short_first_rows;
    auto& plan = round.plan;
    planner_.plan(step_lanes, limits, plan);
    for (const auto l : plan.overtaken) ++lanes_[l]->overtakes;
    double prompt_ms = 0;
    std::uint64_t prompt_rows = 0;
    // Grouped prompts: when the planner's head is a whole short prompt, other whole short prompts that arrived after
    // it may share its pass (plan_prefill_group: arrival order, one route class, at most one slice of rows).
    auto& group = round.group;
    group.clear();
    // Burst follow-up: lanes that hold only their prefill's first id (no decode pass ran yet) do not make the burst wait on
    // the priced gap or the decoders' debt: with a multi-pass group capacity the waiting whole prompts of the burst run as
    // one group work, so all lanes then decode together. Once any lane has run a decode pass the plan is the planner's.
    const bool lane_decoding = std::any_of(step_lanes.begin(), step_lanes.end(), [](const StepLane& entry) { return !entry.prompt; });
    const bool burst_follow = lane_decoding && std::all_of(step_lanes.begin(), step_lanes.end(), [&](const StepLane& entry) {
        return entry.prompt || !lanes_[entry.lane]->decoded;
    });
    bool injected = false;  // the planner held the prompts for the decoders' debt: the oldest waiting prompt heads the group
    if (burst_follow && plan.prompt.empty() && config_.prompt_group.groups() && backend_.prompt_group_capacity().passes > 1U) {
        std::int64_t oldest = -1;
        for (std::size_t i = 0; i < lanes_.size(); ++i) {
            const auto& c = lanes_[i];
            if (c && c->prompt_start == 0 && c->prompt_read == 0 && c->prompt_left() > 0 &&
                (oldest < 0 || c->order < lanes_[static_cast<std::size_t>(oldest)]->order))
                oldest = static_cast<std::int64_t>(i);
        }
        if (oldest >= 0) {
            plan.prompt.push_back({static_cast<std::uint32_t>(oldest), 0U});
            injected = true;
        }
    }
    if (!plan.prompt.empty() && config_.prompt_group.groups()) {
        const std::uint32_t head = plan.prompt.front().lane;
        const Lane& h = *lanes_[head];
        // the planner runs the head's whole prompt this step (its entry = the rows it gives the head, 0 = one unit);
        // the group may fill the rest of one slice unit
        const PrefillGroupCapacity capacity = backend_.prompt_group_capacity();
        const bool wide = capacity.passes > 1U && (!lane_decoding || burst_follow);
        const std::uint64_t head_rows = burst_follow && wide ? full_slice
            : plan.prompt.front().rows ? plan.prompt.front().rows : units[0];
        // company fills at most one unit, and under a priced gap never past the unfloored room (the head itself runs
        // even when it alone exceeds it: a floor overshoot, recorded below)
        // (a backend that runs the group as several sub-passes, capacity.passes, offers more rows while no lane decodes: the
        // whole burst of waiting prompts then starts decoding together)
        const std::uint64_t passes = wide ? capacity.passes : 1U;
        const std::uint64_t want = burst_follow && wide ? std::uint64_t{full_slice} * passes
                                                         : std::min<std::uint64_t>(units[0] * passes, gap_room);
        if (capacity.members >= 2U && h.prompt_start == 0 && h.prompt_read == 0 && h.prompt_left() <= head_rows) {
            auto& others = round.others;
            others.clear();
            for (std::size_t i = 0; i < lanes_.size(); ++i) {
                const auto& c = lanes_[i];
                // a multi-pass group also takes waiting prompts older than the head (the planner's short-first pick):
                // they are served earlier, never later, than they would be alone
                if (c && i != head && c->prompt_start == 0 && c->prompt_read == 0 && c->prompt_left() > 0 &&
                    (passes > 1U || c->order > h.order))
                    others.push_back(static_cast<std::uint32_t>(i));
            }
            std::sort(others.begin(), others.end(), [&](auto a, auto b) { return lanes_[a]->order < lanes_[b]->order; });
            auto& candidates = round.candidates;
            candidates.clear();
            std::uint64_t offered = 0;
            const auto offer = [&](std::uint32_t i) {
                const std::uint64_t rows = lanes_[i]->prompt_left();
                GroupCandidate c;
                c.id = i;
                c.rows = static_cast<std::uint32_t>(std::min<std::uint64_t>(rows, std::numeric_limits<std::uint32_t>::max()));
                c.arrival_seq = passes > 1U ? offered++ : lanes_[i]->order;  // the head first, then oldest first
                c.route_class = backend_.prompt_route_class(rows);
                candidates.push_back(c);
            };
            offer(head);
            for (const auto i : others) offer(i);
            PrefillGroupPolicy policy = config_.prompt_group;
            policy.max_rows = static_cast<std::uint32_t>(std::min<std::uint64_t>(policy.max_rows, want));
            plan_prefill_group(candidates, policy, PrefillGroupCost{}, std::numeric_limits<double>::infinity(),
                               capacity, group);
        }
    }
    if (injected && group.size() <= 1U) plan.prompt.clear();  // no company: the planner's hold stands
    // A backend failure of a prompt pass fails the lanes of that pass, as a failed verify does, and ends this step (no
    // first id is pending yet); a contract violation (std::logic_error) stays fatal after the cleanup.
    const auto fail_prompt = [&](auto indices, const std::exception& ex) {
        auto& failed = round.run;
        failed.clear();
        for (const auto i : indices) failed.push_back(lanes_[static_cast<std::size_t>(i)].get());
        fail_lanes(failed, std::string("prefill: ") + ex.what());
    };
    if (group.size() > 1) {
        auto& members = round.members;
        members.clear();
        for (const auto i : group) {
            const Lane& m = *lanes_[static_cast<std::size_t>(i)];
            members.push_back({static_cast<std::uint32_t>(i),
                               std::span<const std::uint32_t>(m.request.prompt.data(), m.prompt_left()), true});
            prompt_rows += m.prompt_left();
        }
        account_gap(prompt_rows, false);
        prompt_rows = 0;
        const double t0 = now_ms();
        try {
            backend_.prefill_group(members);
        } catch (const std::exception& ex) {
            if (dynamic_cast<const std::logic_error*>(&ex) != nullptr) {
                cleanup_keeping_contract_error([&] { fail_prompt(std::span<const std::uint64_t>(group), ex); });
                throw;
            }
            fail_prompt(std::span<const std::uint64_t>(group), ex);
            return;
        }
        prompt_ms = now_ms() - t0;
        prompt_ms_ += prompt_ms;
        ++prompt_groups_;
        grouped_prompts_ += group.size();
        for (const auto i : group) {
            Lane& m = *lanes_[static_cast<std::size_t>(i)];
            const std::uint64_t rows = m.prompt_left();
            prompt_rows += rows;
            m.prompt_read += rows;
            m.position += rows;
            if (m.request.prefilled) m.request.prefilled(m.prompt_read);
        }
        for (const auto i : group) emit_first(static_cast<std::uint32_t>(i));
    } else if (!plan.prompt.empty()) {
        Lane& l = *lanes_[plan.prompt.front().lane];
        const std::uint64_t want = plan.prompt.front().rows ? plan.prompt.front().rows : units[0];
        bool over = false;  // no bit-safe cut fits the priced grant: the smallest bit-safe piece runs
        prompt_rows = config_.prompt_slice_decoding == 0U
            ? prompt_slice_rows(want, units[0], l.prompt_left(), backend_.prompt_tail_rows())
            : piece_rows(l, want, full_slice, bounded_regime, gap_bounded, over);
        account_gap(prompt_rows, over);  // a piece the target cannot hold (the floor, a piece above the grant)
        // a call of ONE backend pass (a chunk or a piece of one) prices the next piece; a region of two chunks (alone,
        // 8192 + 16) carries two fixed costs and is not a sample
        const bool one_pass = config_.prompt_slice_decoding == 0U || l.prompt_read + prompt_rows <= l.chunk_end;
        const double t0 = now_ms();
        try {
            backend_.prefill_scoped(plan.prompt.front().lane,
                                    std::span<const std::uint32_t>(l.request.prompt.data() + l.prompt_read, prompt_rows),
                                    l.prompt_read == 0, l.request.prompt.size());
        } catch (const std::exception& ex) {
            const std::uint32_t one[1] = {plan.prompt.front().lane};
            if (dynamic_cast<const std::logic_error*>(&ex) != nullptr) {
                cleanup_keeping_contract_error([&] { fail_prompt(std::span<const std::uint32_t>(one), ex); });
                throw;
            }
            fail_prompt(std::span<const std::uint32_t>(one), ex);
            return;
        }
        prompt_ms = now_ms() - t0;
        prompt_ms_ += prompt_ms;
        // The gap's row cost is learned from passes of at least the decoding floor only: a shorter pass (a lone short
        // prompt, a remainder) is dominated by the fixed cost it is priced with and under-prices the next piece (gfx1151:
        // a ~280-row lane prompt priced the first piece of an 8K prompt at ~2 s against a 1000 ms target).
        if (one_pass && prompt_rows != 0U && prompt_rows >= std::max<std::uint64_t>(config_.prompt_slice_decoding, 1U))
            last_row_ms_ = std::max(0.0, prompt_ms - config_.prompt_pass_fixed_ms) / static_cast<double>(prompt_rows);
        l.prompt_read += prompt_rows;
        l.position += prompt_rows;
        if (l.request.prefilled) l.request.prefilled(l.prompt_read);
        emit_first(plan.prompt.front().lane);
    }
    for (auto& [l, error] : first_done) finish(*l, std::move(error));

    // 2. Lone-lane speculation (CohortConfig::lane_speculation): while at most max_live requests are live, a greedy
    // decoding lane runs one speculator step (one verify + commit of its own) instead of the round; the ids go through
    // the round's delivery. A lane that takes the round again suspends its speculator (kept: learned walls and
    // acceptance) and its context drafter first reads the ids the speculator emitted. Ends are applied after the round.
    auto& done = round.done;  // speculator ends first, then the round's (applied after the round)
    done.clear();
    auto& decode_ids = round.decode_ids;  // the lanes of the round
    decode_ids.clear();
    double spec_ms = 0;
    std::size_t spec_lanes = 0;
    const bool speculate = config_.lane_speculation.enabled && live() <= config_.lane_speculation.max_live;
    for (const auto id : plan.decode_lanes) {
        Lane& l = *lanes_[id];
        l.decoded = true;  // a decode pass (round or speculator step) runs for it in this step
        if (speculate && !l.sampler && !l.request.selection) {
            const double s0 = now_ms();
            auto& sp = l.speculator;
            if (!sp) sp = std::make_unique<LaneSpeculator>(backend_, id, config_.lane_speculation, lane_spec_rows_);
            if (!sp->active()) {
                // the stream is the prompt then the emitted ids; its last token is the pending one (not committed)
                std::vector<std::uint32_t> stream = l.request.prompt;
                stream.insert(stream.end(), l.result.ids.begin(), l.result.ids.end());
                if (stream.size() != l.position + 1U || stream.back() != l.pending)
                    throw std::logic_error("cohort decoder: lane speculation out of step with the lane");
                const bool resume = sp->started();
                const std::uint64_t refused = sp->ready_refusals();
                sp->start(l.pending, stream);
                lane_spec_ready_false_ += sp->ready_refusals() - refused;
                if (resume) {
                    ++lane_spec_resumes_;
                    ++l.spec_resumes;
                }
                if (sp->lookup_only() && !l.spec_lookup_only) {
                    l.spec_lookup_only = true;
                    ++lane_spec_lookup_only_;
                }
            }
            const std::uint64_t left = token_limit(l.request) - l.result.ids.size();
            const std::uint64_t room_pos = backend_.max_positions() - l.position;  // rows position .. position + R - 1
            const std::uint64_t window = std::max<std::uint32_t>(backend_.max_window(id), 1U);
            const auto room = static_cast<std::uint32_t>(
                std::min<std::uint64_t>({left, room_pos, std::uint64_t{lane_spec_rows_}, window}));
            spec::StepParams params;
            params.remaining = params.max_rows = room;
            params.context = l.position;
            if (l.request.stop.size() == 1U) params.eos = l.request.stop.front();
            std::vector<std::uint32_t> out;
            const spec::StepRecord record = sp->step(out, params);
            if (record.emitted == 0U || out.empty() || out.size() > room)
                throw std::logic_error("cohort decoder: lane speculation produced an invalid window");
            ++lane_spec_steps_;
            ++l.spec_steps;
            rows_ += record.rows;
            l.result.steps += 1;
            l.result.drafted += record.rows - 1U;
            l.result.accepted += record.accepted;
            l.position += out.size();
            if (l.drafter) {
                if (l.spec_unseen.empty()) l.spec_anchor = l.pending;
                l.spec_unseen.insert(l.spec_unseen.end(), out.begin(), out.end());
            }
            if (deliver(l, out)) done.emplace_back(&l, std::string());
            spec_ms += now_ms() - s0;
            ++spec_lanes;
            continue;
        }
        if (l.speculator && l.speculator->active()) {
            l.speculator->suspend();
            ++lane_spec_suspends_;
            ++l.spec_suspends;
        }
        if (!l.spec_unseen.empty()) {
            if (l.drafter) {
                // one observe-only cycle: the context drafter's context follows the lane (pending, then the ids)
                std::vector<double> unused;
                (void)l.drafter->begin_cycle(l.spec_anchor, std::span<const std::uint32_t>(), 0U, 0U, unused,
                                             config_.min_p);
                l.drafter->finish_cycle(l.spec_unseen);
            }
            l.spec_unseen.clear();
        }
        decode_ids.push_back(id);
    }
    if (spec_lanes != 0U) decode_ms_ += spec_ms;

    // 3. Decode round: every decoding lane holds its certain row; draft rows are bought by plan_rows.
    const double t0 = now_ms();
    auto& run = round.run;
    run.clear();
    for (const auto id : decode_ids) run.push_back(lanes_[id].get());
    // trace_steps: one line per step, after its accounting (debt_ms is the decode still owed after this step).
    const auto trace = [&](std::size_t decode_lanes, double decode_ms) {
        if (!config_.trace_steps) return;
        const double end = now_ms();
        std::size_t waiting = 0;
        for (const auto& s : step_lanes) waiting += s.prompt ? 1U : 0U;
        std::fprintf(stderr,
                     "cohort-trace step %llu start_ms %.3f idle_ms %.3f prompt_rows %llu prompt_ms %.3f decode_lanes %zu "
                     "decode_ms %.3f step_ms %.3f prompts_waiting %zu debt_ms %.3f\n",
                     static_cast<unsigned long long>(steps_), step_start_ms,
                     last_step_end_ms_ > 0.0 ? step_start_ms - last_step_end_ms_ : 0.0,
                     static_cast<unsigned long long>(prompt_rows), prompt_ms, decode_lanes, decode_ms,
                     end - step_start_ms, waiting, planner_.debt_ms());
        last_step_end_ms_ = end;
    };
    if (run.empty()) {
        planner_.account(prompt_ms, spec_ms, 0.0, prompt_rows);
        trace(spec_lanes, spec_ms);
        for (auto& [l, error] : done) finish(*l, std::move(error));
        return;
    }
    const std::size_t n = run.size();
    auto& demands = round.demands;
    demands.resize(n);
    const auto backend_rows = backend_.rows_per_forward();
    const auto batch_rows = config_.uniform_verify_windows
        ? std::min(backend_rows, backend_.max_batch_verify_rows()) : 0U;
    const bool uniform = batch_rows >= n;
    auto draft_cap = max_draft_for_live(config_, live());
    if (uniform) draft_cap = std::min(draft_cap, batch_rows / static_cast<std::uint32_t>(n) - 1U);
    auto& device = round.device;
    auto& device_p = round.device_p;
    auto& acc = round.acc;
    auto& device_acc = round.device_acc;
    // Each lane's draft room, first for every lane: a drafter that serves the round's lanes in one pass gets them all
    // at once (CohortBackend::propose_round) before the per-lane proposals below.
    auto& rooms = round.rooms;
    auto& proposals = round.proposals;
    rooms.resize(n);
    proposals.clear();
    const bool drafts_on_device = backend_.device_drafts();
    for (std::size_t k = 0; k < n; ++k) {
        const Lane& l = *run[k];
        const std::uint64_t left = token_limit(l.request) - l.result.ids.size();
        const std::uint64_t room_pos = backend_.max_positions() - l.position - 1U;
        // a selection that reads logits decides one row per step from the lane's final logits
        const bool one_row = l.request.selection && l.request.selection_reads_logits;
        // the backend's own cap on the lane's window (pending row + drafts)
        const std::uint64_t window_cap = std::max<std::uint32_t>(backend_.max_window(decode_ids[k]), 1U) - 1U;
        rooms[k] = one_row ? 0U : static_cast<std::uint32_t>(std::min<std::uint64_t>(
                                      {draft_cap, left - 1U, room_pos, window_cap}));
        if (drafts_on_device && (!l.sampler || l.sampled_drafts) && rooms[k] != 0U)
            proposals.push_back({decode_ids[k], rooms[k]});
    }
    if (!proposals.empty()) backend_.propose_round(proposals);
    for (std::size_t k = 0; k < n; ++k) {
        Lane& l = *run[k];
        auto& temp = round.decode[k];
        auto& drafts = temp.drafts;
        auto& accept = temp.accept;
        auto& device_rows = temp.device_rows;
        drafts.clear();
        accept.clear();
        device_rows = 0;
        const std::uint64_t left = token_limit(l.request) - l.result.ids.size();
        const std::size_t room = rooms[k];
        acc.clear();
        device.clear();
        device_p.clear();
        if (drafts_on_device && (!l.sampler || l.sampled_drafts) && room != 0) {
            backend_.propose(decode_ids[k], static_cast<std::uint32_t>(room), device, device_p);
            if (device.size() > room) device.resize(room);
        }
        device_acc.resize(device.size());
        double device_expected = 0.0, chain_p = 1.0;
        for (std::size_t d = 0; d < device.size(); ++d) {
            const double p = d < device_p.size() && std::isfinite(device_p[d]) ? std::clamp(device_p[d], 0.0, 1.0)
                                                                               : config_.accept_prior;
            device_acc[d] = p;
            device_expected += chain_p *= p;
        }
        const auto* probabilities = &acc;
        if (l.drafter) {
            // room 0 still observes the pending token once (the drafter's context follows the lane). A device proposal
            // is the confident prefix: the context drafter continues it, or replaces it with a long exact copy span
            // expected to deliver more rows.
            const auto replaced = l.drafter->stats().replaced_cycles;
            const auto context = l.drafter->begin_cycle_view(l.pending, device, device.size(), room, acc, config_.min_p,
                                                            device.empty() ? -1.0 : device_expected);
            if (device.empty() || l.drafter->stats().replaced_cycles != replaced) {
                drafts.assign(context.begin(), context.end());
            } else {
                drafts = device;
                drafts.insert(drafts.end(), context.begin(), context.end());
                device_acc.insert(device_acc.end(), acc.begin(), acc.end());
                probabilities = &device_acc;
                device_rows = device.size();
            }
        } else if (!device.empty()) {
            drafts = device;
            probabilities = &device_acc;
            device_rows = device.size();
        }
        if (drafts.size() > room) drafts.resize(room);
        device_rows = std::min(device_rows, drafts.size());
        accept.push_back(1.0F);
        const double w = config_.accept_weight;
        for (std::size_t d = 0; d < drafts.size(); ++d) {
            const double claim = d < probabilities->size() ? (*probabilities)[d] : config_.accept_prior;
            const double p = w > 0.0 && d < l.tried.size() ? (l.kept[d] + w * claim) / (l.tried[d] + w) : claim;
            accept.push_back(static_cast<float>(p));
        }
        // drain rate: tokens per step the lane realized so far, this step's full offer counted as one more step
        // (a lane that is not paced realizes what it is offered, so the slowest lane's rate is its potential)
        double chain = 1.0, expected = 1.0;  // the pending row is certain
        for (std::size_t d = 1; d < accept.size(); ++d) expected += chain *= static_cast<double>(accept[d]);
        demands[k].rate = (static_cast<double>(l.result.ids.size()) + expected) / static_cast<double>(l.result.steps + 1U);
        demands[k].lane = decode_ids[k];
        demands[k].remaining_tokens = static_cast<std::uint32_t>(left);
        demands[k].min_rows = 1;
        demands[k].max_rows = 1U + static_cast<std::uint32_t>(drafts.size());
        demands[k].accept = accept;
        demands[k].waited_ms = l.waited_ms;
    }
    RowLimits row_limits;
    row_limits.rows_per_forward = uniform ? batch_rows : backend_rows;
    row_limits.uniform_windows = uniform;
    row_limits.drain = waiting_.empty();  // nobody waits for a lane: a lane that ends early frees nothing
    const RowPlan& rp = round.row_planner.plan(demands, fitted_cost(sums_, config_, static_cast<std::uint32_t>(n)),
                                              row_limits, config_.accept_prior);

    auto scratch = std::exchange(verification_, {});
    auto& [groups, tokens, top, keep] = scratch;
    groups.resize(n);
    tokens.clear();
    for (std::size_t k = 0; k < n; ++k) {
        const std::uint32_t rows = std::min(rp.rows.at(k), demands[k].max_rows);
        if (rows == 0) throw std::logic_error("cohort decoder: a running lane was parked (lanes > rows of a forward)");
        // a lane has no score consumer only when it is greedy: no sampler (host or device sampling both read the
        // final scores) and no selection that reads logits
        const Lane& l = *run[k];
        const bool reads_scores = l.request.selection ? l.request.selection_reads_logits : l.sampler.has_value();
        groups[k] = {demands[k].lane, rows, reads_scores};
        tokens.push_back(run[k]->pending);
        const auto& drafts = round.decode[k].drafts;
        tokens.insert(tokens.end(), drafts.begin(), drafts.begin() + (rows - 1U));
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
    top.assign(tokens.size(), 0U);
    // A backend failure of the verify or commit fails every lane of this forward (selections discarded, lanes
    // released, budget returned) and leaves the decoder usable; the next step starts clean. Every lane is retired even
    // when a backend release throws (each finish() frees its lane first); the first such error is rethrown once.
    const auto fail_forward = [&](const char* what, const std::exception& ex) {
        for (std::size_t k = 0; k < n; ++k)
            if (auto& selection = run[k]->request.selection) selection->discard();
        verification_ = std::move(scratch);
        fail_lanes(std::span<Lane* const>(run.data(), n), std::string(what) + ": " + ex.what());
        // Lanes that already finished in this step's speculator pass are not in `run`: finish them as well.
        for (auto& [l, error] : done) finish(*l, std::move(error));
        done.clear();
    };
    try {
        backend_.verify(groups, tokens, top);
    } catch (const std::logic_error& ex) {  // contract violations stay fatal after the cleanup
        cleanup_keeping_contract_error([&] { fail_forward("verify", ex); });
        throw;
    } catch (const std::exception& ex) {
        fail_forward("verify", ex);
        return;
    }
    keep.resize(n);
    // Round sampling: a backend that samples on the device may decide every sampled lane of the step in one call.
    auto& samples = round.samples;
    auto& sampled = round.sampled;
    samples.clear();
    sampled.assign(n, 0U);
    if (backend_.samples_on_device() && backend_.samples_in_round()) {
        for (std::size_t k = 0; k < n; ++k) {
            Lane& l = *run[k];
            if (l.request.selection || !l.sampler || groups[k].rows != 1U) continue;  // a drafted window: by row
            samples.push_back({groups[k].lane, &*l.sampler, l.position + 1U, 0U});
            sampled[k] = static_cast<std::uint32_t>(samples.size());
        }
        if (!samples.empty()) backend_.sample_round(samples);
    }
    auto& receipts = round.receipts;
    receipts.resize(n);
    const StepScratch::Receipts clear_receipts{receipts};
    base = 0;
    for (std::size_t k = 0; k < n; ++k) {
        Lane& l = *run[k];
        const std::uint32_t rows = groups[k].rows;
        auto& picked = round.decode[k];
        auto& chosen = picked.chosen;  // the ids each lane emits this step
        picked.refused.clear();
        if (auto& selection = l.request.selection) {
            try {
                if (l.request.selection_reads_logits) {
                    chosen = {selection->select(backend_.read_final_logits(groups[k].lane))};
                    receipts[k] = selection->seal();
                } else {
                    auto window = select_greedy_window(*selection,
                                                       std::span<const std::uint32_t>(tokens.data() + base + 1U, rows - 1U),
                                                       std::span<const std::uint32_t>(top.data() + base, rows));
                    chosen = std::move(window.ids);
                    receipts[k] = std::move(window.receipt);
                }
            } catch (const MtpSelectionError& error) {
                selection->discard();
                picked.refused = std::string("selection: ") + error.what();
                chosen.clear();
            }
            keep[k] = std::max<std::uint32_t>(1U, static_cast<std::uint32_t>(chosen.size()));
        } else {
            std::uint32_t a = 1;  // the pending row is always kept; each draft row equal to the target's choice is too
            if (auto& sampler = l.sampler) {
                if (rows == 1U) {
                    top[base] = sampled[k] != 0U ? samples[sampled[k] - 1U].token
                                                 : sample_lane(backend_, groups[k].lane, *sampler, l.position + 1U);
                } else {
                    // Sampled drafts (coupled): row r is decided as without a draft at its position, so a draft row is
                    // kept while it equals the id before it; rows after a rejection or a stop id are not decided.
                    a = 0;
                    do {
                        top[base + a] = sample_window_row(backend_, groups[k].lane, a, *sampler, l.position + 1U + a);
                        ++a;
                    } while (a < rows && tokens[base + a] == top[base + a - 1U] &&
                             std::ranges::find(l.request.stop, top[base + a - 1U]) == l.request.stop.end());
                }
            } else {
                while (a < rows && tokens[base + a] == top[base + a - 1U]) ++a;
            }
            keep[k] = a;
            chosen.assign(top.begin() + static_cast<std::ptrdiff_t>(base),
                          top.begin() + static_cast<std::ptrdiff_t>(base + a));
        }
        base += rows;
    }
    try {
        backend_.commit(keep);
    } catch (const std::logic_error& ex) {  // contract violations stay fatal after the cleanup
        cleanup_keeping_contract_error([&] { fail_forward("commit", ex); });
        throw;
    } catch (const std::exception& ex) {
        fail_forward("commit", ex);
        return;
    }
    for (auto& receipt : receipts)
        if (receipt) receipt->publish();  // the model prefix is committed: the selection state follows it
    const double decode_ms = now_ms() - t0;
    decode_ms_ += decode_ms;
    std::uint32_t total_rows = 0;
    for (const auto& g : groups) total_rows += g.rows;
    rows_ += total_rows;
    observe_cost(total_rows, static_cast<std::uint32_t>(n), decode_ms);
    // last_decode_ms_ prices the decode round a prompt pass waits beside (the gap bound). A round that carried draft
    // rows is priced without them (the fitted row cost per draft row): a prompt arriving raises the live lanes and the
    // next round runs under the by-live draft cap, so a lone lane's long window must not shrink the next prompt's
    // grant. A round without drafts is recorded as measured, as in a cohort that never drafts.
    const std::uint32_t draft_rows = total_rows - static_cast<std::uint32_t>(n);
    last_decode_ms_ = draft_rows == 0U
        ? decode_ms
        : std::max(0.0, decode_ms - static_cast<double>(draft_rows) *
                                        fitted_cost(sums_, config_, static_cast<std::uint32_t>(n)).row_ms);
    planner_.account(prompt_ms, decode_ms + spec_ms, 0.0, prompt_rows);
    trace(n + spec_lanes, decode_ms + spec_ms);

    for (std::size_t k = 0; k < n; ++k) {
        Lane& l = *run[k];
        auto& picked = round.decode[k];
        if (!picked.refused.empty()) {
            done.emplace_back(&l, std::move(picked.refused));
            continue;
        }
        const std::span<const std::uint32_t> emitted(picked.chosen);
        if (l.drafter) {
            // the context drafter's own rows follow the device rows of the window
            const std::size_t off = picked.device_rows, delivered_rows = groups[k].rows - 1U, hit_rows = keep[k] - 1U;
            l.drafter->finish_cycle(emitted, delivered_rows > off ? delivered_rows - off : 0U,
                                    hit_rows > off ? hit_rows - off : 0U);
        }
        // kept histogram: verified draft rows up to the first rejection; an offer the plan did not buy still tells
        // whether its first row was the target's next token
        const std::uint32_t delivered = groups[k].rows - 1U, hits = keep[k] - 1U;
        if (!l.tried.empty()) {
            for (std::uint32_t r = 0; r < hits; ++r) l.tried[r] += 1.0, l.kept[r] += 1.0;
            if (hits < delivered) l.tried[hits] += 1.0;
            if (delivered == 0 && !picked.drafts.empty()) {
                l.tried[0] += 1.0;
                if (picked.drafts[0] == emitted[0]) l.kept[0] += 1.0;
            }
        }
        l.result.steps += 1;
        l.result.drafted += groups[k].rows - 1U;
        l.result.accepted += keep[k] - 1U;
        l.position += keep[k];
        if (deliver(l, emitted)) done.emplace_back(&l, std::string());
    }
    verification_ = std::move(scratch);
    for (auto& [l, error] : done) finish(*l, std::move(error));
}

void CohortDecoder::run() {
    while (!idle()) step();
}

}  // namespace strix::serve
