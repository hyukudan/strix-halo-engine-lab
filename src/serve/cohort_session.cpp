#include "strix/serve/cohort_session.hpp"
#include "strix/serve/job_sampling.hpp"

#include <algorithm>

namespace strix::serve {
namespace {

session::Hooks request_lease_hooks() {
    session::Hooks hooks;
    hooks.initialize = hooks.reset = hooks.release = [](session::KvHandle) {};
    hooks.view = hooks.capture = [](session::KvHandle, std::uint32_t) -> session::RankState {
        throw std::logic_error("cohort physical KV is owned by the backend");
    };
    hooks.restore = [](session::KvHandle, std::uint32_t, const session::RankState&) {
        throw std::logic_error("cohort restore is unavailable");
    };
    hooks.truncate = [](session::KvHandle, std::uint32_t, std::uint64_t) {
        throw std::logic_error("cohort truncate is unavailable");
    };
    return hooks;
}

} // namespace

CohortSession::CohortSession(GenerationQueue& queue, CohortBackend& backend, CohortConfig config,
                             std::function<void(const GenerationJob&)> validate)
    : backend_(backend), slots_(std::min(backend.lanes(), backend.rows_per_forward())),
      cohort_(backend, std::move(config)), store_(slots_, 1U, request_lease_hooks()),
      runtime_(queue, *this, store_, {.enabled = false, .store = false}, [&] {
          SessionHooks hooks;
          hooks.disable_native_cache = true;
          hooks.control_turn = [this](SessionRuntime& owner) { turn(owner); };
          hooks.validate = std::move(validate);
          return hooks;
      }()) {}

session::Capabilities CohortSession::caps() const {
    session::Capabilities capability;
    capability.max_context = backend_.max_positions();
    capability.max_slots = capability.max_batch = slots_;
    capability.batch_decode = slots_ > 1U;
    const bool samples = backend_.supports_final_logits();
    capability.temperature = capability.top_k = capability.top_p = capability.min_p = capability.seed = samples;
    capability.owner_control = true;
    return capability;
}

void CohortSession::configure_api(ApiConfig& config) const {
    runtime_.configure_api(config);
    config.http_slots = slots_;
    const auto sampling = config.capabilities_extra;
    config.capabilities_extra = [sampling, slots = slots_](Json& value) {
        sampling(value);
        value.set("batch_decode", Json(slots > 1U));
        value.set("decode_slots", Json(static_cast<std::uint64_t>(slots)));
    };
}

void CohortSession::prefill(std::span<const std::uint32_t>, session::KvHandle&) {
    throw std::logic_error("cohort session requires its shared owner");
}

session::StepResult CohortSession::decode(session::KvHandle&, const session::SamplerState&) {
    throw std::logic_error("cohort session requires its shared owner");
}

void CohortSession::turn(SessionRuntime& owner) {
    bool wait = cohort_.idle();
    while (owner.live() < slots_ && owner.reserve_fifo(wait)) {
        wait = false;
        const auto& job = owner.reservation()->job;
        CohortRequest request{++next_request_, job.prompt, job.max_tokens, job.stop_tokens};
        request.sampling = job_sampling_policy(job.sampling);
        session::KvHandle handle;
        try {
            if (job.sampling.temperature > 0.0 && !token_sampling_enabled(request.sampling))
                throw std::invalid_argument("temperature is too small to sample with");
            handle = owner.activate(0, true).state;
        } catch (const std::invalid_argument& error) {
            owner.refuse({}, error.what());
            continue;
        }
        requests_.emplace(request.id, handle);
        request.cancelled = [&owner, handle] { return owner.ticket_cancelled(handle); };
        const auto prompt_size = static_cast<std::uint64_t>(request.prompt.size());
        request.prefilled = [&owner, handle, prompt_size](std::uint64_t rows) {
            owner.publish_progress(handle, {GenerationPhase::prefill, rows, 0U, prompt_size});
        };
        request.publish = [&owner, handle, prompt_size](std::span<const std::uint32_t> ids) {
            owner.publish_progress(handle, {GenerationPhase::decode, prompt_size, 0U, prompt_size});
            return owner.publish(handle, ids);
        };
        owner.notify_admitted(handle);
        cohort_.submit(std::move(request));
    }
    if (!cohort_.idle()) cohort_.step();
    for (auto& result : cohort_.take_finished()) {
        const auto handle = requests_.at(result.id);
        const auto& taken = owner.inspect(handle);
        const auto& job = taken.job;
        GenerationStats stats;
        stats.prompt_tokens = job.prompt.size();
        stats.generated_tokens = result.ids.size();
        stats.cycles = result.steps;
        stats.proposed = result.drafted;
        stats.accepted = result.accepted;
        stats.sampling = result.sampling;
        if (!result.sampling_receipt.empty())
            stats.engine_extra.emplace_back("sampling_receipt", std::move(result.sampling_receipt));
        if (!result.ids.empty()) {
            const std::chrono::duration<double, std::milli> queued = taken.started - taken.submitted;
            stats.first_token_ms = queued.count() + result.first_ms - result.submit_ms;
        }
        stats.finish = result.cancelled ? "cancelled" :
            ((!result.ids.empty() && std::ranges::find(job.stop_tokens, result.ids.back()) != job.stop_tokens.end())
                ? "stop" : "length");
        owner.complete_native(handle, std::move(stats), std::move(result.error));
        requests_.erase(result.id);
    }
}

} // namespace strix::serve
