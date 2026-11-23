#pragma once

#include "strix/serve/cohort_decode.hpp"
#include "strix/serve/session_runtime.hpp"

#include <functional>
#include <map>
#include <type_traits>
#include <utility>

namespace strix::serve {

// Binds a CohortBackend to the shared HTTP session owner. The backend owns
// physical lane state. The runtime holds request leases and publishes tokens.
// This adapter serves request-only cohorts (greedy, or sampled when the backend reads final logits); retained
// prefixes are the backend's own (CohortBackend::restore / capture; no_store requests are not captured).
// Owner hooks of an engine bound to the cohort. Each is optional; none set = the plain request-only session.
struct CohortSessionHooks {
    std::function<void(const GenerationJob&)> validate;
    // Builds the engine's request fields from the job before submit: may add engine default stop ids, set
    // sampler_options, context_tokens and traits. Throwing std::invalid_argument refuses the job like validate.
    std::function<void(const GenerationJob&, CohortRequest&)> prepare;
    // Called once per finished request with its result, before the runtime publishes the end.
    std::function<void(const GenerationJob&, const CohortResult&)> finished;
};

class CohortSession final : private session::Engine {
public:
    CohortSession(GenerationQueue&, CohortBackend&, CohortConfig, CohortSessionHooks = {});
    // A validate callable alone (the round-1 form). A template, so a braced `{}` fourth argument selects the hooks
    // overload above instead of being ambiguous between the two.
    template <class Validate>
        requires(!std::is_same_v<std::remove_cvref_t<Validate>, CohortSessionHooks> &&
                 std::is_constructible_v<std::function<void(const GenerationJob&)>, Validate>)
    CohortSession(GenerationQueue& queue, CohortBackend& backend, CohortConfig config, Validate&& validate)
        : CohortSession(queue, backend, std::move(config),
                        CohortSessionHooks{std::function<void(const GenerationJob&)>(std::forward<Validate>(validate)),
                                           {}, {}}) {}
    SessionRuntime& runtime() { return runtime_; }
    void configure_api(ApiConfig&) const;

private:
    session::Capabilities caps() const override;
    profile::ExecutionReceipt receipt() const override { return {}; }
    void prefill(std::span<const std::uint32_t>, session::KvHandle&) override;
    session::StepResult decode(session::KvHandle&, const session::SamplerState&) override;
    void turn(SessionRuntime&);

    CohortBackend& backend_;
    CohortSessionHooks owner_;
    std::uint32_t slots_;
    CohortDecoder cohort_;
    session::KvStore store_;
    SessionRuntime runtime_;
    std::map<std::uint64_t, session::KvHandle> requests_;
    std::uint64_t next_request_ = 0;
};

} // namespace strix::serve
