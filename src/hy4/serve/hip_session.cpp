#include "hy4/serve/session_service.hpp"

#include "hy4/hip/hy4_hip_forward.hpp"
#include "hy4/hip/hy4_hip_rows.hpp"
#include "hy4/hy4_speculation.hpp"
#include "strix/profiles/hy4.hpp"
#include "strix/serve/tokenizer.hpp"
#include "strix/weight_pack.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <utility>

namespace strix::hy4::serve {
namespace {
namespace hb = hipbind;
namespace rows = hb::rows;
using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point since) {
    return std::chrono::duration<double, std::milli>(Clock::now() - since).count();
}

class Engine final : public sv::GenerationEngine {
    hb::ForwardOptions options_;
    hb::Forward forward_;

public:
    explicit Engine(hb::ForwardOptions options)
        : options_(std::move(options)), forward_(options_) {}

    std::uint64_t context_tokens() const override { return options_.context; }

    sv::GenerationStats generate(sv::GenerationJob& job) override {
        sv::GenerationStats stats;
        stats.prompt_tokens = job.prompt.size();
        if (job.prompt.size() >= options_.context || job.prompt.empty() || job.max_tokens == 0U) {
            stats.finish = "context";
            return stats;
        }
        const auto available = options_.context - static_cast<std::uint32_t>(job.prompt.size());
        const auto length = std::min(available, job.max_tokens);
        const bool speculative = options_.nextn && length > 1U && available - length >= rows::kMaxVerifyRows;
        stats.finish = length < job.max_tokens ? "context" : "length";
        const auto publish = [&](std::span<const std::uint32_t> tokens) {
            const auto stop = std::ranges::find_if(tokens, [&](auto id) {
                return std::ranges::find(job.stop_tokens, id) != job.stop_tokens.end();
            });
            const auto keep = stop == tokens.end() ? tokens.size() : static_cast<std::size_t>(stop - tokens.begin()) + 1U;
            stats.generated_tokens += keep;
            if (job.on_tokens && !job.on_tokens(tokens.first(keep))) {
                stats.finish = "cancelled";
                return false;
            }
            if (stop == tokens.end()) return true;
            stats.finish = "stop";
            return false;
        };

        if (job.on_admitted) job.on_admitted();
        const auto start = Clock::now();
        forward_.sync();
        forward_.reset();
        std::unique_ptr<rows::MtpSource<hb::Forward>> mtp;
        if (speculative) {
            mtp = std::make_unique<rows::MtpSource<hb::Forward>>(forward_);
            mtp->install(job.prompt);
        }
        std::vector<float> exact_logits;
        const auto* logits = &exact_logits;
        if (options_.prompt_rows != 0U) logits = &forward_.prefill(job.prompt);
        else exact_logits = rows::prefill_verify_rows(forward_, job.prompt, rows::kMaxVerifyRows, speculative);
        auto next = static_cast<std::uint32_t>(std::ranges::max_element(*logits) - logits->begin());
        stats.prefill_ms = elapsed(start);
        const auto decode_start = Clock::now();
        if (speculative) {
            mtp->finish_prompt(next);
            // Publish the anchor before calibration, then only newly verified rows.
            if (publish(std::span(&next, 1))) {
                double best = std::numeric_limits<double>::infinity();
                for (unsigned trial = 0; trial < 2U; ++trial) {
                    const auto begin = Clock::now();
                    (void)forward_.step_greedy(next);
                    best = std::min(best, elapsed(begin));
                    forward_.rewind();
                }
                rows::SpecOptions spec;
                rows::seed_spec_options(spec, best);
                spec.publish = [&, first = true](std::span<const std::uint32_t> ids) mutable {
                    if (std::exchange(first, false)) return true;
                    return publish(ids);
                };
                const auto result = rows::spec_decode(forward_, job.prompt, next, length, spec, *mtp);
                stats.cycles = result.steps.size();
                stats.proposed = result.proposed;
                stats.accepted = result.accepted;
            }
        } else {
            for (std::uint32_t i = 0; i < length; ++i) {
                if (!publish(std::span(&next, 1)) || i + 1U == length) break;
                next = forward_.step_greedy(next);
                ++stats.cycles;
            }
        }
        stats.decode_ms = elapsed(decode_start);
        stats.engine_extra.emplace_back("hy4_speculation", speculative ? "true" : "false");
        return stats;
    }
};
} // namespace

SessionAssets load_hip_service(const Options& options, const PackFacts& pack) {
    const auto context = options.ctx ? options.ctx : pack.context_length;
    if (context == 0U || context > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("hy4_flash serve needs a valid --ctx or GGUF context_length");

    hipDeviceProp_t prop{};
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&prop, device) != hipSuccess)
        throw std::runtime_error("hy4_flash serve could not query the selected HIP device");
    namespace hp = ::strix::hy4::profile;
    // Price the largest admitted prompt when sizing a persistent forward.
    const auto resolved = hp::resolve({hp::device_of(prop.gcnArchName), hp::Topology::one_node, context - 1U}, options.profile);
    hb::ForwardOptions native;
    native.context = static_cast<std::uint32_t>(context);
    native.verify_rows = rows::kMaxVerifyRows;
    for (const auto& part : weight_pack::gguf_parts(options.gguf)) native.parts.emplace_back(part);
    if (const auto* value = resolved->find("prompt_rows"); value && ::strix::profile::applied(value->status))
        native.prompt_rows = static_cast<std::uint32_t>(std::stoul(value->value));
    if (const auto* value = resolved->find("head_q8"); value && ::strix::profile::applied(value->status))
        native.head_q8 = value->value == "1";
    native.prompt_rows = options.prompt_rows.value_or(native.prompt_rows);
    native.head_q8 = options.head_q8.value_or(native.head_q8);
    SpeculationRun run;
    run.block_in_pack = pack.next_token_block;
    run.no_mtp = options.no_mtp;
    run.environment = envcfg::text(::strix::session::speculation_environment);
    native.nextn = speculation(run).enabled;
    ::strix::profile::apply(resolved->resolution);
    auto tokenizer = std::make_shared<sv::BpeApiTokenizer>(sv::BpeTokenizer::load_gguf(options.gguf));
    return {std::move(tokenizer), std::make_unique<Engine>(std::move(native))};
}
} // namespace strix::hy4::serve
