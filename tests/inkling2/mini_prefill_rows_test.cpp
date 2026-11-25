// The declared prefill-row selection of Inkling 2 Mini (inkling2::mini_prefill_rows) and what the shared admission
// (serve::admit_prefill_chunk, load_with_prefill_admission) makes of it for the CLI. The HTTP server reads the same
// table (pinned in the served-wiring test). No GPU.
#include "inkling2/inkling2_mini_prefill_rows.hpp"
#include "strix/device_allocation_error.hpp"
#include "strix/serve/prefill_chunk_fallback.hpp"

#include <cstdint>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

namespace mini = strix::inkling2;
namespace sv = strix::serve;
using strix::profile::Name;
using Backend = mini::MiniPrefillBackend;

void require(bool passed, const char* detail) {
    if (!passed) throw std::runtime_error(detail);
}

// The declared cells.
static_assert(mini::mini_prefill_rows(Backend::hip, Name::standard).cli == 8192U);
static_assert(mini::mini_prefill_rows(Backend::hip, Name::standard).cli_pressure == 0U);
static_assert(mini::mini_prefill_rows(Backend::hip, Name::fast).cli == 8192U);
static_assert(mini::mini_prefill_rows(Backend::cuda, Name::standard).cli == 8192U);
static_assert(mini::mini_prefill_rows(Backend::cuda, Name::standard).cli_pressure == 2048U);
static_assert(mini::mini_prefill_rows(Backend::cuda, Name::fast).cli == 2048U);
static_assert(mini::mini_prefill_rows(Backend::cuda, Name::quality).cli == 2048U);
static_assert(mini::mini_prefill_rows(Backend::cuda, Name::turbo).cli == 2048U);
static_assert(mini::mini_prefill_rows(Backend::cuda, Name::fast).cli_pressure == 0U);
static_assert(mini::mini_prefill_rows(Backend::hip, Name::standard).http == 128U);
static_assert(mini::mini_prefill_rows(Backend::cuda, Name::standard).http == 128U);

// The rows each load sees for a prompt. A device that cannot hold more than `capacity` rows refuses larger ones with
// the typed allocation failure; `status` is the whole outcome (loads attempted, whether the load finally succeeded).
struct Outcome {
    std::vector<std::uint32_t> loads;
    bool loaded = false;
};

Outcome admit(Backend backend, Name tier, std::optional<std::uint32_t> requested, std::uint32_t prompt,
              std::uint32_t capacity) {
    const auto rows = mini::mini_prefill_rows(backend, tier);
    Outcome outcome;
    std::ostringstream log;
    try {
        sv::load_with_prefill_admission(requested, rows.cli, rows.cli_pressure, prompt, log, [&](std::uint32_t chunk) {
            outcome.loads.push_back(chunk);
            if (chunk > capacity) throw strix::DeviceAllocationError("allocate: out of memory", 2);
            return chunk;
        });
        outcome.loaded = true;
    } catch (const strix::DeviceAllocationError&) {
    }
    return outcome;
}

using Loads = std::vector<std::uint32_t>;

}  // namespace

int main() try {
    constexpr std::uint32_t roomy = 1U << 20;
    // EXACT on CUDA: the implicit 2050-token prompt is one 2050-row chunk (it was 2048 + 2 rows at the incumbent
    // chunk); an 8K prompt is one 8192-row chunk.
    auto out = admit(Backend::cuda, Name::standard, std::nullopt, 2050U, roomy);
    require(out.loaded && out.loads == Loads{2050U}, "CUDA EXACT 2050-token prompt selects one 2050-row chunk");
    out = admit(Backend::cuda, Name::standard, std::nullopt, 8192U, roomy);
    require(out.loaded && out.loads == Loads{8192U}, "CUDA EXACT 8K prompt selects one 8192-row chunk");
    // Pressure: the device refuses the tall workspaces; the load retries once at the incumbent 2048 rows.
    out = admit(Backend::cuda, Name::standard, std::nullopt, 8192U, 4096U);
    require(out.loaded && out.loads == Loads{8192U, 2048U}, "CUDA EXACT pressure retries at 2048 rows");
    out = admit(Backend::cuda, Name::standard, std::nullopt, 2050U, 2048U);
    require(out.loaded && out.loads == Loads{2050U, 2048U}, "CUDA EXACT 2050 under pressure retries at 2048 rows");
    out = admit(Backend::cuda, Name::standard, std::nullopt, 1000U, 512U);
    require(!out.loaded && out.loads == Loads{1000U}, "a chunk already at the fallback is not retried");
    // An explicit chunk wins and is never retried.
    out = admit(Backend::cuda, Name::standard, 2048U, 2050U, roomy);
    require(out.loaded && out.loads == Loads{2048U}, "explicit 2048 is the incumbent schedule");
    out = admit(Backend::cuda, Name::standard, 4096U, 8192U, 2048U);
    require(!out.loaded && out.loads == Loads{4096U}, "an explicit chunk is never retried");
    // Every other CUDA profile keeps the incumbent chunk and has no retry.
    for (const auto tier : {Name::fast, Name::quality, Name::turbo}) {
        out = admit(Backend::cuda, tier, std::nullopt, 2050U, roomy);
        require(out.loaded && out.loads == Loads{2048U}, "CUDA non-EXACT profiles keep the 2048-row chunk");
        out = admit(Backend::cuda, tier, std::nullopt, 8192U, 1024U);
        require(!out.loaded && out.loads == Loads{2048U}, "CUDA non-EXACT profiles have no pressure retry");
    }
    // HIP keeps its tall chunk for every profile, without a retry.
    for (const auto tier : {Name::standard, Name::fast, Name::turbo}) {
        out = admit(Backend::hip, tier, std::nullopt, 8192U, roomy);
        require(out.loaded && out.loads == Loads{8192U}, "HIP keeps the 8192-row chunk");
        out = admit(Backend::hip, tier, std::nullopt, 8192U, 2048U);
        require(!out.loaded && out.loads == Loads{8192U}, "HIP has no pressure retry");
    }
    std::cout << "ok\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
