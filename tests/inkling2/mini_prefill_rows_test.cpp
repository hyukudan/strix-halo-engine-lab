// The declared implicit CLI prefill chunk of Inkling 2 Mini (inkling2::mini_cli_prefill_rows) and what the shared
// serial-generation load makes of it (serve::load_with_chunk_fallback with no declared fallback). No GPU.
#include "inkling2/inkling2_mini_prefill_rows.hpp"
#include "strix/device_allocation_error.hpp"
#include "strix/serve/prefill_chunk_fallback.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
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

static_assert(mini::mini_cli_prefill_rows(Backend::hip, Name::standard) == 8192U);
static_assert(mini::mini_cli_prefill_rows(Backend::hip, Name::fast) == 8192U);
static_assert(mini::mini_cli_prefill_rows(Backend::hip, Name::turbo) == 8192U);
static_assert(mini::mini_cli_prefill_rows(Backend::cuda, Name::standard) == 8192U);
static_assert(mini::mini_cli_prefill_rows(Backend::cuda, Name::fast) == 2048U);
static_assert(mini::mini_cli_prefill_rows(Backend::cuda, Name::quality) == 2048U);
static_assert(mini::mini_cli_prefill_rows(Backend::cuda, Name::turbo) == 2048U);

using Loads = std::vector<std::uint32_t>;

// The loads the shared CLI attempts for a prompt (chunk capped by the prompt, the family declares no fallback). A device
// that cannot hold more than `capacity` rows refuses larger ones with the typed allocation failure.
Loads loads(Backend backend, Name tier, std::uint32_t prompt, std::uint32_t capacity, bool& loaded) {
    Loads seen;
    std::ostringstream log;
    loaded = false;
    try {
        const auto chunk = std::min(mini::mini_cli_prefill_rows(backend, tier), prompt);
        sv::load_with_chunk_fallback(chunk, false, 0U, log, [&](std::uint32_t rows) {
            seen.push_back(rows);
            if (rows > capacity) throw strix::DeviceAllocationError("allocate: out of memory", 2);
            return rows;
        });
        loaded = true;
    } catch (const strix::DeviceAllocationError&) {
    }
    return seen;
}

}  // namespace

int main() try {
    constexpr std::uint32_t roomy = 1U << 20;
    bool loaded = false;
    // CUDA EXACT: a 2050-token prompt is one 2050-row chunk (it was 2048 + 2 rows); an 8194-token prompt is 8192 + 2.
    require(loads(Backend::cuda, Name::standard, 2050U, roomy, loaded) == Loads{2050U} && loaded,
            "CUDA EXACT 2050-token prompt selects one 2050-row chunk");
    require(loads(Backend::cuda, Name::standard, 8194U, roomy, loaded) == Loads{8192U} && loaded,
            "CUDA EXACT 8194-token prompt selects 8192-row chunks");
    // No pressure retry on any profile: the typed failure reports after the one attempt.
    require(loads(Backend::cuda, Name::standard, 8192U, 4096U, loaded) == Loads{8192U} && !loaded,
            "CUDA EXACT has no pressure retry");
    for (const auto tier : {Name::fast, Name::quality, Name::turbo}) {
        require(loads(Backend::cuda, tier, 8192U, roomy, loaded) == Loads{2048U} && loaded,
                "CUDA non-EXACT profiles keep the 2048-row chunk");
        require(loads(Backend::cuda, tier, 8192U, 1024U, loaded) == Loads{2048U} && !loaded,
                "CUDA non-EXACT profiles have no pressure retry");
    }
    for (const auto tier : {Name::standard, Name::fast, Name::turbo})
        require(loads(Backend::hip, tier, 8192U, roomy, loaded) == Loads{8192U} && loaded, "HIP keeps the 8192-row chunk");
    std::cout << "ok\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
