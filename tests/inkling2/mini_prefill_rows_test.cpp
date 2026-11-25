// The declared implicit CLI prefill chunk of Inkling 2 Mini and its pressure fallback (inkling2::mini_cli_prefill_rows,
// mini_cli_prefill_fallback_rows) and what the shared serial-generation load makes of them (serve::load_with_chunk_fallback
// with the chunk and fallback both capped by the prompt, as the shared CLI does). Each attempted load allocates the
// decoder's real row-scaled MoE workspace through the native device runtime (device::native::Buffer) compiled against the
// host CUDA runtime stub, whose free device memory holds the 2048-row workspace but not the 8192-row one. No GPU.
#include "inkling2/inkling2_mini_moe_dispatch.hpp"
#include "inkling2/inkling2_mini_prefill_rows.hpp"
#include "strix/device_allocation_error.hpp"
#include "strix/serve/prefill_chunk_fallback.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// The native runtime header against the host runtime stub (tests/stubs/cuda_status): keep it last, after the headers the
// runtime header itself includes; the macro only selects the native section of the runtime header.
#include "strix/device/declaration.hpp"
#include "strix/session/step_replay.hpp"
#define __CUDACC__ 1
#include "strix/device/runtime.hpp"

namespace {

namespace mini = strix::inkling2;
namespace sv = strix::serve;
namespace stub = cuda_status_stub;
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
// Only the CUDA EXACT tall default falls back (to the incumbent 2048 rows); no other row has a pressure retry.
static_assert(mini::mini_cli_prefill_fallback_rows(Backend::cuda, Name::standard) == 2048U);
static_assert(mini::mini_cli_prefill_fallback_rows(Backend::cuda, Name::fast) == 0U);
static_assert(mini::mini_cli_prefill_fallback_rows(Backend::cuda, Name::quality) == 0U);
static_assert(mini::mini_cli_prefill_fallback_rows(Backend::cuda, Name::turbo) == 0U);
static_assert(mini::mini_cli_prefill_fallback_rows(Backend::hip, Name::standard) == 0U);
static_assert(mini::mini_cli_prefill_fallback_rows(Backend::hip, Name::fast) == 0U);
static_assert(mini::mini_cli_prefill_fallback_rows(Backend::hip, Name::turbo) == 0U);

using Loads = std::vector<std::uint32_t>;

struct Attempt {
    Loads seen;                           // rows of every attempted load
    std::optional<std::uint32_t> loaded;  // rows of the load that completed
    std::string log;                      // what the shared load printed
    bool typed_failure = false, other_failure = false;
};

// The bytes of the decoder's row-scaled MoE workspace (MiniResidentBackend::moe_fp32_) at `rows`.
std::size_t workspace_bytes(std::uint32_t rows) { return mini::mini_moe_workspace_elements(rows) * sizeof(float); }

// The loads the shared CLI attempts for a prompt: the chunk (the declared implicit rows, or `explicit_rows`) and the
// declared fallback, both capped by the prompt. Every load allocates the real row-scaled workspace on the stub device.
Attempt attempt(Backend backend, Name tier, std::uint32_t prompt, std::optional<std::uint32_t> explicit_rows = {}) {
    Attempt result;
    std::ostringstream log;
    try {
        const auto chunk = std::min(explicit_rows.value_or(mini::mini_cli_prefill_rows(backend, tier)), prompt);
        const auto fallback = std::min(mini::mini_cli_prefill_fallback_rows(backend, tier), prompt);
        result.loaded = sv::load_with_chunk_fallback(chunk, explicit_rows.has_value(), fallback, log,
                                                     [&](std::uint32_t rows) {
            result.seen.push_back(rows);
            const strix::device::native::Buffer<float> workspace(mini::mini_moe_workspace_elements(rows));
            return rows;
        });
    } catch (const strix::DeviceAllocationError&) {
        result.typed_failure = true;
    } catch (const std::runtime_error&) {
        result.other_failure = true;
    }
    result.log = log.str();
    return result;
}

}  // namespace

int main() try {
    // The free device memory holds exactly the 2048-row workspace: neither the 8192-row one nor a 5000-row one.
    const auto pressure = workspace_bytes(2048U);
    require(workspace_bytes(5000U) > pressure && workspace_bytes(8192U) > pressure, "pressure budget");

    // Roomy device: CUDA EXACT keeps the tall chunk (a 2050-token prompt is one 2050-row chunk), one load, no line.
    auto roomy = attempt(Backend::cuda, Name::standard, 2050U);
    require(roomy.seen == Loads{2050U} && roomy.loaded == 2050U && roomy.log.empty(),
            "CUDA EXACT 2050-token prompt selects one 2050-row chunk");
    roomy = attempt(Backend::cuda, Name::standard, 8194U);
    require(roomy.seen == Loads{8192U} && roomy.loaded == 8192U && roomy.log.empty(),
            "CUDA EXACT 8194-token prompt selects 8192-row chunks");

    // Under pressure the 8192-row workspace is refused with the typed device allocation failure at the real allocation
    // call; the implicit EXACT chunk loads once more at 2048 rows and the shared load reports it.
    stub::malloc_free_bytes = pressure;
    stub::malloc_calls = 0;
    auto fell_back = attempt(Backend::cuda, Name::standard, 8195U);
    require(fell_back.seen == Loads({8192U, 2048U}) && fell_back.loaded == 2048U && stub::malloc_calls == 2U &&
                fell_back.log == "strix-prefill-chunk 8192 -> 2048 (device memory)\n",
            "CUDA EXACT under pressure loads at 2048 rows and says so");
    // A prompt between the two chunks names its own size.
    fell_back = attempt(Backend::cuda, Name::standard, 5000U);
    require(fell_back.seen == Loads({5000U, 2048U}) && fell_back.loaded == 2048U &&
                fell_back.log == "strix-prefill-chunk 5000 -> 2048 (device memory)\n",
            "CUDA EXACT 5000-token prompt under pressure loads at 2048 rows");
    // An explicit chunk is never changed: the typed failure propagates after the one attempt.
    const auto explicit_chunk = attempt(Backend::cuda, Name::standard, 8195U, 8192U);
    require(explicit_chunk.seen == Loads{8192U} && !explicit_chunk.loaded && explicit_chunk.typed_failure &&
                explicit_chunk.log.empty(), "explicit --prefill-chunk 8192 is never retried");
    require(attempt(Backend::cuda, Name::standard, 8195U, 2048U).loaded == 2048U, "explicit --prefill-chunk 2048 loads");
    // A non-memory runtime failure at the same allocation never falls back.
    stub::malloc_status = cudaErrorInvalidValue;
    const auto other = attempt(Backend::cuda, Name::standard, 8195U);
    require(other.seen == Loads{8192U} && !other.loaded && other.other_failure && !other.typed_failure && other.log.empty(),
            "a non-memory failure is not retried");
    stub::malloc_status = cudaSuccess;
    // The other CUDA profiles already run 2048 rows: they load under the same pressure, with no retry route.
    for (const auto tier : {Name::fast, Name::quality, Name::turbo}) {
        const auto incumbent = attempt(Backend::cuda, tier, 8195U);
        require(incumbent.seen == Loads{2048U} && incumbent.loaded == 2048U && incumbent.log.empty(),
                "CUDA non-EXACT profiles keep the 2048-row chunk");
    }
    // HIP keeps its single attempt: the typed failure reports after it.
    for (const auto tier : {Name::standard, Name::fast, Name::turbo}) {
        const auto hip = attempt(Backend::hip, tier, 8195U);
        require(hip.seen == Loads{8192U} && !hip.loaded && hip.typed_failure, "HIP has no pressure retry");
    }
    stub::malloc_free_bytes = static_cast<std::size_t>(-1);
    for (const auto tier : {Name::standard, Name::fast, Name::turbo})
        require(attempt(Backend::hip, tier, 8195U).loaded == 8192U, "HIP keeps the 8192-row chunk");
    std::cout << "ok\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
