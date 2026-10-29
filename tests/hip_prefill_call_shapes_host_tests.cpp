#include "hip_prefill_pool.hpp"
#include "hip_decode_options.hpp"
#include "strix/known_prompt_chunks.hpp"
#include "strix/q51_known_rows.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {
using namespace strix;
void expect(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
constexpr unsigned context = 262144U, experts = 512U, top_k = 10U, ratio = 4U;
constexpr unsigned round_rows(unsigned rows, unsigned multiple) { return (rows + multiple - 1U) / multiple * multiple; }

// Exercise the shared HC/MoE owner used by native setup, without allocating GPU
// images. Snapshot the initial reservation separately from the mutable metadata:
// a later tail setup must preserve both the byte extent AND its indexing stride.
struct Pools {
    HipPrefillPoolOwner hc, moe;
    unsigned hc_reserved = 0, moe_reserved = 0, qsa_reserved = 0;
    unsigned hc_setups = 0, moe_setups = 0;
    void setup(unsigned width) {
        hc.setup(width, [&] { hc_reserved = hc.rows(); ++hc_setups; });
        moe.setup(width, [&] { moe_reserved = moe.rows(); ++moe_setups; });
        // QSA stage setup already retains the first (largest) row capacity.
        if (!qsa_reserved) qsa_reserved = known_prefill_pool_rows(width);
        expect(known_prefill_pool_rows(width) <= qsa_reserved, "QSA owner order increased capacity");
    }
    void check(unsigned rows, unsigned width, std::uint64_t position) const {
        expect(rows >= 16U && rows % 8U == 0U && rows <= width, "call exceeds its compiled owner");
        expect(position + rows <= context, "call exceeds context");
        expect(hc.rows(rows) == hc_reserved, "HC tail setup changed capacity/packed overflow offset");
        // HC wide2 uses a transposed row pitch rounded UP to 32, not just count.
        expect(hc.rows(round_rows(rows, 32U)) == hc_reserved, "HC padded pitch exceeds reservation");
        expect(moe.rows(rows) == moe_reserved, "MoE tail setup changed shared pool extents");
        const auto count = static_cast<std::uint64_t>(rows), capacity = std::uint64_t{moe_reserved};
        expect(count * 2560U * 2U <= capacity * 2560U * 2U, "MoE A16 exceeds reservation");
        expect(count * top_k * 640U * 2U <= capacity * top_k * 640U * 2U, "MoE DA16 exceeds reservation");
        // CSR batches pad each expert independently (cap >= 64 on P1/DA16).
        // This bound includes skewed routing and every expert's partial batch.
        const auto batches = std::min(count * top_k, experts + (count * top_k + 63U) / 64U);
        const auto counter_words = capacity * top_k / 64U + 1024U;
        expect(batches <= counter_words, "DA16 batch counters exceed reservation");
        expect(batches + 2U + 8U <= counter_words, "P1 ticket/done/stats exceed reservation");
        expect(count * 16U + 1U <= capacity * 16U + 64U, "core2 merge counters exceed reservation");
        expect(514U + (count * top_k + 2047U) / 2048U * experts <=
            514U + (capacity * top_k + 2047U) / 2048U * experts, "multi-CTA CSR exceeds reservation");
        expect(q51_known_rows::prefill_scratch_bytes(rows, experts, top_k) <=
            q51_known_rows::prefill_scratch_bytes(width, experts, top_k), "owner/partial CSR exceeds reservation");
        // Stage batching admits only whole 64-row groups; rounding up here is a
        // conservative bound also covering calls which take the ordinary path.
        const auto padded = round_rows(rows, 64U);
        const auto groups = padded / 64U, reserved_groups = known_prefill_stage_groups(qsa_reserved);
        expect(padded <= qsa_reserved && groups <= reserved_groups, "QSA padded groups exceed reservation");
        expect(std::uint64_t{groups} * 64U * 64U * 8U <=
            std::uint64_t{reserved_groups} * 64U * 64U * 8U, "QSA tables exceed reservation");
        // Each scored group takes its largest complete-block prefix, as in the
        // stage launcher. qsa_flash::kGroup = 4 blocks per maxima word per row.
        std::uint64_t maxima = 0;
        if (rows % 64U == 0U) for (unsigned first = 0; first < rows; first += 64U) {
            const auto blocks = (position + first + 64U) / ratio;
            maxima += 64U * ((blocks + 3U) / 4U);
        }
        expect(maxima <= std::uint64_t{qsa_reserved} * ((context / ratio + 3U) / 4U),
            "QSA maxima exceed context/row reservation");
        // PLE staging, hashes, projections and gates are per compiled owner, not
        // process-wide. Merges/balanced calls must stay on the T owner. Its batch
        // pack uses complete groups of 64 and processes the remainder separately.
        expect(count <= width && rows / 64U * 64U <= width, "PLE staging/projections/packed rows exceed owner");
        const auto staged_next = std::min<std::uint64_t>(width, context - position - rows);
        expect(staged_next <= width, "PLE prefetch exceeds staging owner");
    }
};
struct Call { unsigned owner, rows; std::uint64_t position; };
class Backend final : public KnownPromptWindowBackend, public SerialGreedyStepBackend {
    unsigned width_;
    Pools& pools_;
    std::vector<Call>& calls_;
    void record(unsigned rows, std::uint64_t position) {
        pools_.check(rows, width_, position);
        calls_.push_back({width_, rows, position});
    }
public:
    Backend(unsigned width, Pools& pools, std::vector<Call>& calls) : width_(width), pools_(pools), calls_(calls) {
        pools_.setup(width);
    }
    unsigned logical_rows() const noexcept override { return width_; }
    bool supports_tiled_prefill() const noexcept override { return true; }
    KnownWindowOutputPolicy output_policy() const noexcept override { return KnownWindowOutputPolicy::terminal_row; }
    std::vector<float> execute(std::uint32_t, std::uint64_t position) override {
        calls_.push_back({1U, 1U, position}); return std::vector<float>(16U, 0.0F);
    }
    std::vector<float> execute_window(std::array<std::uint32_t, 2>, std::uint64_t) override {
        throw std::logic_error("unexpected pair");
    }
    std::vector<float> execute_rows(std::span<const std::uint32_t> tokens, std::uint64_t position) override {
        expect(tokens.size() == width_, "full call differs from owner width");
        record(static_cast<unsigned>(tokens.size()), position); return std::vector<float>(16U, 0.0F);
    }
    void execute_state_rows(std::span<const std::uint32_t> tokens, std::uint64_t position) override {
        (void)execute_rows(tokens, position);
    }
    std::vector<float> execute_rows_partial(std::span<const std::uint32_t> tokens, std::uint64_t position) override {
        record(static_cast<unsigned>(tokens.size()), position); return std::vector<float>(16U, 0.0F);
    }
    void execute_state_rows_partial(std::span<const std::uint32_t> tokens, std::uint64_t position, bool) override {
        (void)execute_rows_partial(tokens, position);
    }
};
void exercise(unsigned count, unsigned width, const char* schedule) {
    decode_cli::Options options;
    options.known_prefix_window = options.known_prefix_prefill_explicit = options.known_prefix_full_rows = true;
    options.known_prefix_prefill_rows = width;
    options.known_prefix_output_policy = KnownWindowOutputPolicy::terminal_row;
    options.tokens.assign(count, 1U);
    decode_cli::apply_default_known_prefill_tails(options, schedule);
    Pools pools;
    std::vector<Call> calls;
    Backend main(width, pools, calls);
    std::vector<std::unique_ptr<Backend>> tails;
    for (const auto tail : options.known_prefix_tail_rows) tails.push_back(std::make_unique<Backend>(tail, pools, calls));
    KnownPromptChunkRequest request{options.tokens, 0U, 16U, context, 1U, 15U, width,
        KnownWindowOutputPolicy::terminal_row, true};
    request.schedule_extensions = true;
    request.ladder_tail_rows = options.known_prefix_ladder_tail_rows;
    KnownPromptChunkCursor cursor(request, main, main);
    for (auto& tail : tails) cursor.set_tail_backend(*tail);
    while (cursor.advance()) {}
    expect(cursor.progress().consumed_prompt_tokens == count && cursor.progress().prefill_complete,
        "cursor did not consume the complete prompt");
    expect(pools.hc_setups == 1U && pools.moe_setups == 1U, "tail setup reserved pools again");
    expect(pools.hc_reserved == width && pools.moe_reserved == width && pools.qsa_reserved == width,
        "T8192/T16384 reservation extents changed");
    std::uint64_t next = 0;
    for (const auto& call : calls) {
        expect(call.position == next, "call shapes skipped or replayed rows"); next += call.rows;
    }
    expect(next == count, "call shape coverage differs from prompt");
    // Exact reproducer: construction of the surviving 64/16 owners used to
    // lower shared metadata to 8192 before even the FIRST 16384-row preflight.
    if (count == 262143U && width == 16384U && std::string(schedule) == "12") {
        expect(options.known_prefix_tail_rows == std::vector<unsigned>{64U, 16U}, "reproducer owners changed");
        expect(calls.size() == 37U, "reproducer call count changed");
        for (unsigned i = 0; i < 15U; ++i)
            expect(calls[i].owner == width && calls[i].rows == width, "reproducer lost a full T call");
        expect(calls[15].owner == width && calls[15].rows == 16128U, "merged rows left the T owner");
        for (unsigned i = 16U; i < 19U; ++i) expect(calls[i].rows == 64U, "reproducer lost 64-row tail");
        for (unsigned i = 19U; i < 22U; ++i) expect(calls[i].rows == 16U, "reproducer lost 16-row tail");
        for (unsigned i = 22U; i < 37U; ++i) expect(calls[i].rows == 1U, "reproducer lost serial suffix");
    }
    std::cout << "T" << width << " prompt=" << count << " owners=" << 1U + tails.size() << " calls:";
    for (std::size_t i = 0; i < calls.size();) {
        std::size_t end = i + 1U;
        while (end < calls.size() && calls[end].owner == calls[i].owner && calls[end].rows == calls[i].rows) ++end;
        std::cout << ' ' << calls[i].rows << 'x' << end - i << "@" << calls[i].owner; i = end;
    }
    std::cout << '\n';
}
}
int main(int argc, char** argv) try {
    const char* schedule = argc > 1 ? argv[1] : "12";
#ifdef _WIN32
    expect(_putenv_s("STRIX_QPF2K_SCHED", schedule) == 0, "set schedule");
#else
    expect(::setenv("STRIX_QPF2K_SCHED", schedule, 1) == 0, "set schedule");
#endif
    // Each schedule runs in its own process: the real cursor caches the reader.
    for (const auto width : {16384U, 8192U})
        for (const auto count : {16385U, 24577U, 100000U, 131072U, 196608U, 262143U,
                                131073U, 131136U, 132096U, 147455U}) exercise(count, width, schedule);
    std::cout << "prefill call shape host contracts PASS (SCHED=" << schedule << ")\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
}
