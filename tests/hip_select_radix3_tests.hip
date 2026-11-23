// Device fixture of the shared Radix3 select schedule (src/attn_core/select_radix3.hpp): the real select_blocks<1024>
// entry point, eager (NoControl) and graph (control) forms, against the host statement select_radix3_rule.hpp.
// Boundaries: blocks <= k is the full-visible row (ratio 4, visible 9, k 3, two finite scores: nine sequential ids, no
// write past them, scores untouched); k == 0 emits the partial-block tail alone without reading the radix state (ratio 4,
// visible 5, one finite complete block: token 4 only, scores untouched). Random rows (ties, -inf, +-0, zero fill) compare
// tokens bit for bit; preserve_scores leaves the score buffer bitwise unchanged. Output buffers carry a sentinel guard.
// Prints one PASS / FAIL line; exit 0 only on PASS (exit 77: no device (ctest skip); exit 2: runtime error).
#include "attn_core/select_radix3.hpp"
#include "attn_core/select_radix3_rule.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {

namespace radix3 = strix::attn_core::select::radix3;
namespace rule = radix3::rule;

constexpr unsigned kThreads = 1024;
constexpr std::uint32_t kSentinel = 0xDEADBEEFU, kGuard = 64;
int g_failures = 0;

#define CHECK(cond, ...)                                                                   \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            ++g_failures;                                                                  \
            std::fprintf(stderr, "FAIL %s:%d: %s  [", __FILE__, __LINE__, #cond);          \
            std::fprintf(stderr, __VA_ARGS__);                                             \
            std::fprintf(stderr, "]\n");                                                   \
        }                                                                                  \
    } while (0)

void hok(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
        std::exit(2);
    }
}

struct GraphControl {
    std::uint32_t visible_tokens, topk;
    bool launch;
    __device__ bool graph() const { return true; }
    __device__ bool visible(std::uint32_t& tokens) const {
        if (!launch) return false;
        tokens = visible_tokens;
        return true;
    }
    __device__ std::uint32_t block_topk() const { return topk; }
};

template <class Control>
__global__ __launch_bounds__(kThreads) void select_kernel(float* scores, std::uint32_t num_blocks, std::uint32_t k,
                                                          std::uint32_t ratio, std::uint32_t visible, std::uint32_t* out,
                                                          Control control, bool preserve) {
    radix3::select_blocks<kThreads, Control>(scores, num_blocks, k, ratio, visible, out, control, preserve);
}

struct Result {
    std::vector<std::uint32_t> out;   // out_words + kGuard words, sentinel filled before the launch
    std::vector<float> scores;        // after the launch
};

template <class Control>
Result run(const std::vector<float>& scores_in, std::uint32_t k, std::uint32_t ratio, std::uint32_t visible,
           std::uint32_t out_words, Control control, bool preserve) {
    float* d_scores = nullptr;
    std::uint32_t* d_out = nullptr;
    const std::size_t n = scores_in.empty() ? 1 : scores_in.size();
    hok(hipMalloc(&d_scores, n * sizeof(float)), "malloc scores");
    hok(hipMalloc(&d_out, (out_words + kGuard) * sizeof(std::uint32_t)), "malloc out");
    std::vector<std::uint32_t> init(out_words + kGuard, kSentinel);
    hok(hipMemcpy(d_out, init.data(), init.size() * 4, hipMemcpyHostToDevice), "init out");
    if (!scores_in.empty()) hok(hipMemcpy(d_scores, scores_in.data(), scores_in.size() * 4, hipMemcpyHostToDevice), "init scores");
    select_kernel<Control><<<1, kThreads>>>(d_scores, static_cast<std::uint32_t>(scores_in.size()), k, ratio, visible, d_out, control, preserve);
    hok(hipGetLastError(), "launch");
    hok(hipDeviceSynchronize(), "sync");
    Result r;
    r.out.resize(out_words + kGuard);
    r.scores.resize(scores_in.size());
    hok(hipMemcpy(r.out.data(), d_out, r.out.size() * 4, hipMemcpyDeviceToHost), "read out");
    if (!scores_in.empty()) hok(hipMemcpy(r.scores.data(), d_scores, scores_in.size() * 4, hipMemcpyDeviceToHost), "read scores");
    hok(hipFree(d_scores), "free scores");
    hok(hipFree(d_out), "free out");
    return r;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

// Tokens equal the expected list, the guard is untouched (nothing written past the expected size).
void check_tokens(const Result& r, const std::vector<std::uint32_t>& want, std::uint32_t out_words, const char* tag) {
    bool head = true;
    for (std::size_t i = 0; i < want.size(); ++i) head = head && r.out[i] == want[i];
    CHECK(head, "%s: tokens differ from the host rule (want %zu ids)", tag, want.size());
    bool rest = true;
    for (std::size_t i = want.size(); i < r.out.size(); ++i) rest = rest && r.out[i] == kSentinel;
    CHECK(rest, "%s: write past the %zu expected ids (out words %u)", tag, want.size(), out_words);
}

std::vector<std::uint32_t> sequential(std::uint32_t n) {
    std::vector<std::uint32_t> v(n);
    for (std::uint32_t i = 0; i < n; ++i) v[i] = i;
    return v;
}

void test_boundaries() {
    const radix3::NoControl eager{};
    const float inf = std::numeric_limits<float>::infinity();
    // (1) blocks <= k, eager and graph: the full-visible row. ratio 4, visible 9, two finite scores, k 3.
    {
        const std::vector<float> s{1.0F, 2.0F};
        const auto want = rule::select_tokens(s, 3U, 4U, 9U);
        CHECK(want == sequential(9), "host rule is the nine sequential ids");
        const auto r = run(s, 3U, 4U, 9U, 9U, eager, false);
        check_tokens(r, want, 9U, "eager blocks<k");
        CHECK(same_bits(r.scores, s), "eager blocks<k: scores mutated");
        const auto g = run(s, 0U, 4U, 9U, 9U, GraphControl{9U, 3U, true}, false);
        check_tokens(g, want, 9U, "graph blocks<k");
        CHECK(same_bits(g.scores, s), "graph blocks<k: scores mutated");
    }
    // (1b) blocks == k, with and without a tail; blocks == 0.
    for (const std::uint32_t visible : {12U, 14U}) {
        const std::vector<float> s{3.0F, -inf, 1.0F};
        const auto r = run(s, 3U, 4U, visible, visible, eager, false);
        check_tokens(r, sequential(visible), visible, "eager blocks==k");
        CHECK(same_bits(r.scores, s), "eager blocks==k: scores mutated");
    }
    {
        const std::vector<float> none;
        check_tokens(run(none, 2U, 4U, 3U, 3U, eager, false), sequential(3U), 3U, "eager blocks==0");
    }
    // (2) k == 0: the partial-block tail alone, scores untouched (no uninitialised LDS reaches the output).
    {
        const std::vector<float> s{1.0F};
        const auto want = rule::select_tokens(s, 0U, 4U, 5U);
        CHECK(want == std::vector<std::uint32_t>{4U}, "host rule is token 4 only");
        for (const bool preserve : {false, true}) {
            const auto r = run(s, 0U, 4U, 5U, 1U, eager, preserve);
            check_tokens(r, want, 1U, "eager k=0");
            CHECK(same_bits(r.scores, s), "eager k=0: scores mutated");
        }
        const auto g = run(s, 0U, 4U, 5U, 1U, GraphControl{5U, 0U, true}, false);
        check_tokens(g, want, 1U, "graph k=0");
        CHECK(same_bits(g.scores, s), "graph k=0: scores mutated");
    }
    {
        std::vector<float> s(3000);
        for (std::size_t i = 0; i < s.size(); ++i) s[i] = static_cast<float>(i % 17U) - 8.0F;
        const auto r = run(s, 0U, 4U, 12001U, 1U, eager, false);
        check_tokens(r, std::vector<std::uint32_t>{12000U}, 1U, "eager k=0 wide");
        CHECK(same_bits(r.scores, s), "eager k=0 wide: scores mutated");
        // An invisible graph launch writes nothing.
        const auto g = run(s, 0U, 4U, 12001U, 1U, GraphControl{12001U, 0U, false}, false);
        check_tokens(g, {}, 1U, "graph not visible");
        CHECK(same_bits(g.scores, s), "graph not visible: scores mutated");
    }
}

std::vector<float> make_scores(std::mt19937& rng, std::uint32_t blocks, unsigned mode) {
    std::vector<float> s(blocks);
    std::normal_distribution<float> normal(0.0F, 3.0F);
    std::uniform_int_distribution<int> few(-3, 3), pick(0, 9);
    const float ninf = -std::numeric_limits<float>::infinity();
    for (auto& v : s) {
        if (mode == 0U) v = normal(rng);
        else if (mode == 1U) v = static_cast<float>(few(rng));
        else if (mode == 2U) v = pick(rng) < 6 ? ninf : normal(rng);
        else v = pick(rng) < 3 ? (pick(rng) < 5 ? 0.0F : -0.0F) : static_cast<float>(few(rng));
    }
    return s;
}

void test_random(std::mt19937& rng) {
    const radix3::NoControl eager{};
    for (unsigned mode = 0; mode < 4U; ++mode)
        for (const std::uint32_t k : {0U, 1U, 5U, 16U, 64U, 300U})
            for (const std::uint32_t blocks : {1U, 3U, 17U, 100U, 513U, 2049U, 5000U}) {
                const std::uint32_t ratio = 4U, visible = blocks * ratio + blocks % 3U;
                const auto s = make_scores(rng, blocks, mode);
                const auto want = rule::select_tokens(s, k, ratio, visible);
                char tag[96];
                std::snprintf(tag, sizeof(tag), "mode %u k %u blocks %u", mode, k, blocks);
                const auto words = static_cast<std::uint32_t>(want.size());
                const auto keep = run(s, k, ratio, visible, words, eager, true);
                check_tokens(keep, want, words, tag);
                CHECK(same_bits(keep.scores, s), "%s: preserve_scores mutated the scores", tag);
                const auto eaten = run(s, k, ratio, visible, words, eager, false);
                check_tokens(eaten, want, words, tag);
                const auto g = run(s, 0U, ratio, visible, words, GraphControl{visible, k, true}, false);
                check_tokens(g, want, words, tag);
            }
}

}  // namespace

int main() {
    int devices = 0;
    hok(hipGetDeviceCount(&devices), "hipGetDeviceCount");
    if (devices < 1) {
        std::fprintf(stderr, "hip_select_radix3_tests: no device\n");
        return 77;
    }
    std::mt19937 rng(20261123U);
    test_boundaries();
    test_random(rng);
    if (g_failures != 0) {
        std::printf("hip_select_radix3_tests: FAIL (%d)\n", g_failures);
        return 1;
    }
    std::printf("hip_select_radix3_tests: PASS\n");
    return 0;
}
