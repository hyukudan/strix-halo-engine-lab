// Residual bank batch, HIP body: every job of residual_mix::launch_bank_batch stores the bits of
// residual_mix::launch_mix(job_args(args, j)), the single-query one-row readout it replaces (incumbent kernel,
// incumbent schedule choice). Cases cover the decode shape of the bound engine (2 normalized bank rows, 2 sets of 24
// learned queries, width 2304), the paired-bank register form limit (width 4096), the loop form above it (5000),
// odd widths, one bank row (serial schedule), unnormalized banks and non-finite values. A last case runs the batch
// inside a captured graph and replays it. No model. Prints "residual bank batch hip: PASS" when every case matches.
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "core/residual_mix.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

namespace mix = strix::core::residual_mix;

bool ok(hipError_t status) { return status == hipSuccess; }

template <class T>
T* upload(const std::vector<T>& host) {
    T* device = nullptr;
    if (!ok(hipMalloc(&device, host.size() * sizeof(T)))) return nullptr;
    if (!ok(hipMemcpy(device, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice))) return nullptr;
    return device;
}

std::vector<float> values(std::size_t count, std::mt19937& random, float range, bool specials) {
    std::uniform_real_distribution<float> value(-range, range);
    std::vector<float> out(count);
    for (auto& x : out) x = value(random);
    if (specials) {
        const std::uint32_t bits[] = {0x7F800000U, 0xFF800000U, 0x7FC00000U, 0x80000000U, 0x00000001U, 0x7F7FFFFFU};
        for (std::size_t i = 5; i < out.size(); i += 613U) std::memcpy(&out[i], &bits[random() % 6U], sizeof(float));
    }
    return out;
}

struct Case {
    const char* name;
    unsigned width;
    unsigned banks;
    unsigned sets;
    unsigned per_set;
    bool normalized;
    bool specials;
    bool graph;
};

// Bitwise comparison; NaN payloads included.
std::size_t mismatches(const std::vector<float>& a, const std::vector<float>& b) {
    std::size_t bad = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        std::uint32_t x = 0, y = 0;
        std::memcpy(&x, &a[i], sizeof x);
        std::memcpy(&y, &b[i], sizeof y);
        bad += x != y ? 1U : 0U;
    }
    return bad;
}

bool run_case(const Case& test, std::uint32_t seed, hipStream_t stream) {
    std::mt19937 random(seed);
    const auto width = test.width;
    std::vector<float*> banks;
    for (unsigned b = 0; b < test.banks; ++b) banks.push_back(upload(values(width, random, 2.0F, test.specials)));
    std::vector<float*> queries, batch_out, single_out;
    std::vector<std::size_t> counts;
    for (unsigned s = 0; s < test.sets; ++s) {
        const auto count = test.per_set + s;  // unequal sets exercise locate()
        counts.push_back(count);
        queries.push_back(upload(values(std::size_t{count} * width, random, 0.25F, test.specials)));
        const std::vector<float> poison(std::size_t{count} * width, -7.0F);
        batch_out.push_back(upload(poison));
        single_out.push_back(upload(poison));
    }
    for (auto* p : banks) if (!p) return false;
    for (std::size_t s = 0; s < queries.size(); ++s)
        if (!queries[s] || !batch_out[s] || !single_out[s]) return false;

    mix::BankBatchArgs args{};
    for (unsigned b = 0; b < test.banks; ++b) args.banks[b] = banks[b];
    for (unsigned s = 0; s < test.sets; ++s)
        args.sets[s] = {queries[s], batch_out[s], static_cast<unsigned>(counts[s])};
    args.width = width;
    args.bank_count = test.banks;
    args.set_count = test.sets;
    args.epsilon = 1e-6F;
    args.normalized = test.normalized;
    if (!mix::fits(args)) {
        std::printf("residual bank batch hip %s: FAIL (fits)\n", test.name);
        return false;
    }

    // Incumbent: one launch_mix per job, into the single-query image.
    auto single = args;
    for (unsigned s = 0; s < test.sets; ++s) single.sets[s].output = single_out[s];
    const auto jobs = mix::job_count(args);
    for (unsigned job = 0; job < jobs; ++job)
        if (!ok(mix::launch_mix(mix::job_args(single, job), stream))) {
            std::printf("residual bank batch hip %s: FAIL (incumbent launch)\n", test.name);
            return false;
        }

    if (test.graph) {
        hipGraph_t graph{};
        hipGraphExec_t exec{};
        if (!ok(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal))) return false;
        const auto status = mix::launch_bank_batch(args, stream);
        if (!ok(hipStreamEndCapture(stream, &graph)) || !ok(status) ||
            !ok(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0))) {
            std::printf("residual bank batch hip %s: FAIL (graph capture)\n", test.name);
            return false;
        }
        for (int replay = 0; replay < 3; ++replay)
            if (!ok(hipGraphLaunch(exec, stream))) return false;
        (void)hipGraphExecDestroy(exec);
        (void)hipGraphDestroy(graph);
    } else if (!ok(mix::launch_bank_batch(args, stream))) {
        std::printf("residual bank batch hip %s: FAIL (batch launch)\n", test.name);
        return false;
    }
    if (!ok(hipStreamSynchronize(stream))) return false;

    std::size_t bad = 0, total = 0;
    for (unsigned s = 0; s < test.sets; ++s) {
        std::vector<float> a(counts[s] * width), b(counts[s] * width);
        if (!ok(hipMemcpy(a.data(), batch_out[s], a.size() * sizeof(float), hipMemcpyDeviceToHost)) ||
            !ok(hipMemcpy(b.data(), single_out[s], b.size() * sizeof(float), hipMemcpyDeviceToHost)))
            return false;
        bad += mismatches(a, b);
        total += a.size();
    }
    for (auto* p : banks) (void)hipFree(p);
    for (std::size_t s = 0; s < queries.size(); ++s) {
        (void)hipFree(queries[s]);
        (void)hipFree(batch_out[s]);
        (void)hipFree(single_out[s]);
    }
    std::printf("residual bank batch hip %s: %s jobs %u values %zu mismatches %zu\n", test.name,
                bad == 0U ? "PASS" : "FAIL", jobs, total, bad);
    return bad == 0U;
}

// Refusals: a batch whose output overlaps a bank or a query image, or an empty batch, launches nothing.
bool refusals(hipStream_t stream) {
    std::vector<float> host(4U * 64U, 1.0F);
    auto* buffer = upload(host);
    if (!buffer) return false;
    mix::BankBatchArgs args{};
    args.banks[0] = buffer;
    args.banks[1] = buffer + 64;
    args.sets[0] = {buffer + 128, buffer + 64, 1U};  // output == bank 1
    args.width = 64;
    args.bank_count = 2;
    args.set_count = 1;
    args.epsilon = 1e-6F;
    args.normalized = true;
    bool good = mix::launch_bank_batch(args, stream) == hipErrorInvalidValue;
    args.sets[0] = {buffer + 128, buffer + 128, 1U};  // output == query
    good = good && mix::launch_bank_batch(args, stream) == hipErrorInvalidValue;
    args.set_count = 0;
    good = good && mix::launch_bank_batch(args, stream) == hipErrorInvalidValue;
    (void)hipFree(buffer);
    std::printf("residual bank batch hip refusals: %s\n", good ? "PASS" : "FAIL");
    return good;
}

}  // namespace

int main() {
    hipStream_t stream{};
    if (!ok(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking))) {
        std::printf("residual bank batch hip: FAIL (stream)\n");
        return 2;
    }
    const Case cases[] = {
        {"decode-2x24-w2304", 2304, 2, 2, 24, true, false, false},
        {"decode-2x24-w2304-graph", 2304, 2, 2, 24, true, false, true},
        {"specials-w2304", 2304, 2, 2, 5, true, true, false},
        {"unnormalized-3banks-w2304", 2304, 3, 3, 4, false, false, false},
        {"register-limit-w4096", 4096, 2, 1, 6, true, false, false},
        {"loop-form-w5000", 5000, 2, 2, 3, true, false, false},
        {"odd-w333-8banks", 333, 8, 4, 2, false, true, false},
        {"one-bank-serial-w2304", 2304, 1, 2, 3, true, false, false},
        {"one-bank-unnormalized-w7", 7, 1, 1, 3, false, false, false},
    };
    bool good = true;
    std::uint32_t seed = 20261122U;
    for (const auto& test : cases) good = run_case(test, seed++, stream) && good;
    good = refusals(stream) && good;
    (void)hipStreamDestroy(stream);
    std::printf("residual bank batch hip: %s\n", good ? "PASS" : "FAIL");
    return good ? 0 : 1;
}
