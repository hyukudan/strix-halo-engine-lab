// Same-TU diagnostic, as in the existing QSA projection fixtures. No new public API.
#include "../src/hip_indexed_attention_launcher.hip"
#include <chrono>
#include <iostream>
#include <iomanip>

namespace {
using namespace strix;
void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
struct Memory {
    void* p = nullptr;
    explicit Memory(std::size_t bytes) { check(hipMalloc(&p, bytes)); }
    ~Memory() { if (p) static_cast<void>(hipFree(p)); }
    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;
};
struct Events {
    hipEvent_t begin = nullptr, end = nullptr;
    Events() { check(hipEventCreate(&begin)); check(hipEventCreate(&end)); }
    ~Events() { static_cast<void>(hipEventDestroy(begin)); static_cast<void>(hipEventDestroy(end)); }
};
struct Stream {
    hipStream_t value = nullptr;
    Stream() { check(hipStreamCreateWithFlags(&value, hipStreamNonBlocking)); }
    ~Stream() { static_cast<void>(hipStreamSynchronize(value)); static_cast<void>(hipStreamDestroy(value)); }
};
constexpr unsigned T = 8192U, Rows = 64U, Guard = 16U;
constexpr std::uint32_t Poison = 0x7fa12345U;
using Views = QsaRowsInputViews<Rows>;
// Distinct payloads, signed zeros, both subnormal signs, normals and overflow operands.
constexpr std::array<std::uint16_t, 16> special{
    0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080,
    0x7f80, 0xff80, 0x7fc1, 0xffc2, 0x7f81, 0xff82, 0x7f7f, 0xff7f};
std::uint16_t finite(std::size_t i) {
    // Stable integer hash: independent signs/mantissas and enough exponent range
    // to expose cancellation and intermediate rounding, without finite overflow.
    auto hash = static_cast<std::uint32_t>(i);
    hash = (hash ^ (hash >> 16U)) * 0x7feb352dU;
    hash = (hash ^ (hash >> 15U)) * 0x846ca68bU;
    hash ^= hash >> 16U;
    return static_cast<std::uint16_t>((hash & 0x807fU) | ((117U + (hash >> 8U) % 21U) << 7U));
}
double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return (v[(v.size() - 1U) / 2U] + v[v.size() / 2U]) * 0.5;
}
// Independent oracle: serial FP64 dot over the unmodified BF16 operand bits.
// Not timed; BF16 products fit exactly in FP64, including the full exponent range.
__host__ __device__ double oracle_bf16(std::uint16_t value) {
    const unsigned exponent = (value >> 7U) & 255U, mantissa = value & 127U;
    if (exponent == 0U) {
        const double magnitude = static_cast<double>(mantissa) * 0x1p-133;
        return (value & 0x8000U) ? -magnitude : magnitude;
    }
    // Decode directly into FP64 so the oracle does not depend on FP32 denormal
    // conversion policy. Infinity/NaN classification is retained, payload ignored.
    const std::uint64_t bits = (static_cast<std::uint64_t>(value & 0x8000U) << 48U) |
        (static_cast<std::uint64_t>(exponent == 255U ? 2047U : exponent + 896U) << 52U) |
        (static_cast<std::uint64_t>(mantissa) << 45U);
    return __builtin_bit_cast(double, bits);
}
__global__ void reference_kernel(const std::uint16_t* x, const std::uint16_t* w,
    double* out, unsigned width, unsigned outputs) {
    const unsigned column = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned row = blockIdx.y;
    if (column >= outputs) return;
    double sum = 0.0;
    for (unsigned f = 0U; f < width; ++f)
        sum += oracle_bf16(x[static_cast<std::size_t>(row) * width + f]) *
            oracle_bf16(w[static_cast<std::size_t>(column) * width + f]);
    out[static_cast<std::size_t>(row) * outputs + column] = sum;
}
float from_bits(std::uint32_t bits) {
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
__global__ __launch_bounds__(128) void mapping_trace_kernel(const Views* groups, unsigned width,
    std::uint16_t* operands, float* steps, unsigned row, unsigned column) {
    qsa_index::rows_wmma_body<true>(groups, width, operands, steps, row, column);
}
__global__ __launch_bounds__(128) void mapping_feedback_control_kernel(const Views* groups, unsigned width) {
    qsa_index::rows_wmma_body<false, true>(groups, width);
}
// Diagnose one known failing cell even when the corrected production route passes.
// Recompute from captured fragment bits, NOT from regenerated integer weights.
void operand_diagnostics(const std::vector<std::uint16_t>& operands, const std::vector<float>& steps,
    const std::vector<std::uint16_t>& x, const std::vector<std::uint16_t>& w,
    unsigned width, unsigned row, unsigned column, float expected) {
    double sum = 0.0;
    unsigned nonzero = 0U, input_mismatches = 0U, replica_mismatches = 0U;
    std::cerr << std::setprecision(17);
    for (unsigned k = 0U; k < width; ++k) {
        const auto a = operands[k], b = operands[width + k];
        input_mismatches += static_cast<unsigned>(a != x[static_cast<std::size_t>(row) * width + k]);
        input_mismatches += static_cast<unsigned>(b != w[static_cast<std::size_t>(column) * width + k]);
        replica_mismatches += static_cast<unsigned>(a != operands[2U * width + k]);
        replica_mismatches += static_cast<unsigned>(b != operands[3U * width + k]);
        const double product = oracle_bf16(a) * oracle_bf16(b);
        sum += product;
        if (product == 0.0) continue;
        ++nonzero;
        std::cerr << "INDEX_OPERAND_PRODUCT r=" << row << " o=" << column << " k=" << k
            << " a_bits=0x" << std::hex << a << " b_bits=0x" << b << std::dec
            << " a=" << oracle_bf16(a) << " b=" << oracle_bf16(b)
            << " product=" << product << " fp64_prefix=" << sum << '\n';
    }
    float previous = 0.0F, previous_feedback = 0.0F;
    unsigned panel_mismatches = 0U;
    for (unsigned k = 0U; k < width; k += 16U) {
        const unsigned at = k / 16U * 3U;
        const float panel = steps[at], outer = steps[at + 1U], feedback = steps[at + 2U];
        double fp64_panel = 0.0;
        for (unsigned f = 0U; f < 16U; ++f)
            fp64_panel += oracle_bf16(operands[k + f]) * oracle_bf16(operands[width + k + f]);
        panel_mismatches += static_cast<unsigned>(static_cast<double>(panel) != fp64_panel);
        if (panel != 0.0F || outer != previous || feedback != previous_feedback || static_cast<double>(panel) != fp64_panel)
            std::cerr << "INDEX_OPERAND_STEP k=" << k << " panel=" << panel << " fp64_panel=" << fp64_panel
                << " fp32_outer=" << outer << " native_feedback=" << feedback << '\n';
        previous = outer; previous_feedback = feedback;
    }
    std::cerr << "INDEX_OPERAND_FP64 r=" << row << " o=" << column << " width=" << width
        << " nonzero_products=" << nonzero << " sum=" << sum << " expected=" << expected
        << " input_bit_mismatches=" << input_mismatches << " replica_bit_mismatches=" << replica_mismatches
        << " zero_c_panel_mismatches=" << panel_mismatches
        << " fp32_outer=" << previous << " native_feedback=" << previous_feedback
        << " feedback_differs_from_outer=" << (previous_feedback != previous) << std::endl;
    std::cerr << std::setprecision(6);
    require(input_mismatches == 0U && replica_mismatches == 0U, "WMMA captured operand bits differ");
    require(sum == static_cast<double>(expected), "captured operand FP64 mapping mismatch");
    require(panel_mismatches == 0U, "zero-C WMMA panel differs from captured FP64 products");
    require(previous == expected, "zero-C WMMA panel accumulation mapping mismatch");
}
// Metrics against unrounded FP64 for every output column, plus each q/k tensor.
// Nonfinites are counted separately: no NaN-payload/signed-zero equivalence claim.
struct Error {
    double max_abs = 0.0, sq = 0.0, norm = 0.0;
    std::size_t finite = 0U, special = 0U, classification_mismatch = 0U;
    void add(double expected, float actual) {
        if (std::isfinite(expected) && std::isfinite(actual)) {
            const double delta = static_cast<double>(actual) - expected;
            max_abs = std::max(max_abs, std::fabs(delta));
            sq += delta * delta; norm += expected * expected; ++finite;
        } else {
            ++special;
            // Overflow to FP32/BF16 is expected; NaN payloads and denormal
            // handling of WMMA are architecture policy, not the old tree's.
            const float fp32 = static_cast<float>(expected);
            std::uint32_t range_bits = 0U;
            std::memcpy(&range_bits, &fp32, sizeof(range_bits));
            const float rounded_range = from_bits((range_bits + 0x7fffU + ((range_bits >> 16U) & 1U)) & 0xffff0000U);
            if (!((std::isnan(rounded_range) && std::isnan(actual)) ||
                  (std::isinf(rounded_range) && std::isinf(actual) && std::signbit(rounded_range) == std::signbit(actual))))
                ++classification_mismatch;
        }
    }
    void print(unsigned kind, const std::string& output) const {
        const double rel = norm != 0.0 ? std::sqrt(sq / norm) : (sq == 0.0 ? 0.0 : std::numeric_limits<double>::infinity());
        std::cout << "WMMA_FP64 case=" << kind << " output=" << output << " max_abs=" << max_abs
            << " rel_l2=" << rel << " finite=" << finite << " special=" << special
            << " classification_mismatch=" << classification_mismatch << '\n';
    }
};
// Report bounded evidence before throwing: repeated values can have many matching
// coordinates, so these are mapping/K-chunk hints, not a proof of a permutation.
void mapping_diagnostics(const std::vector<float>& got, const std::vector<float>& expected,
    unsigned rows, unsigned outputs, const char* route, unsigned pattern) {
    std::size_t mismatches = 0U;
    bool scaled = true, have_scale = false;
    double scale = 0.0;
    for (unsigned r = 0U; r < rows; ++r) for (unsigned o = 0U; o < outputs; ++o) {
        const auto at = static_cast<std::size_t>(r) * outputs + o;
        const float actual = got[at], want = expected[at];
        if (want == 0.0F) scaled = scaled && actual == 0.0F;
        else {
            const double ratio = static_cast<double>(actual) / static_cast<double>(want);
            if (!have_scale) { scale = ratio; have_scale = true; }
            scaled = scaled && std::isfinite(ratio) && ratio == scale;
        }
        if (actual == want) continue;
        if (mismatches++ >= 8U) continue;
        std::cerr << "INDEX_MAPPING_MISMATCH route=" << route << " pattern=" << pattern
            << " r=" << r << " o=" << o << " got=" << actual << " expected=" << want;
        // Search the current 16x16 fragment; print the first match and total
        // multiplicity so a collision cannot masquerade as a unique lane map.
        unsigned matches = 0U, match_r = 0U, match_o = 0U;
        for (unsigned rr = r / 16U * 16U; rr < r / 16U * 16U + 16U; ++rr)
            for (unsigned oo = o / 16U * 16U; oo < o / 16U * 16U + 16U; ++oo)
                if (got[at] == expected[static_cast<std::size_t>(rr) * outputs + oo]) {
                    if (matches == 0U) { match_r = rr; match_o = oo; }
                    ++matches;
                }
        if (matches != 0U) std::cerr << " matches_expected_at=(" << match_r << ',' << match_o
            << ") tile_matches=" << matches << " possible_lane_permutation";
        const auto transposed = static_cast<std::size_t>(r / 16U * 16U + o % 16U) * outputs + o / 16U * 16U + r % 16U;
        if (actual == expected[transposed]) std::cerr << " matches_tile_transpose";
        if (want != 0.0F) std::cerr << " got/expected=" << static_cast<double>(actual) / static_cast<double>(want);
        std::cerr << '\n';
    }
    if (mismatches != 0U) {
        std::cerr << "INDEX_MAPPING_DIAG route=" << route << " pattern=" << pattern
            << " mismatches=" << mismatches << '/' << expected.size();
        if (scaled && have_scale) std::cerr << " uniform_scale=" << scale << " possible_K_chunk_loss_or_duplication";
        else std::cerr << " nonuniform_error: inspect_lane_map_operand_bits_and_K_chunks";
        std::cerr << std::endl;
    }
    require(mismatches == 0U, "sparse integer mapping oracle mismatch");
}
// Pattern 0 checks widely separated K16 planes; pattern 1 labels every cell in
// a 16x16 tile uniquely (and uses powers of two for BF16-exact row scaling).
void mapping_oracle(hipStream_t stream, unsigned pattern) {
    constexpr unsigned width = 2560U, outputs = 640U;
    const auto bits = [](int value) {
        const float f = static_cast<float>(value);
        std::uint32_t u = 0U; std::memcpy(&u, &f, sizeof(u));
        return static_cast<std::uint16_t>(u >> 16U);
    };
    std::vector<std::uint16_t> x(Rows * width, 0U), w(outputs * width);
    constexpr std::array<unsigned, 4> planes{0U, 16U, 1280U, 2544U};
    for (unsigned r = 0U; r < Rows; ++r) for (unsigned plane : planes) {
        if (pattern == 0U || plane == planes.back())
            x[r * width + plane + r % 16U] = bits(static_cast<int>(pattern == 0U ? r / 16U + 1U : 1U << (r / 16U)));
    }
    for (unsigned o = 0U; o < outputs; ++o) for (unsigned f = 0U; f < width; ++f)
        w[o * width + f] = bits(pattern == 0U ? static_cast<int>((o * 3U + f * 5U) % 11U) - 5 :
            static_cast<int>(16U * (o % 16U) + f % 16U + 1U) * (o / 16U % 2U == 0U ? 1 : -1));
    std::vector<float> expected(Rows * outputs);
    for (unsigned r = 0U; r < Rows; ++r) for (unsigned o = 0U; o < outputs; ++o) {
        int sum = 0;
        for (unsigned plane : planes) {
            if (pattern != 0U && plane != planes.back()) continue;
            const unsigned f = plane + r % 16U;
            const int weight = pattern == 0U ? static_cast<int>((o * 3U + f * 5U) % 11U) - 5 :
                static_cast<int>(16U * (o % 16U) + f % 16U + 1U) * (o / 16U % 2U == 0U ? 1 : -1);
            sum += weight * static_cast<int>(pattern == 0U ? r / 16U + 1U : 1U << (r / 16U));
        }
        expected[r * outputs + o] = static_cast<float>(sum);
    }
    Memory dx(x.size() * sizeof(x[0])), dw(w.size() * sizeof(w[0])), out(Rows * outputs * sizeof(float));
    check(hipMemcpyAsync(dx.p, x.data(), x.size() * sizeof(x[0]), hipMemcpyHostToDevice, stream));
    check(hipMemcpyAsync(dw.p, w.data(), w.size() * sizeof(w[0]), hipMemcpyHostToDevice, stream));
    Views v{}; v.count = 2U; v.first[1] = 512U; v.first[2] = outputs;
    v.weights[0] = dw.p; v.weights[1] = static_cast<const std::uint16_t*>(dw.p) + 512U * width;
    for (unsigned r = 0U; r < Rows; ++r) {
        v.input[r] = static_cast<const std::uint16_t*>(dx.p) + r * width;
        v.output[0][r] = static_cast<float*>(out.p) + r * outputs;
        v.output[1][r] = v.output[0][r] + 512U;
    }
    std::vector<float> result(Rows * outputs);
    float traced = 0.0F;
    // Check the unchanged exact route first, even if native WMMA subsequently fails.
    for (bool native : {false, true}) {
        if (native && pattern == 0U) {
            // Same body, staging, lane loads and stores as production. Extra
            // native-C chain is diagnostic only; never used in timed launches.
            Memory groups(sizeof(v)), operands(4U * width * sizeof(std::uint16_t)),
                steps(width / 16U * 3U * sizeof(float));
            check(hipMemcpyAsync(groups.p, &v, sizeof(v), hipMemcpyHostToDevice, stream));
            check(hipMemsetAsync(operands.p, 0xff, 4U * width * sizeof(std::uint16_t), stream));
            check(hipMemsetAsync(steps.p, 0xff, width / 16U * 3U * sizeof(float), stream));
            hipLaunchKernelGGL(mapping_trace_kernel, dim3(outputs / 64U), dim3(128), 0, stream,
                static_cast<const Views*>(groups.p), width, static_cast<std::uint16_t*>(operands.p),
                static_cast<float*>(steps.p), 0U, 4U);
            check(hipGetLastError()); check(hipStreamSynchronize(stream));
            std::vector<std::uint16_t> captured(4U * width);
            std::vector<float> captured_steps(width / 16U * 3U);
            check(hipMemcpy(captured.data(), operands.p, captured.size() * sizeof(captured[0]), hipMemcpyDeviceToHost));
            check(hipMemcpy(captured_steps.data(), steps.p, captured_steps.size() * sizeof(float), hipMemcpyDeviceToHost));
            check(hipMemcpy(&traced, static_cast<float*>(out.p) + 4U, sizeof(float), hipMemcpyDeviceToHost));
            // Uninstrumented old feedback identity: detect a trace that masks a
            // scheduling-sensitive failure instead of blaming arithmetic.
            hipLaunchKernelGGL(mapping_feedback_control_kernel, dim3(outputs / 64U), dim3(128), 0, stream,
                static_cast<const Views*>(groups.p), width);
            check(hipGetLastError()); check(hipStreamSynchronize(stream));
            float control = 0.0F;
            check(hipMemcpy(&control, static_cast<float*>(out.p) + 4U, sizeof(float), hipMemcpyDeviceToHost));
            std::cerr << std::setprecision(17) << "INDEX_FEEDBACK_CONTROL r=0 o=4 uninstrumented=" << control
                << " traced=" << captured_steps.back() << " expected=" << expected[4U] << '\n';
            operand_diagnostics(captured, captured_steps, x, w, width, 0U, 4U, expected[4U]);
        }
        auto& ib = qpf2k_index_batch();
        require(!ib.active && ib.groups.empty(), "pending batch before mapping oracle");
        check(hipMemsetAsync(out.p, 0xff, result.size() * sizeof(float), stream));
        ib.active = true; ib.wmma = native;
        const auto launched = qpf2k_launch_index_wmma.load();
        const auto batches = qpf2k_launch_index_batch.load();
        launch_qsa_rows_input<WeightKind::bf16, Rows>(false, false, v, width, stream);
        qpf2k_index_flush(stream); ib.active = false; ib.wmma = false;
        require(qpf2k_launch_index_wmma.load() == launched + (native ? 1U : 0U), "mapping oracle WMMA route mismatch");
        require(qpf2k_launch_index_batch.load() == batches + 1U, "mapping oracle batch launch mismatch");
        check(hipStreamSynchronize(stream));
        check(hipMemcpy(result.data(), out.p, result.size() * sizeof(float), hipMemcpyDeviceToHost));
        if (native && pattern == 0U) require(result[4U] == traced, "traced/production WMMA mismatch");
        mapping_diagnostics(result, expected, Rows, outputs, native ? "wmma" : "exact_batch", pattern);
        std::cout << (native ? "WMMA_MAPPING_EXACT" : "BATCH_MAPPING_EXACT")
            << " pattern=" << pattern << " rows=64 outputs=640 width=2560" << std::endl;
    }
}
// Compare the governed production scope against the original inactive dispatcher,
// including every output guard. Rejected WMMA layouts may still batch exactly.
template<unsigned N>
void fallback_case(hipStream_t stream, unsigned width, unsigned q, unsigned k,
                   bool unaligned, bool change_matrix, bool change_width = false) {
    const unsigned outputs = q + k, stride = outputs + 2U * Guard;
    const unsigned allocation_width = std::max(width, 2560U);
    std::vector<std::uint16_t> x(N * allocation_width + 8U), w(outputs * allocation_width * 2U);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = finite(i);
    for (std::size_t i = 0; i < w.size(); ++i) w[i] = finite(i * 3U + 1U);
    Memory dx(x.size() * sizeof(x[0])), dw(w.size() * sizeof(w[0]));
    check(hipMemcpyAsync(dx.p, x.data(), x.size() * sizeof(x[0]), hipMemcpyHostToDevice, stream));
    check(hipMemcpyAsync(dw.p, w.data(), w.size() * sizeof(w[0]), hipMemcpyHostToDevice, stream));
    const std::size_t words = 2U * N * stride;
    Memory old(words * sizeof(float)), next(words * sizeof(float));
    std::vector<std::uint32_t> poison(words, Poison), ref(words), got(words);
    for (void* out : {old.p, next.p})
        check(hipMemcpyAsync(out, poison.data(), words * sizeof(float), hipMemcpyHostToDevice, stream));
    const auto launch = [&](void* out) {
        for (unsigned group = 0; group < 2U; ++group) {
            const unsigned hidden = change_width && group == 1U ? 2560U : width;
            QsaRowsInputViews<N> v{};
            v.count = k == 0U ? 1U : 2U; v.first[1] = q;
            if (k != 0U) v.first[2] = outputs;
            const auto* weights = static_cast<const std::uint16_t*>(dw.p) +
                (change_matrix && group == 1U ? outputs * allocation_width : 0U);
            v.weights[0] = weights; v.weights[1] = weights + q * hidden;
            for (unsigned r = 0; r < N; ++r) {
                v.input[r] = static_cast<const std::uint16_t*>(dx.p) + r * allocation_width + (unaligned ? 1U : 0U);
                v.output[0][r] = static_cast<float*>(out) + (group * N + r) * stride + Guard;
                v.output[1][r] = v.output[0][r] + q;
            }
            launch_qsa_rows_input<WeightKind::bf16, N>(false, false, v, hidden, stream);
            if constexpr (N == 64U) if (!unaligned && width % 256U == 0U && qpf2k_index_batch().active) {
                require(qpf2k_index_batch().groups.size() == (group == 0U || change_matrix || change_width ? 1U : 2U),
                    "matrix/width transition did not flush pending group");
            }
        }
    };
    launch(old.p);
    const auto batches = qpf2k_launch_index_batch.load(), wmma = qpf2k_launch_index_wmma.load();
    {
        const Qpf2kIndexScope scope(0U);
        require(qpf2k_index_batch().active && qpf2k_index_batch().wmma, "production scope did not activate");
        launch(next.p);
        qpf2k_index_flush(stream);
    }
    require(!qpf2k_index_batch().active && !qpf2k_index_batch().wmma && qpf2k_index_batch().groups.empty(),
        "production scope did not reset");
    require(qpf2k_launch_index_wmma.load() == wmma, "fallback launched WMMA");
    const unsigned expected = N == 64U && !unaligned && width % 256U == 0U ? (change_matrix || change_width ? 2U : 1U) : 0U;
    require(qpf2k_launch_index_batch.load() == batches + expected, "wrong fallback batch count");
    check(hipStreamSynchronize(stream));
    check(hipMemcpy(ref.data(), old.p, words * sizeof(float), hipMemcpyDeviceToHost));
    check(hipMemcpy(got.data(), next.p, words * sizeof(float), hipMemcpyDeviceToHost));
    require(ref == got, "fallback differs from original route bytes");
    for (unsigned row = 0; row < 2U * N; ++row) for (unsigned o = 0; o < stride; ++o)
        require((got[row * stride + o] == Poison) == (o < Guard || o >= Guard + outputs), "fallback guard/missing output");
    std::cout << "FALLBACK_EXACT rows=" << N << " width=" << width << " q=" << q << " k=" << k
        << " unaligned=" << unaligned << " matrix_change=" << change_matrix << " width_change=" << change_width << '\n';
}
void negative_dispatch(hipStream_t stream) {
    {
        const Qpf2kIndexScope scope(0U, false);
        require(!qpf2k_index_batch().wmma, "non-fast profile activated candidate (outside READY) arithmetic");
    }
    for (unsigned mode : {1U, 2U, 3U}) {
        const Qpf2kIndexScope scope(mode);
        require(!qpf2k_index_batch().active && !qpf2k_index_batch().wmma, "non-eager scope activated batching");
    }
    try {
        const Qpf2kIndexScope scope(0U);
        qpf2k_index_batch().groups.emplace_back();
        throw std::runtime_error("scope unwind probe");
    } catch (const std::runtime_error&) {}
    require(!qpf2k_index_batch().active && !qpf2k_index_batch().wmma && qpf2k_index_batch().groups.empty(),
        "exception did not reset production scope");
    Views invalid{}; invalid.count = 1U; invalid.first[1] = 128U; invalid.first[0] = 1U;
    require(!qpf2k_index_wmma_layout(invalid, 2560U), "nonzero segment origin admitted");
    invalid.first[0] = 0U; invalid.weights[0] = reinterpret_cast<void*>(1U);
    require(!qpf2k_index_wmma_layout(invalid, 2560U), "unaligned matrix admitted");
    fallback_case<64U>(stream, 2304U, 512U, 128U, false, false);
    fallback_case<64U>(stream, 2558U, 512U, 128U, false, false);
    fallback_case<64U>(stream, 2560U, 256U, 128U, false, false);
    fallback_case<64U>(stream, 2560U, 64U, 0U, false, true);
    fallback_case<64U>(stream, 2304U, 64U, 0U, false, false, true);
    fallback_case<64U>(stream, 2560U, 512U, 128U, true, false);
    fallback_case<32U>(stream, 2560U, 512U, 128U, false, false);
    fallback_case<16U>(stream, 2560U, 512U, 128U, false, false);
    fallback_case<8U>(stream, 2560U, 512U, 128U, false, false);
}
void run(unsigned kind, unsigned repetitions, hipStream_t stream) {
    const ModelContract model{};
    const auto width = static_cast<unsigned>(model.hidden_size);
    const auto q = static_cast<unsigned>(model.indexer_n_heads * model.indexer_head_dim);
    const auto k = static_cast<unsigned>(model.indexer_kv_heads * model.indexer_head_dim);
    require(width == 2560U && q == 512U && k == 128U, "update fixture for changed model contract");
    const unsigned outputs = q + k, stride = outputs + 2U * Guard;
    const std::size_t elements = static_cast<std::size_t>(T) * stride;
    std::vector<std::uint16_t> x(static_cast<std::size_t>(T) * width);
    std::vector<std::uint16_t> w(static_cast<std::size_t>(outputs) * width);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = finite(i);
    for (std::size_t i = 0; i < w.size(); ++i) w[i] = finite(i * 3U + 1U);
    // Spread each special across leaves and steps without making every row/output NaN.
    if (kind == 1U) for (unsigned r = 0; r < T; r += 31U)
        x[static_cast<std::size_t>(r) * width + (r * 257U) % width] = special[(r / 31U) % special.size()];
    if (kind == 2U) for (unsigned o = 0; o < outputs; o += 7U)
        w[static_cast<std::size_t>(o) * width + (o * 257U) % width] = special[(o / 7U) % special.size()];
    if (kind == 1U) {
        // Entire rows exercise signed-zero chains and underflow, independently of finite noise.
        for (unsigned r = 0; r < 8U; ++r)
            std::fill_n(x.data() + static_cast<std::size_t>(r) * width, width, special[r]);
    }
    Memory dx(x.size() * sizeof(x[0])), dw(w.size() * sizeof(w[0]));
    Memory baseline(elements * sizeof(float)), batched(elements * sizeof(float)), legacy(elements * sizeof(float));
    Memory wmma(elements * sizeof(float)), oracle(static_cast<std::size_t>(T) * outputs * sizeof(double));
    check(hipMemcpyAsync(dx.p, x.data(), x.size() * sizeof(x[0]), hipMemcpyHostToDevice, stream));
    check(hipMemcpyAsync(dw.p, w.data(), w.size() * sizeof(w[0]), hipMemcpyHostToDevice, stream));
    std::vector<std::uint32_t> initial(elements, Poison), ref(elements), result(elements);
    for (void* p : {baseline.p, batched.p, legacy.p, wmma.p})
        check(hipMemcpyAsync(p, initial.data(), elements * sizeof(float), hipMemcpyHostToDevice, stream));
    check(hipStreamSynchronize(stream));
    const auto views = [&](void* output) {
        std::vector<Views> groups(T / Rows);
        for (unsigned g = 0; g < T / Rows; ++g) {
            auto& v = groups[g]; v.count = 2U; v.first[1] = q; v.first[2] = outputs;
            v.weights[0] = dw.p; v.weights[1] = static_cast<const std::uint16_t*>(dw.p) + static_cast<std::size_t>(q) * width;
            for (unsigned r = 0; r < Rows; ++r) {
                const auto row = g * Rows + r;
                v.input[r] = static_cast<const std::uint16_t*>(dx.p) + static_cast<std::size_t>(row) * width;
                v.output[0][r] = static_cast<float*>(output) + static_cast<std::size_t>(row) * stride + Guard;
                v.output[1][r] = v.output[0][r] + q;
            }
            // Emulate a dense-prefix query skip followed by normal q/k groups.
            if (kind == 3U && g < T / Rows / 2U) {
                v.count = 1U; v.weights[0] = v.weights[1]; v.first[1] = k; v.first[2] = 0U;
                for (unsigned r = 0; r < Rows; ++r) v.output[0][r] = v.output[1][r];
            }
        }
        return groups;
    };
    const auto old = views(baseline.p), next = views(batched.p), serial = views(legacy.p), native = views(wmma.p);
    const auto launch_old = [&] {
        for (const auto& v : old)
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_index::rows_kernel<Rows, 4U, 2U, 16U>),
                dim3((v.first[v.count] + 31U) / 32U, Rows / 4U), dim3(512), width * 4U * sizeof(float), stream, v, width);
        check(hipGetLastError());
    };
    const auto launch_batch = [&](bool use_wmma = false) {
        auto& ib = qpf2k_index_batch();
        require(!ib.active && ib.groups.empty(), "unexpected pending production batch");
        ib.active = true; ib.wmma = use_wmma;
        const auto deferred = ib.deferred;
        const auto launches = qpf2k_launch_index_batch.load();
        const auto wmma_launches = qpf2k_launch_index_wmma.load();
        // Exercise the production deferral and ring, not a separate test-only launcher.
        for (const auto& v : (use_wmma ? native : next)) launch_qsa_rows_input<WeightKind::bf16, Rows>(false, false, v, width, stream);
        require(ib.deferred - deferred == T / Rows, "candidate did not defer every group");
        qpf2k_index_flush(stream);
        ib.active = false; ib.wmma = false;
        require(qpf2k_launch_index_batch.load() - launches == (kind == 3U ? 2U : 1U), "wrong batched q/k launch count");
        require(qpf2k_launch_index_wmma.load() - wmma_launches == (use_wmma ? (kind == 3U ? 2U : 1U) : 0U),
            "wrong WMMA launch count / precedence");
    };
    const auto launch_legacy = [&] {
        for (const auto& v : serial) for (unsigned first = 0; first < Rows; first += 16U) {
            QsaRowsInputViews<16U> slice{}; slice.count = v.count;
            std::copy(std::begin(v.weights), std::end(v.weights), slice.weights);
            std::copy(std::begin(v.first), std::end(v.first), slice.first);
            for (unsigned r = 0; r < 16U; ++r) {
                slice.input[r] = v.input[first + r];
                for (unsigned m = 0; m < v.count; ++m) slice.output[m][r] = v.output[m][first + r];
            }
            hipLaunchKernelGGL(HIP_KERNEL_NAME(linear_to_float_rows_kernel<WeightKind::bf16, 16U>),
                dim3(slice.first[slice.count]), dim3(256), 0, stream, slice, width);
        }
        check(hipGetLastError());
    };
    launch_old(); launch_legacy();
    // More than eight launches wrap the production pinned-table ring.
    for (unsigned i = 0; i < 10U; ++i) launch_batch();
    check(hipStreamSynchronize(stream));
    check(hipMemcpy(ref.data(), baseline.p, elements * sizeof(float), hipMemcpyDeviceToHost));
    for (void* p : {batched.p, legacy.p}) {
        check(hipMemcpy(result.data(), p, elements * sizeof(float), hipMemcpyDeviceToHost));
        for (std::size_t i = 0; i < elements; ++i) {
            if (ref[i] != result[i]) throw std::runtime_error("bit mismatch case=" + std::to_string(kind) +
                " element=" + std::to_string(i) + " expected=" + std::to_string(ref[i]) + " actual=" + std::to_string(result[i]));
        }
    }
    for (unsigned r = 0; r < T; ++r) for (unsigned j = 0; j < stride; ++j) {
        const auto value = ref[static_cast<std::size_t>(r) * stride + j];
        if (j < Guard || j >= Guard + outputs || (kind == 3U && r < T / 2U && j < Guard + q)) require(value == Poison, "output guard overwritten");
        else require(value != Poison, "output not written");
    }
    // Two complete production-ring launches must produce identical output words,
    // including special values and all poison guards. More warmups wrap the ring.
    launch_batch(true);
    check(hipStreamSynchronize(stream));
    std::vector<std::uint32_t> native_ref(elements);
    check(hipMemcpy(native_ref.data(), wmma.p, elements * sizeof(float), hipMemcpyDeviceToHost));
    for (unsigned i = 0U; i < 10U; ++i) launch_batch(true);
    check(hipStreamSynchronize(stream));
    check(hipMemcpy(result.data(), wmma.p, elements * sizeof(float), hipMemcpyDeviceToHost));
    require(result == native_ref, "WMMA nondeterminism");
    for (unsigned r = 0U; r < T; ++r) for (unsigned j = 0U; j < stride; ++j) {
        const auto value = native_ref[static_cast<std::size_t>(r) * stride + j];
        if (j < Guard || j >= Guard + outputs || (kind == 3U && r < T / 2U && j < Guard + q))
            require(value == Poison, "WMMA guard overwritten");
        else require(value != Poison && (value & 0xffffU) == 0U, "WMMA missing/invalid BF16 epilogue");
    }
    hipLaunchKernelGGL(reference_kernel, dim3((outputs + 127U) / 128U, T), dim3(128), 0, stream,
        static_cast<const std::uint16_t*>(dx.p), static_cast<const std::uint16_t*>(dw.p),
        static_cast<double*>(oracle.p), width, outputs);
    check(hipGetLastError()); check(hipStreamSynchronize(stream));
    std::vector<double> fp64(static_cast<std::size_t>(T) * outputs);
    check(hipMemcpy(fp64.data(), oracle.p, fp64.size() * sizeof(double), hipMemcpyDeviceToHost));
    std::array<Error, 2> totals{};
    for (unsigned o = 0U; o < outputs; ++o) {
        Error error;
        for (unsigned r = 0U; r < T; ++r) {
            if (kind == 3U && r < T / 2U && o < q) continue;
            const double expected = fp64[static_cast<std::size_t>(r) * outputs + o];
            const float actual = from_bits(native_ref[static_cast<std::size_t>(r) * stride + Guard + o]);
            error.add(expected, actual); totals[o < q ? 0U : 1U].add(expected, actual);
        }
        error.print(kind, std::to_string(o));
        if (kind == 0U || kind == 3U)
            require(error.norm > 0.0 && error.sq <= 0.0001 * error.norm,
                "WMMA FP64 relative-L2 exceeds 1% arithmetic sanity bound");
    }
    totals[0].print(kind, "q"); totals[1].print(kind, "k");
    if (kind == 0U || kind == 3U) require(totals[0].special + totals[1].special == 0U, "finite WMMA case became nonfinite");
    Events event;
    std::array<std::vector<double>, 3> gpu_ms, wall_ms;
    const auto measure = [&](unsigned arm) {
        const auto start = std::chrono::steady_clock::now();
        check(hipEventRecord(event.begin, stream));
        if (arm == 0U) launch_old(); else launch_batch(arm == 2U);
        check(hipEventRecord(event.end, stream)); check(hipEventSynchronize(event.end));
        float elapsed = 0.0F; check(hipEventElapsedTime(&elapsed, event.begin, event.end));
        gpu_ms[arm].push_back(static_cast<double>(elapsed));
        wall_ms[arm].push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    };
    // Rewarm every arm after the untimed, bandwidth-heavy FP64 oracle.
    for (unsigned i = 0U; i < 3U; ++i) { launch_old(); launch_batch(); launch_batch(true); }
    check(hipStreamSynchronize(stream));
    for (unsigned i = 0; i < repetitions; ++i) {
        for (unsigned a = 0U; a < 3U; ++a) measure((i + a) % 3U);
    }
    // Timing reuses the output buffers: verify the final timed writes too.
    check(hipMemcpy(result.data(), batched.p, elements * sizeof(float), hipMemcpyDeviceToHost));
    require(result == ref, "bit mismatch after timing");
    check(hipMemcpy(result.data(), wmma.p, elements * sizeof(float), hipMemcpyDeviceToHost));
    require(result == native_ref, "WMMA bit mismatch after timing");
    std::cout << "EXACT_BATCH WMMA_DETERMINISTIC case=" << kind << " T=" << T << " width=" << width << " q=" << q << " k=" << k
        << " groups=" << T / Rows << " launches_old=" << T / Rows << " launches_batch=" << (kind == 3U ? 2U : 1U) << " launches_wmma=" << (kind == 3U ? 2U : 1U) << " N=" << repetitions
        << " old_gpu_ms=" << median(gpu_ms[0]) << " batch_gpu_ms=" << median(gpu_ms[1])
        << " wmma_gpu_ms=" << median(gpu_ms[2])
        << " wmma_wall_ms=" << median(wall_ms[2]) << " old_wall_ms=" << median(wall_ms[0]) << " batch_wall_ms=" << median(wall_ms[1]) << '\n';
}
} // namespace
int main(int argc, char** argv) try {
    const auto n = argc > 1 ? std::stoul(argv[1]) : 7UL;
    require(n > 0UL && n <= 101UL, "N must be 1..101");
    require(qsa_index_rows_enabled(), "run with STRIX_QSA_INDEX_ROWS=1 for the original rows reference");
    require(qpf2k_index_batch_enabled(), "run with STRIX_QPF2K_INDEX_BATCH=1");
    require(qpf2k_index_wmma_enabled(), "run with STRIX_PREFILL_EXACT_FP16=0 and STRIX_PREFILL_INDEX_PROJ_WMMA=1 on gfx1151 wave32");
    Stream stream;
    negative_dispatch(stream.value);
    for (unsigned pattern = 0U; pattern < 2U; ++pattern) mapping_oracle(stream.value, pattern);
    for (unsigned kind = 0; kind < 4U; ++kind) run(kind, static_cast<unsigned>(n), stream.value);
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
