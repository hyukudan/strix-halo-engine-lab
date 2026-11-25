#include "cuda/core/attn_selected_blocks.cuh"
#include "mmattn_minimax_frozen.cuh"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strix::attn;
using namespace strix::cuda::core::attn;
namespace frozen = mmattn_frozen;

using Shape = SelectedBlockShape<128, 128, frozen::kBlock, frozen::kTopK, 16>;
constexpr unsigned groups = 4, heads = groups * Shape::heads_per_group;
constexpr unsigned rows = 9, layer = 1, guards = 16;
constexpr unsigned outputs = rows * heads * Shape::value_dim;
constexpr std::uint32_t canary = 0x4e6a5b3cU;
constexpr float scale = 0.08838834764831845F;
static_assert(Shape::k_stride == frozen::kKStride);

void check(cudaError_t status, const char* action) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
}

template <class T> class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t size) : bytes_(size * sizeof(T)) {
        check(cudaMalloc(reinterpret_cast<void**>(&data_), bytes_), "cudaMalloc");
    }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() const { return data_; }
    void write(std::span<const T> host) const {
        if (host.size_bytes() != bytes_) throw std::runtime_error("bad host upload size");
        check(cudaMemcpy(data_, host.data(), bytes_, cudaMemcpyHostToDevice), "host to device");
    }
    void read(std::span<T> host) const {
        if (host.size_bytes() != bytes_) throw std::runtime_error("bad host download size");
        check(cudaMemcpy(host.data(), data_, bytes_, cudaMemcpyDeviceToHost), "device to host");
    }

private:
    std::size_t bytes_;
    T* data_ = nullptr;
};

struct Sequence {
    std::uint32_t row0, rows, cap, pos0;
};

constexpr std::array sequences{
    Sequence{0, 3, 273, 143}, Sequence{3, 3, 333, 286},
    Sequence{6, 2, 400, 383}, Sequence{8, 1, 19, 0},
};

enum class Data { finite, signed_zero, nonfinite, empty_list };

struct Random {
    std::uint32_t state = 0x7557331U;
    float next() {
        state = state * 1664525U + 1013904223U;
        return static_cast<float>(static_cast<int>(state >> 21U) - 1024) / 256.0F;
    }
    std::uint16_t bf16() { return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(next()) >> 16U); }
};

void compare(const char* label, std::span<const float> a, std::span<const float> b) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        const auto ab = std::bit_cast<std::uint32_t>(a[i]), bb = std::bit_cast<std::uint32_t>(b[i]);
        if (ab != bb) {
            std::fprintf(stderr, "%s index=%zu bits=%08x expected=%08x\n", label, i,
                         static_cast<unsigned>(ab), static_cast<unsigned>(bb));
            throw std::runtime_error("byte mismatch");
        }
    }
}

void check_output(std::span<const float> image, Data fixture) {
    for (std::size_t i = 0; i < image.size(); ++i) {
        const bool guard = i < guards || i >= guards + outputs;
        const bool untouched = std::bit_cast<std::uint32_t>(image[i]) == canary;
        if (guard != untouched) throw std::runtime_error("output or guard was not written as expected");
        if (!guard && fixture == Data::finite && !std::isfinite(image[i]))
            throw std::runtime_error("finite fixture produced nonfinite output");
        if (!guard && fixture == Data::empty_list) {
            const auto group_row = (i - guards) / (Shape::value_dim * Shape::heads_per_group);
            if (group_row % groups == 0 && !std::isnan(image[i]))
                throw std::runtime_error("empty list did not produce NaN");
        }
    }
}

void run(Data fixture) {
    Random random;
    std::vector<float> query(rows * heads * Shape::qk_dim);
    for (auto& q : query) q = random.next();
    std::array<std::uint32_t, rows> row_sequence{};
    std::array<std::uint32_t, rows * groups> counts{};
    std::array<std::uint32_t, rows * groups * Shape::list_capacity> blocks{};
    blocks.fill(0xffffffffU);
    std::array<SelectedBlockSequence, sequences.size()> shared_views{};
    std::array<frozen::SeqView, sequences.size()> frozen_views{};

    std::size_t total = 0;
    for (const auto& seq : sequences) total += (layer + 1U) * seq.cap * groups * Shape::qk_dim;
    std::vector<std::uint16_t> k(total), v(total);
    DeviceBuffer<std::uint16_t> dk(total), dv(total);
    for (auto& x : k) x = random.bf16();
    for (auto& x : v) x = random.bf16();
    if (fixture == Data::signed_zero) {
        // Products of the least-magnitude FP32 subnormal and +/-0.25
        // round to signed zero. Whole keys alternate the sign across ties.
        std::fill(query.begin(), query.end(), std::bit_cast<float>(0x80000001U));
        for (std::size_t i = 0; i < k.size(); ++i)
            k[i] = ((i / (groups * Shape::qk_dim)) % 2U == 0U) ? 0x3e80U : 0xbe80U;
        for (std::size_t i = 0; i < v.size(); ++i) {
            if (i % 3U == 0U) v[i] = 0x8000U;
            if (i % 3U == 1U) v[i] = 0x0000U;
        }
        query[0] = -0.0F;
    }
    if (fixture == Data::nonfinite) {
        query[0] = std::bit_cast<float>(0x7fc12000U);
        query[Shape::qk_dim] = std::bit_cast<float>(0xff800000U);
    }

    std::size_t offset = 0;
    for (std::size_t s = 0; s < sequences.size(); ++s) {
        const auto seq = sequences[s];
        shared_views[s] = {dk.data() + offset, dv.data() + offset, seq.row0, seq.pos0, seq.cap};
        frozen_views[s] = {seq.row0, seq.rows, seq.cap, dk.data() + offset, dv.data() + offset, nullptr, seq.pos0};
        if (fixture == Data::nonfinite) {
            const auto at = offset + layer * seq.cap * groups * Shape::qk_dim;
            const auto pitch = groups * Shape::qk_dim;
            k[at] = 0x7f80U;
            k[at + Shape::qk_dim] = 0xff80U;
            k[at + Shape::qk_dim * 2U + 17U] = 0x7f81U;
            k[at + Shape::qk_dim * 3U + 100U] = 0xffc3U;
            if (seq.cap > 16U) {
                k[at + pitch * 15U + 3U] = 0x7fc1U;
                k[at + pitch * 16U + 4U] = 0xff80U;
            }
            v[at + 4U] = 0x7f80U;
            v[at + 5U] = 0xff80U;
            v[at + Shape::value_dim + 9U] = 0x7fc9U;
        }
        offset += (layer + 1U) * seq.cap * groups * Shape::qk_dim;
        for (unsigned row = seq.row0; row < seq.row0 + seq.rows; ++row) {
            row_sequence[row] = static_cast<std::uint32_t>(s);
            const auto tail = (seq.pos0 + row - seq.row0) / Shape::block_keys;
            for (unsigned g = 0; g < groups; ++g) {
                const auto i = row * groups + g;
                if (fixture == Data::empty_list && g == 0) continue;
                // A group-specific rotation followed by a reverse walk keeps
                // the last causal block in a different slot across groups.
                for (unsigned j = 0; j <= tail; ++j) {
                    const auto id = (tail - j + g) % (tail + 1U);
                    if (tail >= 2U && g == 2U && id == 1U) continue;
                    blocks[i * Shape::list_capacity + counts[i]++] = id;
                }
                if (tail > 0U && g == 3U) blocks[i * Shape::list_capacity + counts[i]++] = 0U;
            }
        }
    }

    DeviceBuffer<SelectedBlockSequence> ds(shared_views.size());
    DeviceBuffer<frozen::SeqView> df(frozen_views.size());
    DeviceBuffer<float> dq(query.size()), out(outputs + 2U * guards);
    DeviceBuffer<std::uint32_t> dr(row_sequence.size()), db(blocks.size()), dc(counts.size());
    dk.write(k);
    dv.write(v);
    ds.write(shared_views);
    df.write(frozen_views);
    dq.write(query);
    dr.write(row_sequence);
    db.write(blocks);
    dc.write(counts);
    const std::vector<float> initial(outputs + 2U * guards, std::bit_cast<float>(canary));
    std::array<std::vector<float>, 2> frozen_output{
        std::vector<float>(initial.size()), std::vector<float>(initial.size())};
    std::vector<float> result(initial.size());
    auto* dst = out.data() + guards;
    for (const auto schedule : {SelectedBlockSchedule::warp, SelectedBlockSchedule::staged}) {
        const auto staged = schedule == SelectedBlockSchedule::staged;
        auto& expected = frozen_output[staged ? 1U : 0U];
        out.write(initial);
        if (staged)
            frozen::attend_shared_kernel<<<dim3(rows, groups), 512, Shape::staged_bytes>>>(
                df.data(), dr.data(), layer, dq.data(), db.data(), dc.data(), dst, 0);
        else
            frozen::attend_kernel<<<(rows * heads + 7U) / 8U, 256>>>(
                df.data(), dr.data(), rows, layer, dq.data(), db.data(), dc.data(), dst);
        check(cudaGetLastError(), "frozen launch");
        out.read(expected);
        check_output(expected, fixture);
        out.write(initial);
        const SelectedBlockInput input{ds.data(), dr.data(), dq.data(), db.data(), dc.data(), dst, layer, scale};
        check(enqueue_selected_blocks<Shape>(schedule, rows, groups, input, nullptr), "selected-block launch");
        out.read(result);
        compare(staged ? "staged vs frozen staged" : "warp vs frozen warp", result, expected);
    }
    if (fixture == Data::finite) compare("finite frozen warp vs frozen staged", frozen_output[0], frozen_output[1]);
}
}  // namespace

int main() {
    try {
        check(prepare_selected_blocks<Shape>(), "staged shared memory");
        check(cudaFuncSetAttribute(frozen::attend_shared_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(Shape::staged_bytes)), "frozen shared memory");
        check(enqueue_selected_blocks<Shape>(SelectedBlockSchedule::warp, 0, 0, {}, nullptr), "empty launch");
        run(Data::finite);
        std::puts("MMATTN PASS finite tails, block order, groups and sequences");
        run(Data::signed_zero);
        std::puts("MMATTN PASS signed-zero inputs vs both frozen schedules");
        run(Data::nonfinite);
        std::puts("MMATTN PASS nonfinite inputs vs both frozen schedules");
        run(Data::empty_list);
        std::puts("MMATTN PASS empty lists, zero rows and output guards");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "MMATTN FAIL %s\n", error.what());
        return 1;
    }
}
