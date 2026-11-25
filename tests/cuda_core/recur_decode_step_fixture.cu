#include "cuda/core/recur_cuda.cuh"
#include "recur_decode_frozen.cuh"
#include "strix/cuda/amd_math_tables.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace {

namespace rec = strix::cuda::recur;
namespace exec = strix::exec;
namespace frozen = rec::test::frozen;
namespace am = strix::cuda::amdmath;
using strix::RecurrentStateType;
constexpr auto kShape = strix::ops::recur::shape::gdn_2560_16x48x128;
using L = strix::ops::recur::GdnDecodeLayout<kShape>;

constexpr unsigned kRows = 16, kSlots = 2, kStride = L::channels + 16;
constexpr unsigned kRecBlocks = L::value_heads * L::head_dim / 32U;
constexpr unsigned kRecThreads = frozen::RecShape<4>::Threads;
constexpr strix::ops::recur::DecodeRowStrides kStrides{L::hidden + 3, L::channels + 4, L::value_width + 2,
    L::channels + 8, L::value_heads + 3, L::value_width + 5, L::channels + 6, L::value_width + 7};
static_assert(L::channels == frozen::kConvWidth && L::value_width == frozen::kValueWidth &&
              L::matrix_elements == frozen::kMatrixFloats && L::history_elements == frozen::kConvStateHalfs);

template <class T> struct Image {
    std::uint16_t history[L::history_elements];
    T matrix[L::matrix_elements];
};

template <class T> struct Input {
    std::uint16_t hidden[kRows][kStrides.input];
    float qkv[kRows][L::channels], z[kRows][kStrides.z];
    float beta[L::value_heads * L::hidden], alpha[L::value_heads * L::hidden], dt[L::value_heads],
        a[L::value_heads], conv[L::history_elements], gain[L::head_dim];
    Image<T> image[kSlots];
    exec::StepBlock step;
    strix::state::SlotEntry directory[2 * kSlots];
};

template <class T> struct Pending {
    alignas(16) float qkv[kRows][kStrides.qkv], prepared[kRows][kStrides.prepared];
    float beta[kRows][kStrides.gate], gate[kRows][kStrides.gate], out[kRows][kStrides.out],
        raw[kRows][kStrides.raw_capture], deltas[L::a8_deltas];
    std::int8_t codes[kRows][kStrides.a8_codes];
    std::uint16_t history[kRows][L::history_elements];
    T matrix[kRows][L::matrix_elements];
};

template <class T> struct Donor {
    std::uint16_t input[kRows][kStrides.input];
    alignas(16) float qkv[kRows][kStride], z[kRows][kStride], beta[kRows][kStride],
        gate[kRows][kStride], out[kRows][kStride];
    float raw[kRows][L::channels], deltas[L::a8_deltas], decay[kRows * L::value_heads];
    std::int8_t codes[kRows][L::value_width];
    Image<T> image;
};

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <class T> auto allocate_device() {
    T* storage = nullptr;
    check(cudaMalloc(&storage, sizeof(T)));
    const auto release = [](T* p) { (void)cudaFree(p); };
    return std::unique_ptr<T, decltype(release)>(storage, release);
}

// On WDDM, only the CPU mirror is host-accessible. Launch pointers and
// addresses embedded in descriptors always refer to the device allocation.
template <class T> class Mirrored {
    static_assert(std::is_trivially_copyable_v<T>);

public:
    T& host() const { return *host_; }
    T* device() const { return device_.get(); }

    void upload(std::size_t bytes = sizeof(T)) const {
        check(cudaMemcpy(device(), &host(), bytes, cudaMemcpyHostToDevice));
    }

    // The caller synchronizes the producing work before reading device state.
    void download(std::size_t bytes = sizeof(T)) const {
        check(cudaMemcpy(&host(), device(), bytes, cudaMemcpyDeviceToHost));
    }

private:
    std::unique_ptr<T> host_ = std::make_unique<T>();
    decltype(allocate_device<T>()) device_ = allocate_device<T>();
};

template <class T>
void compare(const char* label, const T* actual, const T* expected, std::size_t count) {
    if (std::memcmp(actual, expected, count * sizeof(T)) == 0) return;
    for (std::size_t i = 0; i < count; ++i) {
        if (std::memcmp(actual + i, expected + i, sizeof(T)) == 0) continue;
        std::fprintf(stderr, "%s byte offset %zu differs\n", label, i * sizeof(T));
        throw std::runtime_error("bitwise mismatch");
    }
}

struct Random {
    std::uint32_t bits = 0xe537a2d1U;
    float next(float scale) {
        bits ^= bits << 13U;
        bits ^= bits >> 17U;
        bits ^= bits << 5U;
        return static_cast<float>(static_cast<int>(bits & 0xffffU) - 32768) * (scale * 0x1p-15F);
    }
};

std::uint16_t narrow(float x) {
    const auto bits = std::bit_cast<std::uint32_t>(x);
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}

template <RecurrentStateType Type, class T>
void initialize(Input<T>& input) {
    Random rng;
    for (auto& row : input.hidden) for (auto& x : row) x = narrow(rng.next(1.0F));
    for (auto& row : input.qkv) {
        for (auto& x : row) x = rng.next(1.0F);
        row[0] = 0x1.01p-2F;
        row[1] = -0.0F;
    }
    for (auto& row : input.z)
        for (auto& x : row) x = std::bit_cast<float>(std::uint32_t{narrow(rng.next(4.0F))} << 16U);
    for (auto& x : input.beta) x = rng.next(0.125F);
    for (auto& x : input.alpha) x = rng.next(0.125F);
    for (auto& x : input.dt) x = rng.next(1.0F);
    input.dt[0] = -24.0F;
    input.dt[1] = 24.0F;
    for (auto& x : input.a) x = -0.125F - std::abs(rng.next(0.75F));
    for (auto& x : input.conv) x = rng.next(0.75F);
    for (auto& x : input.gain) x = 0.75F + rng.next(0.25F);
    for (auto& image : input.image) {
        for (auto& x : image.history) x = narrow(rng.next(1.0F));
        for (auto& x : image.matrix) {
            const float value = rng.next(0.0625F);
            if constexpr (Type == RecurrentStateType::f32) x = value;
            else if constexpr (Type == RecurrentStateType::bf16) x = narrow(value);
            else x = static_cast<__half_raw>(__float2half_rn(value)).x;
        }
    }
}

template <class T>
void describe(Input<T>& input, const Input<T>* device, unsigned width, bool chain) {
    auto& step = input.step;
    step = {};
    step.h.rows = width;
    step.h.valid_mask = (1U << width) - 1U;
    constexpr unsigned start[kSlots] = {37U, 541U};
    for (unsigned slot = 0; slot < kSlots; ++slot) {
        input.directory[slot] = {reinterpret_cast<std::uintptr_t>(device->image[slot].history),
                                 start[slot], width, start[slot] + width, 1};
        input.directory[kSlots + slot] = {reinterpret_cast<std::uintptr_t>(device->image[slot].matrix),
                                          start[slot], width, start[slot] + width, 1};
    }
    // Interleaved slots, siblings (3 and 4), a hole in each validity channel.
    constexpr unsigned n = exec::kStepNoParent;
    constexpr unsigned cohort_parents[kRows] = {n, n, n, 0, 0, 1, n, 4, 5, 7, 8, 9, 10, 11, 12, 13};
    for (unsigned r = 0; r < width; ++r) {
        const unsigned parent = chain ? (r == 0U ? n : r - 1U) : cohort_parents[r];
        const unsigned slot = parent == n ? (chain ? 0U : (r == 0U ? 1U : 0U)) : step.row[parent].slot;
        const unsigned pos = parent == n ? start[slot] : step.row[parent].position + 1U;
        const unsigned ancestors = (1U << r) | (parent == n ? 0U : step.row[parent].ancestors);
        step.row[r] = {0, pos, slot, parent, ancestors, exec::step_valid, 0, 0};
    }
    if (!chain && width >= 4U) step.h.valid_mask &= ~(1U << 2U);
    if (!chain && width >= 8U) step.row[6].flags = 0U;
}

template <RecurrentStateType Type, class T>
void serial(const exec::StepBlock& step, const Input<T>* device_input, Pending<T>* device_work,
            const rec::DecodeWeights& w, const am::Tables& tables, bool alias) {
    for (unsigned r = 0; r < step.h.rows; ++r) {
        if (!exec::step_row_valid(step, r)) continue;
        const auto& desc = step.row[r];
        const auto& image = device_input->image[desc.slot];
        const bool root = desc.parent == exec::kStepNoParent;
        check(cudaMemcpyAsync(device_work->history[r], root ? image.history : device_work->history[desc.parent],
                              sizeof(device_work->history[r]), cudaMemcpyDeviceToDevice, nullptr));
        const rec::DecodeRow row{device_input->hidden[r], device_work->qkv[r], device_input->z[r],
            alias ? device_work->qkv[r] : device_work->prepared[r], device_work->beta[r], device_work->gate[r],
            device_work->out[r], device_work->raw[r]};
        const rec::DecodeState state{device_work->history[r], root ? image.matrix : device_work->matrix[desc.parent],
                                     device_work->matrix[r], Type};
        check(rec::enqueue_decode(kShape, w, row, state, tables, {device_work->codes[r], device_work->deltas + r}, nullptr));
    }
}

template <unsigned R, RecurrentStateType Type, class T>
void donor_prefix(const Mirrored<Input<T>>& inputs, const Mirrored<Donor<T>>& donors,
                  const Pending<T>& work, unsigned target,
                  const am::Tables& tables, bool alias) {
    const auto& input = inputs.host();
    auto& donor = donors.host();
    std::array<unsigned, R> path{};
    unsigned count = 0U;
    for (unsigned r = target; r != exec::kStepNoParent; r = input.step.row[r].parent) path[count++] = r;
    std::reverse(path.begin(), path.begin() + count);
    const unsigned slot = input.step.row[target].slot;
    const auto& published = input.image[slot];
    std::memcpy(donor.image.history, published.history, sizeof(published.history));
    for (unsigned i = 0; i < R; ++i) {
        const unsigned r = i < count ? path[i] : target; // Extra donor rows cannot affect the prefix.
        std::copy_n(input.hidden[r], L::hidden, donor.input[i]);
        std::copy_n(input.qkv[r], L::channels, donor.qkv[i]);
        std::copy_n(input.z[r], L::value_width, donor.z[i]);
    }
    donors.upload();
    const auto* device_input = inputs.device();
    auto* device_donor = donors.device();
    const auto& device_published = device_input->image[slot];
    const frozen::Rows rows{device_donor->qkv[0], device_donor->z[0], device_donor->beta[0],
                            device_donor->gate[0], device_donor->out[0], kStride};
    frozen::decode_front_kernel<R><<<frozen::kDecConvBlocks + frozen::kDecGateBlocks, 256>>>(
        device_donor->input[0], kStrides.input, rows, device_input->beta, device_input->alpha, device_input->dt,
        device_input->a, device_input->conv, 1e-6F, tables, {device_donor->image.history}, device_donor->raw[0]);
    check(cudaGetLastError());

    // Frozen donor accept replay: recompute the matrix and roll taps back to
    // count, independently of the full-width destructive front above.
    constexpr unsigned rollback_blocks = (L::channels + kRecThreads - 1U) / kRecThreads;
    const frozen::DevRef<float> in{reinterpret_cast<float*>(const_cast<T*>(device_published.matrix))};
    const frozen::DevRef<float> out{reinterpret_cast<float*>(device_donor->image.matrix)};
    if constexpr (Type == RecurrentStateType::f32) {
        frozen::recurrence_dec_kernel<4><<<kRecBlocks + rollback_blocks, kRecThreads>>>(
            rows, count, in, out, tables, device_donor->raw[0], device_published.history, device_donor->image.history);
    } else {
        frozen::decay_kernel<<<(count * L::value_heads + 255U) / 256U, 256>>>(rows, count, device_donor->decay, tables);
        check(cudaGetLastError());
        frozen::recurrence_kernel<4, Type, true><<<kRecBlocks, kRecThreads>>>(rows, count, in, out, device_donor->decay);
        check(cudaGetLastError());
        frozen::conv_rollback_kernel<><<<(L::channels + 255U) / 256U, 256>>>(
            device_donor->raw[0], count, device_published.history, device_donor->image.history);
    }
    check(cudaGetLastError());
    frozen::norm_kernel<true><<<count * L::value_heads / 8U, 256>>>(
        rows, count, device_input->gain, 1e-6F, tables, device_donor->codes[0], device_donor->deltas);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    donors.download();

    for (unsigned i = 0; i < count; ++i) {
        const unsigned r = path[i];
        compare("frozen QKV", alias ? work.qkv[r] : work.prepared[r], donor.qkv[i], L::channels);
        compare("frozen raw", work.raw[r], donor.raw[i], L::channels);
        compare("frozen beta", work.beta[r], donor.beta[i], L::value_heads);
        compare("frozen gate", work.gate[r], donor.gate[i], L::value_heads);
        compare("frozen output", work.out[r], donor.out[i], L::value_width);
        compare("frozen A8 codes", work.codes[r], donor.codes[i], L::value_width);
        for (unsigned j = 0; j < L::a8_deltas; j += L::a8_pitch)
            compare("frozen A8 deltas", work.deltas + j + r, donor.deltas + j + i, 1);
    }
    compare("frozen accepted taps", work.history[target], donor.image.history, L::history_elements);
    compare("frozen accepted matrix", work.matrix[target], donor.image.matrix, L::matrix_elements);
}

template <RecurrentStateType Type, class T>
void resume(const Mirrored<Input<T>>& inputs, const Mirrored<Pending<T>>& pending,
            const Mirrored<Pending<T>>& reference, rec::DecodeStep args,
            const rec::DecodeWeights& w, const am::Tables& tables) {
    auto& input = inputs.host();
    auto* device_input = inputs.device();
    auto* device_work = pending.device();
    auto* device_seq = reference.device();
    constexpr unsigned accepted = 4U, next = kRows - 1U;
    const unsigned position = input.step.row[accepted].position + 1U;
    input.directory[0].base = reinterpret_cast<std::uintptr_t>(device_work->history[accepted]);
    input.directory[0].frontier = position;
    input.directory[kSlots].base = reinterpret_cast<std::uintptr_t>(device_work->matrix[accepted]);
    input.directory[kSlots].frontier = position;
    input.step.h.rows = 1U;
    input.step.h.valid_mask = 1U;
    input.step.row[0] = {0, position, 0, exec::kStepNoParent, 1, exec::step_valid, 0, 0};
    inputs.upload();

    // Reuse the last rejected row's buffers for a new token after acceptance.
    check(cudaMemcpy(device_work->qkv[next], input.qkv[next], sizeof(input.qkv[next]), cudaMemcpyHostToDevice));
    check(cudaMemcpy(device_seq->qkv[next], input.qkv[next], sizeof(input.qkv[next]), cudaMemcpyHostToDevice));
    args.rows = strix::ops::recur::decode_row(args, next);
    args.pending_history = device_work->history[next];
    args.pending_matrix = device_work->matrix[next];
    args.a8 = {device_work->codes[next], device_work->deltas + next};
    check(rec::enqueue_decode_step(kShape, 1U, w, args, tables, nullptr));

    check(cudaMemcpyAsync(device_seq->history[next], device_seq->history[accepted], sizeof(device_seq->history[next]),
                          cudaMemcpyDeviceToDevice, nullptr));
    const rec::DecodeRow row{device_input->hidden[next], device_seq->qkv[next], device_input->z[next],
        device_seq->qkv[next], device_seq->beta[next], device_seq->gate[next], device_seq->out[next], device_seq->raw[next]};
    const rec::DecodeState state{device_seq->history[next], device_seq->matrix[accepted], device_seq->matrix[next], Type};
    check(rec::enqueue_decode(kShape, w, row, state, tables, {device_seq->codes[next], device_seq->deltas + next}, nullptr));
    check(cudaDeviceSynchronize());
    pending.download();
    reference.download();
    inputs.download();
    compare("resumed after partial acceptance", reinterpret_cast<const unsigned char*>(&pending.host()),
            reinterpret_cast<const unsigned char*>(&reference.host()), sizeof(Pending<T>));
}

template <unsigned R, RecurrentStateType Type, class T>
void run(const Mirrored<Input<T>>& inputs, const Mirrored<Pending<T>>& pending,
         const Mirrored<Pending<T>>& reference, const Mirrored<Donor<T>>& donors,
         const am::Tables& tables, bool chain) {
    std::fprintf(stderr, "K9G X3 decode %s R=%u case=%s\n", strix::recurrent_state_type_name(Type), R,
                 chain ? "verify" : "cohort");
    std::fflush(stderr);
    auto& input = inputs.host();
    auto& work = pending.host();
    auto& seq = reference.host();
    auto* device_input = inputs.device();
    auto* device_work = pending.device();
    describe(input, device_input, R, chain);
    std::memset(&work, 0xa5, sizeof(work));
    for (unsigned r = 0; r < R; ++r) std::copy_n(input.qkv[r], L::channels, work.qkv[r]);
    std::memcpy(&seq, &work, sizeof(work));
    const auto published = std::make_unique<std::array<Image<T>, kSlots>>();
    std::memcpy(published->data(), input.image, sizeof(input.image));
    const auto directory = std::to_array(input.directory);
    const strix::state::StateTable table{device_input->directory, 2, kSlots};
    const rec::DecodeWeights w{device_input->beta, device_input->alpha, device_input->dt, device_input->a,
                               device_input->conv, device_input->gain};
    const bool alias = chain;
    auto strides = kStrides;
    if (alias) strides.prepared = strides.qkv;
    const rec::DecodeStep args{&device_input->step,
        {device_input->hidden[0], device_work->qkv[0], device_input->z[0],
         alias ? device_work->qkv[0] : device_work->prepared[0],
         device_work->beta[0], device_work->gate[0], device_work->out[0], device_work->raw[0]},
        strides, {table, 0}, {table, 1}, device_work->history[0], device_work->matrix[0], Type,
        {device_work->codes[0], device_work->deltas}};
    pending.upload();
    reference.upload();

    // Zero acceptance keeps the published images, including for a fully
    // padded launch. For a nonempty accepted path select its last row image.
    if constexpr (R == 1U) {
        input.step.h.valid_mask = 0U;
        inputs.upload();
        check(rec::enqueue_decode_step(kShape, R, w, args, tables, nullptr));
        check(cudaDeviceSynchronize());
        pending.download();
        inputs.download();
        compare("fully padded", reinterpret_cast<const unsigned char*>(&work),
                reinterpret_cast<const unsigned char*>(&seq), sizeof(work));
        input.step.h.valid_mask = 1U;
    }
    inputs.upload();
    serial<Type>(input.step, device_input, reference.device(), w, tables, alias);
    check(rec::enqueue_decode_step(kShape, R, w, args, tables, nullptr));
    check(cudaDeviceSynchronize());
    pending.download();
    reference.download();
    inputs.download();
    compare("sequential K9G (including all padding)", reinterpret_cast<const unsigned char*>(&work),
            reinterpret_cast<const unsigned char*>(&seq), sizeof(work));
    compare("published directory", input.directory, directory.data(), directory.size());
    for (unsigned slot = 0; slot < kSlots; ++slot) {
        compare("published taps", input.image[slot].history, (*published)[slot].history, L::history_elements);
        compare("published matrix", input.image[slot].matrix, (*published)[slot].matrix, L::matrix_elements);
    }
    for (unsigned r = 0; r < R; ++r)
        if (exec::step_row_valid(input.step, r)) {
            try {
                donor_prefix<R, Type>(inputs, donors, work, r, tables, alias);
            } catch (...) {
                std::fprintf(stderr, "R=%u row=%u %s\n", R, r, chain ? "verify" : "cohort");
                throw;
            }
        }
    if constexpr (R == kRows) {
        if (chain) resume<Type>(inputs, pending, reference, args, w, tables);
    }
}

template <RecurrentStateType Type>
void test(const am::Tables& tables) {
    using T = std::conditional_t<Type == RecurrentStateType::f32, float, std::uint16_t>;
    const Mirrored<Input<T>> input;
    const Mirrored<Pending<T>> work, seq;
    const Mirrored<Donor<T>> donor;
    initialize<Type>(input.host());
    const auto width = [&]<unsigned R>() {
        run<R, Type>(input, work, seq, donor, tables, true);
        run<R, Type>(input, work, seq, donor, tables, false);
    };
    width.template operator()<1>();
    width.template operator()<2>();
    width.template operator()<4>();
    width.template operator()<8>();
    width.template operator()<16>();
    std::printf("K9G X3 decode %s EXACT (R=1,2,4,8,16, verify/cohort, partial accept, padding)\n",
                strix::recurrent_state_type_name(Type));
    std::fflush(stdout);
}

template <class T>
struct ReplayBanks {
    // Layer 0 guards the nonzero layer offset on every selected state.
    std::uint16_t before[kSlots][2][L::history_elements], taps[kSlots][2][L::history_elements];
    alignas(float) T matrix[kSlots][2][2][L::matrix_elements]; // slot, parity, layer, element
};

struct ReplayWords {
    std::uint16_t* taps;
    float* matrix;
};

struct ReplayGraph {
    cudaStream_t stream{};
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};

    ~ReplayGraph() {
        if (executable) (void)cudaGraphExecDestroy(executable);
        if (graph) (void)cudaGraphDestroy(graph);
        if (stream) (void)cudaStreamDestroy(stream);
    }
};

template <unsigned R, RecurrentStateType Type, class T, std::size_t Count>
void run_replay(const Mirrored<Input<T>>& inputs, const Mirrored<Pending<T>>& pending,
                const Mirrored<Pending<T>>& reference, const Mirrored<Donor<T>>& donors,
                const am::Tables& tables, const std::array<unsigned, Count>& counts) {
    auto& input = inputs.host();
    auto& work = pending.host();
    auto& seq = reference.host();
    auto* device_input = inputs.device();
    auto* device_work = pending.device();
    const Mirrored<ReplayBanks<T>> replay_banks;
    auto* banks = &replay_banks.host();
    auto* device_banks = replay_banks.device();
    const Mirrored<ReplayWords> words;
    auto expected = std::make_unique_for_overwrite<ReplayBanks<T>>();
    std::memset(banks, 0xa5, sizeof(*banks));
    for (unsigned slot = 0U; slot < kSlots; ++slot) {
        std::memcpy(banks->before[slot][1], input.image[slot].history, sizeof(input.image[slot].history));
        for (auto& parity : banks->matrix[slot])
            std::memcpy(parity[1], input.image[slot].matrix, sizeof(input.image[slot].matrix));
    }

    constexpr auto tap_offset = sizeof(banks->taps[0][0]);
    constexpr auto matrix_offset = sizeof(banks->matrix[0][0][0]);
    const strix::state::StateTable table{device_input->directory, 2, kSlots};
    const rec::DecodeWeights w{device_input->beta, device_input->alpha, device_input->dt, device_input->a,
                               device_input->conv, device_input->gain};
    auto strides = kStrides;
    strides.prepared = strides.qkv;
    const rec::DecodeReplayStep args{&device_input->step,
        {device_input->hidden[0], device_work->qkv[0], device_input->z[0], device_work->qkv[0], device_work->beta[0],
         device_work->gate[0], device_work->out[0], device_work->raw[0]},
        strides, {table, 0}, {table, 1}, {nullptr, &words.device()->taps, tap_offset},
        {nullptr, &words.device()->matrix, matrix_offset}, Type, {device_work->codes[0], device_work->deltas},
        tap_offset, matrix_offset};

    ReplayGraph graph;
    for (std::size_t replay = 0; replay < counts.size(); ++replay) {
        const auto count = counts[replay];
        const unsigned slot = R == 1U && replay == 1U ? 1U : 0U;
        const unsigned parity = R == 1U && replay != 0U ? 1U : 0U;
        std::fprintf(stderr, "K9G X3 replay %s R=%u case=%zu count=%u slot=%u parity=%u\n",
                     strix::recurrent_state_type_name(Type), R, replay, count, slot, parity);
        std::fflush(stderr);
        auto& before = banks->before[slot];
        auto& taps = banks->taps[slot];
        auto& matrix_in = banks->matrix[slot][parity];
        if (R == 1U && replay == 2U) {
            std::memcpy(before[1], taps[1], tap_offset); // Continue session 0 from replay 0.
            check(cudaMemcpy(device_banks->before[slot][1], before[1], tap_offset, cudaMemcpyHostToDevice));
        }
        std::memcpy(input.image[slot].history, before[1], tap_offset);
        std::memcpy(input.image[slot].matrix, matrix_in[1], matrix_offset);
        describe(input, device_input, R, true);
        input.step.h.valid_mask = (1U << count) - 1U;
        for (unsigned r = 0; r < R; ++r) {
            input.step.row[r].slot = slot;
            input.step.row[r].position = input.directory[slot].frontier + r;
        }
        input.directory[slot].base = reinterpret_cast<std::uintptr_t>(device_banks->before[slot][0]);
        input.directory[kSlots + slot].base = reinterpret_cast<std::uintptr_t>(device_banks->matrix[slot][parity][0]);
        words.host() = {device_banks->taps[slot][0], reinterpret_cast<float*>(device_banks->matrix[slot][parity ^ 1U][0])};

        constexpr auto row_bytes = offsetof(Pending<T>, history);
        std::memset(&work, 0xa5, row_bytes);
        for (unsigned r = 0; r < R; ++r) std::copy_n(input.qkv[r], L::channels, work.qkv[r]);
        std::memcpy(&seq, &work, row_bytes);
        std::memcpy(expected.get(), banks, sizeof(*banks));
        const auto directory = std::to_array(input.directory);
        inputs.upload();
        pending.upload(row_bytes);
        reference.upload(row_bytes);
        words.upload();
        if (replay == 0U) replay_banks.upload();
        // The nonblocking graph stream does not wait for default-stream uploads.
        check(cudaDeviceSynchronize());
        if (replay == 0U) {
            check(cudaStreamCreateWithFlags(&graph.stream, cudaStreamNonBlocking));
            check(cudaStreamBeginCapture(graph.stream, cudaStreamCaptureModeThreadLocal));
            check(rec::enqueue_decode_replay_step(kShape, R, w, args, tables, graph.stream));
            check(cudaStreamEndCapture(graph.stream, &graph.graph));
            check(cudaGraphInstantiate(&graph.executable, graph.graph, 0));
        }
        check(cudaGraphLaunch(graph.executable, graph.stream));
        check(cudaDeviceSynchronize());
        inputs.download();

        serial<Type>(input.step, device_input, reference.device(), w, tables, true);
        check(cudaDeviceSynchronize());
        pending.download(row_bytes);
        reference.download();
        compare("captured replay rows and padding vs direct", reinterpret_cast<const unsigned char*>(&work),
                reinterpret_cast<const unsigned char*>(&seq), row_bytes);
        if (count != 0U) {
            const unsigned last = count - 1U;
            donor_prefix<R, Type>(inputs, donors, seq, last, tables, true);
            std::memcpy(expected->taps[slot][1], seq.history[last], tap_offset);
            std::memcpy(expected->matrix[slot][parity ^ 1U][1], seq.matrix[last], matrix_offset);
        }
        replay_banks.download();
        inputs.download();
        compare("captured replay images and inactive banks vs direct",
                reinterpret_cast<const unsigned char*>(banks),
                reinterpret_cast<const unsigned char*>(expected.get()), sizeof(*banks));
        compare("captured replay published directory", input.directory, directory.data(), directory.size());
    }
}

template <RecurrentStateType Type>
void test_replay(const am::Tables& tables) {
    using T = std::conditional_t<Type == RecurrentStateType::f32, float, std::uint16_t>;
    const Mirrored<Input<T>> input;
    const Mirrored<Pending<T>> work, seq;
    const Mirrored<Donor<T>> donor;
    initialize<Type>(input.host());
    run_replay<1, Type>(input, work, seq, donor, tables, std::array{1U, 1U, 1U});
    std::printf("K9G X3 replay %s R=1 EXACT (3 graph launches, 2 slots, parity, layer offsets)\n",
                strix::recurrent_state_type_name(Type));
    std::fflush(stdout);
    run_replay<2, Type>(input, work, seq, donor, tables, std::array{2U, 1U, 0U});
    run_replay<4, Type>(input, work, seq, donor, tables, std::array{4U, 3U, 1U});
    run_replay<8, Type>(input, work, seq, donor, tables, std::array{8U, 7U, 6U, 5U, 2U});
    run_replay<16, Type>(input, work, seq, donor, tables, std::array{16U, 9U, 4U, 0U});
    std::printf("K9G X3 replay %s R=2,4,8,16 EXACT (one spare, prefix replay, padding, zero accept)\n",
                strix::recurrent_state_type_name(Type));
    std::fflush(stdout);
}

#if defined(STRIX_K9G_REPLAY_FIXTURE)
constexpr auto kFixture = "strix-k9g-decode-replay-fixture";
template <RecurrentStateType Type>
void run_fixture(const am::Tables& tables) { test_replay<Type>(tables); }
#else
constexpr auto kFixture = "strix-k9g-decode-step-fixture";
template <RecurrentStateType Type>
void run_fixture(const am::Tables& tables) { test<Type>(tables); }
#endif

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s TABLES_DIR\n", kFixture);
        return 2;
    }
    try {
        auto tables = am::load_tables(argv[1]);
        try {
            run_fixture<RecurrentStateType::f32>(tables);
            run_fixture<RecurrentStateType::f16>(tables);
            run_fixture<RecurrentStateType::bf16>(tables);
        } catch (...) {
            am::free_tables(tables);
            throw;
        }
        am::free_tables(tables);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s FAIL: %s\n", kFixture, error.what());
        return 1;
    }
}
