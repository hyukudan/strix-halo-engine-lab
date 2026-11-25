#include "ops/recur/recur_decode_step.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>

namespace {

namespace rec = strix::ops::recur;
namespace exec = strix::exec;
constexpr auto kShape = rec::shape::gdn_2560_16x48x128;
using strix::RecurrentStateType;

void require(bool condition, const char* message) {
    if (condition) return;
    std::fprintf(stderr, "K9G X3 replay plan: %s\n", message);
    std::exit(1);
}

template <class T>
T* address(std::uintptr_t value) {
    return reinterpret_cast<T*>(value);
}

}  // namespace

int main() {
    constexpr std::uintptr_t tap_offset = sizeof(std::uint16_t), matrix_offset = sizeof(float);
    constexpr std::uintptr_t tap_in = 0x100000, tap_out = 0x200000, matrix_a = 0x400000, matrix_b = 0x800000;
    std::array<strix::state::SlotEntry, 4> entries{{
        {tap_in, 17, 0, 33, 0}, {tap_out, 42, 0, 58, 0},
        {matrix_a, 17, 0, 33, 0}, {matrix_b, 42, 0, 58, 0},
    }};
    const strix::state::StateTable table{entries.data(), 2, 2};
    exec::StepBlock desc{};
    desc.h.rows = 8;
    desc.h.valid_mask = 0x1f;
    for (unsigned r = 0; r < 8U; ++r)
        desc.row[r] = {0, 17U + r, 0, r == 0U ? exec::kStepNoParent : r - 1U, 0, exec::step_valid, 0, 0};

    // The host planner sees layouts and addresses only.
    auto* tap_word = address<std::uint16_t>(tap_out);
    auto* matrix_word = address<float>(matrix_b);
    rec::DecodeReplayStep args{&desc,
        {address<std::uint16_t>(0x100), address<float>(0x200), address<float>(0x300),
         address<float>(0x400), address<float>(0x500), address<float>(0x600), address<float>(0x700)},
        rec::packed_decode_strides(kShape), {table, 0}, {table, 1}, {nullptr, &tap_word, tap_offset}, {nullptr, &matrix_word, matrix_offset},
        RecurrentStateType::f32, {}, tap_offset, matrix_offset};
    for (unsigned width : {1U, 2U, 4U, 8U, 16U})
        require(rec::valid_decode_replay_step(kShape, width, args), "rejected a supported captured width");
    for (unsigned width : {0U, 3U, 17U, 32U})
        require(!rec::valid_decode_replay_step(kShape, width, args), "accepted an unsupported captured width");

    auto images = rec::decode_replay_images<kShape>(8U, args);
    require(images.rows == 5U && images.history == address<std::uint16_t>(tap_in + tap_offset) &&
            images.pending_history == address<std::uint16_t>(tap_out + tap_offset) &&
            images.matrix == address<float>(matrix_a + matrix_offset) &&
            images.pending_matrix == address<float>(matrix_b + matrix_offset), "misresolved a padded prefix");
    require(!rec::decode_replay_images<kShape>(4U, args).rows, "accepted a descriptor wider than its graph");
    desc.h.valid_mask = 0x17;
    require(!rec::decode_replay_images<kShape>(8U, args).rows, "accepted a hole in the chain");
    desc.h.valid_mask = 0x1f;
    desc.row[4].parent = 2U;
    require(!rec::decode_replay_images<kShape>(8U, args).rows, "accepted a branch");
    desc.row[4].parent = 3U;
    desc.row[4].slot = 1U;
    require(!rec::decode_replay_images<kShape>(8U, args).rows, "accepted a cross-slot parent");
    desc.row[4].slot = 0U;
    ++desc.row[4].position;
    require(!rec::decode_replay_images<kShape>(8U, args).rows, "accepted a position gap");
    --desc.row[4].position;
    entries[2].capacity = desc.row[4].position;
    require(!rec::decode_replay_images<kShape>(8U, args).rows, "accepted a row beyond capacity");
    entries[2].capacity = 33;

    desc.h.rows = 1;
    desc.h.valid_mask = 1U;
    desc.row[0].slot = 1U;
    desc.row[0].position = 42U;
    tap_word = address<std::uint16_t>(tap_in);
    matrix_word = address<float>(matrix_a);
    images = rec::decode_replay_images<kShape>(1U, args);
    require(images.rows == 1U && images.history == address<std::uint16_t>(tap_out + tap_offset) &&
            images.matrix == address<float>(matrix_b + matrix_offset) &&
            images.pending_history == address<std::uint16_t>(tap_in + tap_offset) &&
            images.pending_matrix == address<float>(matrix_a + matrix_offset), "retained a captured slot or parity");
    desc.h.valid_mask = 0U;
    require(!rec::decode_replay_images<kShape>(1U, args).rows, "accepted an empty prefix");
    desc.h.valid_mask = 1U;

    args.pending_history = {tap_word, nullptr, tap_offset};
    args.pending_matrix = {matrix_word, nullptr, matrix_offset};
    require(rec::valid_decode_replay_step(kShape, 1U, args) && rec::decode_replay_images<kShape>(1U, args).rows == 1U,
            "rejected direct output with an offset");
    args.pending_history.slot = &tap_word;
    require(!rec::valid_decode_replay_step(kShape, 1U, args), "accepted two output address sources");
    args.pending_history.direct = nullptr;
    tap_word = nullptr;
    require(rec::valid_decode_replay_step(kShape, 1U, args) && !rec::decode_replay_images<kShape>(1U, args).rows,
            "host read a device word or replay accepted a null target");
    tap_word = address<std::uint16_t>(tap_out - sizeof(std::uint16_t));
    require(!rec::decode_replay_images<kShape>(1U, args).rows, "accepted a partial tap overlap");
    constexpr rec::GdnDecodeShape smaller{512, 2, 8, 64, 3, 2};
    tap_word = address<std::uint16_t>(tap_out + smaller.history_elements() * sizeof(std::uint16_t));
    require(rec::decode_replay_images<smaller>(1U, args).rows == 1U &&
            !rec::decode_replay_images<kShape>(1U, args).rows, "ignored the shape when checking tap overlap");
    tap_word = address<std::uint16_t>(tap_in);
    const auto narrow_bytes = kShape.matrix_bytes(RecurrentStateType::f16);
    args.pending_matrix = {nullptr, &matrix_word, matrix_offset};
    matrix_word = address<float>(matrix_b + narrow_bytes);
    require(!rec::decode_replay_images<kShape>(1U, args).rows, "accepted overlapping FP32 matrix images");
    for (const auto type : {RecurrentStateType::f16, RecurrentStateType::bf16}) {
        args.type = type;
        require(rec::decode_replay_images<kShape>(1U, args).rows == 1U, "rejected adjacent 16-bit matrix images");
    }
    matrix_word = address<float>(tap_in);
    require(!rec::decode_replay_images<kShape>(1U, args).rows, "accepted overlapping output images");
    matrix_word = address<float>(UINTPTR_MAX - 3U);
    require(!rec::decode_replay_images<kShape>(1U, args).rows, "accepted a wrapped offset");
    matrix_word = address<float>(UINTPTR_MAX - 7U);
    require(!rec::decode_replay_images<kShape>(1U, args).rows, "accepted a wrapped byte range");
    args.pending_matrix.byte_offset = sizeof(std::uint16_t);
    require(!rec::valid_decode_replay_step(kShape, 1U, args), "accepted a misaligned matrix offset");
    args.pending_matrix.byte_offset = matrix_offset;
    args.history_byte_offset = 1U;
    require(!rec::valid_decode_replay_step(kShape, 1U, args), "accepted a misaligned published tap offset");
    std::puts("K9G X3 replay plan PASS");
}
