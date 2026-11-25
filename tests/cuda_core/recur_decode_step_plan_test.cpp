#include "ops/recur/recur_decode_step.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

namespace rec = strix::ops::recur;
namespace exec = strix::exec;
constexpr auto kShape = rec::shape::gdn_2560_16x48x128;
using L = rec::GdnDecodeLayout<kShape>;

void require(bool condition, const char* message) {
    if (condition) return;
    std::fprintf(stderr, "K9G X3 decode plan: %s\n", message);
    std::exit(1);
}

void test_sources() {
    std::array<strix::state::SlotEntry, 6> directory{{
        {0x1000, 12, 3, 15, 0}, {0x2000, 99, 1, 100, 0}, {},
        {0x4000, 12, 3, 15, 0}, {0x5000, 99, 1, 100, 0}, {},
    }};
    exec::StepBlock desc{};
    desc.h.rows = 16;
    desc.h.valid_mask = 0x7fff;
    const auto set = [&](unsigned r, unsigned pos, unsigned slot, unsigned parent) {
        desc.row[r] = {0, pos, slot, parent, 0, exec::step_valid, 0, 0};
    };
    set(0, 12, 0, exec::kStepNoParent);
    set(1, 99, 1, exec::kStepNoParent);
    set(2, 13, 0, 0);
    set(3, 13, 0, 0);
    set(4, 14, 0, 2);
    set(5, 14, 0, 3);
    set(6, 15, 0, 5); // Past capacity.
    set(7, 100, 0, 1); // Parent from another slot.
    set(8, 12, 0, 8); // Self-parent.
    set(9, 12, 0, 15); // Forward parent.
    set(10, 12, 0, exec::kStepNoParent);
    desc.row[10].flags = 0; // Mask says valid, flag says padded.
    set(11, 13, 0, 10);
    set(12, 14, 0, 11); // A valid immediate parent with an invalid ancestor.
    set(13, 14, 0, 0); // Gap.
    set(14, 12, 3, exec::kStepNoParent); // Outside directory.
    set(15, 12, 0, exec::kStepNoParent); // Flag says valid, mask says padded.

    const strix::state::StateTable table{directory.data(), 2, 3};
    rec::DecodeStep step{};
    step.descriptor = &desc;
    step.history = {table, 0};
    step.matrix = {table, 1};
    for (unsigned r = 0; r < 6; ++r) {
        const auto src = rec::decode_source(step, r);
        const auto slot = desc.row[r].slot;
        require(src.history == &directory[slot] && src.matrix == &directory[3U + slot] &&
                src.parent == desc.row[r].parent, "rejected an admissible X3 row");
    }
    for (unsigned r = 6; r < 16; ++r)
        require(!rec::decode_source(step, r).history, "admitted an invalid X3 row");
    directory[3].frontier = 13;
    require(!rec::decode_source(step, 4).history, "admitted mismatched published frontiers");
    directory[3].frontier = 12;
    directory[3].base = 0;
    require(!rec::decode_source(step, 0).matrix, "admitted an unbound matrix");
    require(rec::decode_source(step, 1).matrix == &directory[4], "changed another slot");
}

void test_layout() {
    const rec::DecodeRowStrides strides{L::hidden + 1, L::channels + 4, L::value_width + 2,
        L::channels + 8, L::value_heads + 9, L::value_width + 3, L::channels + 6, L::value_width + 7};
    std::vector<std::uint16_t> input(2 * strides.input);
    alignas(16) std::array<float, 2 * (L::channels + 8)> prepared{};
    std::vector<float> qkv(2 * strides.qkv), z(2 * strides.z), beta(2 * strides.gate),
        decay(2 * strides.gate), out(2 * strides.out), raw(2 * strides.raw_capture), deltas(L::a8_deltas);
    std::vector<std::int8_t> codes(2 * strides.a8_codes);
    exec::StepBlock desc{};
    strix::state::SlotEntry directory[2]{};
    const strix::state::StateTable table{directory, 2, 1};
    rec::DecodeStep step{&desc, {input.data(), qkv.data(), z.data(), prepared.data(), beta.data(), decay.data(),
        out.data(), raw.data()}, strides, {table, 0}, {table, 1}, input.data(), qkv.data(),
        strix::RecurrentStateType::bf16, {codes.data(), deltas.data()}};

    for (unsigned width : {1U, 2U, 4U, 8U, 16U})
        require(rec::valid_decode_step(kShape, width, step), "rejected a supported width");
    for (unsigned width : {0U, 3U, 17U, 32U})
        require(!rec::valid_decode_step(kShape, width, step), "admitted an unsupported width");

    constexpr rec::GdnDecodeShape smaller{512, 2, 8, 64, 3, 2};
    auto smaller_step = step;
    smaller_step.strides = rec::packed_decode_strides(smaller);
    require(rec::valid_decode_step(smaller, 2U, smaller_step), "rejected smaller packed rows");
    require(!rec::valid_decode_step(kShape, 2U, smaller_step), "admitted packed rows for a different shape");
    require(!rec::valid_decode_step(smaller, 4U, smaller_step), "admitted more columns than the A8 pitch");
    smaller_step.a8 = {};
    require(rec::valid_decode_step(smaller, 4U, smaller_step), "applied A8 pitch limits to an FP32 output");

    const auto row = rec::decode_row(step, 1);
    require(row.input == input.data() + strides.input && row.qkv == qkv.data() + strides.qkv &&
            row.z == z.data() + strides.z && row.prepared == prepared.data() + strides.prepared &&
            row.beta == beta.data() + strides.gate && row.log_decay == decay.data() + strides.gate &&
            row.out == out.data() + strides.out && row.raw_capture == raw.data() + strides.raw_capture,
            "incorrect strided row");
    step.rows.raw_capture = nullptr;
    require(!rec::decode_row(step, 1).raw_capture, "shifted a null raw capture");
    step.a8.codes = nullptr;
    require(!rec::valid_decode_step(kShape, 2, step), "admitted half of the A8 view");
    step.a8.deltas = nullptr;
    step.strides.prepared += 1;
    require(!rec::valid_decode_step(kShape, 2, step), "admitted a misaligned prepared stride");
    step.strides.prepared = strides.prepared;
    step.rows.qkv = step.rows.prepared;
    require(!rec::valid_decode_step(kShape, 2, step), "admitted mismatched QKV alias strides");
    step.strides.qkv = strides.prepared;
    require(rec::valid_decode_step(kShape, 2, step), "rejected an exact QKV alias");
    step.strides.input = std::numeric_limits<std::size_t>::max();
    require(!rec::valid_decode_step(kShape, 16, step), "admitted overflowing row offsets");
}

}  // namespace

int main() {
    test_sources();
    test_layout();
    std::puts("K9G X3 decode layout and descriptor PASS");
}
