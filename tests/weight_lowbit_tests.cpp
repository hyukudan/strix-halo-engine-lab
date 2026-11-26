// Low-bit resident row storage (strix/weight/host_lowbit.hpp): the host encoder, the loader that produces it from BF16
// rows (whole and partial reads) and the incumbent sweep decoder agree; hand-written bytes
// decode to their specified codes, and ties round to even with
// fixed bytes in any rounding mode; a scale beyond FP16 is refused; the chosen scale never reconstructs worse than the
// unclipped scale; the Q8_0 loader is unchanged by the shared block source.
#include "core/iu4_onfly.hpp"
#include "lowbit_folded_fixture.hpp"
#include "core/lowbit_f16_staging.hpp"
#include "inkling2/inkling2_mini_dense_hip.hpp"
#include "core/resident_row_plan.hpp"
#include "core/weight_decode.hpp"
#include "strix/weight/host_lowbit.hpp"
#include "strix/weight/host_q8.hpp"
#include "strix/weight/levers.hpp"
#include "strix/device/declaration.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace wei = strix::weight;
namespace wd = strix::core::weight_decode;

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

std::uint16_t to_bf16(float f) {
    std::uint32_t x;
    std::memcpy(&x, &f, 4);
    return static_cast<std::uint16_t>((x + 0x7fffU + ((x >> 16U) & 1U)) >> 16U);
}

std::vector<std::uint16_t> random_matrix(std::size_t n, std::size_t k, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 0.02F);
    std::vector<std::uint16_t> bits(n * k);
    for (std::size_t i = 0; i < bits.size(); ++i) {
        float v = normal(rng);
        if (i % 97U == 5U) v *= 12.0F;  // outliers exercise the clip search
        bits[i] = to_bf16(v);
    }
    // One all-zero group (row 0, group 1) and one single-signed group (row 1, group 0).
    for (std::size_t j = 64; j < 128; ++j) bits[j] = 0;
    for (std::size_t j = 0; j < 64; ++j) bits[k + j] = to_bf16(std::fabs(strix::core::onfly::bf16_bits_to_f32(bits[k + j])));
    return bits;
}

template <class Storage, wd::Format F>
void check_format(const char* name) {
    constexpr std::size_t n = 3, k = 384;
    const auto bits = random_matrix(n, k, 7U + Storage::high_bytes);
    const wei::Shape shape{{n, k, 0, 0}, 2};
    const auto storage = F == wd::Format::q4_g64 ? wei::kQ4G64WeightStorage : wei::kQ6G64WeightStorage;

    wei::MatrixLoadGeometry geometry{};
    check(bool(wei::matrix_load_geometry(storage, shape, geometry)), "geometry");
    check(geometry.row_bytes == k / 128U * Storage::bytes && geometry.bytes == n * geometry.row_bytes, "row bytes");
    check(geometry.row_bytes == wd::row_bytes(F, k), "dense family row bytes");
    wei::MatrixLoadGeometry odd{};
    check(!wei::matrix_load_geometry(storage, {{n, 192, 0, 0}, 2}, odd), "K % 128 refused");

    const auto source = wei::memory_source(std::as_bytes(std::span(bits)));
    const wei::Slice slice{source, {0, bits.size() * 2U}};
    std::vector<std::byte> loaded(geometry.bytes);
    check(bool(wei::load_bf16_matrix(slice, shape, storage, loaded)), "load");

    // Direct encoder == loader; reference decode == device chunk decoder; codes and error bounds.
    std::vector<std::uint8_t> direct(geometry.bytes);
    std::vector<float> w(n * k);
    for (std::size_t i = 0; i < w.size(); ++i) w[i] = strix::core::onfly::bf16_bits_to_f32(bits[i]);
    for (std::size_t b = 0; b < w.size() / 128U; ++b)
        check(wei::lowbit::encode_block<Storage>(w.data() + b * 128U, direct.data() + b * Storage::bytes), "encode");
    check(std::memcmp(direct.data(), loaded.data(), direct.size()) == 0, "loader bytes == encoder bytes");
    // The bytes do not depend on the caller's rounding mode, and the caller's mode survives the encoder.
    for (const int mode : {FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
        std::vector<std::uint8_t> moded(geometry.bytes);
        if (std::fesetround(mode) != 0) continue;
        for (std::size_t b = 0; b < w.size() / 128U; ++b)
            check(wei::lowbit::encode_block<Storage>(w.data() + b * 128U, moded.data() + b * Storage::bytes), "encode moded");
        check(std::fegetround() == mode, "rounding mode restored");
        std::fesetround(FE_TONEAREST);
        check(moded == direct, "bytes independent of rounding mode");
    }

    for (std::size_t r = 0; r < n; ++r) {
        const auto* row = direct.data() + r * geometry.row_bytes;
        for (unsigned i = 0; i < k; i += 16U) {
            const wd::Chunk<F> chunk(row, i);
            for (unsigned e = 0; e < 16U; ++e) {
                const unsigned col = i + e;
                const auto* block = row + (col / 128U) * Storage::bytes;
                const float ref = wei::lowbit::decode<Storage>(block, col % 128U);
                const float dev = chunk.at(e);
                check(std::memcmp(&ref, &dev, sizeof(ref)) == 0, "chunk decode == reference decode");
            }
        }
        for (unsigned g = 0; g < k / 64U; ++g) {
            const auto* block = row + (g / 2U) * Storage::bytes;
            const float s = wei::lowbit::f16_bits_to_f32(Storage::scale_bits(block, g % 2U));
            const float* src = w.data() + r * k + g * 64U;
            float pos = 0, neg = 0, err = 0, err_full = 0;
            for (unsigned j = 0; j < 64; ++j) {
                pos = std::max(pos, src[j]);
                neg = std::max(neg, -src[j]);
            }
            const float base = std::max(pos / float(Storage::qmax), neg / float(-Storage::qmin));
            if (base == 0.0F) {
                check(s == 0.0F, "zero group has zero scale");
                continue;
            }
            const float full = wei::lowbit::f16_bits_to_f32(wei::lowbit::f32_to_f16_bits(base));
            for (unsigned j = 0; j < 64; ++j) {
                const float d = src[j] - wei::lowbit::decode<Storage>(block, (g % 2U) * 64U + j);
                err += d * d;
                const float df = src[j] - full * float(wei::lowbit::quantise<Storage>(src[j], full));
                err_full += df * df;
            }
            check(err <= err_full, "chosen scale no worse than unclipped");
        }
    }

    // Partial reads of the staged source (start and end inside blocks) equal the whole image.
    wei::Slice packed{};
    check(bool(wei::prepare_bf16_matrix_slice(slice, shape, storage, packed)), "prepare");
    check(packed.range.bytes == geometry.bytes, "packed size");
    for (std::uint64_t offset : {0ULL, 1ULL, 67ULL, 99ULL, 150ULL}) {
        const std::uint64_t count = std::min<std::uint64_t>(geometry.bytes - offset, 211ULL);
        std::vector<std::byte> part(count);
        check(bool(packed.source->read({offset, count}, part)), "partial read");
        check(std::memcmp(part.data(), loaded.data() + offset, count) == 0, "partial read bytes");
    }

    // Non-finite input is refused.
    auto bad = bits;
    bad[200] = 0x7f80U;
    const auto bad_source = wei::memory_source(std::as_bytes(std::span(bad)));
    check(!wei::load_bf16_matrix({bad_source, {0, bad.size() * 2U}}, shape, storage, loaded), "nonfinite refused");
    std::cout << name << " ok\n";
}

void check_q8_unchanged() {
    constexpr std::size_t n = 2, k = 96;
    const auto bits = random_matrix(n, k + 32U, 3U);  // reuse the generator, keep the first n * k values
    std::vector<std::uint16_t> m(bits.begin(), bits.begin() + static_cast<std::ptrdiff_t>(n * k));
    const wei::Shape shape{{n, k, 0, 0}, 2};
    const auto source = wei::memory_source(std::as_bytes(std::span(m)));
    const wei::Slice slice{source, {0, m.size() * 2U}};
    const auto expected = strix::core::onfly::bf16_to_q8_0_host(m.data(), n, k);
    std::vector<std::byte> loaded(expected.size());
    check(bool(wei::load_bf16_matrix(slice, shape, wei::kQ8WeightStorage, loaded)), "q8 load");
    check(std::memcmp(loaded.data(), expected.data(), expected.size()) == 0, "q8 bytes unchanged");
    wei::Slice packed{};
    check(bool(wei::prepare_bf16_matrix_slice(slice, shape, wei::kQ8WeightStorage, packed)), "q8 prepare");
    std::vector<std::byte> part(50);
    check(bool(packed.source->read({17, 50}, part)), "q8 partial read");
    check(std::memcmp(part.data(), expected.data() + 17, 50) == 0, "q8 partial bytes unchanged");
}

// Route plan of low-bit launches (core/resident_row_plan.hpp): device data, a function of the row count (and of the
// panels for the small block); one arithmetic identity per storage for both routes (the
// device fixture tests/hip_dense_lowbit_bench.hip checks that the routes give the same bits).
void check_lowbit_plan() {
    namespace co = strix::core;
    {
        // The route plan is data of the device's declaration (strix/device/declaration.hpp `lowbit_rows`), never a
        // constant of the core header and never a compiled catalogue row: two declarations of the same device (one
        // arch, one profile, one identity) carry their own switch points, a declaration without the key runs one
        // route at every row count, and the plan a launch reads is the one bound to its device ordinal.
        const auto plan_of = [](const std::string& extra) {
            const auto d = strix::device::parse_declaration(
                "schema = \"strix-devices-v1\"\n[devices.igpu]\nvendor = \"amd\"\narch = \"gfx1151\"\n"
                "pci = \"0000:c5:00.0\"\nprofile = \"strix-halo-128\"\n" + extra);
            const auto& rows = d.devices.at(0).lowbit_rows;
            return co::LowBitRowPlan{rows.small_rows, rows.wide_panels};
        };
        const auto measured = plan_of("lowbit_rows = { small_rows = 64, wide_panels = 64 }\n");
        const auto other = plan_of("lowbit_rows = { small_rows = 32, wide_panels = 8 }\n");
        const auto absent = plan_of("");
        check(measured.small_rows == 64U && measured.wide_panels == 64U && other.small_rows == 32U &&
                  other.wide_panels == 8U,
              "plan: two declarations of one device key keep their own switch points");
        check(co::plan_lowbit_rows(48U, measured) == co::LowBitRoute::small_tile &&
                  co::plan_lowbit_rows(48U, other) == co::LowBitRoute::tile && co::lowbit_wide_block(17U, 8U, other) &&
                  !co::lowbit_wide_block(17U, 8U, measured),
              "plan: each declaration's switch points decide its own routes");
        for (unsigned rows = 1U; rows <= 64U; ++rows)
            check(co::plan_lowbit_rows(rows, measured) == co::LowBitRoute::small_tile, "plan: declared rows <= 64 run small");
        for (const unsigned rows : {65U, 128U, 512U, 65535U * 128U})
            check(co::plan_lowbit_rows(rows, measured) == co::LowBitRoute::tile, "plan: declared wider windows run the tile");
        check(!co::lowbit_wide_block(16U, 96U, measured) && co::lowbit_wide_block(17U, 64U, measured) &&
                  co::lowbit_wide_block(32U, 96U, measured) && !co::lowbit_wide_block(17U, 63U, measured) &&
                  !co::lowbit_wide_block(33U, 96U, measured),
              "plan: declared 32-row blocks only for 17..32 rows of >= 64 panels");
        for (const auto& one_route : {absent, co::LowBitRowPlan{}}) {
            for (const unsigned rows : {1U, 16U, 17U, 32U, 64U, 65U, 512U})
                check(co::plan_lowbit_rows(rows, one_route) == co::LowBitRoute::tile, "plan: no declared row = one route");
            check(!co::lowbit_wide_block(24U, 1000U, one_route), "plan: no declared row keeps 16-row blocks");
        }
        // Binding (the device gate binds the declared points with the device): per device ordinal, unbound = one
        // route, out-of-range ordinals bind nothing; the launchers read the bound points.
        namespace dv = strix::device;
        const dv::LowBitRows declared{measured.small_rows, measured.wide_panels};
        check(co::bound_lowbit_row_plan(3).small_rows == 0U && dv::bind_lowbit_rows(3, declared) &&
                  co::bound_lowbit_row_plan(3).small_rows == 64U && co::bound_lowbit_row_plan(3).wide_panels == 64U &&
                  co::bound_lowbit_row_plan(4).small_rows == 0U && dv::bind_lowbit_rows(3, {}) &&
                  co::bound_lowbit_row_plan(3).small_rows == 0U && !dv::bind_lowbit_rows(-1, declared) &&
                  !dv::bind_lowbit_rows(dv::kLowBitRowSlots, declared) &&
                  co::bound_lowbit_row_plan(dv::kLowBitRowSlots).small_rows == 0U,
              "plan: bound per device ordinal, unbound and out-of-range = one route");
    }
    const auto& ids = co::kResidentLowBitTileArithmeticHip;
    check(ids.size() == 2U && ids[0].name != ids[1].name && ids[0].revision && ids[1].revision,
          "plan: one arithmetic identity per low-bit storage");
}

// Byte ranges of resident launches (core/resident_row_plan.hpp): a projection never writes its input, and a grouped
// list never writes one byte twice, whatever the storages; a refused list calls no backend (nothing is written).
template <class Storage>
void check_folded_scale(wei::FormatId format) {
    namespace co = strix::core;
    constexpr auto folded = co::LowBitArithmetic::f16_folded_scale;
    constexpr auto grouped = co::LowBitArithmetic::f16_group_scale;
    std::array<int, Storage::elements> codes{};
    std::array<std::uint8_t, 2U * Storage::bytes> bytes{};
    codes[0] = Storage::qmin;
    const auto limit = 65504.0F / static_cast<float>(-Storage::qmin);
    const std::uint16_t admissible[]{wei::lowbit::f32_to_f16_bits(limit), 0U};
    const std::uint16_t overflowing[]{wei::lowbit::f32_to_f16_bits(65536.0F / static_cast<float>(-Storage::qmin)), 0U};
    wei::lowbit::pack_block<Storage>(codes.data(), admissible, bytes.data());
    wei::lowbit::pack_block<Storage>(codes.data(), overflowing, bytes.data() + Storage::bytes);
    const auto saved = bytes;
    const auto read = [](std::span<const std::uint8_t> src, std::span<std::uint8_t> dst) {
        std::ranges::copy(src, dst.begin());
    };
    wei::ResidentMatrixView matrix{bytes, format, 2U, Storage::elements, Storage::bytes};
    check(!co::certify_folded_scale(matrix, read).empty() && co::admitted_lowbit_arithmetic(folded, matrix) == grouped,
          "one overflowing group declines folded staging for the matrix");
    const float incumbent = wei::lowbit::decode<Storage>(saved.data() + Storage::bytes, 0U);
    const auto from_staging = [&](float value) {
        return co::admitted_lowbit_arithmetic(folded, matrix) == grouped ? value
            : wei::lowbit::f16_bits_to_f32(wei::lowbit::f32_to_f16_bits(value));
    };
    check(incumbent == -65536.0F && std::bit_cast<std::uint32_t>(from_staging(incumbent)) ==
          std::bit_cast<std::uint32_t>(incumbent) && bytes == saved, "declined staging keeps incumbent bytes");
    matrix = {std::span(bytes).first(Storage::bytes), format, 1U, Storage::elements, Storage::bytes};
    check(co::certify_folded_scale(matrix, read).empty() && co::admitted_lowbit_arithmetic(folded, matrix) == folded &&
          from_staging(wei::lowbit::decode<Storage>(bytes.data(), 0U)) == -65504.0F,
          "largest finite FP16 product folds exactly");
    codes[0] = 7;
    const std::uint16_t inexact[]{wei::lowbit::f32_to_f16_bits(1.0F + 0x1p-10F), 0U};
    wei::lowbit::pack_block<Storage>(codes.data(), inexact, bytes.data());
    check(!co::certify_folded_scale(matrix, read).empty() && co::admitted_lowbit_arithmetic(folded, matrix) == grouped,
          "finite product that would round declines folded staging");
}

void check_grouped_fixture_poison() {
    float first = 1.0F, second = 2.0F;
    const auto poison = [](float* output, std::size_t count) { std::fill_n(output, count, -1.0F); };
    const bool ran = strix::test::launch_poisoned_folded_group(&first, &second, 1U, poison, [&] {
        check(second == -1.0F, "second grouped destination is poisoned before launch");
        second = 2.0F;  // Model a launch that only wrote the second destination.
        return true;
    });
    check(ran && first != 1.0F && second == 2.0F, "grouped fixture catches a skipped standalone destination");
}

void check_resident_ranges() {
    namespace co = strix::core;
    namespace w = strix::weight;
    constexpr unsigned width = 128U, outputs = 64U, rows = 2U;
    constexpr std::size_t q4_row = w::Q4G64Storage::row_bytes(width), q8_row = w::Q8Storage::row_bytes(width);
    alignas(16) static std::uint8_t weights_a[outputs * q4_row]{};
    alignas(16) static std::uint8_t weights_b[outputs * q4_row]{};
    alignas(16) static std::uint8_t weights_q8[outputs * q8_row]{};
    alignas(16) static float arena[rows * width + 4U * rows * outputs + 64U]{};
    const auto view = [](std::span<const std::uint8_t> bytes, w::FormatId storage, std::uint64_t row_bytes) {
        return w::ResidentMatrixView{bytes, storage, outputs, width, row_bytes};
    };
    const auto a = view(weights_a, w::kQ4G64WeightStorage, q4_row);
    const auto b = view(weights_b, w::kQ4G64WeightStorage, q4_row);
    const auto q8 = view(weights_q8, w::kQ8WeightStorage, q8_row);
    const float* input = arena;
    float* out0 = arena + rows * width;
    float* out1 = out0 + rows * outputs;  // adjacent: disjoint
    check(co::valid_resident_lowbit_rows(a, outputs, input, rows, out0) &&
              co::valid_resident_q8_rows(q8, outputs, input, rows, out0),
          "ranges: tight output after the input is valid");
    // An aligned partial overlap of the output with the input (output = input + 4 floats), and the input itself.
    float* inside = arena + 4;
    check(!co::valid_resident_lowbit_rows(a, outputs, input, rows, inside) &&
              !co::valid_resident_q8_rows(q8, outputs, input, rows, inside) &&
              !co::valid_resident_lowbit_rows(a, outputs, input, rows, arena),
          "ranges: an output overlapping the input is refused (low-bit and Q8_0)");
    // An output ending exactly where the input starts is disjoint.
    check(co::valid_resident_lowbit_rows(a, outputs, arena + outputs + 16U, 1U, arena + 16U),
          "ranges: output ending at the input is valid");
    using P = co::ResidentRowProjection;
    const std::array<P, 2> same{{{a, outputs, out0}, {b, outputs, out0}}};
    const std::array<P, 2> same_reversed{{{b, outputs, out0}, {a, outputs, out0}}};
    const std::array<P, 2> partial{{{a, outputs, out0}, {q8, outputs, out0 + 4}}};
    const std::array<P, 2> adjacent{{{a, outputs, out0}, {b, outputs, out1}}};
    check(!co::resident_list_ranges_disjoint(input, width, rows, same) &&
              !co::resident_list_ranges_disjoint(input, width, rows, same_reversed),
          "ranges: two projections into one destination are refused (both orders)");
    check(!co::resident_list_ranges_disjoint(input, width, rows, partial),
          "ranges: a partial overlap across storages is refused");
    check(co::resident_list_ranges_disjoint(input, width, rows, adjacent), "ranges: adjacent outputs are valid");
    const std::array<P, 2> into_input{{{a, outputs, out0}, {b, outputs, inside}}};
    check(!co::resident_list_ranges_disjoint(input, width, rows, into_input),
          "ranges: a list output over the input is refused");
    // An output over weights the list reads (1 row, K = 128, two Q4 projections of 64 outputs, aligned independent
    // input, disjoint destinations otherwise): A writes from the first byte of B's weights; also over its own weights
    // and over a Q8_0 run's weights, in both orders. The standalone validators refuse it too.
    {
        alignas(16) static std::uint8_t shared[outputs * q4_row + outputs * sizeof(float)]{};
        const auto b_alias = view(std::span<const std::uint8_t>(shared, outputs * q4_row), w::kQ4G64WeightStorage, q4_row);
        float* over_b = reinterpret_cast<float*>(shared);
        float* over_q8 = reinterpret_cast<float*>(weights_q8 + 16);
        for (const auto& list : {std::array<P, 2>{{{a, outputs, over_b}, {b_alias, outputs, out1}}},
                                 std::array<P, 2>{{{b_alias, outputs, out1}, {a, outputs, over_b}}},
                                 std::array<P, 2>{{{a, outputs, over_q8}, {q8, outputs, out1}}},
                                 std::array<P, 2>{{{q8, outputs, out1}, {a, outputs, over_q8}}}})
            check(!co::resident_list_ranges_disjoint(input, 1U, 1U, list) &&
                      !co::resident_list_ranges_disjoint(input, width, 1U, list),
                  "ranges: an output over weights another projection reads is refused (both orders)");
        check(!co::valid_resident_lowbit_rows(b_alias, outputs, input, 1U, over_b) &&
                  !co::valid_resident_q8_rows(q8, outputs, input, 1U, over_q8),
              "ranges: a standalone output over its own weights is refused");
        static const std::uint16_t bf16_w[outputs * width]{};
        float* over_bf16 = const_cast<float*>(reinterpret_cast<const float*>(bf16_w + 8));
        const std::array<P, 2> on_bf16{{{bf16_w, outputs, out1}, {a, outputs, over_bf16}}};
        check(!co::resident_list_ranges_disjoint(input, width, 1U, on_bf16),
              "ranges: an output over a BF16 run's weights is refused");
    }
    // The mixed BF16 / compact dispatcher refuses an overlapping list before any run is dispatched.
    static const std::uint16_t bf16_bits[outputs * width]{};
    const auto run = [](unsigned& calls) { return [&calls](auto) { ++calls; return 0; }; };
    const auto admit = [](auto) { return true; };
    for (const auto& list : {std::array<P, 2>{{{bf16_bits, outputs, out0}, {a, outputs, out0 + 8}}},
                             std::array<P, 2>{{{a, outputs, out0 + 8}, {bf16_bits, outputs, out0}}}}) {
        unsigned calls = 0;
        const int status = co::dispatch_resident_rows<4>(input, width, rows, std::span<const P>(list), 0, 1,
                                                         run(calls), run(calls), admit, admit);
        check(status == 1 && calls == 0U, "ranges: an overlapping mixed list dispatches nothing");
    }
    unsigned calls = 0;
    const std::array<P, 2> mixed{{{bf16_bits, outputs, out0}, {a, outputs, out1}}};
    check(co::dispatch_resident_rows<4>(input, width, rows, std::span<const P>(mixed), 0, 1, run(calls), run(calls), admit, admit) == 0 &&
              calls == 2U,
          "ranges: a disjoint mixed list dispatches both runs");
    // A valid BF16 run followed by a compact run the backend refuses (a Q4 view with a wrong row pitch): the whole
    // list is refused before the BF16 run is submitted.
    auto bad_pitch = a;
    bad_pitch.row_bytes += 16U;
    const auto backend_admits = [](std::span<const P> compact_run) {
        for (const auto& p : compact_run) {
            const auto& m = std::get<w::ResidentMatrixView>(p.weights);
            if (!co::valid_resident_lowbit_rows(m, p.outputs, arena, rows, p.output)) return false;
        }
        return true;
    };
    for (const auto& list : {std::array<P, 2>{{{bf16_bits, outputs, out0}, {bad_pitch, outputs, out1}}},
                             std::array<P, 2>{{{bad_pitch, outputs, out1}, {bf16_bits, outputs, out0}}}}) {
        unsigned refused_calls = 0;
        const int status = co::dispatch_resident_rows<4>(input, width, rows, std::span<const P>(list), 0, 1,
                                                         run(refused_calls), run(refused_calls), admit, backend_admits);
        check(status == 1 && refused_calls == 0U, "ranges: a malformed compact run after a BF16 run dispatches nothing");
    }
}

void check_bf16_preflight() {
    namespace co = strix::core;
    namespace w = strix::weight;
    using P = co::ResidentRowProjection;
    constexpr unsigned width = 128U;
    constexpr unsigned too_wide = strix::inkling2::kVocabulary + 1U;
    alignas(16) static const float input[width]{};
    alignas(16) static const std::uint8_t compact_bytes[w::Q4G64Storage::row_bytes(width)]{};
    alignas(16) static const std::uint16_t bf16_weights[std::size_t{too_wide} * width]{};
    static float compact_output;
    static std::array<float, too_wide> bf16_output;
    const w::ResidentMatrixView matrix{compact_bytes, w::kQ4G64WeightStorage, 1U, width, sizeof(compact_bytes)};
    const P compact{matrix, 1U, &compact_output};

    for (bool malformed : {false, true}) {
        const P bf16{bf16_weights, malformed ? too_wide : 1U, bf16_output.data()};
        for (const auto& list : {std::array{compact, bf16}, std::array{bf16, compact}}) {
            compact_output = -1.0F;
            bf16_output.fill(-1.0F);
            const auto bf16_admitted = [](std::span<const co::Bf16RowProjection> run) {
                return strix::inkling2::mini_hip_dense_rows_admitted(input, width, 1U, run);
            };
            const int status = co::dispatch_resident_rows<2>(input, width, 1U, list, 0, 1,
                [&](std::span<const co::Bf16RowProjection> run) {
                    if (!bf16_admitted(run)) return 1;  // Same rejection as the HIP launcher.
                    run.front().output[0] = 0.0F;
                    return 0;
                },
                [](std::span<const P> run) { run.front().output[0] = 0.0F; return 0; },
                bf16_admitted,
                [](std::span<const P> run) {
                    return co::valid_resident_lowbit_rows(std::get<w::ResidentMatrixView>(run.front().weights),
                        run.front().outputs, input, 1U, run.front().output, co::LowBitArithmetic::f16_group_scale);
                });
            if (malformed)
                check(status == 1 && compact_output == -1.0F &&
                          std::ranges::all_of(bf16_output, [](float value) { return value == -1.0F; }),
                      "malformed BF16 run leaves both destinations untouched");
            else
                check(status == 0 && compact_output == 0.0F && bf16_output.front() == 0.0F,
                      "valid mixed list submits both runs");
        }
    }

    alignas(16) std::array<float, 17U * width> wide_input{};
    std::array<float, 17U> wide_output{};
    check(!co::valid_resident_lowbit_rows(matrix, 1U, wide_input.data(), 17U, wide_output.data()) &&
              co::valid_resident_lowbit_rows(matrix, 1U, wide_input.data(), 17U, wide_output.data(),
                                             co::LowBitArithmetic::f16_group_scale),
          "wide low-bit rows require the explicit FP16 form");
}

void check_group_accumulator() {
    using Fragment = std::array<float, 8>;
    struct Unit {
        Fragment operator()(Fragment a, Fragment b, Fragment sum) const {
            for (std::size_t lane = 0; lane < sum.size(); ++lane)
                sum[lane] = std::fma(a[lane], b[lane], sum[lane]);
            return sum;
        }
    };
    using Group = strix::core::lowbit_f16::GroupAccumulator<Fragment, Unit>;
    Fragment acc{};
    Fragment left{}, right{};
    left.fill(1.0F);
    const auto add = [&](Group& group, float unit) {
        right.fill(unit);
        group.add(left, right);
    };
    Group first{};
    for (float unit : {0x1p25F, 1.0F, -0x1p25F, 3.0F}) add(first, unit);
    first.apply(2.0F, acc);
    check(acc == Fragment{6, 6, 6, 6, 6, 6, 6, 6}, "K16 units accumulate in insertion order before scaling");
    Group next{};
    add(next, 2.0F);
    next.apply(-3.0F, acc);
    check(acc == Fragment{}, "each group starts from zero and scales once into the previous sum");
    add(next, -1.0F + 0x1p-23F);
    acc.fill(-1.0F);
    next.apply(1.0F - 0x1p-23F, acc);
    check(std::ranges::all_of(acc, [](float value) { return value == -0x1p-46F; }),
          "group scale uses a fused add on every lane");
}

// The resolver admits a per-role mix when the consumer supports each compact format.
void check_lowbit_policies() {
    const wei::WeightLoadScope scope{};
    constexpr std::array all{wei::kBf16WeightStorage, wei::kQ8WeightStorage, wei::kQ4G64WeightStorage,
                             wei::kQ6G64WeightStorage};
    constexpr std::array no_q6{wei::kBf16WeightStorage, wei::kQ8WeightStorage, wei::kQ4G64WeightStorage};
    const auto resolve = [&](const wei::WeightLoadPolicy& policy, strix::profile::Name tier,
                             std::span<const wei::FormatId> consumers) {
        return wei::resolve_weight_load(wei::builtin_registry(), policy, {tier, scope, consumers, true, true}, {});
    };
    const auto mix = resolve(wei::kAttentionQ4HeadQ6Policy, strix::profile::Name::fast, all);
    check(mix.storage == wei::kQ4G64WeightStorage && mix.head_storage == wei::kQ6G64WeightStorage &&
              mix.status == strix::profile::Status::unqualified, "mix resolves q4 attention + q6 head");
    check(mix.storage_for(wei::Role::head) == wei::kQ6G64WeightStorage &&
              mix.storage_for(wei::Role::attention_o) == wei::kQ4G64WeightStorage &&
              mix.storage_for(std::nullopt) == wei::kBf16WeightStorage, "mix storage per role");
    const auto missing = resolve(wei::kAttentionQ4HeadQ6Policy, strix::profile::Name::fast, no_q6);
    check(missing.storage == wei::kBf16WeightStorage && missing.head_storage == wei::kBf16WeightStorage,
          "mix without a q6 consumer stays BF16");
    const auto turbo = resolve(wei::kAttentionHeadQ4Policy, strix::profile::Name::turbo, all);
    check(turbo.storage == wei::kQ4G64WeightStorage && turbo.head_storage == wei::kQ4G64WeightStorage, "q4 turbo arm");
    const auto exact = resolve(wei::kAttentionHeadQ6Policy, strix::profile::Name::standard, all);
    check(exact.storage == wei::kBf16WeightStorage && exact.head_storage == wei::kBf16WeightStorage, "EXACT stays BF16");
    const auto q8 = resolve(wei::kAttentionHeadQ8Policy, strix::profile::Name::fast, all);
    check(q8.storage == wei::kQ8WeightStorage && q8.head_storage == wei::kQ8WeightStorage, "Q8_0 policy unchanged");
    const auto bound = wei::with_legacy(wei::kAttentionQ4HeadQ6Policy, {"legacy-mix", 1});
    check(bound.head_target() == wei::kQ6G64WeightStorage && bound.target == wei::kQ4G64WeightStorage,
          "a legacy binding keeps the head format");
    check(wei::find_weight_load_policy("weights.attention_q4_g64.head_q6_g64@1") == &wei::kAttentionQ4HeadQ6Policy,
          "mix key finds its shared row");
    // Head-only rows: every other tagged role stays BF16, the head takes the low-bit storage; EXACT stays BF16.
    for (const auto* policy : {&wei::kHeadQ6Policy, &wei::kHeadQ4Policy}) {
        const auto head = resolve(*policy, strix::profile::Name::fast, all);
        check(head.storage == wei::kBf16WeightStorage && head.head_storage == policy->head_target() &&
                  head.storage_for(wei::Role::head) == policy->head_target() &&
                  head.storage_for(std::nullopt) == wei::kBf16WeightStorage,
              "head-only row: low-bit head, BF16 elsewhere");
        const auto off = resolve(*policy, strix::profile::Name::fast, no_q6);
        check(policy == &wei::kHeadQ4Policy || off.head_storage == wei::kBf16WeightStorage,
              "head-only q6 row needs a q6 consumer");
        const auto standard = resolve(*policy, strix::profile::Name::standard, all);
        check(standard.head_storage == wei::kBf16WeightStorage, "head-only row: EXACT stays BF16");
    }
    check(wei::find_weight_load_policy("weights.head.q4_g64@1") == &wei::kHeadQ4Policy &&
              wei::find_weight_load_policy("weights.head.q6_g64@1") == &wei::kHeadQ6Policy,
          "head-only keys find their shared rows");
}


// Independently specified superblock bytes: signed codes at the ends of the planes (j = 0, 127) and at the scale-group
// boundary (j = 63 | 64), read through the storage accessor, the reference decode and the device chunk decoder.
struct Expect {
    unsigned j;
    int code;
    float value;
};
template <class Storage, wd::Format F>
void check_layout_bytes(const char* name, const std::array<Expect, 4>& expect) {
    std::array<std::uint8_t, Storage::bytes> block{};
    block[0] = 0x01;  // j = 0: +1
    if constexpr (Storage::high_bytes == 0U) {
        block[31] = 0x80;  // j = 63: high nibble of byte 31 = 8 -> -8
        block[32] = 0x07;  // j = 64: +7
        block[63] = 0xF0;  // j = 127: 15 -> -1
    } else {
        block[79] = 0x80;  // j = 63: low 0, high 2 (byte 64 + 15, bits 6..7) -> 32 -> -32
        block[32] = 0x0F;  // j = 64: low 15
        block[80] = 0x01;  //         high 1 (byte 64 + 16, bits 0..1) -> +31
        block[63] = 0xF0;  // j = 127: low 15
        block[95] = 0xC0;  //          high 3 (byte 64 + 31, bits 6..7) -> 63 -> -1
    }
    block[Storage::scale_offset + 1U] = 0x3C;  // group 0 scale 1.0
    block[Storage::scale_offset + 3U] = 0x40;  // group 1 scale 2.0
    for (unsigned j = 0; j < 128U; ++j) {
        int code = 0;
        float value = 0.0F;
        for (const auto& e : expect)
            if (e.j == j) {
                code = e.code;
                value = e.value;
            }
        check(Storage::code(block.data(), j) == code, "layout code");
        check(wei::lowbit::decode<Storage>(block.data(), j) == value, "layout reference decode");
        const wd::Chunk<F> chunk(block.data(), j / 16U * 16U);
        check(chunk.at(j % 16U) == value, "layout chunk decode");
    }
    std::cout << name << " layout ok\n";
}

// Fixed-byte round-to-nearest-even ties: quotients of exactly k + 1/2 against the stored scale (u = 2^-24).
void check_ties() {
    constexpr float u = 0x1p-24F;
    using Q4 = wei::Q4G64Storage;
    using Q6 = wei::Q6G64Storage;
    {  // q4: 63 x 49u and one 45.5u -> scale 7u, codes 7 and 6 (45.5 / 7 = 6.5)
        std::array<float, 128> w{};
        for (unsigned j = 0; j < 63U; ++j) w[j] = 49.0F * u;
        w[63] = 45.5F * u;
        std::array<std::uint8_t, Q4::bytes> expected{}, out{};
        for (unsigned b = 0; b < 31U; ++b) expected[b] = 0x77;
        expected[31] = 0x67;
        expected[64] = 0x07;
        check(wei::lowbit::encode_block<Q4>(w.data(), out.data()), "q4 tie encode");
        check(out == expected, "q4 tie bytes");
        check(wei::lowbit::quantise<Q4>(45.5F * u, 7.0F * u) == 6, "q4 tie 6.5 -> 6");
        check(wei::lowbit::quantise<Q4>(-45.5F * u, 7.0F * u) == -6, "q4 tie -6.5 -> -6");
        check(wei::lowbit::quantise<Q4>(38.5F * u, 7.0F * u) == 6, "q4 tie 5.5 -> 6");
        check(wei::lowbit::quantise<Q4>(-59.5F * u, 7.0F * u) == -8, "q4 tie -8.5 -> -8");
    }
    {  // q6: 62 x 93u, +37.5u, -37.5u -> scale 3u, codes 31, 12, -12 (37.5 / 3 = 12.5)
        std::array<float, 128> w{};
        for (unsigned j = 0; j < 62U; ++j) w[j] = 93.0F * u;
        w[62] = 37.5F * u;
        w[63] = -37.5F * u;
        std::array<std::uint8_t, Q6::bytes> expected{}, out{};
        for (unsigned b = 0; b < 31U; ++b) expected[b] = 0xFF;
        expected[31] = 0x4C;
        for (unsigned b = 64; b < 79U; ++b) expected[b] = 0x55;
        expected[79] = 0xC5;
        expected[96] = 0x03;
        check(wei::lowbit::encode_block<Q6>(w.data(), out.data()), "q6 tie encode");
        check(out == expected, "q6 tie bytes");
    }
    std::cout << "ties ok\n";
}

// A nonzero group whose unclipped scale exceeds FP16 is refused by the encoder, the direct load and the staged read;
// the largest power-of-two scale that fits still encodes exactly.
template <class Storage>
void check_scale_overflow(wei::FormatId storage, const char* name) {
    std::array<float, 128> w{};
    w[5] = 0x1p22F;
    std::array<std::uint8_t, Storage::bytes> out{};
    out.fill(0xA5);
    const auto status = wei::lowbit::encode_block<Storage>(w.data(), out.data());
    check(!status && status.code == wei::lowbit::EncodeStatus::scale_overflow, "overflow status");
    check(std::all_of(out.begin(), out.end(), [](std::uint8_t b) { return b == 0xA5; }), "output untouched on overflow");

    std::vector<std::uint16_t> bits(128, 0);
    bits[5] = to_bf16(0x1p22F);
    const wei::Shape shape{{1, 128, 0, 0}, 2};
    const auto source = wei::memory_source(std::as_bytes(std::span(bits)));
    const wei::Slice slice{source, {0, bits.size() * 2U}};
    std::vector<std::byte> loaded(Storage::bytes);
    const auto direct = wei::load_bf16_matrix(slice, shape, storage, loaded);
    check(!direct && direct.detail.find("FP16") != std::string::npos, "direct load refuses scale overflow");
    wei::Slice packed{};
    const bool prepared = bool(wei::prepare_bf16_matrix_slice(slice, shape, storage, packed));
    check(prepared, "prepare staged slice for scale overflow read");
    if (prepared) {
        std::vector<std::byte> part(Storage::bytes);
        const auto staged = packed.source->read({0, Storage::bytes}, part);
        check(!staged && staged.detail.find("FP16") != std::string::npos, "staged read refuses scale overflow");
    }

    w[5] = 0x1p15F * static_cast<float>(Storage::qmax);  // scale 2^15 fits FP16
    check(wei::lowbit::encode_block<Storage>(w.data(), out.data()), "largest scale encodes");
    check(wei::lowbit::decode<Storage>(out.data(), 5) == w[5] && wei::lowbit::decode<Storage>(out.data(), 6) == 0.0F,
          "largest scale exact");
    std::cout << name << " overflow ok\n";
}
}  // namespace

int main() {
    check_format<wei::Q4G64Storage, wd::Format::q4_g64>("q4_g64");
    check_format<wei::Q6G64Storage, wd::Format::q6_g64>("q6_g64");
    check_q8_unchanged();
    check_lowbit_plan();
    check_folded_scale<wei::Q4G64Storage>(wei::kQ4G64WeightStorage);
    check_folded_scale<wei::Q6G64Storage>(wei::kQ6G64WeightStorage);
    check_grouped_fixture_poison();
    check_resident_ranges();
    check_bf16_preflight();
    check_group_accumulator();
    check_lowbit_policies();
    check_layout_bytes<wei::Q4G64Storage, wd::Format::q4_g64>(
        "q4_g64", {{{0, 1, 1.0F}, {63, -8, -8.0F}, {64, 7, 14.0F}, {127, -1, -2.0F}}});
    check_layout_bytes<wei::Q6G64Storage, wd::Format::q6_g64>(
        "q6_g64", {{{0, 1, 1.0F}, {63, -32, -32.0F}, {64, 31, 62.0F}, {127, -1, -2.0F}}});
    check_ties();
    check_scale_overflow<wei::Q4G64Storage>(wei::kQ4G64WeightStorage, "q4_g64");
    check_scale_overflow<wei::Q6G64Storage>(wei::kQ6G64WeightStorage, "q6_g64");
    if (failures) {
        std::cerr << failures << " failures\n";
        return 1;
    }
    std::cout << "weight lowbit tests passed\n";
    return 0;
}
