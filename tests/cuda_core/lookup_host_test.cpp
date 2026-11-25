#include "lookup_cases.hpp"

#include "core/lookup_catalog.hpp"

#include <algorithm>
#include <cstdio>

namespace {

using namespace lookup_test;

void scalar_oracle() {
    using std::bit_cast;
    require(bit_cast<std::uint32_t>(half(0x8000)) == 0x80000000U, "F16 negative zero");
    require(bit_cast<std::uint32_t>(half(0x0001)) == 0x33800000U, "F16 smallest subnormal");
    require(bit_cast<std::uint32_t>(half(0xfe23, true)) == 0xffc46000U, "F16 NaN payload");
    require(bit_cast<std::uint32_t>(half(0x7d45)) == 0x7fffffffU, "F16 native NaN");
    const std::array<std::uint16_t, 4> bf{0x8000, 0x0001, 0xffc2, 0x3f80};
    for (std::size_t i = 0; i < bf.size(); ++i)
        require(bit_cast<std::uint32_t>(plain_value(Form::bf16_to_f32, bf.data(), i)) == std::uint32_t{bf[i]} << 16U,
                "BF16 widen");
    const auto table = q8(64U);
    require(q8_value(table.data(), 0U) == 1.00390625F && q8_value(table.data(), 1U) == -1.00390625F, "Q8 signed codes");
    require(bf16(q8_value(table.data(), 0U), true) == 0x3f80U &&
            bf16(q8_value(table.data(), 32U), true) == 0x3f82U, "BF16 ties round to even");
    require(bf16(bit_cast<float>(0x7fffffffU), true) == 0x7fc0U &&
            bf16(bit_cast<float>(0x7fffffffU), false) == 0x8000U, "Q8 BF16 NaN forms");

    const auto tiles = tile_data(256U);
    require(tiles.bytes.size() == 4U * 1152U, "NVFP4 record extent");
    require(tiles.bytes[0] == 0x70U && tiles.bytes[1024] == 0x38U, "NVFP4 first pair and scale");
    require(tiles.values[1] == 6.0F * kTensorScale, "NVFP4 first scaled nonzero");
    require(tiles.bytes[1152U + 1024U] == 0x7eU, "NVFP4 second column tile scale");
    require(tiles.bytes[2U * 1152U] == 0x70U, "NVFP4 second row tile");
}

void geometry() {
    const auto step = make_step(kIds);
    for (std::uint32_t i = 0; i < kRows; ++i)
        require(exec::step_row_valid(step, i) == active(i), "X3 row mask, flag and header width");
    const std::vector<std::uint32_t> ids(kIds.begin(), kIds.end());
    const auto b = plain16(13U, false);
    const auto direct = reference<float>(Form::bf16_to_f32, b.data(), 13U, 4U, ids);
    const auto x3 = reference<float>(Form::bf16_to_f32, b.data(), 13U, 4U, ids, true);
    require(x3[4U * 13U] == sentinel<float>(), "masked X3 row preserves poison");
    require(direct[2U * 4U * 13U] == x3[2U * 4U * 13U], "valid X3 row matches ID gather");
    require(std::memcmp(direct.data(), direct.data() + 13U, 13U * sizeof(float)) == 0, "broadcast bytes");
    const auto packed = q8(64U);
    const std::vector<std::uint32_t> invalid{0U, kVocabulary};
    const auto fallback = reference<std::uint16_t>(Form::q8_0_to_bf16_row_zero, packed.data(), 64U, 1U, invalid);
    require(std::equal(fallback.begin(), fallback.begin() + 64, fallback.begin() + 64), "invalid MTP ID selects row zero");

    alignas(float) std::array<std::uint8_t, 64> scratch{};
    auto* out = scratch.data() + 16U;
    lookup::View view{{scratch.data(), 32U, 128U}, Form::nvfp4_tile_to_f32, out, kRows};
    require(lookup::valid(view, exec::kStepMaxRows), "valid NVFP4 shape");
    view.table.vocabulary = 31U;
    require(!lookup::valid(view, exec::kStepMaxRows), "NVFP4 partial row tile rejected");
    view.table.vocabulary = 32U;
    view.table.tensor_scale = std::numeric_limits<float>::infinity();
    require(!lookup::valid(view, exec::kStepMaxRows), "NVFP4 invalid tensor scale rejected");
    view.form = Form::q8_0_to_f32;
    view.table.width = 63U;
    require(!lookup::valid(view, lookup::kMaxIdRows), "Q8 partial block rejected");
    view.form = Form::f32;
    view.rows = exec::kStepMaxRows + 1U;
    require(!lookup::valid(view, exec::kStepMaxRows) && lookup::valid(view, lookup::kMaxIdRows), "X3 max rows");
    view.rows = lookup::kMaxIdRows;
    view.copies = std::numeric_limits<std::uint32_t>::max();
    require(!lookup::valid(view, lookup::kMaxIdRows), "output byte extent overflow rejected");
    view.rows = view.copies = 1U;
    view.output = out + 1U;
    require(!lookup::valid(view, lookup::kMaxIdRows), "output alignment checked");
}

int fake_enqueue(void*, void*) { return 0; }

exec::EntryLaunch fake_resolve(void* context, const exec::LauncherRef& ref, const exec::LaunchSite&) {
    *static_cast<const exec::LauncherRef**>(context) = &ref;
    return {fake_enqueue, nullptr};
}

void catalogue() {
    const auto entries = lookup::catalog_entries();
    require(entries.size() == 56U, "eight forms across both ID sources and row classes");
    exec::Catalog catalog;
    catalog.add(entries);
    const exec::KernelKey e1{strix::graph::Op::gather, "embedding.bf16_to_f32", {"plain.bf16", 1},
                            exec::OperandClass::f32_fma, exec::RowsClass::r1};
    require(catalog.find(e1, strix::device::ArchMask::sm120).size() == 1U, "D4 E1 BF16 R1 replacement key");
    for (const auto& entry : entries) {
        const bool x3 = entry.key.form.ends_with(".x3");
        constexpr std::array<std::uint32_t, 4> lows{1U, 2U, 9U, 33U};
        const auto first = lows[static_cast<std::size_t>(entry.key.rows)];
        const exec::LauncherRef* resolved = nullptr;
        exec::EntryBinder binder{fake_resolve, &resolved};
        exec::LaunchSite site{};
        site.rows = first;
        require(static_cast<bool>(entry.bind(site, binder)) && resolved == entry.launcher, "bind row class lower end");
        site.rows = first - 1U;
        require(!entry.bind(site, binder), "reject preceding row class");
        constexpr std::array<std::uint32_t, 3> highs{1U, 8U, 32U};
        site.rows = first < 33U ? highs[static_cast<std::size_t>(entry.key.rows)] : lookup::kMaxIdRows;
        require(static_cast<bool>(entry.bind(site, binder)), "bind row class upper end");
        ++site.rows;
        require(!entry.bind(site, binder), "reject following row class");
        require(entry.launcher->max_rows == (x3 ? exec::kStepMaxRows : lookup::kMaxIdRows), "launcher source bound");
        exec::WorkspaceNeed scratch{32U, 16U};
        entry.workspace(site, scratch);
        require(scratch.bytes == 0U && entry.cost(site, {}).launches == 1U, "single launch and no scratch");
    }
}

}  // namespace

int main() {
    try {
        scalar_oracle();
        geometry();
        catalogue();
        std::puts("PASS K10c lookup CPU oracle, geometry and D4 fragment");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL K10c lookup host: %s\n", error.what());
        return 1;
    }
}
