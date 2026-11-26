#include "moe_core/grouped_strixpack.hpp"
#include "strix/weight/calib_moments.hpp"

#include <array>
#include <bit>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {
namespace wei = strix::weight;
namespace calib = wei::calib;
namespace sp = wei::sp;
namespace grouped = strix::moe_core::grouped_strixpack;

void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}

template <class F>
void refused(F&& invoke) {
    try {
        invoke();
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error("memory limit should have rejected request");
}

void moment_layers() {
    using calib::MomentLayout;
    constexpr std::size_t limit = 3U * 4U * sizeof(double);
    calib::MomentSet moments(1, MomentLayout::diagonal, limit);
    const std::array<std::string, 1> tensors{"weights"};
    const std::array<std::uint32_t, 1> ids{72};
    const std::array<float, 8> x{1, 2, 3, 4, -2, 3, 0, 1};
    const auto window = moments.add_window(ids);
    for (const auto* path : {"first.moments", "second.moments"}) {
        auto* lhs = &moments.declare("lhs", 4, tensors);
        auto* rhs = &moments.declare("rhs", 4, tensors);
        const std::array targets{lhs, rhs};
        moments.add(window, targets, x.data(), 2);
        require(moments.moment_bytes() == 64U && lhs->h == std::vector<double>{5, 13, 9, 17} &&
                lhs->h == rhs->h && lhs->rows == 2, "diagonal sums or scratch bound");
        moments.flush_layer(path);
        require(moments.moment_bytes() == 0 && moments.inputs().empty(), "flush retained old buffers");
        const auto reopened = calib::MomentSet::read(path, 64);
        require(reopened.layout() == MomentLayout::diagonal && reopened.find("lhs")->diagonal(1) == 13 &&
                reopened.find("rhs")->rows == 2 && reopened.provenance().windows().size() == 1,
                "diagonal serialization lost moments or provenance");
        refused([&] { (void)calib::MomentSet::read(path, 63); });
        std::filesystem::remove(path);
    }
    calib::MomentSet too_small(1, MomentLayout::diagonal, 63);
    too_small.add_window(ids);
    auto* input = &too_small.declare("lhs", 4, tensors);
    const std::array targets{input};
    refused([&] { too_small.add(0, targets, x.data(), 2); });
    require(input->rows == 0 && input->h == std::vector<double>(4), "failed update changed moments");
    refused([&] { (void)too_small.declare("rhs", 4, tensors); });
    calib::MomentSet full(1, MomentLayout::full, 2U * 16U * sizeof(double));
    full.add_window(ids);
    auto* dense = &full.declare("lhs", 4, tensors);
    const std::array full_target{dense};
    full.add(0, full_target, x.data(), 2);
    require(dense->h[4] == -4 && dense->diagonal(1) == 13, "full and diagonal accumulation disagree");
    full.flush_layer("full.moments");
    require(calib::MomentSet::read("full.moments", 128).find("lhs")->h[4] == -4, "legacy full round trip");
    std::filesystem::remove("full.moments");
}

struct Fixture {
    sp::Header header;
    std::vector<std::uint8_t> packed;
    std::array<float, 256> decoded{};
};

template <class Storage>
Fixture make_operand() {
    Fixture fixture;
    fixture.header.bits = Storage::bits;
    fixture.header.base = -2;
    fixture.header.table = sp::linear_table<Storage::bits>();
    fixture.header.flags = sp::kFlagLinear;
    fixture.packed.resize(2U * Storage::bytes);
    for (unsigned row = 0; row < 2; ++row) {
        std::array<unsigned, 128> codes{};
        const std::array<std::uint8_t, 4> scales{2, 8, 17, 26};
        for (unsigned k = 0; k < codes.size(); ++k) {
            codes[k] = (k + row) % (2U * Storage::levels);
            const auto scale = scales[k / 32U];
            const float reference = std::ldexp(static_cast<float>(2U * (codes[k] % Storage::levels) + 1U) *
                                                (1.0F + static_cast<float>(scale % 8U) / 8.0F),
                                                static_cast<int>(scale / 8U) - 2);
            fixture.decoded[row * 128U + k] = codes[k] < Storage::levels ? reference : -reference;
        }
        auto* block = fixture.packed.data() + row * Storage::bytes;
        sp::pack_block<Storage>(codes.data(), scales.data(), block);
        for (unsigned k = 0; k < 128; ++k)
            require(std::bit_cast<std::uint32_t>(sp::decode<Storage>(block, k, fixture.header)) ==
                    std::bit_cast<std::uint32_t>(fixture.decoded[row * 128U + k]), "packed decode vs arithmetic");
    }
    return fixture;
}

void mixed_group() {
    const std::array images{make_operand<sp::SP2Storage>(), make_operand<sp::SP3Storage>(),
                            make_operand<sp::SP4Storage>()};
    const std::array formats{wei::kSP2WeightStorage, wei::kSP3WeightStorage, wei::kSP4WeightStorage};
    const std::array<std::string_view, 3> names{"cold", "hot", "wide"};
    std::vector<wei::Byte> source_bytes(2U * 256U, wei::Byte{7});
    const wei::Slice codes{wei::memory_source(source_bytes), {0, source_bytes.size()}};
    const wei::PackMatrixSource source{wei::PackMatrixSource::Encoding::bf16, 2, 128, codes, {}};
    wei::Digest256 digest{};
    require(bool(source.digest(digest)), "source digest");
    const auto file = "grouped.lp";
    const auto publish = [&] {
        wei::LowbitPackWriter writer(digest, file, names);
        for (std::size_t e = 0; e < names.size(); ++e)
            writer.add(std::string(names[e]), formats[e], 2, 128, digest, images[e].packed, images[e].header);
        require(bool(writer.finish()), "stream pack publish");
    };
    publish();

    wei::LowbitPack pack;
    require(wei::LowbitPack::open(file, digest, pack).code == wei::Error::unsupported_arithmetic,
            "native reader accepted host-only storages");
    require(bool(wei::LowbitPack::open_host(file, digest, pack)) && bool(pack.verify_identity()), "pack admission");
    std::array<grouped::Matrix, 3> staged;
    std::array<grouped::MatrixView, 3> views;
    for (std::size_t e = 0; e < names.size(); ++e) {
        const auto& entry = pack.entries()[e];
        require(entry.tensor == names[e] && bool(staged[e].load(pack, entry, source)), "stage packed expert");
        views[e] = staged[e].view();
        std::array<std::uint8_t, sp::kHeaderBytes> expected{}, actual{};
        images[e].header.write(expected.data());
        views[e].header.write(actual.data());
        require(std::ranges::equal(views[e].bytes, images[e].packed) && expected == actual, "container byte round trip");
    }
    std::array<std::uint32_t, 6> ids{2, 0, 1, 2, 0, 1};
    const std::array<float, 6> route_weights{1, 1, 1, 1, 1, 1};
    const auto plan = strix::moe_core::build_expert_major_plan({3, 2, 3, 0, ids}, route_weights, 1);
    std::vector<float> x(3U * 128U), y(ids.size() * 2U), gather(ids.size() * 128U);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>((i % 67U) + 1U) / 128.0F;
    require(bool(grouped::project(plan, views, grouped::InputRows::token, 128, 2, x, y)), "grouped projection");
    for (const auto& tile : plan.tiles) {
        const auto a = tile.first_assignment;
        const auto* token = x.data() + 128U * plan.grouping.assignment_rows[a];
        std::copy_n(token, 128, gather.data() + 128U * a);
        for (unsigned out = 0; out < 2; ++out) {
            float sum = 0;
            for (unsigned col = 0; col < 128; ++col)
                sum = std::fma(token[col], images[tile.expert].decoded[out * 128U + col], sum);
            require(std::bit_cast<std::uint32_t>(sum) == std::bit_cast<std::uint32_t>(y[a * 2U + out]),
                    "mixed grouped projection vs independent arithmetic");
        }
    }
    const auto reference = y;
    require(bool(grouped::project(plan, views, grouped::InputRows::assignment, 128, 2, gather, y)) &&
            y == reference, "assignment-major input differs");
    views.back().header.bits = 2;
    require(!grouped::project(plan, views, grouped::InputRows::token, 128, 2, x, y) && y == reference,
            "bad mixed format modified output");
    pack = wei::LowbitPack{};
    std::filesystem::remove(file);
}
}  // namespace

int main() {
    try {
        moment_layers();
        mixed_group();
        std::cout << "weight pack CPU reference tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
