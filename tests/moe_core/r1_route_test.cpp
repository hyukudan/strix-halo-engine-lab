// Host test of the R1 route sources and expert union plan (moe_core/r1_route.hpp, microscaled_r1.hpp):
//   expert_union    groups by ascending expert id, assignments ascending, failed / invalid rows and ids excluded,
//                   repeated experts in one row kept as separate assignments;
//   admits_route    the inline route is legal only for its own policy shape, as a checked route publishing into the
//                   Args route buffers, with no written buffer overlapping a buffer the launch reads or writes;
//   fits_union      rows and expert bounds of the union launches; union_groups; the union geometry table.
// The device kernels are checked bitwise against the route launch and the rows launches by
// tests/cuda_core/moe_r1_route_union_bench.cu (CUDA, RTX ticket).

#include "moe_core/microscaled_r1.hpp"
#include "moe_core/r1_route.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

namespace r1 = strix::moe_core::microscaled_r1;
namespace route = strix::moe_core::r1_route;
namespace moe = strix::moe;
using strix::weight::microscaled::MatrixFormat;

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

struct Policy8 {
    static constexpr moe::RoutePolicy value{moe::RouteScore::sigmoid, true, moe::RouteKey::float_serial,
                                            moe::RouteOrder::selection, 128U, 8U, 1U, 1U, true, true, 1e-20F, 2.5F};
};
struct Policy64 {  // another expert count: refused for a 128-expert launch
    static constexpr moe::RoutePolicy value{moe::RouteScore::sigmoid, true, moe::RouteKey::float_serial,
                                            moe::RouteOrder::selection, 64U, 8U, 1U, 1U, true, true, 1e-20F, 2.5F};
};
struct Math {
    static float exp(float v);
};
using Checked = moe::RouteWaveTraits<true>;
using Unchecked = moe::RouteWaveTraits<false>;

void check_union() {
    // rows 2, top_k 3, experts 8: row 0 {5, 1, 3}, row 1 {3, 7, 5}
    const std::vector<std::uint32_t> ids{5, 1, 3, 3, 7, 5};
    auto groups = route::expert_union(ids, {}, {}, 2U, 3U, 8U);
    check(groups.size() == 4U, "union: four distinct experts");
    const std::vector<std::uint32_t> experts{1, 3, 5, 7};
    const std::vector<std::vector<std::uint32_t>> assignments{{1}, {2, 3}, {0, 5}, {4}};
    for (std::size_t g = 0; g < groups.size() && g < 4U; ++g) {
        check(groups[g].expert == experts[g], "union: ascending expert ids");
        check(groups[g].assignments == assignments[g], "union: ascending assignments of a group");
    }
    // a failed row takes no part
    const std::vector<unsigned> status{0U, 1U};
    groups = route::expert_union(ids, status, {}, 2U, 3U, 8U);
    check(groups.size() == 3U && groups[1].assignments == std::vector<std::uint32_t>{2}, "union: failed row excluded");
    // a row the step marks invalid takes no part
    const std::vector<unsigned char> valid{0U, 1U};
    groups = route::expert_union(ids, {}, valid, 2U, 3U, 8U);
    check(groups.size() == 3U && groups[0].expert == 3U && groups[0].assignments == std::vector<std::uint32_t>{3},
          "union: invalid step row excluded");
    // invalid ids are excluded; a repeated expert in one row keeps both assignments
    const std::vector<std::uint32_t> repeated{2, 8, 2, 6, 6, 9};
    groups = route::expert_union(repeated, {}, {}, 2U, 3U, 8U);
    check(groups.size() == 2U && groups[0].expert == 2U && groups[0].assignments == std::vector<std::uint32_t>{0, 2} &&
              groups[1].expert == 6U && groups[1].assignments == std::vector<std::uint32_t>{3, 4},
          "union: repeated experts and invalid ids");
    check(route::union_groups(1U, 8U, 128U) == 8U && route::union_groups(8U, 8U, 128U) == 64U &&
              route::union_groups(8U, 8U, 32U) == 32U, "union: group bound");
}

struct Buffers {
    std::vector<float> logits = std::vector<float>(16U * 128U), bias = std::vector<float>(128U),
                       weights = std::vector<float>(16U * 8U), input = std::vector<float>(16U * 256U),
                       output = std::vector<float>(16U * 256U);
    std::vector<std::uint32_t> ids = std::vector<std::uint32_t>(16U * 8U);
    std::vector<unsigned> failure = std::vector<unsigned>(16U);
    std::vector<std::uint16_t> activation = std::vector<std::uint16_t>(16U * 8U * 128U);
    strix::moe_core::device_grouped::ExpertWeights matrix{};

    r1::Args args() {
        return {&matrix, ids.data(), weights.data(), failure.data(), input.data(), activation.data(), output.data(),
                128U, 256U, 128U, 30.0F, 30.0F};
    }
    template <class P = Policy8, class T = Checked>
    route::Inline<P, Math, T> source() {
        return {logits.data(), bias.data(), ids.data(), weights.data(), failure.data()};
    }
};

void check_route_legality() {
    Buffers b;
    auto args = b.args();
    auto src = b.source();
    check(r1::admits_route<8U>(args, src, 1U) && r1::admits_route<8U>(args, src, 8U), "route: legal rows 1 and 8");
    check(!r1::admits_route<8U>(args, src, 0U) && !r1::admits_route<8U>(args, src, 9U), "route: rows bounds");
    check(!r1::admits_route<4U>(args, src, 1U), "route: policy top_k differs from the launch");
    check(!r1::admits_route<8U>(args, b.source<Policy64>(), 1U), "route: policy experts differ from the launch");
    check(!r1::admits_route<8U>(args, b.source<Policy8, Unchecked>(), 1U), "route: unchecked route refused");
    {
        auto s = src;
        s.logits = nullptr;
        check(!r1::admits_route<8U>(args, s, 1U), "route: missing logits");
        s = src;
        s.bias = nullptr;
        check(!r1::admits_route<8U>(args, s, 1U), "route: missing bias");
    }
    {
        auto a = args;
        auto s = src;
        a.failure = nullptr;
        s.failure = nullptr;
        check(!r1::admits_route<8U>(a, s, 1U), "route: a checked route needs its status words");
    }
    {
        std::vector<std::uint32_t> other(64U);
        auto s = src;
        s.ids = other.data();
        check(!r1::admits_route<8U>(args, s, 1U), "route: publishes into another ids buffer than down_combine reads");
        s = src;
        std::vector<float> w(64U);
        s.weights = w.data();
        check(!r1::admits_route<8U>(args, s, 1U), "route: publishes into another weights buffer");
    }
    {
        // logits overlapping the published ids (CTAs read logits while one CTA per row publishes)
        auto s = src;
        auto a = args;
        a.ids = s.ids = reinterpret_cast<std::uint32_t*>(b.logits.data() + 100U);
        check(!r1::admits_route<8U>(a, s, 1U), "route: ids overlap the logits");
        s = src;
        a = args;
        a.weights = s.weights = b.bias.data() + 4U;
        check(!r1::admits_route<8U>(a, s, 1U), "route: weights overlap the bias");
        s = src;
        a = args;
        a.failure = s.failure = reinterpret_cast<unsigned*>(b.activation.data() + 10U);
        check(!r1::admits_route<8U>(a, s, 1U), "route: status words overlap the activations");
        s = src;
        a = args;
        a.ids = s.ids = reinterpret_cast<std::uint32_t*>(b.input.data() + 256U);
        check(!r1::admits_route<8U>(a, s, 2U), "route: ids overlap the input rows");
        check(r1::admits_route<8U>(a, s, 1U), "route: ids past the input rows of a one-row launch");
        s = src;
        a = args;
        a.weights = s.weights = reinterpret_cast<float*>(b.ids.data() + 4U);
        check(!r1::admits_route<8U>(a, s, 1U), "route: weights overlap the ids");
        s = src;
        a = args;
        a.failure = s.failure = reinterpret_cast<unsigned*>(b.weights.data() + 7U);
        check(!r1::admits_route<8U>(a, s, 1U), "route: status overlaps the weights");
    }
}

void check_union_fit() {
    constexpr auto cuda = r1::kCudaGridLimits;  // the union launches are CUDA launches
    Buffers b;
    auto args = b.args();
    check(r1::fits_union<MatrixFormat::nvfp4, 8U, 8U>(args, 8U, cuda) && r1::fits_union<MatrixFormat::nvfp4, 8U, 4U>(args, 4U, cuda),
          "union: fits");
    check(!r1::fits_union<MatrixFormat::nvfp4, 8U, 4U>(args, 5U, cuda) && !r1::fits_union<MatrixFormat::nvfp4, 8U, 8U>(args, 0U, cuda),
          "union: rows bounds");
    check(r1::fits_union<MatrixFormat::nvfp4, 8U, 16U>(args, 16U, cuda) && !r1::fits_union<MatrixFormat::nvfp4, 8U, 16U>(args, 17U, cuda),
          "union: verify width 16");
    args.experts = route::kMaxUnionExperts + 32U;
    check(!r1::fits_union<MatrixFormat::nvfp4, 8U, 8U>(args, 8U, cuda), "union: expert mask bound");
    // Host / CUDA union: two weight rows per warp at every union width (one gate and one up, or two down).
    check(r1::kUnionGeometry<4U>.gate_columns == 1U && r1::kUnionGeometry<4U>.down_columns == 2U &&
              r1::kUnionGeometry<8U>.gate_columns == 1U && r1::kUnionGeometry<8U>.down_columns == 2U &&
              r1::kUnionGeometry<16U>.gate_columns == 1U && r1::kUnionGeometry<16U>.down_columns == 2U,
          "union: geometry table");
}

}  // namespace

int main() {
    check_union();
    check_route_legality();
    check_union_fit();
    if (failures != 0) {
        std::fprintf(stderr, "strix-r1-route-test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("PASS strix-r1-route-test: expert union plan, inline route legality, union bounds\n");
    return 0;
}
