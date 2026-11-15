// Host tests of the Intel backend scaffold: binary16 rounding, the dense-q8a8 reference kernel, the xpu-smi parser,
// arch selection and the no-SYCL stubs. No device, no oneAPI toolchain.
#include "intel/arch/xe2/caps.hpp"
#include "intel/core/backend.hpp"
#include "strix/device/intel_enumerator.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _MSC_VER
#include <crtdbg.h>
#endif

namespace {
using namespace strix;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

void half_rounding() {
    namespace h = intel::fp16;
    require(h::from_float(1.0F) == 0x3c00U, "1.0");
    require(h::from_float(-2.0F) == 0xc000U, "-2.0");
    require(h::from_float(65504.0F) == 0x7bffU, "max finite");
    require(h::from_float(65519.0F) == 0x7bffU, "below the inf boundary");
    require(h::from_float(65520.0F) == 0x7c00U, "inf boundary");
    require(h::from_float(std::numeric_limits<float>::infinity()) == 0x7c00U, "inf");
    require((h::from_float(std::numeric_limits<float>::quiet_NaN()) & 0x7c00U) == 0x7c00U &&
                (h::from_float(std::numeric_limits<float>::quiet_NaN()) & 0x3ffU) != 0U, "nan");
    require(h::from_float(1.0F + 1.0F / 2048.0F) == 0x3c00U, "tie to even (down)");
    require(h::from_float(1.0F + 3.0F / 2048.0F) == 0x3c02U, "tie to even (up)");
    require(h::from_float(std::ldexp(1.0F, -24)) == 0x0001U, "smallest subnormal");
    require(h::from_float(std::ldexp(1.0F, -25)) == 0x0000U, "half of smallest subnormal ties to zero");
    require(h::from_float(std::ldexp(1.0F, -14)) == 0x0400U, "smallest normal");
    require(h::from_float(-0.0F) == 0x8000U, "-0");
    for (std::uint32_t b = 0; b < 0x10000U; ++b) {  // every finite half round-trips
        const auto bits = static_cast<std::uint16_t>(b);
        if ((bits & 0x7c00U) == 0x7c00U) continue;
        require(h::from_float(h::to_float(bits)) == bits, "round trip");
    }
}

void dense_reference() {
    const std::uint32_t in = 128, count = 3, rows = 2, x_stride = 160, blocks = in / 32;
    const std::uint32_t k_bytes = 34, row_bytes = k_bytes + blocks * 34 + 6;  // a K part offset and row padding
    std::vector<std::uint8_t> weight(count * row_bytes, 0xee);
    std::vector<std::int8_t> codes(rows * x_stride);
    std::vector<float> deltas(rows * (x_stride / 32));
    for (std::uint32_t o = 0; o < count; ++o)
        for (std::uint32_t b = 0; b < blocks; ++b) {
            std::uint8_t* blk = weight.data() + o * row_bytes + k_bytes + b * 34;
            const std::uint16_t d = intel::fp16::from_float(0.0625F * static_cast<float>(1 + (o + b) % 3));
            std::memcpy(blk, &d, 2);
            for (std::uint32_t j = 0; j < 32; ++j) blk[2 + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>((static_cast<int>(o * 7 + b * 5 + j * 3) % 255) - 127));
        }
    for (std::uint32_t i = 0; i < codes.size(); ++i) codes[i] = static_cast<std::int8_t>((static_cast<int>(i * 11) % 255) - 127);
    for (std::uint32_t i = 0; i < deltas.size(); ++i) deltas[i] = 0.01F * static_cast<float>(1 + i % 4);
    const std::uint32_t out_stride = 4;
    std::vector<float> out(rows * out_stride, 7.0F);
    intel::dense::Q8A8View v;
    v.weight = weight.data();
    v.codes = codes.data();
    v.deltas = deltas.data();
    v.out = out.data();
    v.row_bytes = row_bytes;
    v.k_bytes = k_bytes;
    v.in = in;
    v.count = count;
    v.rows = rows;
    v.x_stride = x_stride;
    v.out_stride = out_stride;
    require(intel::dense::supported(v), "view supported");
    intel::dense::run_reference(v);
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t o = 0; o < count; ++o) {
            double want = 0.0;
            for (std::uint32_t k = 0; k < in; ++k) {
                const std::uint8_t* blk = weight.data() + o * row_bytes + k_bytes + (k / 32) * 34;
                std::uint16_t d = 0;
                std::memcpy(&d, blk, 2);
                const float w16 = intel::fp16::round(static_cast<float>(static_cast<std::int8_t>(blk[2 + k % 32])) * intel::fp16::to_float(d));
                const float a16 = intel::fp16::round(static_cast<float>(codes[r * x_stride + k]) * deltas[r * (x_stride / 32) + k / 32]);
                want += static_cast<double>(w16) * static_cast<double>(a16);
            }
            const float got = out[r * out_stride + o];
            require(std::fabs(static_cast<double>(got) - want) <= 1e-4 * (1.0 + std::fabs(want)), "reference sum");
        }
        require(out[r * out_stride + 3] == 7.0F, "padding column untouched");
    }
    const auto first = out;
    v.accumulate = 1;
    intel::dense::run_reference(v);
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t o = 0; o < count; ++o) require(out[r * out_stride + o] == first[r * out_stride + o] * 2.0F, "accumulate");
    v.in = 96;
    require(!intel::dense::supported(v), "in % 128 refused");
}

void listing() {
    const char* csv =
        "Device ID,Device Name,UUID,PCI Device ID,PCI BDF Address,Memory Physical Size\n"
        "0,\"Intel(R) Arc(TM) B580 Graphics\",00000000-0000-0003-0000-0000e20b8086,0xe20b,0000:03:00.0,12216.00 MiB\n"
        "1,Intel(R) Arc(TM) 140V GPU (16GB),00000000-0000-0000-0000-000064a08086,0x64a0,0000:00:02.0,16 GiB\r\n";
    const auto v = device::parse_xpu_smi(csv);
    require(v.size() == 2, "two devices");
    require(v[0].ordinal == 0 && v[0].vendor == device::Vendor::intel, "ordinal and vendor");
    require(v[0].name == "Intel(R) Arc(TM) B580 Graphics", "quoted name");
    require(v[0].arch == "xe2-bmg" && !v[0].integrated, "discrete arch");
    require(v[0].pci == "0000:03:00.0", "pci");
    require(v[0].memory_bytes == 12216ULL << 20, "memory MiB");
    require(v[0].uuid == "00000000-0000-0003-0000-0000e20b8086", "uuid");
    require(v[1].arch == "xe2-lnl" && v[1].integrated, "integrated arch");
    require(v[1].memory_bytes == 16ULL << 30, "memory GiB");
    bool refused = false;
    try {
        (void)device::parse_xpu_smi("Device ID,Device Name\nx,card\n");
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    require(refused, "bad index refused");
    require(device::parse_xpu_smi("").empty(), "empty listing");
    require(device::intel_arch_from_device_id(0xb080U) == "xe3-ptl", "ptl");
    require(device::intel_arch_from_device_id(0x56a0U).empty(), "Alchemist is not a target");
    require(device::level_zero_selector(1) == "level_zero:1", "selector");
}

void arch_and_stubs() {
    require(intel::arch::xe2::selects("xe2-bmg") && intel::arch::xe2::selects("xe3-ptl") && !intel::arch::xe2::selects(""), "selects");
    require(intel::arch::xe2::kSubgroup == 16U, "sub-group");
    if (intel::sycl_built()) return;  // the stubs below describe the build without SYCL
    intel::Queue q;
    intel::Capabilities caps;
    require(intel::open_queue(0, &q, &caps) == intel::Status::not_built && q.impl == nullptr, "open_queue stub");
    void* p = &q;
    require(intel::allocate(q, intel::MemoryKind::shared, 64, &p) == intel::Status::not_built && p == nullptr, "allocate stub");
    require(intel::synchronize(q) == intel::Status::not_built, "synchronize stub");
    require(intel::enqueue_dense_q8a8(q, intel::dense::Q8A8View{}) == intel::Status::unsupported, "invalid view first");
    require(std::string(intel::status_name(intel::Status::not_built)) == "not_built", "status name");
}
} // namespace

int main() {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    try {
        half_rounding();
        dense_reference();
        listing();
        arch_and_stubs();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "intel backend host tests: FAIL: %s\n", e.what());
        return 1;
    }
    std::printf("intel backend host tests: ok (sycl %s)\n", strix::intel::sycl_built() ? "built" : "absent");
    return 0;
}
