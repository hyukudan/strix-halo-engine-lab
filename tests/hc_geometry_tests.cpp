// Shared HC: DeclaredArithmetic equals, bit for bit, the two model-named arithmetic structs it replaces (the DSpark
// raw-with-squares form and the GLM projected form); WideGeometry derives the wide-projection constants from the
// model description.

#include "core/hc_core.hpp"
#include "core/hc_geometry.hpp"
#include "glm/common/glm_math.hpp"
#include "strix/ds41_dspark.hpp"
#include "strix/model/bind.hpp"
#include "strix/model_data/qwen3_8_flash_next.hpp"

#include <bit>
#include <cstdint>
#include <cstdio>

namespace {
namespace hc = strix::core::hc;
namespace ds = strix::ds41::dspark;
namespace glm = strix::glm;
namespace data = strix::model_data::qwen3_8_flash_next;

int failures = 0;
void same(float actual, float expected, const char* what, unsigned index) {
    if (std::bit_cast<std::uint32_t>(actual) != std::bit_cast<std::uint32_t>(expected)) {
        ++failures;
        std::fprintf(stderr, "%s[%u]: %a != %a\n", what, index, static_cast<double>(actual), static_cast<double>(expected));
    }
}

using Geometry = hc::WideGeometry<strix::model::narrow<unsigned>(data::dims::hidden), strix::model::narrow<unsigned>(data::hc::count),
                                  strix::model::narrow<unsigned>(data::hc::lowrank)>;
static_assert(Geometry::width == 10240U && Geometry::stages == 320U && Geometry::stage_pitch == 40U);
static_assert(Geometry::rank_blocks == 10U && Geometry::down_pack_halves == 320U * 10240U);
static_assert(Geometry::up_pack_halves == 4U * 2560U * 320U);

}  // namespace

int main() {
    unsigned state = 12345U;
    for (unsigned i = 0; i < 20000U; ++i) {
        state = state * 1664525U + 1013904223U;
        const float x = (static_cast<float>(state >> 8U) / 8388608.0F - 1.0F) * (i % 7U == 0U ? 120.0F : 12.0F);
        same(hc::DeclaredArithmetic::sigmoid(x), ds::sigmoid(x), "sigmoid/dspark", i);
        same(hc::DeclaredArithmetic::sigmoid(x), glm::sigmoid(x), "sigmoid/glm", i);
        same(hc::DeclaredArithmetic::exp(x), glm::declared_exp(x), "exp/glm", i);
        same(hc::DeclaredArithmetic::exp(x), strix::ds41::declared_exp(x), "exp/ds41", i);
        const float ss = (x < 0.0F ? -x : x) * 977.0F;
        same(hc::DeclaredArithmetic::rstd(ss, 5120U, 1.0e-6F), ds::rstd_of(ss, 5120U, 1.0e-6F), "rstd/dspark", i);
        same(hc::DeclaredArithmetic::rstd(ss, 4096U, 1.0e-6F), glm::inv_sqrt(ss / static_cast<float>(4096U) + 1.0e-6F), "rstd/glm", i);
    }
    if (failures != 0) return 1;
    std::puts("hc geometry tests PASS");
    return 0;
}
