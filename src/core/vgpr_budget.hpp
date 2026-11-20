#pragma once
// VGPR budget of a wave32 kernel on gfx11 (RDNA3 / RDNA3.5, gfx1151: 1,536 VGPRs per SIMD lane, 24-register allocation
// granule, at most 16 waves per SIMD): waves per SIMD a kernel's VGPR count allows, i.e. the amdgpu_waves_per_eu target
// at which the compiler keeps that many registers without spilling: 8 waves up to 192 VGPRs, 7 up to 216, 6 up to 240,
// 5 up to 256 (the wave32 maximum). Pure arithmetic: device cores use it for their occupancy targets, host tests check it.
namespace strix::core::vgpr_budget {

inline constexpr unsigned kFile = 1536U, kGranule = 24U, kMaxWaves = 16U, kMaxVgprs = 256U;

constexpr unsigned waves_for(unsigned vgprs) {
    const unsigned allocated = (vgprs + kGranule - 1U) / kGranule * kGranule;
    const unsigned waves = allocated == 0U ? kMaxWaves : kFile / allocated;
    return waves < kMaxWaves ? waves : kMaxWaves;
}

}  // namespace strix::core::vgpr_budget
