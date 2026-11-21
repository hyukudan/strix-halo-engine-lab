include_guard(GLOBAL)

# Semicolon-separated targets, for example -DSTRIX_HIP_ARCHS="gfx1151;gfx1152".
# Compilation support does not qualify a device's tuning or numerical paths.
if(DEFINED STRIX_HIP_ARCHITECTURES AND NOT DEFINED STRIX_HIP_ARCHS)
    set(STRIX_HIP_ARCHS "${STRIX_HIP_ARCHITECTURES}" CACHE STRING "AMD GPU targets for every HIP target")
    message(DEPRECATION "STRIX_HIP_ARCHITECTURES is now STRIX_HIP_ARCHS; migrated the existing target list")
endif()
set(STRIX_HIP_ARCHS "gfx1151" CACHE STRING "AMD GPU targets for every HIP target")
if(STRIX_HIP_ARCHS STREQUAL "")
    message(FATAL_ERROR "STRIX_HIP_ARCHS must contain at least one HIP architecture")
endif()

# The flag applies to every device pass in a HIP compiler invocation. CDNA
# requires wave64. In a mixed build each declared architecture uses its
# native default (wave32 on RDNA and wave64 on CDNA).
set(STRIX_HIP_WAVE32_FLAG -mno-wavefrontsize64)
if("${STRIX_HIP_ARCHS};${CMAKE_HIP_ARCHITECTURES}" MATCHES "(^|;)(gfx942|gfx950)(;|$)")
    set(STRIX_HIP_WAVE32_FLAG "")
endif()
