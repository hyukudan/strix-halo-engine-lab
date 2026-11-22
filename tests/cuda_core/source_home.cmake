if(EXISTS "${STRIX_SOURCE_ROOT}/src/cuda_core")
    message(FATAL_ERROR "Legacy CUDA kernel home exists")
endif()

file(GLOB_RECURSE shared_sources LIST_DIRECTORIES false "${STRIX_SOURCE_ROOT}/src/cuda/core/*")
foreach(source IN LISTS shared_sources)
    file(STRINGS "${source}" legacy_includes
        REGEX "^[ \t]*#[ \t]*include[ \t]*[<\"]([^>\"]*/)?cuda_core/")
    foreach(include IN LISTS legacy_includes)
        # strix/cuda_core remains the public caller API under include/.
        if(NOT include MATCHES "[<\"]strix/cuda_core/")
            file(RELATIVE_PATH path "${STRIX_SOURCE_ROOT}" "${source}")
            message(FATAL_ERROR "${path} includes the legacy CUDA kernel home: ${include}")
        endif()
    endforeach()
endforeach()
