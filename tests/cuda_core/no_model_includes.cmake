file(GLOB_RECURSE shared_sources LIST_DIRECTORIES false "${STRIX_SOURCE_ROOT}/src/cuda/core/*")
if(NOT shared_sources)
    message(FATAL_ERROR "CUDA core source directory is empty or missing")
endif()

foreach(source IN LISTS shared_sources)
    file(STRINGS "${source}" model_includes
        REGEX "^[ \t]*#[ \t]*include[ \t]*[<\"]([^>\"]*/)?models/")
    if(model_includes)
        file(RELATIVE_PATH path "${STRIX_SOURCE_ROOT}" "${source}")
        message(FATAL_ERROR "${path} includes a model source: ${model_includes}")
    endif()
endforeach()
