include_guard(GLOBAL)
get_filename_component(STRIX_WEIGHT_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
add_library(strix-weight STATIC
    ${STRIX_WEIGHT_ROOT}/src/weight/types.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/contracts.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/registry.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/reference.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/formats/plain.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/formats/ggml.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/formats/gsq.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/formats/ds41.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/formats/packed.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/formats/fp8.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/formats/exl3.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/formats/channel_quant.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/formats/nvfp4.cpp
    # Existing numerical implementations remain the only scalar delegates.
    ${STRIX_WEIGHT_ROOT}/src/ggml_dequant.cpp
    ${STRIX_WEIGHT_ROOT}/src/gguf_reader.cpp
    ${STRIX_WEIGHT_ROOT}/src/iq4_nl.cpp
    ${STRIX_WEIGHT_ROOT}/src/ds41_formats.cpp
    ${STRIX_WEIGHT_ROOT}/src/ds41_moe.cpp
    ${STRIX_WEIGHT_ROOT}/src/ds41_expert_tp.cpp
    ${STRIX_WEIGHT_ROOT}/src/ds41_packed.cpp
    ${STRIX_WEIGHT_ROOT}/src/ds41_native_fp8.cpp
    ${STRIX_WEIGHT_ROOT}/src/q6p_registry.cpp
    ${STRIX_WEIGHT_ROOT}/src/cuda/core/nvfp4_reference.cpp
    ${STRIX_WEIGHT_ROOT}/src/gemma4/exl3/host.cpp
    ${STRIX_WEIGHT_ROOT}/src/h3/te/te_sha256.cpp
    ${STRIX_WEIGHT_ROOT}/src/serve/json.cpp)
target_include_directories(strix-weight PUBLIC ${STRIX_WEIGHT_ROOT}/include PRIVATE ${STRIX_WEIGHT_ROOT}/src)
target_compile_features(strix-weight PUBLIC cxx_std_20)
if(MSVC)
    target_compile_options(strix-weight PRIVATE /fp:strict /D_CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(strix-weight PRIVATE -ffp-contract=off)
endif()
