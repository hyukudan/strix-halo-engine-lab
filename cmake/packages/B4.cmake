if(BUILD_TESTING)
    add_executable(strix-b4-tall-host-test tests/cuda_core/b4_tall_host_test.cpp)
    target_include_directories(strix-b4-tall-host-test PRIVATE include src)
    target_compile_features(strix-b4-tall-host-test PRIVATE cxx_std_20)
    strix_target_warnings(strix-b4-tall-host-test)
    if(MSVC)
        target_compile_options(strix-b4-tall-host-test PRIVATE /fp:strict)
    else()
        target_compile_options(strix-b4-tall-host-test PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME strix-b4-tall-host-test COMMAND strix-b4-tall-host-test)
    set_tests_properties(strix-b4-tall-host-test PROPERTIES LABELS "cpu;plan")
endif()

option(STRIX_B4_CUDA_FIXTURE "Build the B4 three-plane tall conformance fixture for sm_120a" OFF)
if(STRIX_B4_CUDA_FIXTURE)
    enable_language(CUDA)
    find_package(Git REQUIRED)

    set(b4_core "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/core")
    set(b4_baseline "${CMAKE_CURRENT_BINARY_DIR}/b4-baseline")
    foreach(path IN ITEMS dense/dense_big.cu tall_k_plan.hpp)
        execute_process(COMMAND "${GIT_EXECUTABLE}" show "6224587f7:src/cuda/core/${path}"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            OUTPUT_VARIABLE body COMMAND_ERROR_IS_FATAL ANY)
        string(REPLACE "#include \"row_plan.hpp\"" "#include \"strix/cuda_core/dense.hpp\"" body "${body}")
        string(REPLACE "#include \"tall_family.cuh\"" "#include \"core/tall_family.hpp\"" body "${body}")
        file(CONFIGURE OUTPUT "${b4_baseline}/${path}" CONTENT "${body}" @ONLY)
    endforeach()

    add_library(strix-b4-tall-baseline OBJECT "${b4_baseline}/dense/dense_big.cu")
    add_library(strix-b4-tall-candidate OBJECT "${b4_core}/dense/dense_big.cu")
    target_compile_definitions(strix-b4-tall-candidate PRIVATE STRIX_B4_CUDA_FIXTURE=1)
    add_executable(strix-b4-tall-fixture tests/cuda_core/b4_tall_cuda_fixture.cu
        "${b4_core}/dense/dense_mma.cu" "${b4_core}/dense/nvfp4.cu" "${b4_core}/nvfp4_tile.cu")
    target_link_libraries(strix-b4-tall-fixture PRIVATE strix-b4-tall-candidate)

    foreach(target IN ITEMS strix-b4-tall-baseline strix-b4-tall-candidate strix-b4-tall-fixture)
        target_include_directories(${target} PRIVATE include src "${b4_core}/dense" "${b4_core}")
        target_compile_definitions(${target} PRIVATE STRIX_CUDA_FAST8_SM120A=1)
        set_target_properties(${target} PROPERTIES
            CUDA_ARCHITECTURES 120a-real CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
        target_compile_options(${target} PRIVATE --fmad=false --prec-div=true --prec-sqrt=true --ftz=false)
        if(MSVC)
            target_compile_options(${target} PRIVATE -Xcompiler=/EHsc,/Zc:preprocessor,/utf-8)
        endif()
    endforeach()
    # Native execution requires explicit operator invocation.
endif()
