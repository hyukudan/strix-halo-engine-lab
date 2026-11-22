if(BUILD_TESTING)
    add_executable(strix-k2c-tall-host-test tests/cuda_core/tall_host_test.cpp)
    target_include_directories(strix-k2c-tall-host-test PRIVATE src)
    target_compile_features(strix-k2c-tall-host-test PRIVATE cxx_std_20)
    strix_target_warnings(strix-k2c-tall-host-test)
    add_test(NAME strix-k2c-tall-host-test COMMAND strix-k2c-tall-host-test)
    set_tests_properties(strix-k2c-tall-host-test PROPERTIES LABELS "cpu;plan")
endif()

option(STRIX_K2C_CUDA_FIXTURE "Build the K2c family and incumbent CUDA fixtures for sm_120a" OFF)
if(STRIX_K2C_CUDA_FIXTURE)
    enable_language(CUDA)
    find_package(Git REQUIRED)

    # Compile the actual pre-K2c sources independently with the same flags.
    # The fixture uses the same inputs and sequence in both executables.
    set(incumbent_ref af82eded07f80fcda9fa65d82b479f058465e416)
    set(core "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/core")
    set(incumbent "${CMAKE_CURRENT_BINARY_DIR}/k2c-incumbent")
    file(MAKE_DIRECTORY "${incumbent}/dense" "${incumbent}/arch/sm120")
    foreach(path IN ITEMS dense/dense_big.cu nvfp4_tile.cu arch/sm120/nvfp4_tile.cuh)
        execute_process(COMMAND "${GIT_EXECUTABLE}" show "${incumbent_ref}:src/cuda/core/${path}"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            OUTPUT_VARIABLE body COMMAND_ERROR_IS_FATAL ANY)
        file(CONFIGURE OUTPUT "${incumbent}/${path}" CONTENT "${body}" @ONLY)
    endforeach()

    add_library(strix-k2c-tall-support OBJECT "${core}/dense/nvfp4.cu" "${core}/dense/dense_mma.cu")
    target_include_directories(strix-k2c-tall-support PRIVATE include src)
    foreach(arm IN ITEMS family incumbent)
        set(sources "${core}")
        if(arm STREQUAL "incumbent")
            set(sources "${incumbent}")
        endif()
        add_executable(strix-k2c-tall-${arm} tests/cuda_core/tall_cuda_fixture.cu
            "${sources}/dense/dense_big.cu" "${sources}/nvfp4_tile.cu")
        target_link_libraries(strix-k2c-tall-${arm} PRIVATE strix-k2c-tall-support)
        target_include_directories(strix-k2c-tall-${arm} PRIVATE include src
            "${core}/dense" "${core}/arch/sm120" "${core}")
        target_compile_definitions(strix-k2c-tall-${arm} PRIVATE STRIX_CUDA_FAST8_SM120A=1)
    endforeach()
    foreach(target IN ITEMS strix-k2c-tall-family strix-k2c-tall-incumbent strix-k2c-tall-support)
        set_target_properties(${target} PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "120a-real")
        target_compile_options(${target} PRIVATE --fmad=false --prec-div=true --prec-sqrt=true --ftz=false)
        if(MSVC)
            target_compile_options(${target} PRIVATE -Xcompiler=/EHsc,/Zc:preprocessor,/utf-8)
        endif()
    endforeach()
    # Native execution is separate from CTest.
endif()
