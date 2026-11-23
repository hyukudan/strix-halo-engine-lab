# Descriptor-aware exact leaves for X2W CUDA recipes.
add_library(strix_wr1_recipe STATIC src/exec/catalog_cuda_wr1.cpp)
target_include_directories(strix_wr1_recipe PUBLIC include src)
target_compile_features(strix_wr1_recipe PUBLIC cxx_std_20)
target_link_libraries(strix_wr1_recipe PUBLIC strix_exec)
strix_target_warnings(strix_wr1_recipe)

if(BUILD_TESTING)
    add_executable(strix-wr1-host-test tests/cuda_core/wr1_host_test.cpp)
    target_link_libraries(strix-wr1-host-test PRIVATE strix_wr1_recipe)
    strix_target_warnings(strix-wr1-host-test)
    add_test(NAME wr1_host COMMAND strix-wr1-host-test)
    set_tests_properties(wr1_host PROPERTIES LABELS "cpu;plan")
endif()

if(TARGET strix_exec_cuda_decode_catalog)
    add_library(strix_wr1_cuda STATIC src/exec/catalog_cuda_wr1.cu)
    # This target is available for an adapter to link when it adopts WR1.
    target_link_libraries(strix_wr1_cuda PUBLIC strix_wr1_recipe PRIVATE strix_cuda_core_kernels)
    get_target_property(wr1_archs strix_exec_cuda_decode_catalog CUDA_ARCHITECTURES)
    set_target_properties(strix_wr1_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${wr1_archs}")

    add_executable(strix-wr1-cuda-fixture EXCLUDE_FROM_ALL tests/cuda_core/wr1_cuda_fixture.cu)
    target_link_libraries(strix-wr1-cuda-fixture PRIVATE strix_wr1_cuda strix_exec_cuda_decode_catalog strix_cuda_core_kernels)
    set_target_properties(strix-wr1-cuda-fixture PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "120a-real")
    foreach(wr1_target IN ITEMS strix_wr1_cuda strix-wr1-cuda-fixture)
        target_compile_options(${wr1_target} PRIVATE --fmad=false --prec-div=true --prec-sqrt=true --ftz=false)
        if(MSVC)
            target_compile_options(${wr1_target} PRIVATE -Xcompiler=/EHsc,/Zc:preprocessor,/utf-8)
        endif()
    endforeach()
    # Explicit operator execution. CTest contains only the CPU target.
endif()
