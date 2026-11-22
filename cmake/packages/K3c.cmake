if(BUILD_TESTING)
    add_executable(strix-k3c-r1-host-test tests/moe_core/microscaled_r1_cuda_plan_test.cpp)
    target_compile_features(strix-k3c-r1-host-test PRIVATE cxx_std_20)
    target_include_directories(strix-k3c-r1-host-test PRIVATE include src)
    strix_target_warnings(strix-k3c-r1-host-test)
    add_test(NAME strix-k3c-r1-host-test COMMAND strix-k3c-r1-host-test)

    option(STRIX_K3C_CUDA_FIXTURE "Build the CUDA R1 bitwise fixture" OFF)
    if(STRIX_K3C_CUDA_FIXTURE)
        enable_language(CUDA)
        add_executable(strix-k3c-r1-cuda-fixture
            tests/moe_core/microscaled_r1_cuda_fixture.cu
            tests/moe_core/microscaled_r1_cuda_reference.cu)
        set_target_properties(strix-k3c-r1-cuda-fixture PROPERTIES
            CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "120a-real")
        target_include_directories(strix-k3c-r1-cuda-fixture PRIVATE include src)
        target_compile_options(strix-k3c-r1-cuda-fixture PRIVATE --fmad=false --ftz=false)
        if(MSVC)
            target_compile_definitions(strix-k3c-r1-cuda-fixture PRIVATE NOMINMAX)
            target_compile_options(strix-k3c-r1-cuda-fixture PRIVATE -Xcompiler=/EHsc,/Zc:preprocessor)
        endif()
    endif()
endif()
