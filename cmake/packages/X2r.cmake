add_library(strix_exec_cpu STATIC src/exec/executor_cpu.cpp)
target_include_directories(strix_exec_cpu PUBLIC src/exec)
target_link_libraries(strix_exec_cpu PUBLIC strix_exec)
strix_target_warnings(strix_exec_cpu)

if(BUILD_TESTING AND TARGET strix_model_decl)
    add_executable(strix-exec-reference-tests tests/exec_reference_tests.cpp)
    target_include_directories(strix-exec-reference-tests PRIVATE src)
    target_link_libraries(strix-exec-reference-tests PRIVATE strix_exec_cpu strix_model_decl)
    strix_target_warnings(strix-exec-reference-tests)
    if(MSVC)
        target_compile_options(strix-exec-reference-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-exec-reference-tests PRIVATE -fno-fast-math -ffp-contract=off)
    endif()

    foreach(model IN ITEMS inkling2_mini qwen_27b)
        add_test(NAME exec_reference_${model} COMMAND strix-exec-reference-tests ${model}
            "${CMAKE_CURRENT_SOURCE_DIR}/data/graph/templates"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/model_decl")
        set_tests_properties(exec_reference_${model} PROPERTIES LABELS "cpu;plan" TIMEOUT 30)
    endforeach()
endif()
