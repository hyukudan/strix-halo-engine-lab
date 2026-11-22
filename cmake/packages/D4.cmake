add_library(strix_exec_catalog STATIC src/exec/catalog.cpp)
target_include_directories(strix_exec_catalog PUBLIC include)
target_compile_features(strix_exec_catalog PUBLIC cxx_std_20)
strix_target_warnings(strix_exec_catalog)

if(BUILD_TESTING)
    add_executable(strix-exec-catalog-tests tests/exec_catalog_tests.cpp)
    target_link_libraries(strix-exec-catalog-tests PRIVATE strix_exec_catalog strix-weight)
    strix_target_warnings(strix-exec-catalog-tests)
    add_test(NAME strix-exec-catalog-tests COMMAND strix-exec-catalog-tests)
    set_tests_properties(strix-exec-catalog-tests PROPERTIES LABELS "host")

    find_package(Python3 3.11 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME kernelgen_regen COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_LIST_DIR}/../../tools/strix-kernelgen/kernelgen.py" --check)
        set_tests_properties(kernelgen_regen PROPERTIES LABELS "host")
    endif()
endif()
