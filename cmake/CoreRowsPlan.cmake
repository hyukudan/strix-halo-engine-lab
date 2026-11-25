# Backend-neutral rows planner (core/rows_plan): host code, no device language. One definition for every consumer
# (the full build, the CUDA-only build, the B2 tests and fixture); link strix_core_rows_plan, never list rows_plan.cpp.
if(TARGET strix_core_rows_plan)
    return()
endif()
get_filename_component(_rp_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
add_library(strix_core_rows_plan STATIC ${_rp_root}/src/core/rows_plan.cpp)
target_include_directories(strix_core_rows_plan PUBLIC ${_rp_root}/include PRIVATE ${_rp_root}/src)
target_compile_features(strix_core_rows_plan PUBLIC cxx_std_20)
if(COMMAND strix_target_warnings)
    strix_target_warnings(strix_core_rows_plan)
endif()
