# CPU-only CMake lookup regression for a compiler-selected ROCm toolkit.
# Evaluate the actual find_library call from the root project, not a duplicate implementation.
cmake_minimum_required(VERSION 3.24)
if(NOT STRIX_REPO_ROOT)
    message(FATAL_ERROR "STRIX_REPO_ROOT must point to the source tree")
endif()
file(READ "${STRIX_REPO_ROOT}/CMakeLists.txt" _root_cmake)
string(REGEX MATCH "find_library\\([ \t\r\n]*STRIX_HIPBLASLT_LIBRARY[^)]*\\)" _lookup "${_root_cmake}")
if(NOT _lookup)
    message(FATAL_ERROR "Cannot locate root hipBLASLt find_library statement")
endif()

set(_sandbox "${CMAKE_CURRENT_BINARY_DIR}/strix-hip-toolkit-lookup-fixture")
file(REMOVE_RECURSE "${_sandbox}")
foreach(_path compiler/lib explicit/lib sysroot/opt/rocm/core-10.1/lib)
    file(MAKE_DIRECTORY "${_sandbox}/${_path}")
    file(WRITE "${_sandbox}/${_path}/libhipblaslt.so" "lookup-fixture")
endforeach()

function(check_lookup label expected)
    unset(STRIX_HIPBLASLT_LIBRARY CACHE)
    unset(STRIX_HIPBLASLT_LIBRARY)
    cmake_language(EVAL CODE "${_lookup}")
    if(NOT STRIX_HIPBLASLT_LIBRARY STREQUAL "${expected}")
        message(FATAL_ERROR "${label}: expected ${expected}, got ${STRIX_HIPBLASLT_LIBRARY}")
    endif()
endfunction()

# An explicit compiler root must win over /opt/rocm when ROCM_PATH is unset.
set(ENV{ROCM_PATH} "")
set(CMAKE_HIP_COMPILER_ROCM_ROOT "${_sandbox}/compiler")
check_lookup("compiler-root" "${_sandbox}/compiler/lib/libhipblaslt.so")

# An explicit ROCM_PATH still has priority, preserving the old override contract.
set(ENV{ROCM_PATH} "${_sandbox}/explicit")
check_lookup("explicit-ROCM_PATH" "${_sandbox}/explicit/lib/libhipblaslt.so")

# Do not regress the historical HIP DEB layout /opt/rocm/core-10.1/lib.
# Root all absolute find_library paths in an isolated fake filesystem, with no installed ROCm required.
set(ENV{ROCM_PATH} "")
set(CMAKE_HIP_COMPILER_ROCM_ROOT "")
set(CMAKE_FIND_ROOT_PATH "${_sandbox}/sysroot")
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
check_lookup("legacy-core-10.1" "${_sandbox}/sysroot/opt/rocm/core-10.1/lib/libhipblaslt.so")
file(REMOVE_RECURSE "${_sandbox}")
