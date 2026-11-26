include_guard(GLOBAL)

# The host-reference source contract also applies when a CUDA app compiles
# these .cpp files directly. Source options stay local to C++ compilation.
option(STRIX_HOST_LOGPROB_UNCONTRACTED "Use the opt-in unfused FP64 host logprob combine" OFF)

function(strix_host_reference_fp)
    set(root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/..")
    set(sources
        "${root}/src/iq4_nl.cpp"
        "${root}/src/iq4_xs.cpp"
        "${root}/src/token_sampling.cpp"
        "${root}/src/speculation_sampling.cpp"
        "${root}/src/serial_decode_reference.cpp"
        "${root}/src/mtp_forward_semantics.cpp"
        "${root}/src/oracles.cpp"
        "${root}/src/cuda/core/nvfp4_reference.cpp")
    if(MSVC)
        # Precise without /fp:contract disables implicit contraction and keeps
        # MSVC's existing evaluation mode. Retain a target's stronger policy.
        set(options "$<$<NOT:$<BOOL:$<TARGET_PROPERTY:STRIX_HOST_FP_TARGET_STRICT>>>:/fp:precise>")
        set(logprob_options /fp:strict)
    else()
        set(options -fno-fast-math -ffp-contract=off)
        set(logprob_options ${options})
    endif()
    strix_host_fp_in_directory("${CMAKE_CURRENT_SOURCE_DIR}" "${sources}" "${options}"
        "${root}/src/logprob_partials.cpp" "${logprob_options}")
endfunction()

function(strix_host_fp_in_directory dir sources options logprob_source logprob_options)
    if(MSVC)
        get_property(targets DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
        foreach(target IN LISTS targets)
            get_target_property(target_options ${target} COMPILE_OPTIONS)
            if("/fp:strict" IN_LIST target_options)
                set_property(TARGET ${target} PROPERTY STRIX_HOST_FP_TARGET_STRICT TRUE)
            endif()
        endforeach()
    endif()
    set_property(SOURCE ${sources} DIRECTORY "${dir}" APPEND PROPERTY COMPILE_OPTIONS ${options})
    if(STRIX_HOST_LOGPROB_UNCONTRACTED)
        set_property(SOURCE "${logprob_source}" DIRECTORY "${dir}" APPEND PROPERTY COMPILE_OPTIONS ${logprob_options})
    endif()
    get_property(children DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
    foreach(child IN LISTS children)
        strix_host_fp_in_directory("${child}" "${sources}" "${options}" "${logprob_source}" "${logprob_options}")
    endforeach()
endfunction()
