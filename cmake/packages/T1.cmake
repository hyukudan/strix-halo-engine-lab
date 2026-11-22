add_library(strix_profile_catalog STATIC src/profile/catalog_levers.cpp)
target_link_libraries(strix_profile_catalog PUBLIC strix_exec_catalog)
strix_target_warnings(strix_profile_catalog)

if(BUILD_TESTING)
    add_executable(strix-profile-table-tests tests/profile_table_tests.cpp
        src/gemma4/engine/profile.cpp src/cuda/models/glm/producer/profile.cpp)
    target_include_directories(strix-profile-table-tests PRIVATE src)
    target_link_libraries(strix-profile-table-tests PRIVATE strix_profile_catalog strix_ds41_rtx_profile strix_serve_json)
    strix_target_warnings(strix-profile-table-tests)

    foreach(engine qwen_hip qwen_cuda gemma4 ds41_rtx glm_cuda)
        add_test(NAME profile_table_identity_${engine} COMMAND strix-profile-table-tests ${engine})
        set_tests_properties(profile_table_identity_${engine} PROPERTIES LABELS "host;T1")
    endforeach()
    add_test(NAME profile_catalog_serve_cli_parity COMMAND strix-profile-table-tests catalogue)
    set_tests_properties(profile_catalog_serve_cli_parity PROPERTIES LABELS "host;T1")
endif()
