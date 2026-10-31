# Exact ownership/reuse fallbacks, using the existing small native fixtures.
if(NOT STRIX_GEMMA4_DEVICE_BACKEND)
    add_library(strix_gemma4_draft_lifecycle_host_compile OBJECT EXCLUDE_FROM_ALL tests/gemma4_integration/integration.cpp)
    target_link_libraries(strix_gemma4_draft_lifecycle_host_compile PRIVATE strix_gemma4_engine)
    target_compile_definitions(strix_gemma4_draft_lifecycle_host_compile PRIVATE STRIX_GEMMA4_DEVICE_BACKEND=1)
    strix_target_warnings(strix_gemma4_draft_lifecycle_host_compile)
endif()
if(BUILD_TESTING AND STRIX_GEMMA4_DEVICE_BACKEND)
    add_test(NAME gemma4-integration-draft-copy COMMAND strix-gemma4-integration-tests --device)
    set_tests_properties(gemma4-integration-draft-copy PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
        ENVIRONMENT "STRIX_G4_DRAFT_MODEL=0;STRIX_G4_DRAFT_REUSE=0;STRIX_G4_DRAFT_HEAD_SHARE=0")
    add_test(NAME gemma4-integration-draft-no-reuse COMMAND strix-gemma4-integration-tests --device)
    set_tests_properties(gemma4-integration-draft-no-reuse PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4"
        ENVIRONMENT "STRIX_G4_DRAFT_REUSE=0")
endif()
