# Shared multi-request decoder (join / leave at step boundaries, plan_rows + StepPlanner, context drafts) and its
# host test over a fake engine (each request's ids == its solo run).
strix_servespec_serve_sources(src/serve/cohort_decode.cpp)
if(TARGET strix_session_runtime)
    add_library(strix_cohort_session STATIC src/serve/cohort_session.cpp)
    target_link_libraries(strix_cohort_session PUBLIC strix_session_runtime)
    strix_target_warnings(strix_cohort_session)
endif()
strix_servespec_host_test(NAME strix-cohort-decode-tests
    SOURCES tests/cohort_decode_tests.cpp
    LIBS strix_serve)
strix_servespec_host_test(NAME strix-fixed-kv-cohort-tests
    SOURCES tests/fixed_kv_cohort_tests.cpp
    LIBS strix_serve)
