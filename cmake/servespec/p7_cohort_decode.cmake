# Shared multi-request decoder (join / leave at step boundaries, plan_rows + StepPlanner, context drafts) and its
# host test over a fake engine (each request's ids == its solo run).
strix_servespec_serve_sources(src/serve/cohort_decode.cpp)
if(TARGET strix_session_runtime)
    add_library(strix_cohort_session STATIC src/serve/cohort_session.cpp)
    target_link_libraries(strix_cohort_session PUBLIC strix_session_runtime)
    strix_target_warnings(strix_cohort_session)
    strix_servespec_host_test(NAME strix-cohort-session-tests
        SOURCES tests/cohort_session_tests.cpp
        LIBS strix_cohort_session)
endif()
strix_servespec_host_test(NAME strix-cohort-decode-tests
    SOURCES tests/cohort_decode_tests.cpp
    LIBS strix_serve)
strix_servespec_host_test(NAME strix-cohort-allocation-tests
    SOURCES tests/cohort_allocation_tests.cpp
    LIBS strix_serve)
# Cohort pass plan (balanced passes of at most a backend row cap): coverage, order, cap and no one-row pass.
strix_servespec_host_test(NAME strix-cohort-passes-tests
    SOURCES tests/cohort_passes_tests.cpp)
strix_servespec_host_test(NAME strix-fixed-kv-cohort-tests
    SOURCES tests/fixed_kv_cohort_tests.cpp
    LIBS strix_serve)
# Optional CohortBackend hooks (device drafts, retained prefixes, grouped prompts, per-lane selection) over a fake
# engine: every request's ids == its solo run under its selection policy.
strix_servespec_host_test(NAME strix-cohort-lane-hooks-tests
    SOURCES tests/cohort_lane_hooks_tests.cpp
    LIBS strix_serve)
