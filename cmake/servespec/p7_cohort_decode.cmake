# Shared multi-request decoder (join / leave at step boundaries, plan_rows + StepPlanner, context drafts) and its
# host test over a fake engine (each request's ids == its solo run).
strix_servespec_serve_sources(src/serve/cohort_decode.cpp)
strix_servespec_host_test(NAME strix-cohort-decode-tests
    SOURCES tests/cohort_decode_tests.cpp
    LIBS strix_serve)
