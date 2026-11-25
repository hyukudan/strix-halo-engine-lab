# Two-rank replication of the shared multi-request decoder: rank 0 schedules, rank 1 replays its backend calls
# (one control message per forward), and its host test over an in-memory pair (both ranks make the same calls and end
# in the same state; every request's ids == its solo run; a broken control message stops rank 1 before it acts).
strix_servespec_serve_sources(src/serve/replicated_cohort.cpp)
strix_servespec_host_test(NAME strix-replicated-cohort-tests
    SOURCES tests/replicated_cohort_tests.cpp
    LIBS strix_serve Threads::Threads)
if(TARGET strix-replicated-cohort-tests AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Exercise the real Linux host transports without a device runtime.
    target_sources(strix-replicated-cohort-tests PRIVATE src/dist_communicator.cpp)
    target_include_directories(strix-replicated-cohort-tests PRIVATE tests/stubs/dist_communicator src)
    target_compile_definitions(strix-replicated-cohort-tests PRIVATE STRIX_TEST_REPLICA_TRANSPORTS)
endif()
