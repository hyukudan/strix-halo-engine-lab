# Device-sample hook of the shared multi-request decoder: a backend whose sampled id is decided on the device
# (samples_on_device / sample_final) keeps the host path's call points, ids and sampling receipt; replicated as a
# collective call. Host test.
strix_servespec_host_test(NAME strix-cohort-sample-hook-tests
    SOURCES tests/cohort_sample_hook_tests.cpp
    LIBS strix_serve Threads::Threads)

# Pair glue of the shared scheduler: dist::ExchangeTransport (a Transport over a two-sided exchange, for pair engines
# whose only link is a communicator's host path), ReplicatedCohort over it, and the CohortSession idle hook. Host test.
if(TARGET strix_cohort_session)
    strix_servespec_host_test(NAME strix-cohort-pair-glue-tests
        SOURCES tests/cohort_pair_glue_tests.cpp
        LIBS strix_cohort_session strix_serve Threads::Threads)
endif()
