# Device-sample hook of the shared multi-request decoder: a backend whose sampled id is decided on the device
# (samples_on_device / sample_final) keeps the host path's call points, ids and sampling receipt; replicated as a
# collective call. Host test.
strix_servespec_host_test(NAME strix-cohort-sample-hook-tests
    SOURCES tests/cohort_sample_hook_tests.cpp
    LIBS strix_serve Threads::Threads)
