# Pair-engine contract, round 2, of the shared multi-request decoder: per-request lane bind with engine traits
# (bind_lane), the engine's context ceiling with a finish reason (context_tokens / CohortResult::finish), whole-prompt
# restore for prefill-emits-first backends, and engine control messages. Host test.
strix_servespec_host_test(NAME strix-cohort-pair-contract2-tests
    SOURCES tests/cohort_pair_contract2_tests.cpp
    LIBS strix_serve)
