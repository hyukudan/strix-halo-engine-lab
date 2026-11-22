# Pair-engine contract of the shared multi-request decoder: first id from the prefill (prefill_emits_first /
# prefill_top), per-lane window cap (max_window), request sampler options, and collective reads over the two-rank
# replica (read_final_logits / propose as calls). Host test: ids == solo run, rank 1 makes rank 0's calls, defaults
# unchanged for backends that bind none of it.
strix_servespec_host_test(NAME strix-cohort-pair-contract-tests
    SOURCES tests/cohort_pair_contract_tests.cpp
    LIBS strix_serve Threads::Threads)
