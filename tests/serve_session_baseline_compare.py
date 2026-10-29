"""Compare in-memory API bytes with an executable built from 05fedd80 api.cpp.
Pass current and baseline harness executables. No response normalization is used.
"""
import concurrent.futures
import subprocess
import sys

with concurrent.futures.ThreadPoolExecutor(2) as pool:
    for attempt in range(5):
        futures = [pool.submit(subprocess.check_output, [sys.argv[1], "--session-cache", "off"]),
                   pool.submit(subprocess.check_output, [sys.argv[2]])]
        current, baseline = [f.result() for f in futures]
        if current == baseline:
            assert current, "empty API transcript"
            print(f"PASS session cache OFF (switch) API == 05fedd80: {len(current)} identical bytes; hello and route assertions passed")
            break
        # The /models creation second is sampled at construction. Retry only a
        # clock-boundary mismatch, without altering either captured response.
        import re
        if re.sub(rb'"created":\d+', b'"created":0', current) != re.sub(rb'"created":\d+', b'"created":0', baseline):
            raise AssertionError("disabled API bytes differ from baseline")
    else:
        raise AssertionError("could not construct both services in the same clock second")
