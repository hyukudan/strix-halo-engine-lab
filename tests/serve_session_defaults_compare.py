"""Compare raw API response dumps: resolved defaults ON without sessions vs OFF."""
import concurrent.futures
import re
import subprocess
import sys

exe = sys.argv[1]
with concurrent.futures.ThreadPoolExecutor(2) as pool:
    for attempt in range(5):
        futures = [pool.submit(subprocess.check_output, [exe, *args]) for args in
                   [("--defaults-on-no-session",), ("--session-cache", "off")]]
        enabled, disabled = [f.result() for f in futures]
        if enabled == disabled:
            # The transcript size is platform specific (Windows stdout adds CR); identity is what matters.
            assert len(disabled) > 0, "empty API transcript"
            print(f"PASS default request without session == OFF: {len(enabled)} identical bytes")
            break
        # Retry construction across a wall-clock second; never normalize compared bytes.
        if re.sub(rb'"created":\d+', b'"created":0', enabled) != re.sub(rb'"created":\d+', b'"created":0', disabled):
            raise AssertionError("default request without a session changed API bytes")
    else:
        raise AssertionError("services did not construct in the same clock second")
