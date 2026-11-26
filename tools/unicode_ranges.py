"""Unicode range generation for the pinned CPython tables."""

import unicodedata


UCD_VERSION = "15.1.0"
UNICODE_LIMIT = 0x110000


def require_pinned_ucd():
    if unicodedata.unidata_version != UCD_VERSION:
        raise SystemExit(f"Unicode {UCD_VERSION} required, found {unicodedata.unidata_version}")


def nonzero_ranges(value):
    """Yield (first, last, value) for each maximal nonzero range."""
    first = 0
    previous = 0
    for cp in range(UNICODE_LIMIT):
        current = value(cp)
        if current != previous:
            if previous:
                yield first, cp - 1, previous
            first = cp
            previous = current
    if previous:
        yield first, UNICODE_LIMIT - 1, previous
