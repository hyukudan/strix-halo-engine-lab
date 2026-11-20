"""Run one deterministic shard of a Python unittest module.

Usage: python -B tools/ci/unittest_shard.py MODULE INDEX COUNT

The module's test ids are sorted and test number k runs in shard k % COUNT, so the COUNT shards together
run every test exactly once (a test added later lands in some shard without a list to maintain). A
module that fails to import fails every shard. Lets ctest spread one long module over its -j slots.
"""
import os
import sys
import unittest


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def shard(tests, index, count):
    ordered = sorted(tests, key=lambda t: t.id())
    return [t for k, t in enumerate(ordered) if k % count == index]


def main(argv):
    if len(argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    module, index, count = argv[1], int(argv[2]), int(argv[3])
    if count < 1 or not 0 <= index < count:
        print(f'bad shard {index} of {count}', file=sys.stderr)
        return 2
    sys.path.insert(0, os.getcwd())  # import like `python -m unittest` (package imports from the working directory)
    tests = list(flatten(unittest.defaultTestLoader.loadTestsFromName(module)))
    picked = shard(tests, index, count)
    print(f'{module}: shard {index}/{count} runs {len(picked)} of {len(tests)} test(s)', flush=True)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(picked))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
