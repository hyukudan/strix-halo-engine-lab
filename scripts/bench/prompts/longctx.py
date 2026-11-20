#!/usr/bin/env python3
"""Long-context class prompt: a task prompt behind a long unrelated log.

Usage: longctx.py TASK.txt OUT.txt [--chars 30000]

Writes "Background log, not needed for the task below:", CHARS characters of deterministic numbered sensor-log lines
from a fixed LCG (project-owned, unrelated to any task, so it is no copy source for the answer), then "Task:" and the
task text unchanged. The same arguments always produce the same bytes. Only the standard library.
"""
import argparse


def filler(chars):
    x, lines, size, i = 12345, [], 0, 0
    while size < chars:
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        line = "log %05d station=%03d temp=%d.%d hum=%d wind=%d ok=%d" % (
            i, x % 997, x % 41, (x >> 7) % 10, (x >> 11) % 100, (x >> 17) % 60, x & 1)
        lines.append(line)
        size += len(line) + 1
        i += 1
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task")
    p.add_argument("out")
    p.add_argument("--chars", type=int, default=30000)
    a = p.parse_args()
    if a.chars <= 0:
        raise SystemExit("--chars must be positive")
    with open(a.task, encoding="utf-8") as f:
        task = f.read()
    if not task.strip():
        raise SystemExit("empty task prompt: " + a.task)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("Background log, not needed for the task below:\n" + filler(a.chars) + "\n\nTask:\n" + task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
