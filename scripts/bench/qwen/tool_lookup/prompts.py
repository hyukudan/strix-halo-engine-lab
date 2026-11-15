#!/usr/bin/env python3
"""Agent-style requests for the tool-result lookup source (src/session/tool_source.cpp).

Each class is one chat request in OpenAI form: the model has read a file through a tool and must write it back.
  rewrite_num   read_file result line-numbered (cat -n), whole-file rewrite through write_file
  rewrite_json  read_file result as JSON with an escaped "content" string, whole-file rewrite
  edit_num      numbered read_file result, one edit_file call (old_string / new_string)
  rewrite_raw   control: the tool result is the raw file (nothing to restore, the switch must change nothing)
  prose         control: no tools (nothing to restore)
"""
import json

FILE = '''"""Station readings: load, validate and summarise hourly sensor data."""
import csv
import statistics
import sys
from dataclasses import dataclass


@dataclass
class Reading:
    station: str
    hour: int
    temperature: float
    humidity: float


def load_readings(path):
    readings = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            readings.append(Reading(row["station"], int(row["hour"]), float(row["temperature"]), float(row["humidity"])))
    return readings


def validate(readings):
    bad = [r for r in readings if not 0 <= r.hour < 24 or not 0.0 <= r.humidity <= 100.0]
    if bad:
        raise ValueError(f"{len(bad)} readings out of range")
    return readings


def summarise(readings):
    by_station = {}
    for r in readings:
        by_station.setdefault(r.station, []).append(r)
    summary = {}
    for station, rows in sorted(by_station.items()):
        temps = [r.temperature for r in rows]
        summary[station] = {
            "count": len(rows),
            "mean_temperature": statistics.mean(temps),
            "max_temperature": max(temps),
            "min_temperature": min(temps),
            "mean_humidity": statistics.mean(r.humidity for r in rows),
        }
    return summary


def print_report(summary, out=sys.stdout):
    for station, values in summary.items():
        out.write(f"{station}: {values['count']} readings, mean {values['mean_temperature']:.1f} C\\n")


def main(argv):
    if len(argv) != 2:
        print("usage: stations.py READINGS.csv", file=sys.stderr)
        return 2
    print_report(summarise(validate(load_readings(argv[1]))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''

PROSE = ("Write a short essay (about 250 words) on why public libraries still matter in a digital age. "
         "Use plain prose paragraphs, no lists and no headings.")

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write a whole file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace one exact text span of a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_string": {"type": "string"},
                       "new_string": {"type": "string"}}, "required": ["path", "old_string", "new_string"]}}},
]

REWRITE = ("In stations.py rename the function `summarise` to `summarise_by_station` everywhere and make `validate` "
           "also reject a temperature below -90. Read the file, then write the whole updated file back with write_file.")
EDIT = ("In stations.py make `validate` also reject a temperature below -90. Read the file, then make exactly one "
        "edit_file call whose old_string is the whole current `validate` function and new_string the updated one.")


def numbered(text):
    return "".join("%6d\t%s\n" % (i + 1, line) for i, line in enumerate(text.split("\n")[:-1]))


def agent(task, result):
    call = {"id": "call_1", "type": "function",
            "function": {"name": "read_file", "arguments": json.dumps({"path": "stations.py"})}}
    return {"messages": [{"role": "user", "content": task},
                         {"role": "assistant", "content": "", "tool_calls": [call]},
                         {"role": "tool", "tool_call_id": "call_1", "content": result}],
            "tools": TOOLS}


def request(name):
    if name == "rewrite_num":
        return agent(REWRITE, numbered(FILE))
    if name == "rewrite_json":
        return agent(REWRITE, json.dumps({"path": "stations.py", "content": FILE}))
    if name == "edit_num":
        return agent(EDIT, numbered(FILE))
    if name == "rewrite_raw":
        return agent(REWRITE, FILE)
    if name == "prose":
        return {"messages": [{"role": "user", "content": PROSE}]}
    if name == "warm":
        return {"messages": [{"role": "user", "content": "Say hello in one short sentence."}]}
    raise KeyError(name)


CLASSES = ["rewrite_num", "rewrite_json", "edit_num", "rewrite_raw", "prose"]
