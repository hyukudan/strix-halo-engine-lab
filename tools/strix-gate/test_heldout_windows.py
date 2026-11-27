"""Held-out window set: cut windows readable by gate v2, reference-only pre-filter drops greedy windows (R3)."""
import json
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gate_v2  # noqa: E402
import heldout_windows as hw  # noqa: E402

SPEC = {"docs": [
    {"id": "en-a", "kind": "wiki", "class": "prose-en", "lang": "en", "bands": ["b8"], "chars": 1},
    {"id": "code-a", "kind": "files", "class": "code", "lang": "code", "bands": ["b8", "b2"], "chars": 1, "files": []},
    {"id": "chat-a", "kind": "chat", "class": "chat", "lang": "mixed", "bands": ["b2"], "chars": 1}],
    "templates": {"m": {"begin": "<s>", "turn": "<u>{user}<a>{assistant}"}}}


def write_spec(tmp):
    p = tmp / "spec.json"
    p.write_text(json.dumps(SPEC), encoding="utf-8")
    return p


def test_cut_prefix_manifest_and_classes(tmp_path):
    spec = write_spec(tmp_path)
    ids = tmp_path / "ids"
    ids.mkdir()
    (ids / "en-a.m.ids").write_text(",".join(str(i) for i in range(100, 140)), encoding="utf-8")
    (ids / "code-a.m.ids").write_text(" ".join(str(i) for i in range(7, 40)), encoding="utf-8")
    out = tmp_path / "w"
    hw.main(["cut", "--spec", str(spec), "--ids", str(ids), "--model", "m", "--band", "b8", "--tokens", "16",
             "--arms", "exact,fast", "--out", str(out)])
    man = gate_v2.read_manifest(out / "windows.tsv")
    assert list(man) == ["en-a-16", "code-a-16"]
    assert man["en-a-16"].tolist() == list(range(100, 116))  # the document's first N tokens
    assert (out / "windows.tsv").read_text().splitlines()[0].split("\t")[3] == "exact,fast"
    assert json.loads((out / "classes.json").read_text()) == {"en-a-16": "prose-en", "code-a-16": "code"}
    with pytest.raises(SystemExit):  # a document shorter than the window is refused, never padded
        hw.main(["cut", "--spec", str(spec), "--ids", str(ids), "--model", "m", "--band", "b8", "--tokens", "35",
                 "--out", str(out)])


def test_render_chat_template(tmp_path):
    spec = write_spec(tmp_path)
    (tmp_path / "chat-a.turns.json").write_text(json.dumps([["q1", "a1"], ["q2", "a2"]]), encoding="utf-8")
    hw.main(["render", "--spec", str(spec), "--docs", str(tmp_path), "--model", "m"])
    assert (tmp_path / "chat-a.m.txt").read_text(encoding="utf-8") == "<s><u>q1<a>a1<u>q2<a>a2"


def test_faq_pairs():
    text = "Intro text.\n\nQ: How do I do it?\n\nLike this.\n\nAnd this.\n\nWhy not?\n\nBecause.\n"
    assert hw.faq_pairs(text) == [("How do I do it?", "Like this.\n\nAnd this."), ("Why not?", "Because.")]


def head_dump(path, ids, rows, vocab, greedy_rows):
    """rows x vocab logits; row j predicts ids[n - rows + j + 1]; the first greedy_rows rows put the label on top."""
    n = len(ids)
    z = np.zeros((rows, vocab), dtype=np.float32)
    for j in range(rows):
        label = ids[n - rows + j + 1] if j + 1 < rows else 0
        z[j, (label + 1) % vocab] = 1.0
        if j < greedy_rows:
            z[j, label] = 2.0
    z.tofile(path)


def test_prefilter_drops_greedy_reference_only(tmp_path):
    vocab, rows = 50, 40
    d = tmp_path / "d"
    d.mkdir()
    lines = []
    for name, greedy in (("greedy", rows), ("held", 20), ("edge", 38)):
        ids = [(7 * i + len(name)) % vocab for i in range(64)]
        (tmp_path / f"{name}.ids").write_text(" ".join(map(str, ids)), encoding="utf-8")
        head_dump(d / f"{name}-exact.f32", ids, rows, vocab, greedy)
        lines.append(f"{name}\t64\t{tmp_path / (name + '.ids')}\texact,fast")
    lines.append(f"notrun\t64\t{tmp_path / 'held.ids'}\texact,fast")
    man = tmp_path / "m.tsv"
    man.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = tmp_path / "kept.tsv"
    hw.main(["prefilter", "--dumps", str(d), "--manifest", str(man), "--ref", "exact", "--head-rows", str(rows),
             "--out", str(out)])
    kept = [x.split("\t")[0] for x in out.read_text().splitlines() if x]
    # 39 scored rows: greedy 39/39 -> dropped; edge 38/39 = 0.974 > 0.95 -> dropped; held 20/39 kept; notrun missing
    assert kept == ["held"]
    assert not list(d.glob("*-fast.f32"))  # the candidate arm is never read (none exists)
