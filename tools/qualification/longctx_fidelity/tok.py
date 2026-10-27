"""Tokenizer process adapter and exact single non-thinking Qwen user template.

Contract inspected in apps/tokenize_main.cpp and src/serve/chat_template.cpp:
encode parses control tokens; plain does not; output is comma-separated decimal IDs.
Decode takes a file of comma/space-separated decimal IDs and emits raw text.
Every encode call passes one complete candidate prompt, never separate BPE fragments.
"""
import subprocess
import tempfile
from pathlib import Path
try:
    from .common import sha, file_sha
except ImportError:
    from common import sha, file_sha

PREFIX = "<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
TEMPLATE_SHA256 = sha((PREFIX + "{content}" + SUFFIX).encode())


def render(content):
    return PREFIX + content.strip() + SUFFIX


class FakeTokenizer:
    """UTF-8 bytes as IDs 0..255; deliberately NOT a model tokenizer."""
    identity = {"backend": "fake-byte-v1", "sha256": sha(b"fake-byte-v1")}

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="strict")

    def count(self, text):
        return len(self.encode(text))

    def boundaries(self, ids, offsets):
        # Generated inputs are ASCII; reject a bad offset into a multibyte codepoint.
        return [len(self.decode(ids).encode()[:p].decode().encode()) for p in offsets]


class StrixTokenize:
    def __init__(self, binary, model, model_receipt=None):
        self.binary, self.model = str(Path(binary).resolve()), str(Path(model).resolve())
        self.identity = {"backend": "strix-tokenize-encode", "binary_sha256": file_sha(binary),
                         "model_receipt_sha256": file_sha(model_receipt) if model_receipt else None}
        if model_receipt is None:
            raise ValueError("a frozen model content receipt is required (no filename identity)")

    def _call(self, mode, text):
        with tempfile.TemporaryDirectory(prefix="longctx-tokenizer-") as tmp:
            path = Path(tmp) / "input.txt"
            path.write_bytes(text.encode("utf-8"))
            result = subprocess.run([self.binary, self.model, mode, str(path)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return result.stdout.decode("utf-8", errors="strict")

    def encode(self, text):
        raw = self._call("encode", text).strip()
        return [int(x) for x in raw.split(",")] if raw else []

    def plain(self, text):
        raw = self._call("plain", text).strip()
        return [int(x) for x in raw.split(",")] if raw else []

    def decode(self, ids):
        return self._call("decode", ",".join(map(str, ids)))

    def count(self, text):
        return len(self.encode(text))

    def boundaries(self, ids, offsets):
        """Cover character boundaries with token boundaries using prefix decoding.

        No per-fragment encode subprocesses; one encode per candidate prompt. Decodes
        are CPU-only. ASCII generation avoids partial UTF-8 ambiguity. Starts floor,
        ends ceil are selected by build.py from these floor boundaries.
        """
        cache = {0: 0, len(ids): len(self.decode(ids))}
        def length(n):
            if n not in cache:
                cache[n] = len(self.decode(ids[:n]))
            return cache[n]
        result = []
        for offset in offsets:
            lo, hi = 0, len(ids)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if length(mid) <= offset:
                    lo = mid
                else:
                    hi = mid - 1
            result.append(lo)
        return result


def validate_symbols(tokenizer, rendered):
    base = tokenizer.encode(rendered)
    result = {}
    for symbol in "ABCDEFGH":
        standalone = tokenizer.encode(symbol)
        combined = tokenizer.encode(rendered + symbol)
        if (len(standalone) != 1 or combined[:-1] != base or len(combined) != len(base) + 1
                or tokenizer.decode(combined[-1:]) != symbol):
            raise ValueError("answer set is not single-token in this tokenizer/template")
        result[symbol] = combined[-1]
    return result
