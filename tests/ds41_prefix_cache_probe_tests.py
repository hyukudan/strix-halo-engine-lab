import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("probe", Path(__file__).resolve().parents[1] / "tools/ds41_prefix_cache_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class Response:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return iter(self.lines)

    def __exit__(self, *args):
        pass


class ProbeTests(unittest.TestCase):
    def test_metadata_does_not_start_ttft(self):
        self.assertFalse(probe.content_delta({"choices": [{"delta": {"role": "assistant"}}]}))
        self.assertFalse(probe.content_delta({"strix": {"prefill_ms": 5}}))
        self.assertFalse(probe.content_delta({"choices": [{"delta": {"tool_calls": [{"id": "x"}]}}]}))
        self.assertTrue(probe.content_delta({"choices": [{"delta": {"content": "hello"}}]}))
        self.assertTrue(probe.content_delta({"choices": [{"delta": {"reasoning_content": "thinking"}}]}))

    def test_stream_uses_content_time_and_retains_ids(self):
        events = [{"choices": [{"delta": {"role": "assistant"}}]},
                  {"choices": [{"delta": {"content": "hello"}}]},
                  {"strix": {"token_ids": [7, 8, 9]}}]
        lines = [b"data: " + json.dumps(event).encode() + b"\n" for event in events] + [b"data: [DONE]\n"]
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(probe.urllib.request, "urlopen", return_value=Response(lines)), \
                patch.object(probe.time, "perf_counter", side_effect=[10, 10.1, 10.5, 10.7, 10.8, 10.9]):
            output = Path(directory) / "stream.sse"
            result = probe.probe("http://example.invalid", b"{}", output, 1)
            self.assertAlmostEqual(result["ttft_seconds"], 0.5)
            self.assertEqual(result["token_ids"], [7, 8, 9])
            self.assertNotIn("error", result)
            self.assertEqual(output.read_bytes(), b"".join(lines))

    def test_incomplete_stream_is_retained_as_failure(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(probe.urllib.request, "urlopen", return_value=Response([b": keepalive\n"])):
            result = probe.probe("http://example.invalid", b"{}", Path(directory) / "stream.sse", 1)
            self.assertIn("error", result)
            self.assertIsNone(result["ttft_seconds"])


if __name__ == "__main__":
    unittest.main()
