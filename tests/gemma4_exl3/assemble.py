"""Small synthetic streaming assembly tests; no model or optional packages."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("assemble", Path(__file__).parents[2] / "tools/gemma4/exl3_assemble.py")
assemble = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assemble)


class Assembly(unittest.TestCase):
    def make(self, root):
        recipe = dict(codebook="mul1", backend="exl3", scaled_hessian=True, fake_hess=False,
                      crop=None, seed=787, sigma_reg=0.025, hess="private", model="private", model_index_sha256="0" * 64)
        for k in (4, 5, 6, 8):
            (root / f"verify-K{k}.json").write_text(json.dumps(dict(K=k, errors=[])))
            for part in ("a", "b"):
                directory = root / f"K{k}-{part}"
                directory.mkdir()
                index = dict(recipe=recipe, tensors={}, weight_map={})
                if part == "a":
                    for stem, kind in (("body", "ffn"), ("lm_head", "head")):
                        payload, header = bytearray(), {}
                        for suffix, dtype, shape, data in (
                            ("suh", "F16", [128], struct.pack("<128H", *([0x3c00] * 128))),
                            ("svh", "F16", [128], struct.pack("<128H", *([0x3c00] * 128))),
                            ("trellis", "I16", [8, 8, 16 * k], bytes(128 * 128 * k // 8)),
                            ("mul1", "I32", [], struct.pack("<I", 0x83dcd12d))):
                            key = stem + "." + suffix
                            header[key] = dict(dtype=dtype, shape=shape, data_offsets=[len(payload), len(payload) + len(data)])
                            index["weight_map"][key] = stem + ".safetensors"
                            payload.extend(data)
                        raw = json.dumps(header).encode()
                        (directory / (stem + ".safetensors")).write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
                        index["tensors"][stem] = dict(hf=stem + ".weight", **{"class": kind}, K=k, shape_out_in=[128, 128], proxy_err=0.01)
                (directory / "index.json").write_text(json.dumps(index))

    def test_stream_layout_and_recipe_guards(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.make(root)
            card = root / "card.md"; card.write_text("Synthetic fixture origin notice\n")
            saved = assemble.plan
            assemble.plan = lambda: {"body.weight": ([128, 128], "ffn"), "lm_head.weight": ([128, 128], "head")}
            try:
                m = assemble.assemble(root, root / "arm", "M4", model_card=card)
                self.assertEqual(m["quantized_file_bytes"], 512 * 2 + 128 * 128 * (4 + 6) // 8)
                self.assertNotIn("private", json.dumps(m))
                self.assertEqual(m["tensors"][1]["offset"], m["tensors"][0]["bytes"])
                self.assertEqual((root / "arm/MODEL-CARD.md").read_bytes(), card.read_bytes())
                self.assertEqual(m["model_license"], "OpenMDW-1.1")
                with self.assertRaises(ValueError): assemble.assemble(root, root / "arm", "M4", model_card=card)
                p = root / "K5-a/index.json"
                index = json.loads(p.read_text()); index["recipe"]["scaled_hessian"] = False
                p.write_text(json.dumps(index))
                with self.assertRaises(ValueError): assemble.sources(root)
            finally:
                assemble.plan = saved

    def test_unsafe_and_overlapping_ranges(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ("../x", "x\\y", "C:x", "/x"):
                with self.assertRaises(ValueError): assemble.safe_file(root, name)
            h = {"a": dict(data_offsets=[0, 3]), "b": dict(data_offsets=[2, 4])}
            raw = json.dumps(h).encode(); p = root / "bad.safetensors"
            p.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(4))
            with self.assertRaises(ValueError): assemble.header(p)


if __name__ == "__main__":
    unittest.main()
