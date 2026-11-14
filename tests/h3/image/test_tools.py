"""Host-only negative gates for comparison receipts and edited-source byte checks."""
import array
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[3]
def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/"tools/h3/image"/(name+".py"))
    out=importlib.util.module_from_spec(spec);spec.loader.exec_module(out);return out

compare=module("compare")
source=module("check_source_bytes")

class ReceiptTests(unittest.TestCase):
    def test_png_metrics(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PNG receipt test needs optional Pillow")
        with tempfile.TemporaryDirectory() as directory:
            got,ref=Path(directory)/"got.png",Path(directory)/"ref.png"
            Image.new("RGB",(2,3),(128,64,32)).save(got)
            Image.new("RGB",(2,3),(128,64,32)).save(ref)
            self.assertEqual(compare.png_metrics(got,ref)["psnr"],"infinity")
            Image.new("RGB",(2,3),(129,64,32)).save(ref)
            self.assertGreater(compare.png_metrics(got,ref)["psnr"],35)
            Image.new("RGB",(3,2),(128,64,32)).save(ref)
            with self.assertRaises(ValueError): compare.png_metrics(got,ref)

    def test_metrics(self):
        a=array.array("f",[1,0.5,0.25])
        self.assertEqual(compare.metrics(a,a)["rel_l2"],0)
        self.assertEqual(compare.metrics(a,a)["psnr"],"infinity")
        self.assertTrue(compare.metrics(a,a)["bitwise"])
        for bad in [array.array("f"),array.array("f",[1]),array.array("f",[1,float("nan"),2]),array.array("f",[1,float("inf"),2])]:
            with self.assertRaises(ValueError): compare.metrics(a,bad)
        self.assertIsNone(compare.metrics(array.array("f",[1]),array.array("f",[0]))["rel_l2"])

    def test_byte_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"values.f32";p.write_bytes(b"bad")
            with self.assertRaises(ValueError): compare.read(p)

    def test_literal_checker(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"literal.cpp"
            for text in ['const char* x="one\\ntwo";','const char* x=R"tag(one\ntwo)tag";','// "\nconst char x=\'\\n\';']:
                p.write_text(text,encoding="utf-8");source.check(p)
            for text in ['const char* x="one\ntwo";','const char* x="one\rtwo";','const char* x="unfinished;']:
                p.write_text(text,encoding="utf-8")
                with self.assertRaises(ValueError): source.check(p)
            p.write_bytes(b"//\0")
            with self.assertRaises(ValueError): source.check(p)

if __name__=="__main__": unittest.main()
