#!/usr/bin/env python3
"""CPU-only regression checks for CUDA instruction comparison boundaries."""
import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('sass_compare', Path(__file__).with_name('sass_compare.py'))
sass_compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sass_compare)


def image(symbol=None, instruction='IADD3 R0, R0, 0x1, RZ; /* 0x001 */'):
    header = ('Fatbin ptx code:\narch = sm_120\ncode version = [9,0]\ncompressed\n'
              'ptxasOptions = --fmad false\nFatbin elf code:\narch = sm_120\ncode version = [1,8]\n')
    return header + (f'Function : {symbol}\n/*0000*/ {instruction}\n' if symbol else '')


class SassCompareTests(unittest.TestCase):
    def parse(self, output):
        with patch.object(sass_compare.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=output)):
            return sass_compare.sass('unused', 'unused')

    def test_image_order_and_empty_images_are_not_instructions(self):
        left = image('_Z4leftv')
        right = image('_Z5rightv', 'EXIT; /* 0x002 */')
        self.assertEqual(self.parse(left + right), self.parse(image() + right + image() + left + image()))

    def test_namespace_move_is_normalised(self):
        # A reviewed move is declared explicitly (--symbol-rename OLD=NEW); an undeclared move still differs.
        moved = {'_ZN5model4tileEv': '_ZN4core4tileEv'}
        self.assertEqual(sass_compare.parse_sass(image('_ZN5model4tileEv'), moved),
                         sass_compare.parse_sass(image('_ZN4core4tileEv'), moved))
        self.assertNotEqual(self.parse(image('_ZN5model4tileEv')), self.parse(image('_ZN4core4tileEv')))

    def test_function_count_is_preserved(self):
        self.assertNotEqual(self.parse(image('_Z4leftv')), self.parse(image('_Z4leftv') + image('_Z5rightv')))

    def test_instruction_change_fails_cli_gate(self):
        outputs = [image('_Z4tilev'), image('_Z4tilev', 'IADD3 R0, R0, 0x2, RZ; /* 0x002 */')]
        results = [SimpleNamespace(returncode=0, stdout=s) for s in outputs]
        with patch.object(sass_compare.sys, 'argv', ['sass_compare.py', '--base', 'base', '--new', 'new']), \
             patch.object(sass_compare, 'collect', side_effect=lambda root, _: {'tile.cu.obj': [root]}), \
             patch.object(sass_compare.subprocess, 'run', side_effect=results), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(sass_compare.main(), 1)


if __name__ == '__main__':
    unittest.main()
