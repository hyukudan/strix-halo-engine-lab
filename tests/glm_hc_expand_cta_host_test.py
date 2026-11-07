"""Host proof of row placement, ordered tree DAGs, tails and address coverage.

Run: python tests/glm_hc_expand_cta_host_test.py
This proves organization and source correspondence, not compiler/native FP behavior.
The HIP gate remains mandatory for NaN payloads, signed zeros, resources and speed.
"""

import ast
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src/glm/prefill/glm_prefill.hip").read_text(encoding="utf-8")
CANDIDATE = (ROOT / "src/glm/prefill/glm_hc_expand_cta.inc").read_text(encoding="utf-8")
HIDDEN, STREAMS, LANES = 4096, 4, 256


def body(source, name):
    """Extract a balanced function body, retaining nested scopes."""
    signature = re.search(r"\b" + name + r"\([^(){};]*\)\s*\{", source)
    assert signature, name
    start = signature.end() - 1
    depth = 1
    end = start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start + 1:end - 1]


def symbolic(expr, env):
    """Interpret actual reducer expressions as ordered addition DAGs."""
    def walk(node):
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Subscript):
            return walk(node.value)[walk(node.slice)]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            a, b = walk(node.left), walk(node.right)
            return a + b if isinstance(a, int) and isinstance(b, int) else ("add", a, b)
        raise AssertionError(ast.dump(node))
    return walk(ast.parse(expr.strip(), mode="eval").body)


def source_tree(helper, row):
    # Interpret the four LDS expressions and wave32 levels from the C++ source.
    declarations = re.search(r"const float (t0 = .*?);", helper, re.S).group(1)
    assignments = [part.strip().split(" = ", 1) for part in declarations.split(",")]
    w_expr = re.search(r"w = (\(t0.*?);", helper).group(1)
    assert "w = w + __shfl_down(w, h, 32)" in helper
    assert re.search(r"unsigned h = 16; h > 0; h >>= 1U", helper)
    wave = []
    for lane in range(32):
        env = {"l": lane, "s": [("leaf", row, i) for i in range(256)]}
        for name, expr in assignments:
            env[name] = symbolic(expr, env)
        wave.append(symbolic(w_expr, env))
    for h in (16, 8, 4, 2, 1):
        # Only these lanes contribute to lane zero; avoid unspecified shuffle sources.
        wave = [("add", wave[l], wave[l + h]) for l in range(h)]
    return wave[0]


def full_tree(row):
    lanes = [("leaf", row, i) for i in range(256)]
    for h in (128, 64, 32, 16, 8, 4, 2, 1):
        lanes = [("add", lanes[l], lanes[l + h]) for l in range(h)]
    return lanes[0]


def require_placement_contract(source):
    kernel = body(source, "hc_expand_cta_kernel")
    assert "group = threadIdx.x / 256, l = threadIdx.x % 256" in kernel
    assert "r = blockIdx.x * kRows + group" in kernel
    assert "__shared__ float s[kRows][256]" in kernel
    assert "return" not in kernel
    assert "__syncthreads" not in kernel  # the only barrier is in the uniform helper
    active = kernel.index("if (kRows == 1 || r < rows) {")
    call = "hc_expand_row_tree256_lane0(s[group], a, l)"
    assert call in kernel
    tree = kernel.index(call)
    # Count scopes up to the tree: it must be outside every active-row scope.
    prefix = re.sub(r"//[^\n]*", "", kernel[:tree])
    assert prefix.count("{") == prefix.count("}")
    assert active < kernel.index("const float* m =") < tree
    assert "if ((kRows == 1 || r < rows) && l == 0) rr_out[r]" in kernel
    assert "if (flag && bad) atomicOr(flag, 1U)" in kernel
    assert "yv[b] = kPair ? yr[b * 256 + l] + y1r[b * 256 + l]" in kernel
    assert "const float yd = kHoist != 0 ? yv[b] : kPair ? y[yi] + y1[yi] : y[yi]" in kernel
    return kernel


class ExpandCtaTests(unittest.TestCase):
    def test_tree_source_matches_every_ordered_reference_node(self):
        old = body(SOURCE, "lds_tree256_lane0")
        new = body(CANDIDATE, "hc_expand_row_tree256_lane0")
        for row in (0, 1, 4095, 8191):
            self.assertEqual(source_tree(new, row), full_tree(row))
            self.assertEqual(source_tree(new, row), source_tree(old, row))
        # A reversed commutative add is a failure: NaN operand order matters.
        mutated = new.replace("s[l] + s[l + 128]", "s[l + 128] + s[l]")
        self.assertNotEqual(source_tree(mutated, 0), full_tree(0))

    def test_scope_barrier_and_subgroup_mutations(self):
        require_placement_contract(CANDIDATE)
        for before, after in (
            ("s[group], a, l", "s[0], a, l"),
            ("l = threadIdx.x % 256", "l = threadIdx.x"),
            ("float a = 0.0F;", "float a = 0.0F; if (r >= rows) return;"),
            ("if (kRows == 1 || r < rows) {", "if (kRows == 1 || r < rows) { __syncthreads();"),
            ("y[yi] + y1[yi]", "y1[yi] + y[yi]"),
        ):
            with self.subTest(mutation=after), self.assertRaises(AssertionError):
                require_placement_contract(CANDIDATE.replace(before, after))

    def test_rows_waves_tails_and_barrier_participants(self):
        for count in (0, 1, 2, 3, 7, 65, 4096, 8192):
            for rows_per_cta in (1, 2):
                seen = [0] * count
                for block in range((count + rows_per_cta - 1) // rows_per_cta):
                    participants = []
                    for group in range(rows_per_cta):
                        row = block * rows_per_cta + group
                        # All 256 lanes reach the helper, even if the row is inactive.
                        participants.extend(range(group * 256, (group + 1) * 256))
                        for wave in range(8):
                            self.assertEqual((group * 256 + wave * 32) // 256, group)
                        if row < count:
                            seen[row] += 1
                    self.assertEqual(participants, list(range(rows_per_cta * 256)))
                self.assertEqual(seen, [1] * count)

    def test_output_norm_order_and_guarded_addresses(self):
        old = body(SOURCE, "hc_expand_norm_rows_kernel")
        new = require_placement_contract(CANDIDATE)
        for variable in ("mx", "o"):
            pattern = r"const float " + variable + r" = (.*?);"
            norm = lambda expr: re.sub(r"\s+", "", expr)
            self.assertEqual(norm(re.search(pattern, old, re.S).group(1)), norm(re.search(pattern, new, re.S).group(1)))
        self.assertIn("if (i == 0) a = __builtin_fmaf(o, o, a);", new)
        self.assertRegex(new, r"for \(unsigned i = 0; i < kHc - 1; \+\+i\).*?\n#pragma unroll\n\s*for \(unsigned b = 0; b < B; \+\+b\) a = __builtin_fmaf\(v\[i\]\[b\], v\[i\]\[b\], a\);")
        for lane in range(256):
            chain = [stream * HIDDEN + b * 256 + lane for stream in range(4) for b in range(HIDDEN // 256)]
            self.assertEqual(chain, list(range(lane, STREAMS * HIDDEN, 256)))
        # Exhaustive small odd-tail output placement, preserving arbitrary payload bits.
        payloads = (0, 0x80000000, 0x7FC12345, 0xFFA00001, 0x00000001)
        for count in (1, 2, 3, 7):
            guard = 17
            storage = [None] * (count * STREAMS * HIDDEN + 2 * guard)
            for block in range((count + 1) // 2):
                for tid in range(512):
                    row, lane = block * 2 + tid // 256, tid % 256
                    if row >= count:
                        continue
                    for b in range(HIDDEN // 256):
                        d = b * 256 + lane
                        yi, mi = row * HIDDEN + d, row * 24 + 8
                        self.assertTrue(0 <= yi < count * HIDDEN)
                        self.assertTrue(0 <= mi < count * 24)
                        for stream in range(4):
                            offset = row * STREAMS * HIDDEN + stream * HIDDEN + d
                            self.assertIsNone(storage[guard + offset])
                            storage[guard + offset] = payloads[offset % len(payloads)]
            self.assertEqual(storage[:guard] + storage[-guard:], [None] * (2 * guard))
            self.assertEqual(storage[guard:-guard], [payloads[i % len(payloads)] for i in range(count * STREAMS * HIDDEN)])
        # Row offsets must be size_t: this address exceeds 32 bits.
        self.assertGreater(300000 * STREAMS * HIDDEN, 2**32)
        self.assertIn("static_cast<std::size_t>(r) * n", new)

    def test_default_kernel_and_strict_selector_remain_separate(self):
        selector = body(SOURCE, "hc_expand_cta_rows")
        self.assertIn('e == nullptr || std::string(e) == "1"', selector)
        self.assertIn('std::string(e) == "2"', selector)
        self.assertIn("throw std::invalid_argument", selector)
        wrapper = body(SOURCE, "hc_expand_norm_rows")
        self.assertIn("if (hc_expand_cta_rows() == 2)", wrapper)
        self.assertIn("hc_expand_norm_rows_h(hc_expand_hoist(), X, mix, y, Xo, rr, flag, rows, s);", wrapper)
        self.assertIn("if (hc_expand_cta_rows() == 2)", body(SOURCE, "hc_sum_expand_norm_rows"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
