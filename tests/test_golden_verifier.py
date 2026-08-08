import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.verify_kv260_golden import verify_input_quantization


class GoldenVerifierTests(unittest.TestCase):
    def test_half_up_input_quantization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canvas = np.asarray([0, 1, 2, 3, 254, 255], dtype=np.uint8)
            expected = np.asarray([0, 1, 1, 2, 127, 127], dtype=np.int8)
            canvas.tofile(root / "canvas.bin")
            expected.tofile(root / "input.bin")
            meta = {
                "input_h": "1",
                "input_w": "2",
                "input_scale": "0.5",
                "canvas_file": "canvas.bin",
                "input_file": "input.bin",
            }
            verify_input_quantization(root, meta)

            expected[2] = 2
            expected.tofile(root / "input.bin")
            with self.assertRaises(AssertionError):
                verify_input_quantization(root, meta)


if __name__ == "__main__":
    unittest.main()
