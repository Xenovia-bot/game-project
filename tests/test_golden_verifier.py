import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.verify_kv260_golden import decode_reference, verify_input_quantization


def write_level(root, index, h, w, c, stride, scale, fill=-128):
    """Tek bir stride seviyesi icin bos (hicbir tespit uretmeyen) dokum yazar."""
    name = f"output_stride{stride}.bin"
    np.full((h, w, c), fill, dtype=np.int8).tofile(root / name)
    return {
        f"level{index}_file": name,
        f"level{index}_h": str(h),
        f"level{index}_w": str(w),
        f"level{index}_c": str(c),
        f"level{index}_stride": str(stride),
        f"level{index}_scale": str(scale),
    }


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

    def test_channel_count_follows_class_scheme(self):
        """2 sinifli model 7 kanal uretir; kanal sayisi sabitlenmemeli.

        Eskiden 15 (5 + 10 sinif) sabiti yaziliydi ve dogru bir dokumu
        reddediyordu -- yani zorunlu kabul testi hicbir zaman gecemezdi.
        """
        base = {"num_levels": "3", "conf": "0.15", "nms": "0.45",
                "ratio": "1.0", "frame_w": "64", "frame_h": "64"}
        for num_classes in (2, 10):
            with self.subTest(num_classes=num_classes), \
                    tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                meta = dict(base)
                for i, stride in enumerate((8, 16, 32)):
                    meta.update(write_level(root, i, 64 // stride, 64 // stride,
                                            5 + num_classes, stride, 0.25))
                self.assertEqual(decode_reference(root, meta), [])

    def test_channel_count_must_agree_across_levels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = {"num_levels": "3", "conf": "0.15", "nms": "0.45",
                    "ratio": "1.0", "frame_w": "64", "frame_h": "64"}
            for i, (stride, channels) in enumerate(((8, 7), (16, 7), (32, 8))):
                meta.update(write_level(root, i, 64 // stride, 64 // stride,
                                        channels, stride, 0.25))
            with self.assertRaises(ValueError):
                decode_reference(root, meta)


if __name__ == "__main__":
    unittest.main()
