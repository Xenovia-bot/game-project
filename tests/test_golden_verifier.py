import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.verify_kv260_golden import decode_reference, verify_input_quantization


def write_level(root, index, h, w, nc, stride, scale, fill=-128):
    """Tek bir stride seviyesi icin UC tensorlu bos dokum yazar.

    Kart, seviye basina reg/obj/cls'yi ayri tensorler olarak doker; her
    birinin kendi fix_point'i (dolayisiyla kendi olcegi) vardir.
    """
    meta = {
        f"level{index}_h": str(h),
        f"level{index}_w": str(w),
        f"level{index}_stride": str(stride),
        f"level{index}_nc": str(nc),
    }
    for role, channels in (("reg", 4), ("obj", 1), ("cls", nc)):
        name = f"output_stride{stride}_{role}.bin"
        np.full((h, w, channels), fill, dtype=np.int8).tofile(root / name)
        meta[f"level{index}_{role}_file"] = name
        meta[f"level{index}_{role}_scale"] = str(scale)
    return meta


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

    def test_class_count_is_read_from_dump(self):
        """Sinif sayisi dokumden okunur, koda sabitlenmez.

        Eskiden 15 (5 + 10 sinif) sabiti yaziliydi ve 2 sinifli dogru bir
        dokumu reddediyordu -- zorunlu kabul testi hicbir zaman gecemezdi.
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
                                            num_classes, stride, 0.25))
                self.assertEqual(decode_reference(root, meta), [])

    def test_class_count_must_agree_across_levels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = {"num_levels": "3", "conf": "0.15", "nms": "0.45",
                    "ratio": "1.0", "frame_w": "64", "frame_h": "64"}
            for i, (stride, nc) in enumerate(((8, 2), (16, 2), (32, 3))):
                meta.update(write_level(root, i, 64 // stride, 64 // stride,
                                        nc, stride, 0.25))
            with self.assertRaises(ValueError):
                decode_reference(root, meta)

    def test_each_role_uses_its_own_scale(self):
        """reg/obj/cls ayri fix_point tasir; dogrulayici her birini ayri okumali.

        Ayni ham int8 degeri, olcegi farkli oldugu icin farkli float'a cozulur.
        Dogrulayici tek bir olcek kullansaydi bu test kutu koordinatlarinda
        sapma gosterirdi.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            h = w = 1
            meta = {"num_levels": "1", "conf": "0.0", "nms": "0.45",
                    "ratio": "1.0", "frame_w": "512", "frame_h": "512",
                    "level0_h": str(h), "level0_w": str(w),
                    "level0_stride": "8", "level0_nc": "2"}
            # reg: cx=cy=0 ofset, log(w)=log(h)=8*0.5=4 -> exp(4)*8
            np.asarray([[[0, 0, 8, 8]]], dtype=np.int8).tofile(root / "r.bin")
            np.asarray([[[0]]], dtype=np.int8).tofile(root / "o.bin")
            np.asarray([[[0, 0]]], dtype=np.int8).tofile(root / "c.bin")
            meta.update({
                "level0_reg_file": "r.bin", "level0_reg_scale": "0.5",
                "level0_obj_file": "o.bin", "level0_obj_scale": "0.25",
                "level0_cls_file": "c.bin", "level0_cls_scale": "0.125",
            })
            dets = decode_reference(root, meta)
            self.assertEqual(len(dets), 1)
            # obj=cls=0 -> sigmoid(0)*sigmoid(0) = 0.25
            self.assertAlmostEqual(dets[0]["score"], 0.25, places=6)
            # Kutu (0,0) hucresinde ve merkezi 0'da; sol yarisi kareden tasip
            # kirpilir. Sag kenar reg olceginin dogru uygulandigini gosterir:
            #   yarim genislik = exp(8 * reg_scale) * stride / 2
            #                  = exp(8 * 0.5) * 8 / 2 = exp(4) * 4
            # obj (0.25) veya cls (0.125) olcegi kullanilsaydi bu deger
            # sirasiyla exp(2)*4 veya exp(1)*4 cikardi.
            self.assertAlmostEqual(dets[0]["x1"], 0.0, places=6)
            self.assertAlmostEqual(dets[0]["x2"], float(np.exp(4.0)) * 4,
                                   places=2)


if __name__ == "__main__":
    unittest.main()
