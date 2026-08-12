"""GrayThermalTransform, gri/termal dayanikliliginin TEK garantisi.

Neden ayri bir katman gerekiyor: Albumentations'taki ToGray load_image()
asamasinda, YOLOX'un augment_hsv'sinden ONCE calisiyor; augment_hsv doygunlugu
carpmiyor TOPLUYOR, yani S=0 olan gri bir goruntuyu yeniden renklendirebiliyor
(olculdu 2026-08-12: 200 denemenin 26'si, en buyuk kanal farki 29/255).
Bu sinif zincirin SONUNDA durur, o yuzden onu hicbir sey geri alamaz.

yolox yerelde kurulu olmadigi icin exp modulu sahte (stub) yolox modulleriyle
yuklenir -- test edilen kod gercek proje kodudur, taklit degil.
"""

import sys
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_exp_module():
    """training/exps/yolox_tiny_ship.py'yi sahte yolox ile yukler."""
    for name in ("yolox", "yolox.data", "yolox.exp", "yolox.models",
                 "yolox.utils", "yolox.evaluators"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["yolox.data"].COCODataset = type("COCODataset", (), {})
    sys.modules["yolox.exp"].Exp = type("Exp", (), {"__init__": lambda self: None})

    import importlib.util
    path = ROOT / "training" / "exps" / "yolox_tiny_ship.py"
    spec = importlib.util.spec_from_file_location("ship_exp_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP = load_exp_module()


class Passthrough:
    """Sarmalanan gercek preproc'un yerine gecer: YOLOX sozlesmesi
    (CHW float32 BGR goruntu, hedef dizisi) aynen korunur."""

    def __call__(self, image, target, input_dim):
        return image, target


def color_image():
    """Kanallari BELIRGIN farkli bir goruntu: gri'ye cevrildigi olculebilsin."""
    image = np.zeros((3, 8, 8), dtype=np.float32)
    image[0] = 200.0   # B
    image[1] = 100.0   # G
    image[2] = 30.0    # R
    return image


class GrayThermalTransformTests(unittest.TestCase):
    def test_gray_prob_zero_leaves_image_untouched(self):
        t = EXP.GrayThermalTransform(Passthrough(), gray_prob=0.0)
        out, _ = t(color_image(), None, (8, 8))
        np.testing.assert_array_equal(out, color_image())

    def test_gray_prob_one_equalises_channels(self):
        t = EXP.GrayThermalTransform(Passthrough(), gray_prob=1.0)
        out, _ = t(color_image(), None, (8, 8))
        spread = out.max(axis=0) - out.min(axis=0)
        self.assertEqual(spread.max(), 0.0, "kanallar esitlenmeli (gri)")

    def test_gray_uses_luminance_not_plain_mean(self):
        """Duz ortalama (110) ile BT.601 parlakligi (91.6) farkli sonuc verir;
        gercek bir gri-ton kamera parlaklik agirligi kullanir."""
        t = EXP.GrayThermalTransform(Passthrough(), gray_prob=1.0)
        out, _ = t(color_image(), None, (8, 8))
        expected = 0.114 * 200 + 0.587 * 100 + 0.299 * 30
        self.assertAlmostEqual(float(out[0, 0, 0]), expected, places=3)

    def test_invert_flips_polarity(self):
        """Termal white-hot <-> black-hot: kamera ayari, veri degil."""
        t = EXP.GrayThermalTransform(Passthrough(), gray_prob=1.0, invert_prob=1.0)
        out, _ = t(color_image(), None, (8, 8))
        expected = 255.0 - (0.114 * 200 + 0.587 * 100 + 0.299 * 30)
        self.assertAlmostEqual(float(out[0, 0, 0]), expected, places=3)

    def test_invert_never_happens_without_gray(self):
        """Renkli bir goruntuyu terslemek renk negatifi uretirdi -- fiziksel
        karsiligi yok. Tersleme yalnizca gri'ye cevrilmis ornekte olmali."""
        t = EXP.GrayThermalTransform(Passthrough(), gray_prob=0.0, invert_prob=1.0)
        out, _ = t(color_image(), None, (8, 8))
        np.testing.assert_array_equal(out, color_image())

    def test_output_contract_is_preserved(self):
        t = EXP.GrayThermalTransform(Passthrough(), gray_prob=1.0, invert_prob=1.0)
        out, target = t(color_image(), "hedef", (8, 8))
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.shape, (3, 8, 8))
        self.assertTrue(out.flags["C_CONTIGUOUS"], "YOLOX bitisik dizi bekler")
        self.assertEqual(target, "hedef", "hedefler degistirilmemeli")

    def test_values_stay_in_range(self):
        t = EXP.GrayThermalTransform(Passthrough(), gray_prob=1.0, invert_prob=1.0)
        for value in (0.0, 114.0, 255.0):
            image = np.full((3, 4, 4), value, dtype=np.float32)
            out, _ = t(image, None, (4, 4))
            self.assertGreaterEqual(out.min(), 0.0)
            self.assertLessEqual(out.max(), 255.0)

    def test_probabilities_are_roughly_respected(self):
        """0.4 gri / 0.25 tersleme bir NIYET; kod onu gercekten uygulamali."""
        import random
        random.seed(0)
        t = EXP.GrayThermalTransform(Passthrough(), gray_prob=0.4, invert_prob=0.25)
        gray = inverted = 0
        for _ in range(2000):
            out, _ = t(color_image(), None, (8, 8))
            if (out.max(axis=0) - out.min(axis=0)).max() == 0.0:
                gray += 1
                if out[0, 0, 0] > 127:   # parlaklik 91.6 -> terslenince 163.4
                    inverted += 1
        self.assertAlmostEqual(gray / 2000, 0.40, delta=0.05)
        self.assertAlmostEqual(inverted / max(gray, 1), 0.25, delta=0.06)


class ExpConstantsTests(unittest.TestCase):
    def test_single_class(self):
        self.assertEqual(EXP.TARGET_CLASSES, ("ship",))

    def test_gray_probability_is_meaningful(self):
        """0 olursa katman sessizce olur; 1 olursa model renkli goruntu
        gormez. Ikisi de istenmeyen uc."""
        self.assertGreater(EXP.TRAIN_GRAY_PROB, 0.0)
        self.assertLess(EXP.TRAIN_GRAY_PROB, 1.0)
        self.assertGreater(EXP.TRAIN_INVERT_PROB, 0.0)
        self.assertLess(EXP.TRAIN_INVERT_PROB, 1.0)


if __name__ == "__main__":
    unittest.main()
