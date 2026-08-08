"""shorten_zip_names.py icin testler.

En kritik davranis: bir goruntu yeniden adlandirildiginda **etiketi de** ayni
yeni adi almali. Aksi halde images/ ve labels/ eslesmesi kirilir ve o ornek
sessizce etiketsiz kalir -- egitim calisir ama veri bozulmustur.
"""

import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.shorten_zip_names import (
    apply_renames,
    plan_renames,
    repack,
    split_entry,
)


class SplitEntryTests(unittest.TestCase):
    def test_splits_prefix_stem_extension(self):
        self.assertEqual(split_entry("a/b/c.jpg"), ("a/b/", "c", ".jpg"))
        self.assertEqual(split_entry("c.txt"), ("", "c", ".txt"))
        self.assertEqual(split_entry("a/b/noext"), ("a/b/", "noext", ""))


class PlanRenamesTests(unittest.TestCase):
    def test_short_names_are_untouched(self):
        self.assertEqual(plan_renames(["d/images/a.jpg", "d/labels/a.txt"]), {})

    def test_long_name_gets_shortened_deterministically(self):
        long_stem = "x" * 300
        names = [f"d/images/{long_stem}.jpg"]
        first = plan_renames(names, max_bytes=200)
        second = plan_renames(names, max_bytes=200)
        self.assertEqual(first, second, "kisaltma deterministik olmali")
        self.assertIn(long_stem, first)
        self.assertLess(len(first[long_stem]), len(long_stem))

    def test_result_fits_the_limit(self):
        long_stem = "y" * 400
        names = [f"deep/nested/prefix/images/{long_stem}.jpeg"]
        renames = plan_renames(names, max_bytes=200)
        new_name = apply_renames(names[0], renames)
        self.assertLessEqual(len(new_name.encode("utf-8")), 200)

    def test_image_and_label_get_the_same_new_stem(self):
        long_stem = "z" * 300
        names = [f"d/images/{long_stem}.jpg", f"d/labels/{long_stem}.txt"]
        renames = plan_renames(names, max_bytes=200)
        new_image = apply_renames(names[0], renames)
        new_label = apply_renames(names[1], renames)
        self.assertEqual(split_entry(new_image)[1], split_entry(new_label)[1],
                         "goruntu ve etiket ayni taban adi almali")

    def test_distinct_long_stems_do_not_collide(self):
        a, b = "a" * 300, "b" * 300
        renames = plan_renames([f"d/images/{a}.jpg", f"d/images/{b}.jpg"],
                               max_bytes=200)
        self.assertNotEqual(renames[a], renames[b])


class RepackTests(unittest.TestCase):
    def test_repack_preserves_content_and_pairing(self):
        long_stem = "q" * 300
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.zip"
            dst = Path(tmp) / "out.zip"
            with zipfile.ZipFile(src, "w") as z:
                z.writestr("d/images/short.jpg", b"A")
                z.writestr("d/labels/short.txt", b"0 0.5 0.5 0.1 0.1\n")
                z.writestr(f"d/images/{long_stem}.jpg", b"B")
                z.writestr(f"d/labels/{long_stem}.txt", b"1 0.2 0.2 0.1 0.1\n")
            repack(src, dst, max_bytes=200)

            with zipfile.ZipFile(dst) as z:
                names = z.namelist()
                self.assertEqual(len(names), 4, "hicbir giris kaybolmamali")
                for name in names:
                    self.assertLessEqual(len(name.encode("utf-8")), 200)
                # kisa olanlar aynen kalmali
                self.assertIn("d/images/short.jpg", names)
                self.assertEqual(z.read("d/images/short.jpg"), b"A")
                # uzun olanin goruntu/etiket cifti hala eslesmeli
                images = [n for n in names if n.startswith("d/images/")]
                labels = [n for n in names if n.startswith("d/labels/")]
                image_stems = {split_entry(n)[1] for n in images}
                label_stems = {split_entry(n)[1] for n in labels}
                self.assertEqual(image_stems, label_stems)
                # icerik korunmus mu
                long_image = [n for n in images if n != "d/images/short.jpg"][0]
                self.assertEqual(z.read(long_image), b"B")

    def test_repack_is_noop_when_nothing_is_long(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.zip"
            dst = Path(tmp) / "out.zip"
            with zipfile.ZipFile(src, "w") as z:
                z.writestr("d/images/a.jpg", b"A")
            self.assertEqual(repack(src, dst, max_bytes=200), 0)
            self.assertFalse(dst.exists(), "gereksiz yere dosya uretilmemeli")


if __name__ == "__main__":
    unittest.main()
