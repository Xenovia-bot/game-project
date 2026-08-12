"""Gemi not defterinin %%writefile hucreleri kaynak dosyalarla ayni kalmali,
ve kesif hucresi kaynaklari klasor adina degil kategori imzasina gore bulmali.

tests/test_notebook_embeds.py (aerial) ile ayni gerekce, farkli hedef.
"""

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image
import io

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "training" / "kaggle_ship_yolox.ipynb"

#: tools/_sync_ship_notebook_embeds.py ile ayni tablo olmali.
EMBEDS = {
    3: ROOT / "tools" / "dataset_common.py",
    4: ROOT / "tools" / "build_ship_dataset.py",
    5: ROOT / "training" / "exps" / "yolox_tiny_ship.py",
    6: ROOT / "training" / "ship_metrics.py",
}


def notebook_writefile_body(index):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][index]["source"])
    first, _, body = source.partition("\n")
    if not first.startswith("%%writefile "):
        raise AssertionError("Beklenen writefile hucresi degil: %d" % index)
    return body


def strip_notebook_note(body, path):
    rel = path.relative_to(ROOT).as_posix()
    note = "\n\nBu dosya, projedeki %s ile ayni icerige sahiptir.\n" % rel
    return body.replace(note, "\n")


class ShipNotebookEmbedTests(unittest.TestCase):
    def test_common_helper_uses_its_importable_module_name(self):
        """Kaggle'da build_ship_dataset.py `import dataset_common` yapar.

        Eski aerial dosya adi olan build_dataset.py ile yazmak, notebook'un
        veri birlestirme hucresini daha egitim baslamadan durdururdu.
        """
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        first = "".join(notebook["cells"][3]["source"]).split("\n", 1)[0]
        self.assertEqual(first, "%%writefile dataset_common.py")

    def test_sync_table_matches_the_tool(self):
        """Iki tablo ayrisirsa senkron sessizce yanlis hucreyi yazar."""
        source = (ROOT / "tools" / "_sync_ship_notebook_embeds.py").read_text(
            encoding="utf-8")
        for index, path in EMBEDS.items():
            self.assertIn(f"{index}: ROOT / ", source)
            self.assertIn(f'"{path.name}"', source)

    def test_writefile_cells_match_project_sources(self):
        for index, path in EMBEDS.items():
            body = notebook_writefile_body(index)
            body = strip_notebook_note(body, path)
            self.assertEqual(
                body.replace("\r\n", "\n"),
                path.read_text(encoding="utf-8").replace("\r\n", "\n"),
                msg="Notebook hucresi %d, %s ile ayristi" % (index, path),
            )

    def test_class_maps_match_discovery_signatures(self):
        """build_ship_dataset.py CLASS_MAPS ile notebook'un
        CATEGORY_SIGNATURES'i ayrisirsa kesif yanlis kaynagi eslestirir."""
        import sys
        sys.path.insert(0, str(ROOT / "tools"))
        import build_ship_dataset as bsd

        source = self._discover_cell_source()
        namespace = {"Path": Path}
        exec(compile(source.split("SOURCES = discover_ship_sources()")[0],
                     "cell6", "exec"), namespace)
        sigs = namespace["CATEGORY_SIGNATURES"]
        for key, class_map in bsd.CLASS_MAPS.items():
            self.assertIn(key, sigs, f"{key} icin imza eksik")
            self.assertEqual(sigs[key], frozenset(class_map),
                             f"{key}: notebook imzasi CLASS_MAPS'ten farkli")

    def _discover_cell_source(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            if "SOURCES = discover_ship_sources()" in source:
                return source
        raise AssertionError("Kesif hucresi bulunamadi")

    def _discover_from_notebook(self, input_root):
        import re
        source = self._discover_cell_source()
        source = re.sub(r"^!.*$", "# shell", source, flags=re.M)
        body = source.split("SOURCES = discover_ship_sources()")[0]
        body = body.replace('INPUT = Path("/kaggle/input")',
                            f'INPUT = Path(r"{input_root}")')
        namespace = {"Path": Path}
        exec(compile(body, "ship_discover_cell", "exec"), namespace)
        return namespace["discover_ship_sources"]()


def _coco_zip(path, categories, n_images=1):
    buf = io.BytesIO()
    Image.new("RGB", (64, 48)).save(buf, "JPEG")
    jpg = buf.getvalue()
    cats = [{"id": i, "name": n} for i, n in enumerate(categories)]
    with zipfile.ZipFile(path, "w") as z:
        for split in ("train",):
            for i in range(n_images):
                z.writestr(f"{split}/img{i}.jpg", jpg)
            z.writestr(f"{split}/_annotations.coco.json", json.dumps({
                "images": [{"id": i, "file_name": f"img{i}.jpg",
                           "width": 64, "height": 48} for i in range(n_images)],
                "annotations": [], "categories": cats,
            }))


def _extract_coco(root, categories):
    buf = io.BytesIO()
    Image.new("RGB", (64, 48)).save(buf, "JPEG")
    jpg = buf.getvalue()
    cats = [{"id": i, "name": n} for i, n in enumerate(categories)]
    d = root / "train"
    d.mkdir(parents=True)
    (d / "img0.jpg").write_bytes(jpg)
    (d / "_annotations.coco.json").write_text(json.dumps({
        "images": [{"id": 0, "file_name": "img0.jpg", "width": 64, "height": 48}],
        "annotations": [], "categories": cats,
    }))


def _extract_wutdet(root):
    buf = io.BytesIO()
    Image.new("RGB", (64, 48)).save(buf, "JPEG")
    jpg = buf.getvalue()
    voc = root / "voc"
    (voc / "Annotations").mkdir(parents=True)
    (voc / "JPEGImages").mkdir(parents=True)
    (voc / "JPEGImages" / "0.jpg").write_bytes(jpg)
    (voc / "Annotations" / "0.xml").write_text(
        "<annotation><filename>0.jpg</filename>"
        "<size><width>64</width><height>48</height><depth>3</depth></size>"
        "</annotation>")


CATS = {
    "vais_smd_marvel": ["vessel", "buoy", "object"],
    "singapore_maritime": ["objects", "boat", "buoy", "ferry",
                           "flying bird-plane", "kayak", "other",
                           "sail boat", "speed boat", "vessel-ship"],
    "sea_vessels": ["sea-vessels", "fishing boat", "merchant ship",
                    "military ship", "patrol boat", "sails boat",
                    "submarine", "tugboat", "yacht"],
    "ship_model": ["ship"],
    "ir_thermal": ["boat", "bulk carrier", "canoe", "container ship",
                  "fishing boat", "liner", "sailboat", "warship"],
}


class ShipDatasetDiscoveryTests(ShipNotebookEmbedTests):
    """Kesif hucresi, kaynaklari klasor adina degil kategori imzasina gore
    bulmali (aerial projedeki ayni disiplin, bkz. HANDOFF #7 madde 15)."""

    def test_sources_found_despite_unrelated_folder_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # her kaynagi ALAKASIZ, tanidilamayan bir klasor adiyla koy
            _extract_coco(root / "random-folder-1" / "some-name", CATS["vais_smd_marvel"])
            _extract_coco(root / "random-folder-2" / "xyz", CATS["singapore_maritime"])
            _extract_coco(root / "blob3", CATS["sea_vessels"])
            _extract_coco(root / "nested" / "deep" / "blob4", CATS["ship_model"])
            _extract_coco(root / "blob5", CATS["ir_thermal"])
            _extract_wutdet(root / "totally-unrelated-name")

            found = self._discover_from_notebook(root)

        self.assertEqual(set(found), set(CATS) | {"wutdet"})

    def test_zip_sources_are_recognised(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for key, cats in CATS.items():
                _coco_zip(root / f"{key}_random_name.zip", cats)
            # WUTDet zip
            buf = io.BytesIO()
            Image.new("RGB", (64, 48)).save(buf, "JPEG")
            jpg = buf.getvalue()
            with zipfile.ZipFile(root / "wutdet_blob.zip", "w") as z:
                z.writestr("voc/JPEGImages/0.jpg", jpg)
                z.writestr("voc/Annotations/0.xml",
                          "<annotation><filename>0.jpg</filename>"
                          "<size><width>64</width><height>48</height>"
                          "<depth>3</depth></size></annotation>")

            found = self._discover_from_notebook(root)

        self.assertEqual(set(found), set(CATS) | {"wutdet"})
        self.assertTrue(str(found["vais_smd_marvel"]).endswith(".zip"))

    def test_parent_directory_is_not_mistaken_for_a_source(self):
        """Tum kaynaklari iceren ust klasor tek bir kaynak sanilmamali."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "ship-detection-sources"
            for key, cats in CATS.items():
                _extract_coco(bundle / key / "orig-name", cats)
            _extract_wutdet(bundle / "wutdet" / "orig-name")

            found = self._discover_from_notebook(root)

        for key, path in found.items():
            self.assertNotEqual(Path(path).name, "ship-detection-sources",
                               f"{key} ust klasore cozuldu")

    def test_missing_source_raises_with_listing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as ctx:
                self._discover_from_notebook(Path(tmp))
        message = str(ctx.exception)
        self.assertIn("Bulunamayan kaynaklar", message)
        self.assertIn("wutdet", message)

    def test_ambiguous_category_set_is_not_misclassified(self):
        """ship_model'in tek-kategori imzasi ({'ship'}) baska hicbir kaynakla
        cakismamali; taniyamadigi bir kume None dondurmeli (varsayima gitmemeli).
        discover_ship_sources() TUM kaynaklari zorunlu kildigi icin _classify()
        dogrudan sinaniyor.
        """
        source = self._discover_cell_source()
        namespace = {"Path": Path}
        exec(compile(source.split("SOURCES = discover_ship_sources()")[0],
                     "cell6", "exec"), namespace)
        classify = namespace["_classify"]
        self.assertEqual(classify(CATS["ship_model"]), "ship_model")
        self.assertIsNone(classify(["boat"]),
                          "taninmayan kume varsayima gitmemeli")
        self.assertIsNone(classify(CATS["ship_model"] + ["extra"]),
                          "fazladan kategori tam esitligi bozmali")


if __name__ == "__main__":
    unittest.main()
