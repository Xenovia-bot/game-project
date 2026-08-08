"""Notebook'un %%writefile hucreleri kaynak dosyalarla ayni kalmali.

Kaggle'da calisan kod bu hucrelerdir; repodaki kaynak degisip hucre
guncellenmezse egitim eski kodla yapilir ve fark saatler sonra anlasilir.
`python tools/_sync_notebook_embeds.py` bu hizalamayi yapar.
"""

import json
import unittest
import zipfile
import io
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "training" / "kaggle_visdrone_yolox.ipynb"

#: tools/_sync_notebook_embeds.py ile ayni tablo olmali.
EMBEDS = {
    3: ROOT / "tools" / "build_dataset.py",
    4: ROOT / "training" / "visdrone_eval.py",
    5: ROOT / "training" / "exps" / "yolox_nano_visdrone.py",
}
NOTED = {"build_dataset.py", "yolox_nano_visdrone.py"}


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


class NotebookEmbedTests(unittest.TestCase):
    def test_sync_table_matches_the_tool(self):
        """Iki tablo ayrisirsa senkron sessizce yanlis hucreyi yazar."""
        source = (ROOT / "tools" / "_sync_notebook_embeds.py").read_text(
            encoding="utf-8")
        for index, path in EMBEDS.items():
            self.assertIn(f"{index}: ROOT / ", source)
            self.assertIn(f'"{path.name}"', source)

    def test_writefile_cells_match_project_sources(self):
        for index, path in EMBEDS.items():
            body = notebook_writefile_body(index)
            if path.name in NOTED:
                body = strip_notebook_note(body, path)
            self.assertEqual(
                body.replace("\r\n", "\n"),
                path.read_text(encoding="utf-8").replace("\r\n", "\n"),
                msg="Notebook hucresi %d, %s ile ayristi" % (index, path),
            )

    def test_embedded_builder_maps_classes_and_ignores(self):
        """Gomulu build_dataset.py gercekten calisip dogru esleme yapmali."""
        namespace = {}
        exec(compile(notebook_writefile_body(3), "build_dataset.py", "exec"),
             namespace)
        read_visdrone = namespace["read_visdrone"]
        Archive = namespace["Archive"]

        buf = io.BytesIO()
        Image.new("RGB", (100, 100)).save(buf, "JPEG")
        with tempfile.TemporaryDirectory() as tmp:
            zpath = Path(tmp) / "v.zip"
            with zipfile.ZipFile(zpath, "w") as z:
                z.writestr("images/one.jpg", buf.getvalue())
                z.writestr("annotations/one.txt",
                           "0,0,40,40,1,0,0,0\n"      # ignored-region
                           "50,50,10,10,1,4,0,0\n"    # car     -> land
                           "60,60,10,10,1,1,0,0\n"    # yaya    -> atilir
                           "70,70,10,10,0,4,0,0\n")   # score=0 -> ignore
            with Archive(zpath) as archive:
                records = read_visdrone(archive, "train")

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.ignore_regions, [[0.0, 0.0, 40.0, 40.0]])
        real = [a for a in record.anns if not a["ignore"]]
        ignored = [a for a in record.anns if a["ignore"]]
        self.assertEqual(len(real), 1, "yalnizca car kalmali")
        self.assertEqual(real[0]["bbox"], [50.0, 50.0, 10.0, 10.0])
        self.assertEqual(real[0]["category_id"], namespace["LAND"])
        self.assertEqual(len(ignored), 1, "score=0 kutusu ignore olmali")

    def test_notebook_has_no_stale_source_references(self):
        """Silinen modullere atif kalmamali."""
        text = NOTEBOOK.read_text(encoding="utf-8")
        for gone in ("visdrone2coco", "eval_tiled", "tiling.py"):
            self.assertNotIn(gone, text, f"notebook'ta bayat atif: {gone}")



class DatasetDiscoveryTests(unittest.TestCase):
    """Notebook hucresi 6, kaynaklari klasor adina degil ICERIGE gore bulmali.

    Kaggle'a manuel yuklemede klasor adlari degisir ve Kaggle zip'leri acabilir.
    Ada bagli bir arama bu durumda kaynagi bulamaz ve kullanici saatlerce yol
    hatasiyla ugrasir.
    """

    def _discover_from_notebook(self, input_root):
        import re
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "".join(notebook["cells"][6]["source"])
        source = re.sub(r"^!.*$", "# shell", source, flags=re.M)
        body = source.split("SOURCES = discover()")[0]
        body = body.replace('INPUT = Path("/kaggle/input")',
                            f'INPUT = Path(r"{input_root}")')
        namespace = {"Path": Path}
        exec(compile(body, "cell6", "exec"), namespace)
        return namespace["discover"]()

    def test_sources_found_despite_unrelated_folder_names(self):
        buf = io.BytesIO()
        Image.new("RGB", (64, 48)).save(buf, "JPEG")
        jpg = buf.getvalue()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "val"):
                d = root / f"vd-{split}" / f"VisDrone2019-DET-{split}"
                (d / "images").mkdir(parents=True)
                (d / "annotations").mkdir()
                (d / "images" / "a.jpg").write_bytes(jpg)
                (d / "annotations" / "a.txt").write_text("10,10,20,20,1,4,0,0\n")

            # VESSELimg: acilmis klasor, alakasiz isim
            d = root / "my-boats" / "train"
            d.mkdir(parents=True)
            (d / "_annotations.coco.json").write_text(json.dumps({
                "images": [], "annotations": [],
                "categories": [{"id": 0, "name": "Container"},
                               {"id": 1, "name": "Tugboat"}]}))

            # milrec: zip, alakasiz isim
            (root / "stuff").mkdir()
            with zipfile.ZipFile(root / "stuff" / "mv.zip", "w") as z:
                z.writestr("train/_annotations.coco.json", json.dumps({
                    "images": [], "annotations": [],
                    "categories": [
                        {"id": 0, "name": "tank"},
                        {"id": 1, "name": "armoured personnel carrier"}]}))

            # mendeley: images/ + labels/
            d = root / "military-uav" / "dataset" / "train"
            (d / "images").mkdir(parents=True)
            (d / "labels").mkdir()
            (d / "images" / "x.jpg").write_bytes(jpg)
            (d / "labels" / "x.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            found = self._discover_from_notebook(root)

        self.assertEqual(
            set(found),
            {"visdrone-train", "visdrone-val", "vesselimg", "milrec", "mendeley"})
        # Kokler isaret dosyasindan yukari yurunerek bulunur; ust klasor
        # degil, kaynaga ait en dar dizin secilmeli.
        self.assertEqual(Path(found["vesselimg"]).name, "my-boats")
        self.assertEqual(Path(found["milrec"]).name, "mv.zip")
        # labels/ -> <kok>/dataset/train/labels oldugundan kok '<...>/dataset'
        self.assertEqual(Path(found["mendeley"]).name, "dataset")
        self.assertEqual(Path(found["mendeley"]).parent.name, "military-uav")

    def test_parent_directory_is_not_mistaken_for_a_source(self):
        """Tum kaynaklari iceren ust klasor tek bir kaynak sanilmamali.

        Kaggle'da gercekten yasandi: kok dizin alt agacinda milrec'in
        anotasyon dosyasini bulundurdugu icin 'milrec' olarak secildi ve
        okuma sifir goruntu dondurdu.
        """
        buf = io.BytesIO()
        Image.new("RGB", (64, 48)).save(buf, "JPEG")
        jpg = buf.getvalue()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "aerial-vehicle-sources"

            for split in ("train", "val"):
                d = bundle / "aerial-land" / f"VisDrone2019-DET-{split}"
                (d / "images").mkdir(parents=True)
                (d / "annotations").mkdir()
                (d / "images" / "a.jpg").write_bytes(jpg)
                (d / "annotations" / "a.txt").write_text("10,10,20,20,1,4,0,0\n")

            d = bundle / "aerial-land" / "Mendeley" / "dataset" / "train"
            (d / "images").mkdir(parents=True)
            (d / "labels").mkdir()
            (d / "images" / "x.jpg").write_bytes(jpg)
            (d / "labels" / "x.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            # Kaggle zip'i alt klasore aciyor: <kok>/vesselimg/<zip adi>/train/
            d = bundle / "vesselimg" / "VESSELimg.v4i.coco" / "train"
            d.mkdir(parents=True)
            (d / "_annotations.coco.json").write_text(json.dumps({
                "images": [], "annotations": [],
                "categories": [{"id": 0, "name": "Container"},
                               {"id": 1, "name": "Tugboat"}]}))

            d = bundle / "milrec" / "MilRec.v7i.coco" / "train"
            d.mkdir(parents=True)
            (d / "_annotations.coco.json").write_text(json.dumps({
                "images": [], "annotations": [],
                "categories": [{"id": 0, "name": "tank"},
                               {"id": 1, "name": "armoured personnel carrier"}]}))

            found = self._discover_from_notebook(root)

        self.assertEqual(Path(found["milrec"]).name, "MilRec.v7i.coco")
        self.assertEqual(Path(found["vesselimg"]).name, "VESSELimg.v4i.coco")
        self.assertEqual(Path(found["mendeley"]).name, "dataset")
        self.assertEqual(Path(found["visdrone-train"]).name,
                         "VisDrone2019-DET-train")
        self.assertEqual(Path(found["visdrone-val"]).name,
                         "VisDrone2019-DET-val")
        for key, path in found.items():
            self.assertNotEqual(Path(path).name, "aerial-land",
                                f"{key} ust klasore cozuldu")

    def test_missing_source_raises_with_listing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as ctx:
                self._discover_from_notebook(Path(tmp))
        message = str(ctx.exception)
        self.assertIn("Bulunamayan kaynaklar", message)
        self.assertIn("vesselimg", message)

if __name__ == "__main__":
    unittest.main()
