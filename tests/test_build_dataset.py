"""build_dataset.py icin testler.

Gercek veri setleri 4 GB; burada her formatin kucuk sentetik bir ornegi
uretilip okuyucular ve dogrulama kapilari sinaniyor. Amac, saatlerce surecek
bir egitimi bozacak sessiz hatalari yakalamak: yanlis sinif eslemesi, sinir
disi kutu, oturum sizintisi, eslenmemis kategori.
"""

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from tools.build_dataset import (
    IGNORE,
    LAND,
    SEA,
    Archive,
    MILREC_MAP,
    VESSELIMG_MAP,
    VISDRONE_MAP,
    build_coco,
    clip_box,
    read_mendeley_yolo,
    read_roboflow_coco,
    read_visdrone,
    resplit_by_group,
    session_key,
    validate,
)


def jpeg_bytes(width=100, height=80):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buf, "JPEG")
    return buf.getvalue()


def make_zip(path, entries):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)
    return Archive(path)


class ClipBoxTests(unittest.TestCase):
    def test_box_is_clipped_to_image(self):
        self.assertEqual(clip_box(-10, -10, 50, 50, 100, 80), [0.0, 0.0, 40.0, 40.0])
        self.assertEqual(clip_box(90, 70, 50, 50, 100, 80), [90.0, 70.0, 10.0, 10.0])

    def test_degenerate_boxes_rejected(self):
        self.assertIsNone(clip_box(0, 0, 0, 10, 100, 80))
        self.assertIsNone(clip_box(200, 0, 10, 10, 100, 80))   # tamamen disarida
        self.assertIsNone(clip_box(0, 0, -5, -5, 100, 80))


class SessionKeyTests(unittest.TestCase):
    def test_roboflow_frame_names_group_by_session(self):
        a = "img_cameras_2023-10-13-13-55-15_134_jpg.rf.abc123.jpg"
        b = "img_cameras_2023-10-13-13-55-15_147_jpg.rf.def456.jpg"
        c = "img_cameras_2023-10-13-13-56-57_391_jpg.rf.999999.jpg"
        self.assertEqual(session_key(a), session_key(b))
        self.assertNotEqual(session_key(a), session_key(c))

    def test_plain_names_are_their_own_group(self):
        self.assertEqual(session_key("foo/bar.jpg"), "bar.jpg")


class VisDroneReaderTests(unittest.TestCase):
    def _entries(self):
        anno = (
            "10,10,20,20,1,4,0,0\n"     # car        -> land
            "40,10,20,20,1,9,0,0\n"     # bus        -> land
            "60,10,10,10,1,3,0,0\n"     # bicycle    -> atilir
            "70,10,10,10,1,1,0,0\n"     # pedestrian -> atilir
            "10,40,20,20,0,4,0,0\n"     # car score=0 -> ignore
            "0,60,100,20,1,0,0,0\n"     # ignored-region
            "80,70,50,50,1,6,0,0\n"     # truck, sinir disina tasar -> kirpilir
        )
        return {
            "VisDrone2019-DET-train/images/0000001_x.jpg": jpeg_bytes(),
            "VisDrone2019-DET-train/annotations/0000001_x.txt": anno,
        }

    def test_class_mapping_and_ignore_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            with make_zip(Path(tmp) / "vd.zip", self._entries()) as arc:
                recs = read_visdrone(arc, "train")
        self.assertEqual(len(recs), 1)
        rec = recs[0]
        real = [a for a in rec.anns if not a["ignore"]]
        ign = [a for a in rec.anns if a["ignore"]]
        self.assertEqual(len(real), 3, "car + bus + truck bekleniyordu")
        self.assertTrue(all(a["category_id"] == LAND for a in real))
        self.assertEqual(len(ign), 1, "score=0 kutusu ignore olmali")
        self.assertEqual(rec.ignore_regions, [[0.0, 60.0, 100.0, 20.0]])

    def test_out_of_bounds_box_is_clipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            with make_zip(Path(tmp) / "vd.zip", self._entries()) as arc:
                rec = read_visdrone(arc, "train")[0]
        truck = [a for a in rec.anns
                 if not a["ignore"] and a["bbox"][0] == 80.0][0]
        # goruntu 100x80; 80,70,50,50 -> 20x10'a kirpilmali
        self.assertEqual(truck["bbox"], [80.0, 70.0, 20.0, 10.0])

    def test_every_dropped_class_is_intentional(self):
        # Esleme tablosu eksiksiz olmali: 1..11 arasi her kategori tanimli
        for cat in range(1, 12):
            self.assertIn(cat, VISDRONE_MAP, f"VisDrone kategori {cat} eslenmemis")


class RoboflowReaderTests(unittest.TestCase):
    def _coco(self, files, categories, annotations):
        return json.dumps({
            "images": [{"id": i + 1, "file_name": f, "width": 640, "height": 640}
                       for i, f in enumerate(files)],
            "categories": [{"id": i, "name": n} for i, n in enumerate(categories)],
            "annotations": annotations,
        })

    def test_vesselimg_mapping_pilot_is_ignored_buoy_is_dropped(self):
        cats = ["boats", "Buoy", "Chemical", "Container",
                "Passenger-RoRo", "Pilot", "Tugboat"]
        anns = [
            {"id": 1, "image_id": 1, "category_id": 3, "bbox": [10, 10, 100, 50]},
            {"id": 2, "image_id": 1, "category_id": 6, "bbox": [200, 10, 60, 30]},
            {"id": 3, "image_id": 1, "category_id": 5, "bbox": [300, 10, 15, 15]},
            {"id": 4, "image_id": 1, "category_id": 1, "bbox": [400, 10, 20, 40]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with make_zip(Path(tmp) / "v.zip", {
                "train/_annotations.coco.json":
                    self._coco(["a_1_jpg.rf.x.jpg"], cats, anns),
                "train/a_1_jpg.rf.x.jpg": jpeg_bytes(640, 640),
            }) as arc:
                recs = read_roboflow_coco(arc, VESSELIMG_MAP, "vesselimg")
        anns_out = recs[0].anns
        real = [a for a in anns_out if not a["ignore"]]
        ign = [a for a in anns_out if a["ignore"]]
        self.assertEqual(len(real), 2, "Container + Tugboat")
        self.assertTrue(all(a["category_id"] == SEA for a in real))
        self.assertEqual(len(ign), 1, "Pilot ignore olmali")
        self.assertEqual(len(anns_out), 3, "Buoy tamamen atilmali")

    def test_milrec_keeps_tank_and_apc_only(self):
        cats = ["military-tank-plane-soldier", "air-fighter",
                "armoured personnel carrier", "bomber", "soldier", "tank"]
        anns = [
            {"id": 1, "image_id": 1, "category_id": 5, "bbox": [10, 10, 90, 60]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [200, 10, 70, 40]},
            {"id": 3, "image_id": 1, "category_id": 1, "bbox": [300, 10, 20, 20]},
            {"id": 4, "image_id": 1, "category_id": 4, "bbox": [400, 10, 30, 30]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with make_zip(Path(tmp) / "m.zip", {
                "train/_annotations.coco.json":
                    self._coco(["b.rf.y.jpg"], cats, anns),
                "train/b.rf.y.jpg": jpeg_bytes(640, 640),
            }) as arc:
                recs = read_roboflow_coco(arc, MILREC_MAP, "milrec")
        real = [a for a in recs[0].anns if not a["ignore"]]
        self.assertEqual(len(real), 2, "tank + APC")
        self.assertTrue(all(a["category_id"] == LAND for a in real))

    def test_unmapped_category_is_a_hard_error(self):
        # Yeni bir surum bilinmeyen sinif getirirse sessizce atlanmamali
        with tempfile.TemporaryDirectory() as tmp:
            with make_zip(Path(tmp) / "u.zip", {
                "train/_annotations.coco.json":
                    self._coco(["c.rf.z.jpg"], ["Container", "submarine"], []),
                "train/c.rf.z.jpg": jpeg_bytes(640, 640),
            }) as arc:
                with self.assertRaises(SystemExit) as ctx:
                    read_roboflow_coco(arc, VESSELIMG_MAP, "vesselimg")
        self.assertIn("submarine", str(ctx.exception))


class MendeleyReaderTests(unittest.TestCase):
    def test_yolo_normalised_boxes_become_pixels(self):
        # cls cx cy w h  (normalize) - tank ve soldier
        label = "0 0.5 0.5 0.2 0.4\n3 0.1 0.1 0.05 0.05\n"
        with tempfile.TemporaryDirectory() as tmp:
            with make_zip(Path(tmp) / "d.zip", {
                "d/dataset/train/images/x.jpg": jpeg_bytes(200, 100),
                "d/dataset/train/labels/x.txt": label,
            }) as arc:
                recs = read_mendeley_yolo(arc)
        self.assertEqual(len(recs), 1)
        anns = recs[0].anns
        self.assertEqual(len(anns), 1, "yalnizca tank kalmali")
        # cx=0.5*200=100, w=0.2*200=40 -> x=80 ; cy=50, h=40 -> y=30
        self.assertEqual(anns[0]["bbox"], [80.0, 30.0, 40.0, 40.0])
        self.assertEqual(anns[0]["category_id"], LAND)

    def test_test_split_maps_to_val(self):
        with tempfile.TemporaryDirectory() as tmp:
            with make_zip(Path(tmp) / "d.zip", {
                "d/dataset/test/images/y.jpg": jpeg_bytes(200, 100),
                "d/dataset/test/labels/y.txt": "0 0.5 0.5 0.2 0.4\n",
            }) as arc:
                recs = read_mendeley_yolo(arc)
        self.assertEqual(recs[0].split, "val")


class SplitTests(unittest.TestCase):
    class FakeRecord:
        def __init__(self, group):
            self.group = group
            self.split = "train"

    def test_resplit_keeps_sessions_whole(self):
        recs = [self.FakeRecord(f"s{i}") for i in range(20) for _ in range(5)]
        total, n_val = resplit_by_group(recs, 0.25)
        self.assertEqual(total, 20)
        self.assertEqual(n_val, 5)
        by_group = {}
        for r in recs:
            by_group.setdefault(r.group, set()).add(r.split)
        self.assertTrue(all(len(v) == 1 for v in by_group.values()),
                        "bir oturum hem train hem val'de olmamali")

    def test_split_is_deterministic(self):
        a = [self.FakeRecord(f"s{i}") for i in range(20)]
        b = [self.FakeRecord(f"s{i}") for i in range(20)]
        resplit_by_group(a, 0.25)
        resplit_by_group(b, 0.25)
        self.assertEqual([r.split for r in a], [r.split for r in b])


class ValidationTests(unittest.TestCase):
    def _record(self, **kw):
        from tools.build_dataset import Record
        rec = Record(kw.get("source", "s"), "m", kw.get("file_name", "s/a.jpg"),
                     kw.get("width", 100), kw.get("height", 100),
                     kw.get("split", "train"), kw.get("group", "g1"))
        rec.anns = kw.get("anns", [])
        return rec

    def test_clean_records_pass(self):
        rec = self._record(anns=[{"bbox": [10, 10, 20, 20], "category_id": LAND,
                                  "iscrowd": 0, "ignore": 0}])
        self.assertEqual(validate([rec]), [])

    def test_out_of_bounds_box_is_caught(self):
        rec = self._record(anns=[{"bbox": [90, 90, 50, 50], "category_id": LAND,
                                  "iscrowd": 0, "ignore": 0}])
        self.assertTrue(any("sinir disi" in p for p in validate([rec])))

    def test_duplicate_file_name_is_caught(self):
        a = self._record(file_name="s/dup.jpg", group="g1")
        b = self._record(file_name="s/dup.jpg", group="g2")
        self.assertTrue(any("yinelenen" in p for p in validate([a, b])))

    def test_session_leakage_is_caught(self):
        a = self._record(file_name="s/a.jpg", group="same", split="train")
        b = self._record(file_name="s/b.jpg", group="same", split="val")
        self.assertTrue(any("oturum" in p for p in validate([a, b])))

    def test_bad_category_is_caught(self):
        rec = self._record(anns=[{"bbox": [1, 1, 5, 5], "category_id": 7,
                                  "iscrowd": 0, "ignore": 0}])
        self.assertTrue(any("gecersiz kategori" in p for p in validate([rec])))


class CocoOutputTests(unittest.TestCase):
    def _record(self, file_name, split, anns, regions=()):
        from tools.build_dataset import Record
        rec = Record("s", "m", file_name, 100, 100, split, "g")
        rec.anns = anns
        rec.ignore_regions = list(regions)
        return rec

    def test_ids_are_unique_and_consistent(self):
        recs = [
            self._record("s/a.jpg", "train",
                         [{"bbox": [1, 1, 10, 10], "category_id": LAND,
                           "iscrowd": 0, "ignore": 0}], [[0, 0, 5, 5]]),
            self._record("s/b.jpg", "train",
                         [{"bbox": [2, 2, 10, 10], "category_id": SEA,
                           "iscrowd": 0, "ignore": 0},
                          {"bbox": [3, 3, 4, 4], "category_id": LAND,
                           "iscrowd": 1, "ignore": 1}]),
            self._record("s/c.jpg", "val", []),
        ]
        train = build_coco(recs, "train")
        self.assertEqual(len(train["images"]), 2)
        self.assertEqual(len(train["annotations"]), 3)
        self.assertEqual(len({i["id"] for i in train["images"]}), 2)
        self.assertEqual(len({a["id"] for a in train["annotations"]}), 3)
        image_ids = {i["id"] for i in train["images"]}
        self.assertTrue(all(a["image_id"] in image_ids
                            for a in train["annotations"]))
        self.assertEqual(train["images"][0]["ignore_regions"], [[0, 0, 5, 5]])
        self.assertEqual([c["name"] for c in train["categories"]],
                         ["land_vehicle", "sea_vehicle"])

    def test_val_split_is_separate(self):
        recs = [self._record("s/c.jpg", "val", [])]
        self.assertEqual(len(build_coco(recs, "train")["images"]), 0)
        self.assertEqual(len(build_coco(recs, "val")["images"]), 1)

    def test_area_matches_bbox(self):
        recs = [self._record("s/a.jpg", "train",
                             [{"bbox": [1, 1, 10, 20], "category_id": SEA,
                               "iscrowd": 0, "ignore": 0}])]
        ann = build_coco(recs, "train")["annotations"][0]
        self.assertAlmostEqual(ann["area"], 200.0)


class ArchiveTests(unittest.TestCase):
    def test_zip_and_directory_behave_the_same(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "d" / "images").mkdir(parents=True)
            (root / "d" / "images" / "a.jpg").write_bytes(jpeg_bytes(64, 32))
            with Archive(root / "d") as dir_arc,                  make_zip(root / "z.zip",
                          {"images/a.jpg": jpeg_bytes(64, 32)}) as zip_arc:
                self.assertEqual(dir_arc.image_size("images/a.jpg"), (64, 32))
                self.assertEqual(zip_arc.image_size("images/a.jpg"), (64, 32))
                self.assertTrue(dir_arc.exists("images/a.jpg"))
                self.assertFalse(zip_arc.exists("nope.jpg"))


if __name__ == "__main__":
    unittest.main()


class PathPartTests(unittest.TestCase):
    """Arsiv koku farkli seviyelerde olabilir; yol eslestirmesi buna dayanmali.

    Kaggle'da VisDrone genelde
    '<slug>/VisDrone2019-DET-train/VisDrone2019-DET-train/images/...' seklinde
    ic ice gelir; kok nereye isaret ederse etsin okuyucu calismalidir.
    """

    def test_has_part_matches_whole_component_only(self):
        from tools.build_dataset import has_part
        self.assertTrue(has_part("VisDrone2019-DET-train/images/a.jpg", "images"))
        self.assertTrue(has_part("images/a.jpg", "images"))
        self.assertFalse(has_part("myimages/a.jpg", "images"))
        self.assertFalse(has_part("images.jpg", "images"))

    def test_swap_part_replaces_component(self):
        from tools.build_dataset import swap_part
        self.assertEqual(swap_part("images/a.jpg", "images", "annotations"),
                         "annotations/a.jpg")
        self.assertEqual(swap_part("d/images/a.jpg", "images", "labels"),
                         "d/labels/a.jpg")

    def test_visdrone_reader_works_at_either_root(self):
        anno = "10,10,20,20,1,4,0,0\n"
        for prefix in ("", "VisDrone2019-DET-train/"):
            with tempfile.TemporaryDirectory() as tmp:
                with make_zip(Path(tmp) / "v.zip", {
                    f"{prefix}images/0000001_x.jpg": jpeg_bytes(),
                    f"{prefix}annotations/0000001_x.txt": anno,
                }) as arc:
                    recs = read_visdrone(arc, "train")
                self.assertEqual(len(recs), 1, f"kok={prefix!r}")
                self.assertEqual(len(recs[0].anns), 1, f"kok={prefix!r}")

    def test_mendeley_reader_works_at_either_root(self):
        for prefix in ("", "d/dataset/"):
            with tempfile.TemporaryDirectory() as tmp:
                with make_zip(Path(tmp) / "m.zip", {
                    f"{prefix}train/images/x.jpg": jpeg_bytes(200, 100),
                    f"{prefix}train/labels/x.txt": "0 0.5 0.5 0.2 0.4\n",
                }) as arc:
                    recs = read_mendeley_yolo(arc)
                self.assertEqual(len(recs), 1, f"kok={prefix!r}")
                self.assertEqual(len(recs[0].anns), 1, f"kok={prefix!r}")


class EmptySourceTests(unittest.TestCase):
    """Bos kaynak sessizce gecilmemeli.

    Kaggle'daki bazi VisDrone kopyalari Ultralytics'e cevrilmis: 'annotations/'
    yerine YOLO 'labels/' iceriyorlar ve ignored-region bilgisi silinmis.
    Boyle bir kaynakta okuyucu hicbir sey bulamaz; bu durumda 0 goruntuyle
    devam etmek yerine hata verilmelidir.
    """

    def test_visdrone_reader_returns_nothing_for_yolo_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            with make_zip(Path(tmp) / "v.zip", {
                "images/0000001_x.jpg": jpeg_bytes(),
                "labels/0000001_x.txt": "3 0.5 0.5 0.2 0.2\n",   # YOLO, annotations yok
            }) as arc:
                recs = read_visdrone(arc, "train")
        self.assertEqual(recs, [], "annotations/ yoksa kayit uretilmemeli")
