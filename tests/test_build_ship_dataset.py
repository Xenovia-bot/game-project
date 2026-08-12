"""build_ship_dataset.py icin testler.

Odak, 2026-08-11'de gercek veride yakalanan iki sessiz-hata sinifi: (1)
kaynaklar arasi ayni fotografin/karenin iki kez sayilmasi (dedupe), (2)
asiri kaba oturum gruplamasinin split'i bozmasi (ir_thermal). Ayrica
CLASS_MAPS tablosunun kendini tutarli tuttugunu (butun kaynaklar SHIP veya
None'a esler, baska bir seye degil) ve validate() kapisinin gercekten
durdugunu sinar.
"""

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from tools.dataset_common import split_by_group_targets
from tools.build_ship_dataset import (
    CLASS_MAPS,
    SHIP,
    Record,
    ship_session_key,
    build_coco,
    cross_source_identity,
    dedupe_sources,
    read_ship_coco,
    validate,
)
from tools.dataset_common import Archive


def jpeg_bytes(width=100, height=80):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buf, "JPEG")
    return buf.getvalue()


def make_zip(path, entries):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)
    return Archive(path)


class CrossSourceIdentityTests(unittest.TestCase):
    def test_video_frame_identity_includes_frame_number(self):
        # 2026-08-11'de yakalanan bug: yalnizca video ID yakalanirsa ayni
        # videonun TUM kareleri tek kimlige collapse oluyordu.
        a = cross_source_identity("MVI_1451_VIS_frame0_jpg.rf.aaa.jpg")
        b = cross_source_identity("MVI_1451_VIS_frame10_jpg.rf.bbb.jpg")
        self.assertNotEqual(a, b, "ayni videonun farkli kareleri ayni kimlik olmamali")

    def test_same_video_and_frame_is_same_identity(self):
        a = cross_source_identity("MVI_1451_VIS_frame0_jpg.rf.aaa.jpg")
        b = cross_source_identity("MVI_1451_VIS_frame0_jpg.rf.zzz.jpg")
        self.assertEqual(a, b, "hash farkli olsa da ayni video+kare ayni kimlik olmali")

    def test_ob_variant_distinct_from_plain(self):
        a = cross_source_identity("MVI_0790_VIS_OB_frame185_jpg.rf.aaa.jpg")
        b = cross_source_identity("MVI_0790_VIS_frame185_jpg.rf.bbb.jpg")
        self.assertNotEqual(a, b)

    def test_marvel_numeric_identity(self):
        a = cross_source_identity("100074_jpg.rf.aaa.jpg")
        b = cross_source_identity("100074_jpg.rf.zzz.jpg")
        self.assertEqual(a, b)

    def test_unrelated_naming_returns_none(self):
        self.assertIsNone(cross_source_identity("military_97_jpg.rf.aaa.jpg"))
        self.assertIsNone(cross_source_identity("0000000.jpg"))

    def test_ir_thermal_prefix_not_mistaken_for_marvel(self):
        # "1_5135_jpg..." MARVEL deseniyle ("^\d+_jpg") YANLISLIKLA
        # eslesmemeli -- ilk grup "1" olur ama sonrasi "_jpg" degil "_5135_jpg".
        self.assertIsNone(cross_source_identity("1_5135_jpg.rf.aaa.jpg"))


class DedupeSourcesTests(unittest.TestCase):
    class FakeRecord:
        def __init__(self, file_name):
            self.file_name = file_name

    def test_first_source_in_order_wins(self):
        by_source = {
            "singapore_maritime": [self.FakeRecord("singapore_maritime/MVI_1_VIS_frame0_jpg.rf.a.jpg")],
            "vais_smd_marvel": [self.FakeRecord("vais_smd_marvel/MVI_1_VIS_frame0_jpg.rf.b.jpg")],
        }
        # DEDUPE_ORDER'daki gercek sirayi taklit etmek yerine dogrudan
        # fonksiyonun kendi sirasini kullaniyoruz (import edilen DEDUPE_ORDER).
        from tools.build_ship_dataset import DEDUPE_ORDER
        self.assertLess(DEDUPE_ORDER.index("singapore_maritime"),
                        DEDUPE_ORDER.index("vais_smd_marvel"))
        dropped = dedupe_sources(by_source)
        self.assertEqual(len(by_source["singapore_maritime"]), 1)
        self.assertEqual(len(by_source["vais_smd_marvel"]), 0)
        self.assertEqual(dropped["vais_smd_marvel"], 1)

    def test_no_identity_never_dropped(self):
        by_source = {
            "singapore_maritime": [self.FakeRecord("singapore_maritime/a.jpg")],
            "ship_model": [self.FakeRecord("ship_model/b.jpg")],
        }
        dropped = dedupe_sources(by_source)
        self.assertEqual(len(by_source["singapore_maritime"]), 1)
        self.assertEqual(len(by_source["ship_model"]), 1)
        self.assertEqual(sum(dropped.values()), 0)

    def test_within_source_duplicate_also_caught(self):
        by_source = {
            "ship_model": [self.FakeRecord("ship_model/100074_jpg.rf.a.jpg"),
                          self.FakeRecord("ship_model/100074_jpg.rf.b.jpg")],
        }
        dedupe_sources(by_source)
        self.assertEqual(len(by_source["ship_model"]), 1,
                         "ayni kaynak icinde tekrarlanan numara da yakalanmali")


class ShipSessionKeyTests(unittest.TestCase):
    """Oturum anahtari, bolmenin sizinti onleyen tek dayanagi. Bir kaynagin
    adlandirmasi taninmazsa her goruntu kendi oturumu olur ve grup bazli
    bolme sessizce RASTGELE bolmeye cokerm-- 2026-08-12'de olculen gercek
    hata buydu (singapore_maritime 6350 goruntu -> 6350 oturum).
    """

    def test_smd_video_frames_share_one_session(self):
        a = ship_session_key(
            "singapore_maritime/MVI_1478_VIS_frame90_jpg.rf.a.jpg", "singapore_maritime")
        b = ship_session_key(
            "singapore_maritime/MVI_1478_VIS_frame455_jpg.rf.b.jpg", "singapore_maritime")
        self.assertEqual(a, b, "ayni videonun kareleri ayni oturum olmali")

    def test_smd_different_videos_are_different_sessions(self):
        a = ship_session_key("singapore_maritime/MVI_1478_VIS_frame90_jpg.rf.a.jpg",
                             "singapore_maritime")
        b = ship_session_key("singapore_maritime/MVI_1479_VIS_frame90_jpg.rf.b.jpg",
                             "singapore_maritime")
        self.assertNotEqual(a, b)

    def test_smd_ob_variant_is_its_own_video(self):
        a = ship_session_key("singapore_maritime/MVI_0790_VIS_OB_frame5_jpg.rf.a.jpg",
                             "singapore_maritime")
        b = ship_session_key("singapore_maritime/MVI_0790_VIS_frame5_jpg.rf.b.jpg",
                             "singapore_maritime")
        self.assertNotEqual(a, b, "OB farkli bir cekim")

    def test_smd_key_is_source_independent(self):
        """Ayni video iki kaynakta birden bulunabiliyor; kaynak onekli bir
        anahtar ikisini farkli oturum sayar ve sizinti geri gelirdi."""
        self.assertEqual(
            ship_session_key("singapore_maritime/MVI_1478_VIS_frame90_jpg.rf.a.jpg",
                             "singapore_maritime"),
            ship_session_key("vais_smd_marvel/MVI_1478_VIS_frame95_jpg.rf.b.jpg",
                             "vais_smd_marvel"))

    def test_wutdet_neighbour_ids_share_a_session(self):
        a = ship_session_key("wutdet/0000000.jpg", "wutdet")
        b = ship_session_key("wutdet/0000020.jpg", "wutdet")
        self.assertEqual(a, b, "20 ID arayla ornekleneni ayni sahne sayilmali")

    def test_wutdet_distant_ids_are_different_sessions(self):
        self.assertNotEqual(ship_session_key("wutdet/0000000.jpg", "wutdet"),
                            ship_session_key("wutdet/0050000.jpg", "wutdet"))

    def test_ir_thermal_bucketed_by_prefix_and_range(self):
        near = ship_session_key("ir_thermal/1_100_jpg.rf.a.jpg", "ir_thermal")
        far = ship_session_key("ir_thermal/1_5000_jpg.rf.b.jpg", "ir_thermal")
        other_prefix = ship_session_key("ir_thermal/9_100_jpg.rf.c.jpg", "ir_thermal")
        self.assertNotEqual(near, far, "uzak numaralar ayni deve kalmamali")
        self.assertNotEqual(near, other_prefix, "farkli on-ek ayni deve girmemeli")

    def test_ir_thermal_two_prefixes_produce_many_groups(self):
        keys = {ship_session_key(f"ir_thermal/1_{i}_jpg.rf.a.jpg", "ir_thermal")
                for i in range(0, 2000, 50)}
        self.assertGreater(len(keys), 1)

    def test_unknown_naming_falls_back_to_generic(self):
        """Tanimadigi adi sessizce yanlis gruplamamali; genel isleve dusmeli."""
        self.assertEqual(
            ship_session_key("ship_model/patrol_49_jpg.rf.abc.jpg", "ship_model"),
            "ship_model:patrol")


class SplitByGroupTargetTests(unittest.TestCase):
    """Bolme iki sozu birden tutmali: grup asla bolunmemeli (sizinti) VE
    gerceklesen oran hedefe yakin olmali. Eski resplit_by_group() ikincisini
    tutmuyordu: grup SAYISINA gore boldugu icin gerceklesen goruntu orani
    hedef %25 iken %10,6'ya kadar sapmisti (2026-08-12 olcumu).
    """

    class FakeRecord:
        def __init__(self, group):
            self.group = group
            self.split = "train"

    def make(self, sizes):
        return [self.FakeRecord(f"g{i}") for i, n in enumerate(sizes) for _ in range(n)]

    def test_no_group_is_split_across_partitions(self):
        records = self.make([7] * 60)
        split_by_group_targets(records, 0.15, 0.15)
        by_group = {}
        for r in records:
            by_group.setdefault(r.group, set()).add(r.split)
        for group, splits in by_group.items():
            self.assertEqual(len(splits), 1, f"{group} birden fazla bolumde")

    def test_realised_ratio_is_close_to_target(self):
        records = self.make([3] * 200 + [17] * 20)
        counts = split_by_group_targets(records, 0.15, 0.15)
        total = len(records)
        for name in ("val", "test"):
            self.assertAlmostEqual(counts[name] / total, 0.15, delta=0.03,
                                   msg=f"{name} orani hedeften uzak: {counts}")

    def test_oversized_group_stays_in_train(self):
        """Tek bir dev oturum (or. 7300 karelik IP kamera akisi) kotayi
        patlatmamali; train'de kalmali."""
        records = self.make([500] + [2] * 100)
        counts = split_by_group_targets(records, 0.15, 0.15)
        big = [r for r in records if r.group == "g0"]
        self.assertEqual({r.split for r in big}, {"train"})
        self.assertLess(counts["val"], len(records) * 0.25)

    def test_never_produces_empty_val_or_test(self):
        """Her sey tek gruptaysa bile bos bolum uretmemeli -- bos val sessizce
        'AP olculemiyor' demek olurdu."""
        records = self.make([50, 3, 3])
        counts = split_by_group_targets(records, 0.15, 0.15)
        self.assertGreater(counts.get("val", 0), 0)
        self.assertGreater(counts.get("test", 0), 0)

    def test_same_seed_same_split(self):
        a = self.make([5] * 40)
        b = self.make([5] * 40)
        split_by_group_targets(a, 0.15, 0.15, seed=7)
        split_by_group_targets(b, 0.15, 0.15, seed=7)
        self.assertEqual([r.split for r in a], [r.split for r in b])


class ClassMapConsistencyTests(unittest.TestCase):
    """Her kaynagin haritasi yalnizca SHIP veya None uretmeli; baska bir
    deger (orn. yanlislikla baska bir int) bir yazim hatasi olur.
    """

    def test_every_map_only_targets_ship_or_drop(self):
        for source, class_map in CLASS_MAPS.items():
            for name, target in class_map.items():
                self.assertIn(target, (SHIP, None),
                              f"{source}:{name} beklenmeyen hedef: {target}")

    def test_no_source_maps_zero_categories_to_ship(self):
        # Butun kategorileri None yapan bir kaynak hicbir kutu uretmez;
        # bu genelde bir yazim hatasi isaretidir.
        for source, class_map in CLASS_MAPS.items():
            self.assertTrue(any(v == SHIP for v in class_map.values()),
                            f"{source} hicbir kategoriyi SHIP'e eslemiyor")


class ReadShipCocoTests(unittest.TestCase):
    def test_unmapped_category_is_hard_error(self):
        entries = {
            "train/_annotations.coco.json": json.dumps({
                "images": [{"id": 1, "file_name": "x.jpg", "width": 100, "height": 100}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                                "bbox": [1, 1, 5, 5]}],
                "categories": [{"id": 1, "name": "unknown-thing"}],
            }),
            "train/x.jpg": jpeg_bytes(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            with make_zip(Path(tmp) / "d.zip", entries) as arc:
                with self.assertRaises(SystemExit):
                    read_ship_coco(arc, {"ship": SHIP}, "test_source")

    def test_drop_target_keeps_image_without_box(self):
        entries = {
            "train/_annotations.coco.json": json.dumps({
                "images": [{"id": 1, "file_name": "x.jpg", "width": 100, "height": 100}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                                "bbox": [1, 1, 5, 5]}],
                "categories": [{"id": 1, "name": "buoy"}],
            }),
            "train/x.jpg": jpeg_bytes(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            with make_zip(Path(tmp) / "d.zip", entries) as arc:
                recs = read_ship_coco(arc, {"buoy": None}, "test_source")
        self.assertEqual(len(recs), 1, "goruntu kalmali (hard-negative)")
        self.assertEqual(recs[0].anns, [], "kutu uretilmemeli")

    def test_ship_like_category_produces_box(self):
        entries = {
            "train/_annotations.coco.json": json.dumps({
                "images": [{"id": 1, "file_name": "x.jpg", "width": 100, "height": 100}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                                "bbox": [1, 1, 5, 5]}],
                "categories": [{"id": 1, "name": "fishing boat"}],
            }),
            "train/x.jpg": jpeg_bytes(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            with make_zip(Path(tmp) / "d.zip", entries) as arc:
                recs = read_ship_coco(arc, {"fishing boat": SHIP}, "test_source")
        self.assertEqual(len(recs[0].anns), 1)
        self.assertEqual(recs[0].anns[0]["category_id"], SHIP)


class ValidateTests(unittest.TestCase):
    def _record(self, **kw):
        rec = Record(kw.get("source", "s"), "m", kw.get("file_name", "s/a.jpg"),
                     kw.get("width", 100), kw.get("height", 100),
                     kw.get("split", "train"), kw.get("group", "g1"))
        rec.anns = kw.get("anns", [])
        return rec

    def test_clean_records_pass(self):
        rec = self._record(anns=[{"bbox": [10, 10, 20, 20], "category_id": SHIP,
                                  "iscrowd": 0, "ignore": 0}])
        self.assertEqual(validate([rec]), [])

    def test_group_spanning_two_splits_is_caught(self):
        a = self._record(file_name="s/a.jpg", group="same", split="train")
        b = self._record(file_name="s/b.jpg", group="same", split="val")
        problems = validate([a, b])
        self.assertTrue(any("bolum" in p for p in problems))

    def test_out_of_bounds_box_is_caught(self):
        rec = self._record(anns=[{"bbox": [90, 90, 50, 50], "category_id": SHIP,
                                  "iscrowd": 0, "ignore": 0}])
        self.assertTrue(any("sinir disi" in p for p in validate([rec])))

    def test_wrong_category_is_caught(self):
        rec = self._record(anns=[{"bbox": [1, 1, 5, 5], "category_id": 2,
                                  "iscrowd": 0, "ignore": 0}])
        self.assertTrue(any("gecersiz kategori" in p for p in validate([rec])))


class BuildCocoTests(unittest.TestCase):
    def test_single_category_output(self):
        rec = Record("s", "m", "s/a.jpg", 100, 100, "train", "g1")
        rec.anns = [{"bbox": [1, 1, 5, 5], "category_id": SHIP, "iscrowd": 0, "ignore": 0}]
        coco = build_coco([rec], "train")
        self.assertEqual(len(coco["categories"]), 1)
        self.assertEqual(coco["categories"][0]["name"], "ship")
        self.assertEqual(len(coco["images"]), 1)
        self.assertEqual(len(coco["annotations"]), 1)

    def test_other_split_excluded(self):
        rec = Record("s", "m", "s/a.jpg", 100, 100, "val", "g1")
        coco = build_coco([rec], "train")
        self.assertEqual(coco["images"], [])


if __name__ == "__main__":
    unittest.main()
