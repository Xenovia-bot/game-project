"""ship_metrics dogru sayiyi uretmeli: kuantalama kabul kapisi buna bakiyor.

Yanlis bir AP/F1, INT8 kaybini oldugundan kucuk gosterip bozuk bir modelin
karta gitmesine yol acar. Bu yuzden burada bilinen-cevapli sentetik sahneler
kullanilir (mukemmel tespit, hicbir tespit, yarisi yanlis alarm, esik altinda
kalan tespit) -- her birinin dogru sonucu elle hesaplanabilir.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from ship_metrics import (  # noqa: E402
    evaluate_ship,
    format_metrics,
    prf_at_iou50,
    _iou_xywh,
    _match_image,
)


def make_coco(n_images=4, boxes_per_image=2, sources=None):
    """pycocotools COCO nesnesi kurar (diske yazmadan)."""
    from pycocotools.coco import COCO

    images, annotations = [], []
    ann_id = 1
    for image_id in range(1, n_images + 1):
        source = (sources[(image_id - 1) % len(sources)] if sources
                  else "kaynak_a")
        images.append({"id": image_id, "file_name": f"{source}/{image_id}.jpg",
                       "width": 200, "height": 200, "source": source})
        for k in range(boxes_per_image):
            annotations.append({
                "id": ann_id, "image_id": image_id, "category_id": 1,
                "bbox": [10.0 + 50 * k, 10.0, 30.0, 30.0],
                "area": 900.0, "iscrowd": 0,
            })
            ann_id += 1
    coco = COCO()
    coco.dataset = {"images": images, "annotations": annotations,
                    "categories": [{"id": 1, "name": "ship"}]}
    coco.createIndex()
    return coco


def perfect_detections(coco, score=0.9):
    return [{"image_id": a["image_id"], "category_id": 1,
             "bbox": list(a["bbox"]), "score": score}
            for a in coco.dataset["annotations"]]


class IouTests(unittest.TestCase):
    def test_identical_boxes(self):
        self.assertAlmostEqual(_iou_xywh([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)

    def test_disjoint_boxes(self):
        self.assertEqual(_iou_xywh([0, 0, 10, 10], [50, 50, 10, 10]), 0.0)

    def test_half_overlap(self):
        # kesisim 10x20=200, birlesim 400+400-200=600
        self.assertAlmostEqual(_iou_xywh([0, 0, 20, 20], [10, 0, 20, 20]),
                               200 / 600)

    def test_touching_edges_is_not_overlap(self):
        self.assertEqual(_iou_xywh([0, 0, 10, 10], [10, 0, 10, 10]), 0.0)


class MatchTests(unittest.TestCase):
    def test_each_gt_matches_at_most_once(self):
        """Ayni gemiyi iki kez tespit etmek bir TP + bir YANLIS ALARM'dir."""
        gt = [[0, 0, 10, 10]]
        dets = [{"bbox": [0, 0, 10, 10], "score": 0.9},
                {"bbox": [0, 0, 10, 10], "score": 0.8}]
        rows = _match_image(gt, dets)
        self.assertEqual([r[1] for r in rows], [1, 0])

    def test_low_iou_is_false_positive(self):
        gt = [[0, 0, 20, 20]]
        dets = [{"bbox": [15, 0, 20, 20], "score": 0.9}]   # IoU ~0.14
        self.assertEqual([r[1] for r in _match_image(gt, dets)], [0])

    def test_higher_score_detection_claims_the_gt_first(self):
        gt = [[0, 0, 10, 10]]
        dets = [{"bbox": [1, 1, 10, 10], "score": 0.4},
                {"bbox": [0, 0, 10, 10], "score": 0.95}]
        rows = _match_image(gt, dets)
        self.assertEqual(rows[0][0], 0.95)
        self.assertEqual([r[1] for r in rows], [1, 0])

    def test_missed_gt_is_simply_absent(self):
        self.assertEqual(_match_image([[0, 0, 10, 10]], []), [])


class PerfectAndEmptyTests(unittest.TestCase):
    def test_perfect_detections_give_ap_one(self):
        coco = make_coco()
        metrics = evaluate_ship(coco, perfect_detections(coco), deploy_conf=0.15)
        self.assertAlmostEqual(metrics["ap"], 1.0, places=3)
        self.assertAlmostEqual(metrics["ap50"], 1.0, places=3)
        self.assertAlmostEqual(metrics["f1_at"]["f1"], 1.0, places=6)
        self.assertAlmostEqual(metrics["f1_at"]["precision"], 1.0, places=6)
        self.assertAlmostEqual(metrics["f1_at"]["recall"], 1.0, places=6)

    def test_no_detections_gives_zero_not_crash(self):
        """Kuantalama coktugunde model hic tespit uretmez; bu yol calismali."""
        coco = make_coco()
        metrics = evaluate_ship(coco, [], deploy_conf=0.15)
        self.assertEqual(metrics["ap"], 0.0)
        self.assertEqual(metrics["f1_at"]["f1"], 0.0)
        self.assertEqual(metrics["n_detections"], 0)

    def test_detections_below_deploy_conf_do_not_count_for_f1(self):
        """Kart 0.15 esiginde calisiyor; 0.05 skorlu tespit sahada yok demektir."""
        coco = make_coco()
        dets = perfect_detections(coco, score=0.05)
        at_conf, best = prf_at_iou50(coco, dets, deploy_conf=0.15)
        self.assertEqual(at_conf["recall"], 0.0, "0.15 esiginin altinda kalmali")
        self.assertAlmostEqual(best["f1"], 1.0, places=6)
        self.assertLessEqual(best["score"], 0.05)


class PrecisionRecallTests(unittest.TestCase):
    def test_half_false_positives_halve_precision(self):
        coco = make_coco(n_images=4, boxes_per_image=2)   # 8 GT
        dets = perfect_detections(coco)
        for image_id in range(1, 5):                       # 8 yanlis alarm
            for k in range(2):
                dets.append({"image_id": image_id, "category_id": 1,
                             "bbox": [150.0, 150.0 - 40 * k, 20.0, 20.0],
                             "score": 0.9})
        at_conf, _ = prf_at_iou50(coco, dets, deploy_conf=0.15)
        self.assertAlmostEqual(at_conf["precision"], 0.5, places=6)
        self.assertAlmostEqual(at_conf["recall"], 1.0, places=6)
        self.assertAlmostEqual(at_conf["f1"], 2 / 3, places=6)

    def test_half_missed_halves_recall(self):
        coco = make_coco(n_images=4, boxes_per_image=2)   # 8 GT
        dets = perfect_detections(coco)[::2]              # 4 tanesini bul
        at_conf, _ = prf_at_iou50(coco, dets, deploy_conf=0.15)
        self.assertAlmostEqual(at_conf["recall"], 0.5, places=6)
        self.assertAlmostEqual(at_conf["precision"], 1.0, places=6)

    def test_best_f1_is_never_worse_than_fixed_threshold(self):
        coco = make_coco()
        dets = perfect_detections(coco, score=0.42)
        at_conf, best = prf_at_iou50(coco, dets, deploy_conf=0.15)
        self.assertGreaterEqual(best["f1"], at_conf["f1"])


class PerSourceTests(unittest.TestCase):
    def test_per_source_isolates_a_failing_source(self):
        """Asil amac: termal kaynagin AP'si coktugunde genel sayi bunu
        gizleyebilir; kaynak tablosu gostermeli."""
        coco = make_coco(n_images=4, boxes_per_image=2,
                         sources=["ir_thermal", "ship_model"])
        # ship_model goruntuleri (2 ve 4) bulunur; ir_thermal (1 ve 3) bulunmaz
        dets = [d for d in perfect_detections(coco) if d["image_id"] % 2 == 0]
        metrics = evaluate_ship(coco, dets, deploy_conf=0.15)
        self.assertAlmostEqual(metrics["per_source"]["ship_model"]["ap"], 1.0,
                               places=3)
        self.assertLessEqual(metrics["per_source"]["ir_thermal"]["ap"], 0.0)
        self.assertEqual(metrics["per_source"]["ir_thermal"]["images"], 2)

    def test_source_field_absent_is_not_an_error(self):
        coco = make_coco()
        for image in coco.dataset["images"]:
            image.pop("source")
        coco.createIndex()
        metrics = evaluate_ship(coco, perfect_detections(coco))
        self.assertEqual(metrics["per_source"], {})


class FormatTests(unittest.TestCase):
    def test_report_contains_the_numbers_that_matter(self):
        coco = make_coco()
        text = format_metrics(evaluate_ship(coco, perfect_detections(coco)),
                              "INT8")
        for needle in ("INT8", "AP@[.50:.95]", "AP@0.50", "F1", "kaynak"):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
