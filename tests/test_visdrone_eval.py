import unittest

from training.visdrone_eval import evaluate_visdrone, format_metrics


class FakeCOCO:
    """Uretimdeki gibi 2 sinifli sema tasir: 1=land_vehicle, 2=sea_vehicle.

    Sinif adlari ve sinif sayisi buradan okunur; degerlendiricide sabit bir
    sinif listesi yoktur.
    """

    cats = {1: {"id": 1, "name": "land_vehicle"},
            2: {"id": 2, "name": "sea_vehicle"}}

    def __init__(self, images, annotations):
        self.imgs = {image["id"]: image for image in images}
        self.imgToAnns = {image["id"]: [] for image in images}
        for annotation in annotations:
            self.imgToAnns[annotation["image_id"]].append(annotation)

    def getImgIds(self):
        return sorted(self.imgs)


def annotation(ann_id, image_id, category_id, bbox, **extra):
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        **extra,
    }


def detection(image_id, category_id, bbox, score):
    return {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "score": score,
    }


class VisDroneEvaluationTests(unittest.TestCase):
    def test_perfect_detection_has_unit_ap(self):
        coco = FakeCOCO(
            [{"id": 1, "ignore_regions": []}],
            [annotation(1, 1, 1, [10, 10, 20, 20])],
        )
        metrics = evaluate_visdrone(
            coco, [detection(1, 1, [10, 10, 20, 20], 0.9)]
        )
        self.assertAlmostEqual(metrics["ap"], 1.0)
        self.assertAlmostEqual(metrics["ap50"], 1.0)
        self.assertAlmostEqual(metrics["ap75"], 1.0)

    def test_global_ignore_detection_is_not_false_positive(self):
        coco = FakeCOCO(
            [{"id": 1, "ignore_regions": [[0, 0, 40, 40]]}],
            [annotation(1, 1, 1, [50, 50, 20, 20])],
        )
        metrics = evaluate_visdrone(
            coco,
            [
                detection(1, 1, [5, 5, 10, 10], 0.99),
                detection(1, 1, [50, 50, 20, 20], 0.90),
            ],
        )
        self.assertAlmostEqual(metrics["ap"], 1.0)

    def test_detection_without_gt_stays_false_positive(self):
        # Hedef disi siniflar (insan, bisiklet, others...) GT'ye alinmaz.
        # O alanlarda uretilen yuksek skorlu tespit resmi protokolde
        # false-positive kalmalidir -- sessizce ignore edilmemeli.
        coco = FakeCOCO(
            [{"id": 1, "ignore_regions": []}],
            [annotation(1, 1, 1, [50, 50, 20, 20])],
        )
        metrics = evaluate_visdrone(
            coco,
            [
                detection(1, 1, [5, 5, 10, 10], 0.99),
                detection(1, 1, [50, 50, 20, 20], 0.90),
            ],
        )
        self.assertAlmostEqual(metrics["ap"], 0.5)

    def test_top_500_is_global_across_classes(self):
        coco = FakeCOCO(
            [{"id": 1, "ignore_regions": []}],
            [annotation(1, 1, 1, [50, 50, 20, 20])],
        )
        detections = [
            detection(1, 2, [0, 0, 1, 1], 1.0 - index * 1e-4)
            for index in range(500)
        ]
        detections.append(detection(1, 1, [50, 50, 20, 20], 0.1))
        metrics = evaluate_visdrone(coco, detections)
        self.assertAlmostEqual(metrics["ap"], 0.0)

    def test_class_specific_ignore_uses_detection_area(self):
        coco = FakeCOCO(
            [{"id": 1, "ignore_regions": []}],
            [
                annotation(1, 1, 1, [0, 0, 100, 100], iscrowd=1, ignore=1),
                annotation(2, 1, 1, [150, 150, 20, 20]),
            ],
        )
        metrics = evaluate_visdrone(
            coco,
            [
                detection(1, 1, [10, 10, 10, 10], 0.99),
                detection(1, 1, [150, 150, 20, 20], 0.90),
            ],
        )
        # Toolkit, ignore GT'leri recall paydasinda tutar; iki GT'den yalniz
        # normal olan TP oldugu icin maksimum recall ve AP 0.5 olur.
        self.assertAlmostEqual(metrics["ap"], 0.5)


class F1MetricTests(unittest.TestCase):
    def test_perfect_detection_has_unit_f1(self):
        coco = FakeCOCO(
            [{"id": 1, "ignore_regions": []}],
            [annotation(1, 1, 1, [10, 10, 20, 20])],
        )
        metrics = evaluate_visdrone(
            coco, [detection(1, 1, [10, 10, 20, 20], 0.9)]
        )
        self.assertAlmostEqual(metrics["f1_best"]["f1"], 1.0)
        self.assertAlmostEqual(metrics["f1_best"]["precision"], 1.0)
        self.assertAlmostEqual(metrics["f1_best"]["recall"], 1.0)
        # 0.9 skorlu tespit conf=0.30 esigini gecer
        self.assertAlmostEqual(metrics["f1_at"]["f1"], 1.0)

    def test_f1_recall_denominator_excludes_ignore_gt(self):
        # AP paydasi ignore GT'yi sayar (resmi toolkit), F1 paydasi saymaz.
        coco = FakeCOCO(
            [{"id": 1, "ignore_regions": []}],
            [
                annotation(1, 1, 1, [0, 0, 100, 100], iscrowd=1, ignore=1),
                annotation(2, 1, 1, [150, 150, 20, 20]),
            ],
        )
        metrics = evaluate_visdrone(
            coco, [detection(1, 1, [150, 150, 20, 20], 0.9)]
        )
        self.assertAlmostEqual(metrics["ap"], 0.5)
        self.assertAlmostEqual(metrics["f1_best"]["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1_best"]["f1"], 1.0)

    def test_low_score_detection_missed_at_fixed_threshold(self):
        coco = FakeCOCO(
            [{"id": 1, "ignore_regions": []}],
            [annotation(1, 1, 1, [10, 10, 20, 20])],
        )
        metrics = evaluate_visdrone(
            coco, [detection(1, 1, [10, 10, 20, 20], 0.10)], score_thr=0.30
        )
        # En iyi F1 dusuk esikte 1.0, ama dagitim esiginde tespit elenir
        self.assertAlmostEqual(metrics["f1_best"]["f1"], 1.0)
        self.assertAlmostEqual(metrics["f1_at"]["f1"], 0.0)


class ClassSchemeTests(unittest.TestCase):
    """Sinif semasi COCO kategorilerinden okunur, kodda sabit degildir."""

    def test_class_names_come_from_coco_categories(self):
        coco = FakeCOCO(
            [{"id": 1, "ignore_regions": []}],
            [annotation(1, 1, 2, [10, 10, 20, 20])],
        )
        metrics = evaluate_visdrone(
            coco, [detection(1, 2, [10, 10, 20, 20], 0.9)]
        )
        self.assertAlmostEqual(metrics["ap"], 1.0)
        self.assertEqual(metrics["class_names"][2], "sea_vehicle")
        self.assertIn("sea_vehicle", format_metrics(metrics, "saha"))

    def test_num_classes_defaults_to_category_count(self):
        coco = FakeCOCO(
            [{"id": 1, "ignore_regions": []}],
            [annotation(1, 1, 2, [10, 10, 20, 20])],
        )
        auto = evaluate_visdrone(coco, [detection(1, 2, [10, 10, 20, 20], 0.9)])
        explicit = evaluate_visdrone(
            coco, [detection(1, 2, [10, 10, 20, 20], 0.9)], num_classes=2
        )
        self.assertEqual(auto["ap"], explicit["ap"])

    def test_category_beyond_scheme_is_ignored(self):
        # 2 sinifli semada category_id=4 gecersizdir; GT'de de tespitte de
        # yok sayilmali, sessizce yanlis sinifa yazilmamali.
        coco = FakeCOCO(
            [{"id": 1, "ignore_regions": []}],
            [annotation(1, 1, 1, [10, 10, 20, 20]),
             annotation(2, 1, 4, [90, 90, 20, 20])],
        )
        metrics = evaluate_visdrone(
            coco,
            [detection(1, 1, [10, 10, 20, 20], 0.9),
             detection(1, 4, [90, 90, 20, 20], 0.9)],
            num_classes=2,
        )
        self.assertEqual(list(metrics["per_class"]), [1])
        self.assertAlmostEqual(metrics["ap"], 1.0)

    def test_missing_scheme_fails_loudly(self):
        # Sema ne kategorilerden ne argumandan gelirse sessizce 0 sinifla
        # devam etmek yerine hata vermeli.
        class NoCats(FakeCOCO):
            cats = {}

        coco = NoCats(
            [{"id": 1, "ignore_regions": []}],
            [annotation(1, 1, 1, [10, 10, 20, 20])],
        )
        with self.assertRaises(ValueError):
            evaluate_visdrone(coco, [detection(1, 1, [10, 10, 20, 20], 0.9)])


class FormatMetricsTests(unittest.TestCase):
    def test_format_includes_per_class_rows_and_f1(self):
        coco = FakeCOCO(
            [{"id": 1, "ignore_regions": []}],
            [annotation(1, 1, 2, [10, 10, 20, 20])],
        )
        metrics = evaluate_visdrone(
            coco, [detection(1, 2, [10, 10, 20, 20], 0.9)]
        )
        text = format_metrics(metrics, "test")
        self.assertIn("test:", text)
        self.assertIn("En iyi F1", text)
        self.assertIn("sea_vehicle", text)

    def test_format_handles_empty_metrics(self):
        coco = FakeCOCO([{"id": 1, "ignore_regions": []}], [])
        text = format_metrics(evaluate_visdrone(coco, []), "bos")
        self.assertIn("bos:", text)


if __name__ == "__main__":
    unittest.main()
