import unittest

from training.visdrone_eval import (
    GROUP_3,
    PERSON_ONLY,
    VEHICLE_ONLY,
    evaluate_scenarios,
    evaluate_visdrone,
    format_metrics,
    format_scenarios,
)


class FakeCOCO:
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

    def test_others_area_is_not_ignored(self):
        # Category 11 COCO GT'ye alinmaz; o alandaki yuksek skorlu yanlis
        # category-1 tespiti resmi protokolde false-positive kalmalidir.
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


class ClassGroupingTests(unittest.TestCase):
    def test_grouping_removes_van_car_confusion(self):
        # GT 'car'(4), tespit 'van'(5): ince siniflarda ikisi de sifir alir.
        coco = FakeCOCO(
            [{"id": 1, "ignore_regions": []}],
            [annotation(1, 1, 4, [10, 10, 20, 20])],
        )
        detections = [detection(1, 5, [10, 10, 20, 20], 0.9)]

        fine = evaluate_visdrone(coco, detections)
        self.assertAlmostEqual(fine["ap"], 0.0)

        grouped = evaluate_visdrone(coco, detections, group_map=GROUP_3)
        self.assertAlmostEqual(grouped["ap"], 1.0)
        self.assertEqual(list(grouped["per_class"]), [3])  # vehicle
        self.assertEqual(grouped["class_names"][3], "vehicle")

    def test_grouping_keeps_person_and_vehicle_separate(self):
        # 'pedestrian'(1) GT'sine 'car'(4) tespiti hala yanlistir.
        coco = FakeCOCO(
            [{"id": 1, "ignore_regions": []}],
            [annotation(1, 1, 1, [10, 10, 20, 20])],
        )
        grouped = evaluate_visdrone(
            coco, [detection(1, 4, [10, 10, 20, 20], 0.9)], group_map=GROUP_3
        )
        self.assertAlmostEqual(grouped["ap"], 0.0)


class SingleClassTests(unittest.TestCase):
    def _mixed_scene(self):
        # Bir yaya dogru bulunmus; bir arac kacirilmis; bir arac uydurulmus.
        coco = FakeCOCO(
            [{"id": 1, "ignore_regions": []}],
            [
                annotation(1, 1, 1, [10, 10, 20, 20]),    # pedestrian
                annotation(2, 1, 4, [200, 200, 40, 40]),  # car (kacirilir)
            ],
        )
        detections = [
            detection(1, 1, [10, 10, 20, 20], 0.9),    # dogru yaya
            detection(1, 4, [500, 500, 40, 40], 0.8),  # yanlis yerde arac
        ]
        return coco, detections

    def test_person_only_ignores_vehicle_errors(self):
        coco, detections = self._mixed_scene()
        metrics = evaluate_visdrone(coco, detections, group_map=PERSON_ONLY)
        # Arac hatalari tamamen kapsam disi: yaya mukemmel bulundu
        self.assertAlmostEqual(metrics["ap"], 1.0)
        self.assertAlmostEqual(metrics["f1_best"]["f1"], 1.0)
        self.assertEqual(list(metrics["per_class"]), [1])
        # Ad verilmediyse kaynak sinif adlarindan turetilir
        self.assertEqual(metrics["class_names"][1], "pedestrian+people")

    def test_vehicle_only_ignores_person_success(self):
        coco, detections = self._mixed_scene()
        metrics = evaluate_visdrone(coco, detections, group_map=VEHICLE_ONLY)
        # Arac kacirildi ve yanlis yerde bulundu -> sifir
        self.assertAlmostEqual(metrics["ap"], 0.0)
        # Ad verilmediyse eslemedeki tum kaynak siniflardan turetilir
        self.assertTrue(metrics["class_names"][1].startswith("bicycle+car+van"))

    def test_explicit_names_override_derived_ones(self):
        coco, detections = self._mixed_scene()
        metrics = evaluate_visdrone(
            coco, detections, group_map=VEHICLE_ONLY, class_names={1: "vehicle"}
        )
        self.assertEqual(metrics["class_names"][1], "vehicle")

    def test_unmapped_categories_are_dropped_not_merged(self):
        # Yaya GT'sine yapilan arac tespiti person-only'de FP olmamali,
        # cunku o tespit sistemin kapsami disinda kalir.
        coco = FakeCOCO(
            [{"id": 1, "ignore_regions": []}],
            [annotation(1, 1, 1, [10, 10, 20, 20])],
        )
        metrics = evaluate_visdrone(
            coco,
            [
                detection(1, 1, [10, 10, 20, 20], 0.9),
                detection(1, 4, [10, 10, 20, 20], 0.95),
            ],
            group_map=PERSON_ONLY,
        )
        self.assertAlmostEqual(metrics["f1_best"]["precision"], 1.0)


class ScenarioTests(unittest.TestCase):
    def test_all_scenarios_are_evaluated(self):
        coco = FakeCOCO(
            [{"id": 1, "ignore_regions": []}],
            [annotation(1, 1, 4, [10, 10, 20, 20])],
        )
        results = evaluate_scenarios(
            coco, [detection(1, 5, [10, 10, 20, 20], 0.9)]
        )
        self.assertEqual(
            set(results),
            {"10 sinif (resmi)", "2 sinif (hedef)", "3 sinif (ara)",
             "tek sinif: person", "tek sinif: vehicle"},
        )
        # van/car karisikligi: ince siniflarda 0, gruplananlarda 1
        self.assertAlmostEqual(results["10 sinif (resmi)"]["ap"], 0.0)
        self.assertAlmostEqual(results["2 sinif (hedef)"]["ap"], 1.0)
        self.assertAlmostEqual(results["3 sinif (ara)"]["ap"], 1.0)
        self.assertAlmostEqual(results["tek sinif: vehicle"]["ap"], 1.0)
        self.assertEqual(
            results["2 sinif (hedef)"]["class_names"], {1: "person", 2: "vehicle"}
        )

    def test_scenario_table_has_a_row_per_scenario(self):
        coco = FakeCOCO(
            [{"id": 1, "ignore_regions": []}],
            [annotation(1, 1, 4, [10, 10, 20, 20])],
        )
        text = format_scenarios(
            evaluate_scenarios(coco, [detection(1, 4, [10, 10, 20, 20], 0.9)])
        )
        self.assertIn("senaryo", text)
        for name in ("10 sinif (resmi)", "tek sinif: vehicle"):
            self.assertIn(name, text)


class FormatMetricsTests(unittest.TestCase):
    def test_format_includes_per_class_rows_and_f1(self):
        coco = FakeCOCO(
            [{"id": 1, "ignore_regions": []}],
            [annotation(1, 1, 4, [10, 10, 20, 20])],
        )
        metrics = evaluate_visdrone(
            coco, [detection(1, 4, [10, 10, 20, 20], 0.9)]
        )
        text = format_metrics(metrics, "test")
        self.assertIn("test:", text)
        self.assertIn("En iyi F1", text)
        self.assertIn("car", text)

    def test_format_handles_empty_metrics(self):
        coco = FakeCOCO([{"id": 1, "ignore_regions": []}], [])
        text = format_metrics(evaluate_visdrone(coco, []), "bos")
        self.assertIn("bos:", text)


if __name__ == "__main__":
    unittest.main()


class TwoClassEvaluationTests(unittest.TestCase):
    """2 sinifli veri semasiyla degerlendirme."""

    class TwoClassCOCO(FakeCOCO):
        cats = {1: {"id": 1, "name": "person"}, 2: {"id": 2, "name": "vehicle"}}

    def test_class_names_come_from_coco_categories(self):
        coco = self.TwoClassCOCO(
            [{"id": 1, "ignore_regions": []}],
            [annotation(1, 1, 2, [10, 10, 20, 20])],
        )
        metrics = evaluate_visdrone(
            coco, [detection(1, 2, [10, 10, 20, 20], 0.9)], num_classes=2
        )
        self.assertAlmostEqual(metrics["ap"], 1.0)
        self.assertEqual(metrics["class_names"][2], "vehicle")
        self.assertIn("vehicle", format_metrics(metrics, "saha"))

    def test_category_beyond_scheme_is_ignored(self):
        # 2 sinifli semada category_id=4 gecersizdir; GT'de de tespitte de
        # yok sayilmali, sessizce yanlis sinifa yazilmamali.
        coco = self.TwoClassCOCO(
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
