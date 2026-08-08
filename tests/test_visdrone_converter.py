import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.visdrone2coco import convert


class VisDroneConverterTests(unittest.TestCase):
    def test_ignore_others_and_score_zero_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            annotations = root / "annotations"
            images.mkdir()
            annotations.mkdir()
            Image.new("RGB", (200, 200)).save(images / "sample.jpg")
            (annotations / "sample.txt").write_text(
                "\n".join([
                    "0,0,50,50,0,0,0,0",       # global ignored-region
                    "10,10,10,10,1,1,0,0",     # global ignore icinde: cikar
                    "60,0,10,10,1,11,0,0",     # others: hedef degil
                    "80,0,10,10,0,2,0,0",      # class-specific ignore
                    "100,0,10,10,1,3,0,0",     # normal hedef
                ])
            )
            output = root / "instances.json"

            convert(images, annotations, output)
            data = json.loads(output.read_text())

            self.assertEqual(data["images"][0]["ignore_regions"], [[0, 0, 50, 50]])
            self.assertEqual(len(data["annotations"]), 2)
            by_category = {
                annotation["category_id"]: annotation
                for annotation in data["annotations"]
            }
            self.assertEqual(set(by_category), {2, 3})
            self.assertEqual(by_category[2]["iscrowd"], 1)
            self.assertEqual(by_category[2]["ignore"], 1)
            self.assertEqual(by_category[3]["iscrowd"], 0)


if __name__ == "__main__":
    unittest.main()


class TwoClassSchemeTests(unittest.TestCase):
    def test_two_class_scheme_merges_targets_and_keeps_ignore_logic(self):
        import json
        import tempfile
        from pathlib import Path

        from tools.visdrone2coco import convert

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, annos = root / "images", root / "annotations"
            images.mkdir(), annos.mkdir()
            Image.new("RGB", (200, 200)).save(images / "a.jpg")
            (annos / "a.txt").write_text(
                "10,10,20,20,1,1,0,0\n"    # pedestrian -> person
                "50,50,20,20,1,4,0,0\n"    # car        -> vehicle
                "80,80,20,20,1,10,0,0\n"   # motor      -> vehicle
                "120,120,20,20,0,4,0,0\n"  # score=0    -> vehicle crowd
                "150,0,20,20,1,11,0,0\n"   # others     -> atilir
            )
            out = root / "two.json"
            convert(images, annos, out, classes="2")
            data = json.loads(out.read_text())

        self.assertEqual(
            [c["name"] for c in data["categories"]], ["person", "land_vehicle"]
        )
        targets = [a for a in data["annotations"] if not a.get("iscrowd")]
        self.assertEqual(sorted(a["category_id"] for a in targets), [1, 2, 2])
        crowd = [a for a in data["annotations"] if a.get("iscrowd")]
        self.assertEqual([a["category_id"] for a in crowd], [2])
