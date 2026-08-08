import unittest

import numpy as np

from tools.tiling import merge_tiled, nms_xyxy, offset_boxes, tile_rects


class TileRectTests(unittest.TestCase):
    def test_tiles_cover_whole_image(self):
        width, height = 1920, 1080
        rects = tile_rects(width, height, 2, 2, overlap=0.2, include_full=False)
        self.assertEqual(len(rects), 4)
        covered = np.zeros((height, width), dtype=bool)
        for x0, y0, x1, y1 in rects:
            covered[y0:y1, x0:x1] = True
        self.assertTrue(covered.all(), "parcalar kareyi tam kaplamiyor")

    def test_tiles_stay_inside_image(self):
        for width, height in ((1920, 1080), (640, 480), (2000, 1500)):
            for x0, y0, x1, y1 in tile_rects(width, height, 2, 2):
                self.assertGreaterEqual(x0, 0)
                self.assertGreaterEqual(y0, 0)
                self.assertLessEqual(x1, width)
                self.assertLessEqual(y1, height)
                self.assertGreater(x1, x0)
                self.assertGreater(y1, y0)

    def test_overlap_region_is_wide_enough_for_small_objects(self):
        # 2x2 / %20 ortusmede yatay ortusme payi VisDrone nesnelerinden genis
        # olmali, yoksa parca sinirindaki nesne ikiye bolunur.
        rects = tile_rects(1920, 1080, 2, 2, overlap=0.2, include_full=False)
        left, right = rects[0], rects[1]
        self.assertGreater(left[2] - right[0], 100)

    def test_include_full_appends_whole_frame(self):
        rects = tile_rects(1920, 1080, 2, 2, include_full=True)
        self.assertEqual(len(rects), 5)
        self.assertEqual(rects[-1], (0, 0, 1920, 1080))

    def test_single_tile_is_the_full_frame(self):
        self.assertEqual(
            tile_rects(800, 600, 1, 1, include_full=True), [(0, 0, 800, 600)]
        )

    def test_rejects_invalid_arguments(self):
        with self.assertRaises(ValueError):
            tile_rects(0, 100)
        with self.assertRaises(ValueError):
            tile_rects(100, 100, cols=0)
        with self.assertRaises(ValueError):
            tile_rects(100, 100, overlap=1.0)


class NmsTests(unittest.TestCase):
    def test_duplicate_boxes_collapse(self):
        boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]]
        keep = nms_xyxy(boxes, [0.9, 0.8, 0.7], iou_thr=0.45)
        self.assertEqual(sorted(keep.tolist()), [0, 2])

    def test_empty_input(self):
        self.assertEqual(nms_xyxy(np.zeros((0, 4)), []).size, 0)


class MergeTests(unittest.TestCase):
    def test_boxes_are_shifted_into_frame_coordinates(self):
        shifted = offset_boxes([[10, 20, 30, 40]], (100, 200, 740, 840))
        np.testing.assert_allclose(shifted, [[110, 220, 130, 240]])

    def test_same_object_seen_in_two_tiles_is_merged_once(self):
        # Ortusme bolgesindeki nesne: sol parcada x=900, sag parcada x=60
        # yerel koordinatta; ikisi de tam karede x=900 civarina duser.
        tile_results = [
            ((0, 0, 1056, 594), [[900, 100, 940, 140]], [0.90], [3]),
            ((864, 0, 1920, 594), [[36, 100, 76, 140]], [0.85], [3]),
        ]
        boxes, scores, classes = merge_tiled(tile_results, iou_thr=0.45)
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(scores[0], 0.90)
        np.testing.assert_allclose(boxes[0], [900, 100, 940, 140])
        self.assertEqual(classes[0], 3)

    def test_nms_is_class_aware(self):
        # Ayni konumda farkli sinif: ikisi de korunmali
        tile_results = [
            ((0, 0, 640, 640), [[10, 10, 50, 50], [10, 10, 50, 50]],
             [0.9, 0.8], [1, 3]),
        ]
        boxes, _, classes = merge_tiled(tile_results)
        self.assertEqual(len(boxes), 2)
        self.assertEqual(sorted(classes.tolist()), [1, 3])

    def test_max_dets_truncates_by_score(self):
        boxes = [[i * 100, 0, i * 100 + 20, 20] for i in range(10)]
        scores = [i / 10.0 for i in range(10)]
        tile_results = [((0, 0, 2000, 100), boxes, scores, [1] * 10)]
        _, kept_scores, _ = merge_tiled(tile_results, max_dets=3)
        np.testing.assert_allclose(kept_scores, [0.9, 0.8, 0.7])

    def test_empty_tiles_return_empty_arrays(self):
        boxes, scores, classes = merge_tiled(
            [((0, 0, 640, 640), np.zeros((0, 4)), [], [])]
        )
        self.assertEqual(boxes.shape, (0, 4))
        self.assertEqual(scores.size, 0)
        self.assertEqual(classes.size, 0)


if __name__ == "__main__":
    unittest.main()
