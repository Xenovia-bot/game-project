#!/usr/bin/env python3
"""VisDrone DET toolkit'inin AP@500 protokolunun saf Python karsiligi.

AP hesabi resmi `calcAccuracy.m` ile birebir ayni: ignore GT'ler recall
paydasinda kalir (`rec = tp/max(1,numel(gtMatch))`).

Ek olarak, resmi protokolde bulunmayan bir tamamlayici cikti uretilir:
  * P/R/F1 (IoU 0.50) - sabit esikte ve en iyi F1 noktasinda. Yayinlanmis
    YOLO calismalariyla kiyaslanabilmesi icin bu metrigin recall paydasi
    ignore GT'leri **icermez**; AP'ninkinden farklidir, bilincli.

Sinif semasi COCO kategorilerinden okunur; burada sabit bir sinif listesi
yoktur. Sema `tools/build_dataset.py` tarafindan belirlenir.
"""

from collections import defaultdict

import numpy as np

#: En iyi F1 aramasinda taranan guven esikleri.
SCORE_GRID = np.round(np.arange(0.0, 1.0001, 0.01), 4)


def covered_fraction_xywh(box, regions):
    """Bir xywh kutusunun ignore dikdortgenleri birlesimi icindeki oranini bulur."""
    x, y, w, h = (float(v) for v in box)
    if w <= 0 or h <= 0:
        return 0.0
    clipped = []
    for rx, ry, rw, rh in regions:
        left, top = max(x, rx), max(y, ry)
        right, bottom = min(x + w, rx + rw), min(y + h, ry + rh)
        if right > left and bottom > top:
            clipped.append((left, top, right, bottom))
    if not clipped:
        return 0.0

    # Kesisen dikdortgenlerin birlesim alanini x-sweep ile cift saymadan hesapla.
    xs = sorted({edge for rect in clipped for edge in (rect[0], rect[2])})
    covered = 0.0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals = sorted(
            (top, bottom)
            for x1, top, x2, bottom in clipped
            if x1 < right and x2 > left
        )
        union_y = 0.0
        if intervals:
            start, end = intervals[0]
            for next_start, next_end in intervals[1:]:
                if next_start > end:
                    union_y += end - start
                    start, end = next_start, next_end
                else:
                    end = max(end, next_end)
            union_y += end - start
        covered += (right - left) * union_y
    return covered / (w * h)


def _overlap(dt, gt, gt_ignore):
    dx, dy, dw, dh = dt
    gx, gy, gw, gh = gt
    iw = min(dx + dw, gx + gw) - max(dx, gx)
    ih = min(dy + dh, gy + gh) - max(dy, gy)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    if gt_ignore:
        return inter / max(dw * dh, 1e-12)
    union = dw * dh + gw * gh - inter
    return inter / max(union, 1e-12)


def _eval_image(gt_rows, detections, threshold):
    """VisDrone toolkit evalRes.m ile ayni eslestirme sirasi."""
    # Normal GT once gelir; ignore GT en sona gelir ve birden cok kez eslesebilir.
    gt_rows = sorted(gt_rows, key=lambda row: bool(row["ignore"]))
    gt_state = [-1 if row["ignore"] else 0 for row in gt_rows]
    detections = sorted(detections, key=lambda row: -row["score"])
    dt_matches = []

    for detection in detections:
        best_overlap = threshold
        best_index = None
        best_match = 0
        for index, (ground_truth, state) in enumerate(zip(gt_rows, gt_state)):
            if state == 1:
                continue
            if best_match != 0 and state == -1:
                break
            overlap = _overlap(
                detection["bbox"], ground_truth["bbox"], state == -1
            )
            if overlap < best_overlap:
                continue
            best_overlap = overlap
            best_index = index
            best_match = 1 if state == 0 else -1
        if best_index is not None and best_match == 1:
            gt_state[best_index] = 1
        dt_matches.append((detection["score"], best_match))
    return gt_state, dt_matches


def _voc_ap(recall, precision):
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(precision.size - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    changes = np.where(recall[1:] != recall[:-1])[0] + 1
    return float(
        np.sum((recall[changes] - recall[changes - 1]) * precision[changes])
    )


def prepare_detections(coco, detections, max_dets=500):
    """Global ignore filtresi ve goruntu basina global top-500 uygular."""
    grouped = defaultdict(list)
    for detection in detections:
        image_id = int(detection["image_id"])
        regions = coco.imgs[image_id].get("ignore_regions", ())
        if covered_fraction_xywh(detection["bbox"], regions) >= 0.5:
            continue
        grouped[image_id].append(detection)

    prepared = {}
    for image_id in coco.getImgIds():
        rows = grouped.get(image_id, ())
        prepared[image_id] = sorted(
            rows,
            key=lambda row: (
                -float(row["score"]),
                int(row["category_id"]),
                tuple(float(v) for v in row["bbox"]),
            ),
        )[:max_dets]
    return prepared


def _f1_curve(dt_rows, real_gt_count):
    """Skor esigi izgarasi uzerinde precision/recall/F1 dizileri uretir.

    `dt_rows` skora gore azalan sirali (skor, eslesme) ciftleridir; eslesme
    1=TP, 0=FP, -1=ignore GT ile eslesti (ne TP ne FP sayilir).

    Recall paydasi **ignore olmayan** GT sayisidir. AP'nin paydasindan
    (resmi toolkit ignore'lari da sayar) bilincli olarak farklidir: bu metrik
    yayinlanmis YOLO sonuclariyla kiyaslanabilsin diye COCO/Ultralytics
    kuralini izler.
    """
    zeros = np.zeros_like(SCORE_GRID)
    if real_gt_count <= 0:
        return zeros, zeros, zeros

    scores = np.asarray([row[0] for row in dt_rows], dtype=np.float64)
    matches = np.asarray([row[1] for row in dt_rows], dtype=np.int8)
    # scores azalan sirali -> -scores artan; "skor >= t" sayisi searchsorted ile
    kept = np.searchsorted(-scores, -SCORE_GRID, side="right")
    tp_cum = np.concatenate(([0.0], np.cumsum(matches == 1, dtype=np.float64)))
    fp_cum = np.concatenate(([0.0], np.cumsum(matches == 0, dtype=np.float64)))
    tp, fp = tp_cum[kept], fp_cum[kept]

    precision = tp / np.maximum(1e-12, tp + fp)
    recall = tp / float(real_gt_count)
    f1 = 2 * precision * recall / np.maximum(1e-12, precision + recall)
    return precision, recall, f1


def _at_grid(index, precision, recall, f1):
    return {
        "precision": float(precision[index]),
        "recall": float(recall[index]),
        "f1": float(f1[index]),
        "score": float(SCORE_GRID[index]),
    }


def names_from_coco(coco):
    """Sinif adlarini COCO kategorilerinden okur.

    Boylece veri semasi degistiginde tablolar kendiliginden dogru kalir;
    ad listesi ikinci bir yerde tekrarlanmaz.
    """
    cats = getattr(coco, "cats", None) or {}
    return {int(k): str(v.get("name", k)) for k, v in cats.items()}


def evaluate_visdrone(coco, detections, max_dets=500, score_thr=0.30,
                      num_classes=None):
    """AP@[.50:.95], AP50, AP75 ve tamamlayici P/R/F1 degerlerini hesaplar.

    `num_classes` verilmezse COCO kategori sayisindan okunur. Bu araligin
    disindaki kategori kimlikleri hem GT'den hem tespitlerden dusurulur:
    eski semali bir dosya sessizce yanlis sinifa yazilmasin.
    """
    prepared = prepare_detections(coco, detections, max_dets=max_dets)
    if num_classes is None:
        cats = getattr(coco, "cats", None)
        if not cats:
            raise ValueError(
                "num_classes verilmedi ve COCO nesnesinde kategori yok; "
                "sinif semasi belirlenemiyor."
            )
        num_classes = len(cats)

    def mapped(category_id):
        return category_id if 1 <= category_id <= num_classes else None

    gt_by_image_class = defaultdict(list)
    dt_by_image_class = defaultdict(list)
    real_gt_count = defaultdict(int)
    available_classes = set()

    for image_id in coco.getImgIds():
        regions = coco.imgs[image_id].get("ignore_regions", ())
        for annotation in coco.imgToAnns.get(image_id, ()):
            category_id = mapped(int(annotation["category_id"]))
            if category_id is None:
                continue
            if covered_fraction_xywh(annotation["bbox"], regions) >= 0.5:
                continue
            ignore = bool(
                annotation.get("ignore", 0) or annotation.get("iscrowd", 0)
            )
            gt_by_image_class[(image_id, category_id)].append({
                "bbox": [float(v) for v in annotation["bbox"]],
                "ignore": ignore,
            })
            if not ignore:
                real_gt_count[category_id] += 1
            available_classes.add(category_id)
        # Tespitleri sinifa gore bir kez ayir: esik dongusunde tekrarlanmasin.
        for row in prepared[image_id]:
            category_id = mapped(int(row["category_id"]))
            if category_id is not None:
                dt_by_image_class[(image_id, category_id)].append(row)

    thresholds = np.arange(0.50, 0.951, 0.05)
    image_ids = coco.getImgIds()
    per_class = {}
    iou50_matches = {}
    for category_id in sorted(available_classes):
        aps = []
        for index, threshold in enumerate(thresholds):
            gt_matches = []
            dt_matches = []
            for image_id in image_ids:
                gt_state, image_dt = _eval_image(
                    gt_by_image_class.get((image_id, category_id), ()),
                    dt_by_image_class.get((image_id, category_id), ()),
                    threshold,
                )
                gt_matches.extend(gt_state)
                dt_matches.extend(image_dt)

            dt_matches.sort(key=lambda row: -row[0])
            if index == 0:  # IoU 0.50 -> F1 egrisi buradan cikar
                iou50_matches[category_id] = dt_matches
            matches = np.asarray([row[1] for row in dt_matches], dtype=np.int8)
            tp = np.cumsum(matches == 1, dtype=np.float64)
            fp = np.cumsum(matches == 0, dtype=np.float64)
            recall = tp / max(1, len(gt_matches))
            precision = tp / np.maximum(1.0, tp + fp)
            aps.append(_voc_ap(recall, precision))
        per_class[category_id] = aps

    names = names_from_coco(coco)

    empty = {
        "ap": 0.0, "ap50": 0.0, "ap75": 0.0, "per_class": {},
        "per_class_ap50": {}, "class_names": names,
        "f1_best": None, "f1_at": None, "per_class_f1_best": {},
    }
    if not per_class:
        return empty

    curves = {
        category_id: _f1_curve(
            iou50_matches[category_id], real_gt_count[category_id]
        )
        for category_id in sorted(per_class)
    }
    precision_matrix = np.asarray([c[0] for c in curves.values()])
    recall_matrix = np.asarray([c[1] for c in curves.values()])
    f1_matrix = np.asarray([c[2] for c in curves.values()])
    macro = (
        precision_matrix.mean(axis=0),
        recall_matrix.mean(axis=0),
        f1_matrix.mean(axis=0),
    )
    best_index = int(np.argmax(macro[2]))
    fixed_index = int(np.argmin(np.abs(SCORE_GRID - score_thr)))

    matrix = np.asarray(list(per_class.values()), dtype=np.float64)
    return {
        "ap": float(matrix.mean()),
        "ap50": float(matrix[:, 0].mean()),
        "ap75": float(matrix[:, 5].mean()),
        "per_class": {
            category_id: float(np.mean(values))
            for category_id, values in per_class.items()
        },
        "per_class_ap50": {
            category_id: float(values[0])
            for category_id, values in per_class.items()
        },
        "class_names": names,
        "f1_best": _at_grid(best_index, *macro),
        "f1_at": _at_grid(fixed_index, *macro),
        "per_class_f1_best": {
            category_id: _at_grid(int(np.argmax(curve[2])), *curve)
            for category_id, curve in curves.items()
        },
    }


def format_metrics(metrics, title="VisDrone"):
    """Degerlendirme ciktisini insan okunur tek bir metne cevirir."""
    names = metrics.get("class_names", {})
    lines = [
        f"{title}: AP@[.50:.95]={metrics['ap']:.4f}  "
        f"AP@0.50={metrics['ap50']:.4f}  AP@0.75={metrics['ap75']:.4f}",
    ]
    best, fixed = metrics.get("f1_best"), metrics.get("f1_at")
    if best and fixed:
        lines.append(
            f"  En iyi F1={best['f1']:.4f} (P={best['precision']:.4f} "
            f"R={best['recall']:.4f} @conf={best['score']:.2f})"
        )
        lines.append(
            f"  conf={fixed['score']:.2f}: F1={fixed['f1']:.4f} "
            f"P={fixed['precision']:.4f} R={fixed['recall']:.4f}"
        )
    if metrics.get("per_class"):
        lines.append(f"  {'sinif':<18}{'AP':>8}{'AP50':>8}{'F1':>8}")
        for category_id in sorted(metrics["per_class"]):
            f1 = metrics["per_class_f1_best"].get(category_id, {})
            lines.append(
                f"  {names.get(category_id, category_id):<18}"
                f"{metrics['per_class'][category_id]:>8.4f}"
                f"{metrics['per_class_ap50'][category_id]:>8.4f}"
                f"{f1.get('f1', float('nan')):>8.4f}"
            )
    return "\n".join(lines)
