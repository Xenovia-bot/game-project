#!/usr/bin/env python3
"""Gemi tespitinin son metrikleri: COCO mAP + IoU 0.50'de P/R/F1.

Neden iki ayri motor
--------------------
* **AP** pycocotools ile hesaplanir. Sebep tek: egitim sirasinda YOLOX'un
  COCOEvaluator'u de ayni tanimi kullanir. Boylece Kaggle'daki AP, VM'deki
  float AP ve INT8 AP **ayni sayinin** uc olcumudur; kuantalama kaybi
  ("AP kaybi 0.02'yi gecmesin") ancak boyle anlamli olur. Kendi AP'mizi
  yazsaydik bu uc sayi kiyaslanamazdi.
* **P/R/F1** pycocotools'ta yok. Kart sabit bir guven esiginde calisiyor
  (deploy/src/main.cpp varsayilani 0.15) ve saha sorusu "o esikte kac gemiyi
  kaciriyoruz, kac yanlis alarm veriyoruz" -- bu AP degil, P/R/F1 sorusudur.
  Eslestirme mantigi bu projenin eski `visdrone_eval.py`'sinden devralindi
  (VisDrone'a ozel ignore-region dallari cikarildi; gemi verisinde
  ignore_regions yok, iscrowd hep 0).

AP'nin ve F1'in recall paydasi ayni degildir; ikisi de kendi kuralinca
dogrudur. F1, yayinlanmis YOLO calismalariyla kiyaslanabilsin diye
COCO/Ultralytics kuralini izler: payda gercek GT sayisi.

Kullanim
--------
    from ship_metrics import evaluate_ship, format_metrics
    metrics = evaluate_ship(coco_gt, detections, deploy_conf=0.15)
    print(format_metrics(metrics, "INT8"))
"""

import contextlib
import io
from collections import defaultdict

import numpy as np

#: En iyi F1 aramasinda taranan guven esikleri.
SCORE_GRID = np.round(np.arange(0.0, 1.0001, 0.01), 4)

#: COCO'nun standart maxDets'i. YOLOX COCOEvaluator da bunu kullanir; en
#: kalabalik gemi karesinde 36 kutu olculdu, yani sinir baglayici degil.
MAX_DETS = 100


def _iou_xywh(box_a, box_b):
    ax, ay, aw, ah = (float(v) for v in box_a)
    bx, by, bw, bh = (float(v) for v in box_b)
    iw = min(ax + aw, bx + bw) - max(ax, bx)
    ih = min(ay + ah, by + bh) - max(ay, by)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / max(aw * ah + bw * bh - inter, 1e-12)


def _match_image(gt_boxes, detections, iou_thr=0.50):
    """Bir goruntude tespitleri GT'lere aclikla eslestirir.

    Tespitler skora gore azalan sirada islenir ve her GT en fazla bir kez
    eslesir (COCO ve VisDrone toolkit'inin ortak kurali). Donus: tespit
    basina (skor, 1=TP / 0=FP) ciftleri, skor sirasinda.
    """
    used = [False] * len(gt_boxes)
    rows = []
    for detection in sorted(detections, key=lambda row: -float(row["score"])):
        best_iou = iou_thr
        best_index = None
        for index, gt_box in enumerate(gt_boxes):
            if used[index]:
                continue
            iou = _iou_xywh(detection["bbox"], gt_box)
            if iou >= best_iou:
                best_iou = iou
                best_index = index
        if best_index is None:
            rows.append((float(detection["score"]), 0))
        else:
            used[best_index] = True
            rows.append((float(detection["score"]), 1))
    return rows


def _prf_curves(rows, gt_count):
    """SCORE_GRID uzerinde precision/recall/F1 dizileri.

    `rows` skora gore azalan sirali (skor, eslesme) ciftleridir.
    """
    zeros = np.zeros_like(SCORE_GRID)
    if gt_count <= 0 or not rows:
        return zeros, zeros, zeros

    scores = np.asarray([row[0] for row in rows], dtype=np.float64)
    matches = np.asarray([row[1] for row in rows], dtype=np.int8)
    # scores azalan -> -scores artan; "skor >= t" olan tespit sayisi:
    kept = np.searchsorted(-scores, -SCORE_GRID, side="right")
    tp_cum = np.concatenate(([0.0], np.cumsum(matches == 1, dtype=np.float64)))
    fp_cum = np.concatenate(([0.0], np.cumsum(matches == 0, dtype=np.float64)))
    tp, fp = tp_cum[kept], fp_cum[kept]

    precision = tp / np.maximum(1e-12, tp + fp)
    recall = tp / float(gt_count)
    f1 = 2 * precision * recall / np.maximum(1e-12, precision + recall)
    return precision, recall, f1


def _at(index, precision, recall, f1):
    return {
        "precision": float(precision[index]),
        "recall": float(recall[index]),
        "f1": float(f1[index]),
        "score": float(SCORE_GRID[index]),
    }


def coco_ap(coco_gt, detections, image_ids=None, max_dets=MAX_DETS):
    """(AP@[.50:.95], AP@0.50, AP@0.75) -- pycocotools, egitimle ayni tanim."""
    from pycocotools.cocoeval import COCOeval

    if not detections:
        return 0.0, 0.0, 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(list(detections))
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        if image_ids is not None:
            evaluator.params.imgIds = list(image_ids)
        evaluator.params.maxDets = [1, 10, max_dets]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return (float(evaluator.stats[0]), float(evaluator.stats[1]),
            float(evaluator.stats[2]))


def prf_at_iou50(coco_gt, detections, deploy_conf, image_ids=None,
                 max_dets=MAX_DETS):
    """Sabit esikte ve en iyi F1 noktasinda P/R/F1.

    Sinif bazinda hesaplanip makro ortalanir; tek sinifli ("ship") veride
    bu dogrudan o sinifin degeridir.
    """
    wanted = set(coco_gt.getImgIds() if image_ids is None else image_ids)
    categories = sorted(getattr(coco_gt, "cats", {}) or {1: None})

    detections_by_key = defaultdict(list)
    for detection in detections:
        image_id = int(detection["image_id"])
        if image_id in wanted:
            detections_by_key[(image_id, int(detection["category_id"]))].append(
                detection)

    gt_by_key = defaultdict(list)
    gt_count = defaultdict(int)
    for image_id in wanted:
        for annotation in coco_gt.imgToAnns.get(image_id, ()):
            if annotation.get("iscrowd", 0):
                continue
            key = (image_id, int(annotation["category_id"]))
            gt_by_key[key].append([float(v) for v in annotation["bbox"]])
            gt_count[int(annotation["category_id"])] += 1

    curves = []
    for category_id in categories:
        rows = []
        for image_id in wanted:
            key = (image_id, category_id)
            image_dets = sorted(detections_by_key.get(key, ()),
                                key=lambda row: -float(row["score"]))[:max_dets]
            rows.extend(_match_image(gt_by_key.get(key, ()), image_dets))
        rows.sort(key=lambda row: -row[0])
        curves.append(_prf_curves(rows, gt_count.get(category_id, 0)))

    if not curves:
        zeros = np.zeros_like(SCORE_GRID)
        curves = [(zeros, zeros, zeros)]
    macro = (np.mean([c[0] for c in curves], axis=0),
             np.mean([c[1] for c in curves], axis=0),
             np.mean([c[2] for c in curves], axis=0))
    fixed_index = int(np.argmin(np.abs(SCORE_GRID - float(deploy_conf))))
    best_index = int(np.argmax(macro[2]))
    return _at(fixed_index, *macro), _at(best_index, *macro)


def evaluate_ship(coco_gt, detections, deploy_conf=0.15, max_dets=MAX_DETS,
                  per_source=True):
    """Tek cagrida teslim edilecek metrikler.

    per_source: goruntulerde `source` alani varsa kaynak bazli AP tablosu da
    uretilir. Tek global sayi bu veri setinde yaniltici olabilir -- kaynaklar
    cok farkli (ir_thermal gercek termal, vais'in %60'i gri ton) ve termal
    AP'nin cokup cokmedigini gosteren tek sey budur.
    """
    ap, ap50, ap75 = coco_ap(coco_gt, detections, max_dets=max_dets)
    at_conf, best = prf_at_iou50(coco_gt, detections, deploy_conf,
                                 max_dets=max_dets)
    metrics = {
        "ap": ap, "ap50": ap50, "ap75": ap75,
        "f1_at": at_conf, "f1_best": best,
        "deploy_conf": float(deploy_conf),
        "n_images": len(coco_gt.getImgIds()),
        "n_detections": len(detections),
        "per_source": {},
    }

    if per_source:
        by_source = defaultdict(list)
        for image_id in coco_gt.getImgIds():
            source = coco_gt.imgs[image_id].get("source")
            if source:
                by_source[str(source)].append(image_id)
        for source, image_ids in sorted(by_source.items()):
            source_ap, source_ap50, _ = coco_ap(coco_gt, detections,
                                                image_ids=image_ids,
                                                max_dets=max_dets)
            metrics["per_source"][source] = {
                "images": len(image_ids), "ap": source_ap, "ap50": source_ap50,
            }
    return metrics


def format_metrics(metrics, title="COCO"):
    """Metrikleri insan okunur tek bir metne cevirir."""
    lines = [
        f"{title}: AP@[.50:.95]={metrics['ap']:.4f}  "
        f"AP@0.50={metrics['ap50']:.4f}  AP@0.75={metrics['ap75']:.4f}"
    ]
    at_conf, best = metrics.get("f1_at"), metrics.get("f1_best")
    if at_conf:
        lines.append(
            f"  Kart esiginde (conf={metrics['deploy_conf']:.2f}): "
            f"P={at_conf['precision']:.4f} R={at_conf['recall']:.4f} "
            f"F1={at_conf['f1']:.4f}"
        )
    if best:
        lines.append(
            f"  En iyi F1={best['f1']:.4f} (P={best['precision']:.4f} "
            f"R={best['recall']:.4f}) esik={best['score']:.2f}"
        )
    per_source = metrics.get("per_source") or {}
    if per_source:
        lines.append(f"  {'kaynak':<20}{'goruntu':>9}{'AP':>9}{'AP50':>9}")
        for source, row in sorted(per_source.items()):
            lines.append(f"  {source:<20}{row['images']:>9}"
                         f"{row['ap']:>9.4f}{row['ap50']:>9.4f}")
    return "\n".join(lines)
