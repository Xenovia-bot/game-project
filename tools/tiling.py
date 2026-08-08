#!/usr/bin/env python3
"""Dilimlenmis (tiled) cikarim icin saf geometri ve birlestirme mantigi.

Neden: 1920x1080 bir kareyi 640x640'a kucultmek olcegi ~0.33'e dusurur, yani
20 piksellik bir yaya 7 piksele iner ve stride-8 haritasinda tek hucreye
sikisir. Kareyi ortusen parcalara bolup her parcayi ayri ayri modele vermek
ayni nesneyi ~2 kat buyuk tutar. SAHI calismasi VisDrone'da bu teknigin
+5..7 AP getirdigini olcmustur (arXiv 2202.06934).

Bu modul bilincli olarak torch/cv2 icermez: ayni mantik hem Kaggle'daki
Python degerlendirmesinde hem de KV260'taki C++ uygulamasinda kullanilacak,
o yuzden once burada saf ve test edilebilir halde durur.

Ortusme kurali: parca kenarina denk gelen nesne iki parcada da **tam olarak**
gorunsun diye ortusme payi en buyuk nesneden genis secilmelidir. VisDrone'da
nesneler <50 px oldugundan varsayilan %20 ortusme (1920 genislikte ~192 px)
fazlasiyla yeterlidir; boylece parca sinirinda ikiye bolunmus yarim kutular
olusmaz ve NMS kopyalari sorunsuz birlestirir.
"""

import numpy as np


def tile_rects(width, height, cols=2, rows=2, overlap=0.2, include_full=True):
    """Ortusen parca dikdortgenlerini `(x0, y0, x1, y1)` olarak uretir.

    Parcalar esit araliklarla yerlestirilir ve her biri temel hucre boyutunun
    `(1 + overlap)` katidir. `include_full` ile tum kare de bir "parca" olarak
    listeye eklenir: kucuk nesneleri parcalar, buyuk nesneleri tam kare yakalar.

    Donen dikdortgenler tamsayidir ve goruntu sinirlari icinde kalir.
    """
    if width <= 0 or height <= 0:
        raise ValueError("gecersiz goruntu boyutu")
    if cols < 1 or rows < 1:
        raise ValueError("cols ve rows en az 1 olmali")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap [0, 1) araliginda olmali")

    def axis_positions(total, count):
        base = total / count
        size = min(float(total), base * (1.0 + overlap))
        starts = []
        for index in range(count):
            center = (index + 0.5) * base
            start = min(max(center - size / 2.0, 0.0), total - size)
            starts.append(start)
        return size, starts

    tile_w, xs = axis_positions(width, cols)
    tile_h, ys = axis_positions(height, rows)

    rects = []
    for y0 in ys:
        for x0 in xs:
            x_start, y_start = int(round(x0)), int(round(y0))
            x_end = min(width, x_start + int(round(tile_w)))
            y_end = min(height, y_start + int(round(tile_h)))
            rects.append((x_start, y_start, x_end, y_end))

    if include_full and (cols > 1 or rows > 1):
        rects.append((0, 0, int(width), int(height)))
    # Ayni dikdortgen birden fazla kez uretilmisse (ornegin 1x1) tekrari at.
    unique = []
    for rect in rects:
        if rect not in unique:
            unique.append(rect)
    return unique


def offset_boxes(boxes, rect):
    """Parca-yerel xyxy kutularini tam kare koordinatlarina tasir."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if boxes.size == 0:
        return boxes
    shifted = boxes.copy()
    shifted[:, [0, 2]] += float(rect[0])
    shifted[:, [1, 3]] += float(rect[1])
    return shifted


def nms_xyxy(boxes, scores, iou_thr=0.45):
    """Tek sinif icin standart greedy NMS; kalan kutularin indekslerini doner."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if boxes.shape[0] == 0:
        return np.empty(0, dtype=np.int64)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        current = order[0]
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        inter_w = np.maximum(
            0.0, np.minimum(x2[current], x2[rest]) - np.maximum(x1[current], x1[rest])
        )
        inter_h = np.maximum(
            0.0, np.minimum(y2[current], y2[rest]) - np.maximum(y1[current], y1[rest])
        )
        inter = inter_w * inter_h
        union = areas[current] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def merge_tiled(tile_results, iou_thr=0.45, max_dets=500):
    """Parca sonuclarini tam kare koordinatlarinda birlestirir.

    `tile_results`: `(rect, boxes_xyxy, scores, class_ids)` dizisi. Kutular
    parca-yerel koordinatlardadir (letterbox tersi zaten uygulanmis olmali).

    NMS **sinif bazinda** uygulanir: ust uste binen bir yaya ile bisiklet
    birbirini elemez. Ayni nesnenin komsu parcalardaki kopyalari ise ayni
    sinifta olduklari icin birlesir.

    Doner: `(boxes, scores, class_ids)`, skora gore azalan sirali.
    """
    all_boxes, all_scores, all_classes = [], [], []
    for rect, boxes, scores, class_ids in tile_results:
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        if boxes.shape[0] == 0:
            continue
        all_boxes.append(offset_boxes(boxes, rect))
        all_scores.append(np.asarray(scores, dtype=np.float64).reshape(-1))
        all_classes.append(np.asarray(class_ids).reshape(-1))

    if not all_boxes:
        return (
            np.zeros((0, 4), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
        )

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    classes = np.concatenate(all_classes, axis=0)

    keep = []
    for class_id in np.unique(classes):
        mask = np.nonzero(classes == class_id)[0]
        kept = nms_xyxy(boxes[mask], scores[mask], iou_thr=iou_thr)
        keep.extend(mask[kept].tolist())

    keep = np.asarray(keep, dtype=np.int64)
    order = keep[scores[keep].argsort()[::-1]][:max_dets]
    return boxes[order], scores[order], classes[order]
