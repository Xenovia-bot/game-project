#!/usr/bin/env python3
"""VisDrone-DET etiketlerini COCO JSON formatina donusturur.

VisDrone satir formati:
    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category>,<truncation>,<occlusion>

Kategori esleme:
    0  = ignored-region  -> image.ignore_regions alaninda korunur
    1..10                -> COCO category_id 1..10 (asagidaki sinif listesi)
    11 = others          -> resmi protokol gibi sinif hedeflerinden cikartilir
    score == 0, cat 1..10 -> kendi sinifinda COCO crowd/ignore kutusu

Egitim loader'i ignored-region ve score=0 alanlarini 114 ile maskeler.
Degerlendirici, category=0 bolgesinin en az yuzde 50'si icinde kalan tespitleri
resmi VisDrone dropObjectsInIgr adimina uygun olarak sonuclardan cikartir.

Kullanim:
    python visdrone2coco.py --image-dir VisDrone2019-DET-train/images \
        --anno-dir VisDrone2019-DET-train/annotations \
        --output datasets/visdrone_coco/annotations/instances_train.json
"""

import argparse
import json
from pathlib import Path

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # tqdm yoksa sade dongu
    def tqdm(iterable, **kwargs):
        return iterable

VISDRONE_CLASSES = (
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

#: --classes 2: VisDrone kaynakli saha tanimi. VisDrone yalnizca kara
#: tasiti icerdigi icin sinif adi bilerek `land_vehicle`: deniz araci
#: ayri bir veri setinden ucuncu sinif olarak eklenecek.
#: Ignore ve ignored-region mantigi degismez; yalnizca kimlikler eslenir.
TWO_CLASS_NAMES = ("person", "land_vehicle")
TWO_CLASS_MAP = {
    1: 1, 2: 1,                                  # pedestrian, people
    3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2,  # tum kara tasitlari
}

SCHEMES = {
    "10": (VISDRONE_CLASSES, {i: i for i in range(1, 11)}),
    "2": (TWO_CLASS_NAMES, TWO_CLASS_MAP),
}


def covered_fraction_xywh(box, regions):
    """Kutu alaninin ignore dikdortgenleri birlesimi icindeki orani."""
    x, y, w, h = (float(v) for v in box)
    clipped = []
    for rx, ry, rw, rh in regions:
        left, top = max(x, rx), max(y, ry)
        right, bottom = min(x + w, rx + rw), min(y + h, ry + rh)
        if right > left and bottom > top:
            clipped.append((left, top, right, bottom))
    if w <= 0 or h <= 0 or not clipped:
        return 0.0
    xs = sorted({edge for rect in clipped for edge in (rect[0], rect[2])})
    covered = 0.0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted(
            (top, bottom)
            for x1, top, x2, bottom in clipped
            if x1 < right and x2 > left
        )
        if not intervals:
            continue
        union_y = 0.0
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start > end:
                union_y += end - start
                start, end = next_start, next_end
            else:
                end = max(end, next_end)
        covered += (right - left) * (union_y + end - start)
    return covered / (w * h)


def convert(image_dir, anno_dir, output, classes="10"):
    if classes not in SCHEMES:
        raise SystemExit(f"HATA: bilinmeyen sinif semasi {classes!r}")
    class_names, class_map = SCHEMES[classes]

    image_dir, anno_dir, output = Path(image_dir), Path(anno_dir), Path(output)
    img_files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not img_files:
        raise SystemExit(f"HATA: {image_dir} icinde goruntu bulunamadi")

    images, annotations = [], []
    ann_id = 1
    per_class = {name: 0 for name in class_names}
    skipped = 0
    global_ignored = 0
    class_ignored = 0
    dropped_in_ignore = 0

    for img_id, img_path in enumerate(tqdm(img_files, desc="donusturuluyor"), start=1):
        # PIL yalnizca basligi okur, tum goruntuyu cozmez (hizli)
        with Image.open(img_path) as im:
            width, height = im.size
        image_record = {
            "id": img_id,
            "file_name": img_path.name,
            "width": width,
            "height": height,
            "ignore_regions": [],
        }
        images.append(image_record)
        image_ann_start = len(annotations)

        txt = anno_dir / (img_path.stem + ".txt")
        if not txt.exists():
            continue
        for line in txt.read_text().splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                skipped += 1
                continue
            x, y, w, h, score, cat = (int(float(v)) for v in parts[:6])
            # goruntu sinirlarina kirp
            x2 = min(x + w, width)
            y2 = min(y + h, height)
            x = max(x, 0)
            y = max(y, 0)
            w = x2 - x
            h = y2 - y
            if w <= 0 or h <= 0:
                skipped += 1
                continue
            if cat == 0:
                image_record["ignore_regions"].append([x, y, w, h])
                global_ignored += 1
                continue
            if cat == 11 or cat < 1 or cat > 10:
                skipped += 1
                continue
            category_id = class_map[cat]
            if score == 0:
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 1,
                    "ignore": 1,
                })
                ann_id += 1
                class_ignored += 1
                continue
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            per_class[class_names[category_id - 1]] += 1
            ann_id += 1

        if image_record["ignore_regions"]:
            image_annotations = annotations[image_ann_start:]
            kept = [
                annotation for annotation in image_annotations
                if covered_fraction_xywh(
                    annotation["bbox"], image_record["ignore_regions"]
                ) < 0.5
            ]
            dropped_in_ignore += len(image_annotations) - len(kept)
            annotations[image_ann_start:] = kept

    per_class = {name: 0 for name in class_names}
    for annotation in annotations:
        if not annotation.get("iscrowd", 0):
            per_class[class_names[annotation["category_id"] - 1]] += 1

    coco = {
        "info": {
            "description": "VisDrone2019-DET (COCO formati)",
            "class_scheme": classes,
        },
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i + 1, "name": name, "supercategory": "none"}
            for i, name in enumerate(class_names)
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(coco, f)

    print(f"\nYazildi: {output}")
    print(f"  goruntu : {len(images)}")
    print(f"  kutu    : {len(annotations)} (atlanan: {skipped})")
    print(f"  ignore  : {global_ignored} genel bolge, {class_ignored} sinif-ozel kutu")
    print(f"  maskede : {dropped_in_ignore} hedef resmi protokole gore cikartildi")
    for name, count in per_class.items():
        print(f"    {name:16s} {count}")
    return coco


def parse_args():
    p = argparse.ArgumentParser(
        description="VisDrone-DET etiketlerini COCO JSON formatina cevirir"
    )
    p.add_argument("--image-dir", required=True, help="VisDrone images/ klasoru")
    p.add_argument("--anno-dir", required=True, help="VisDrone annotations/ klasoru")
    p.add_argument("--output", required=True, help="Cikti COCO JSON yolu")
    p.add_argument(
        "--classes", default="10", choices=sorted(SCHEMES),
        help="10 = resmi VisDrone siniflari, 2 = person/vehicle saha tanimi",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.image_dir, args.anno_dir, args.output, classes=args.classes)
