#!/usr/bin/env python3
"""Tam-kare ve dilimlenmis (tiled) cikarimi ayni val setinde karsilastirir.

Amac: tiling'in VisDrone'da kac AP/F1 puani getirdigini **olcmek**. Kazanc
olculmeden KV260 tarafina C++ yazmak korlemesine is olur; oradaki bedel
kare basina DPU cagrisinin 1'den 5'e cikmasidir.

Kullanim (Kaggle, egitim bittikten sonra):
    python eval_tiled.py --exp-file yolox_nano_visdrone.py \\
        --ckpt YOLOX_outputs/yolox_nano_visdrone/best_ckpt.pth \\
        --data-dir datasets/visdrone_coco --grid 2x2 --overlap 0.2
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

from visdrone_eval import evaluate_visdrone, format_metrics

try:  # tiling.py notebook'ta yan yana yazilir, repoda tools/ altindadir
    from tiling import merge_tiled, tile_rects
except ImportError:  # pragma: no cover
    from tools.tiling import merge_tiled, tile_rects


def load_exp(exp_file):
    spec = importlib.util.spec_from_file_location("exp_module", exp_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Exp()


def letterbox(image, size):
    """YOLOX ValTransform ile ayni: en-boy korunur, sol-ust, 114 dolgu."""
    import cv2

    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    ratio = min(size / image.shape[0], size / image.shape[1])
    new_h, new_w = int(image.shape[0] * ratio), int(image.shape[1] * ratio)
    if new_h > 0 and new_w > 0:
        padded[:new_h, :new_w] = cv2.resize(
            image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
    return padded.transpose(2, 0, 1).astype(np.float32), ratio


@torch.no_grad()
def infer_tiles(model, image, rects, size, conf_thr, nms_thr, num_classes, device):
    """Her parcayi modele verir; parca-yerel xyxy tespitlerini dondurur."""
    from yolox.utils import postprocess

    batch, ratios = [], []
    for x0, y0, x1, y1 in rects:
        tensor, ratio = letterbox(image[y0:y1, x0:x1], size)
        batch.append(tensor)
        ratios.append(ratio)

    outputs = postprocess(
        model(torch.from_numpy(np.stack(batch)).to(device)),
        num_classes, conf_thr, nms_thr, class_agnostic=False,
    )

    results = []
    for rect, ratio, output in zip(rects, ratios, outputs):
        if output is None or len(output) == 0:
            results.append((rect, np.zeros((0, 4)), [], []))
            continue
        output = output.cpu().numpy()
        boxes = output[:, 0:4] / ratio          # letterbox tersi -> parca-yerel
        scores = output[:, 4] * output[:, 5]    # objectness * sinif skoru
        classes = output[:, 6].astype(np.int64)
        results.append((rect, boxes, scores, classes))
    return results


def run(args):
    import cv2
    from pycocotools.coco import COCO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp = load_exp(args.exp_file)
    size = exp.test_size[0]

    model = exp.get_model().to(device).eval()
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(checkpoint.get("model", checkpoint))

    data_dir = Path(args.data_dir)
    coco = COCO(str(data_dir / "annotations" / args.val_ann))
    image_dir = data_dir / args.image_folder

    cols, rows = (int(v) for v in args.grid.lower().split("x"))
    configs = {
        "tam kare": (1, 1, False),
        f"tiled {args.grid}": (cols, rows, args.include_full),
    }

    all_metrics = {}
    for label, (grid_cols, grid_rows, include_full) in configs.items():
        detections = []
        started = time.time()
        tile_total = 0
        for image_id in coco.getImgIds():
            info = coco.imgs[image_id]
            image = cv2.imread(str(image_dir / info["file_name"]))
            if image is None:
                raise SystemExit(f"goruntu okunamadi: {info['file_name']}")

            rects = tile_rects(
                info["width"], info["height"], grid_cols, grid_rows,
                overlap=args.overlap, include_full=include_full,
            )
            tile_total += len(rects)
            tile_results = infer_tiles(
                model, image, rects, size, args.conf, args.nms,
                exp.num_classes, device,
            )
            boxes, scores, classes = merge_tiled(
                tile_results, iou_thr=args.nms, max_dets=args.max_dets
            )
            for box, score, class_id in zip(boxes, scores, classes):
                detections.append({
                    "image_id": int(image_id),
                    "category_id": int(class_id) + 1,
                    "bbox": [
                        float(box[0]), float(box[1]),
                        float(box[2] - box[0]), float(box[3] - box[1]),
                    ],
                    "score": float(score),
                })

        elapsed = time.time() - started
        # Sinif semasi COCO'dan okunur; 2 sinifli veride de 10 sinifli veride
        # de dogru calisir. Sinif gruplama burada yapilmaz - bu scriptin isi
        # yalnizca tam kare ile dilimlenmis cikarimi karsilastirmak.
        metrics = evaluate_visdrone(
            coco, detections, max_dets=args.max_dets,
            num_classes=exp.num_classes, score_thr=args.score_thr,
        )
        all_metrics[label] = metrics

        print(f"\n{'=' * 66}\n{label}  "
              f"({tile_total / max(1, len(coco.getImgIds())):.1f} parca/kare, "
              f"{elapsed:.0f} s, {len(detections)} tespit)\n{'=' * 66}")
        print(format_metrics(metrics, label))

        if args.save_json:
            out = Path(args.save_json).with_suffix("")
            slug = label.replace(" ", "_").replace("/", "")
            Path(f"{out}_{slug}.json").write_text(json.dumps(detections))

    print(f"\n{'=' * 66}\nKARSILASTIRMA\n{'=' * 66}")
    print(f"{'yapilandirma':<20}{'AP50':>9}{'AP':>9}{'AP75':>9}"
          f"{'F1':>9}{'R':>9}")
    for label, m in all_metrics.items():
        print(f"{label:<20}{m['ap50']:>9.4f}{m['ap']:>9.4f}{m['ap75']:>9.4f}"
              f"{m['f1_best']['f1']:>9.4f}{m['f1_best']['recall']:>9.4f}")

    labels = list(all_metrics)
    if len(labels) == 2:
        base, tiled = all_metrics[labels[0]], all_metrics[labels[1]]
        print(f"\ntiling kazanci: AP50 {tiled['ap50'] - base['ap50']:+.4f}, "
              f"AP {tiled['ap'] - base['ap']:+.4f}, "
              f"F1 {tiled['f1_best']['f1'] - base['f1_best']['f1']:+.4f}, "
              f"recall {tiled['f1_best']['recall'] - base['f1_best']['recall']:+.4f}")
        print("Karar olcutu: bu kazanc, kare basina ~5 kat DPU maliyetini "
              "ve dusen FPS'i hakli cikariyor mu?")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp-file", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default="datasets/visdrone_coco")
    p.add_argument("--val-ann", default="instances_val.json")
    p.add_argument("--image-folder", default="val_images")
    p.add_argument("--grid", default="2x2", help="parca izgarasi, or. 2x2 / 3x2")
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--include-full", action="store_true", default=True,
                   help="parcalarin yaninda tum kareyi de isle (buyuk nesneler)")
    p.add_argument("--no-include-full", dest="include_full",
                   action="store_false")
    p.add_argument("--conf", type=float, default=0.001,
                   help="AP olcumu icin dusuk tutulur; F1 egrisi zaten taranir")
    p.add_argument("--score-thr", type=float, default=0.15,
                   help="F1'in ayrica raporlanacagi dagitim esigi (kartla ayni)")
    p.add_argument("--nms", type=float, default=0.65)
    p.add_argument("--max-dets", type=int, default=500)
    p.add_argument("--save-json", default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
