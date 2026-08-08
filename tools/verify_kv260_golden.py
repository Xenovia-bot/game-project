#!/usr/bin/env python3
"""KV260 C++ YOLOX decode/NMS/merkez sonucunu bagimsiz Python referansiyla dogrular.

Kartta bir kez:
  ./yolox_visdrone_demo model.xmodel video.mp4 output.avi centers.csv \
      --max-frames 1 --dump-first-frame golden

Sonra ``golden`` klasorunu bu scriptin calistigi makineye kopyalayin:
  python tools/verify_kv260_golden.py --dump-dir golden
"""

import argparse
import csv
from pathlib import Path

import numpy as np


def sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    return np.float32(1.0) / (np.float32(1.0) + np.exp(-x))


def read_meta(path):
    meta = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError("Gecersiz metadata satiri: %r" % line)
        meta[key] = value
    return meta


def verify_input_quantization(dump_dir, meta):
    size = int(meta["input_h"]) * int(meta["input_w"]) * 3
    canvas = np.fromfile(dump_dir / meta["canvas_file"], dtype=np.uint8)
    actual = np.fromfile(dump_dir / meta["input_file"], dtype=np.int8)
    if canvas.size != size or actual.size != size:
        raise AssertionError(
            "Input dump boyutu gecersiz: canvas=%d, int8=%d, beklenen=%d"
            % (canvas.size, actual.size, size)
        )
    scaled = np.clip(
        canvas.astype(np.float32) * np.float32(meta["input_scale"]),
        -128.0,
        127.0,
    )
    # Vitis AI varsayilan HALF_UP; BGR piksel girdileri negatif degildir.
    expected = np.floor(scaled + np.float32(0.5)).astype(np.int8)
    mismatch = np.flatnonzero(expected != actual)
    if mismatch.size:
        index = int(mismatch[0])
        raise AssertionError(
            "Input INT8 donusumu farkli: index=%d, Python=%d, C++=%d "
            "(toplam fark=%d)"
            % (index, expected[index], actual[index], mismatch.size)
        )


def iou(a, b):
    xx1 = max(a["x1"], b["x1"])
    yy1 = max(a["y1"], b["y1"])
    xx2 = min(a["x2"], b["x2"])
    yy2 = min(a["y2"], b["y2"])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter / max(area_a + area_b - inter, 1e-9)


def deterministic_key(det):
    return (
        -det["score"], det["class_id"], det["x1"], det["y1"], det["x2"], det["y2"]
    )


def decode_reference(dump_dir, meta):
    candidates = []
    num_levels = int(meta["num_levels"])
    for i in range(num_levels):
        h = int(meta[f"level{i}_h"])
        w = int(meta[f"level{i}_w"])
        c = int(meta[f"level{i}_c"])
        stride = int(meta[f"level{i}_stride"])
        scale = np.float32(meta[f"level{i}_scale"])
        if c != 15:
            raise ValueError("Beklenen cikti kanali 15, bulunan %d" % c)

        file_path = dump_dir / meta[f"level{i}_file"]
        raw = np.fromfile(file_path, dtype=np.int8)
        if raw.size != h * w * c:
            raise ValueError(
                "%s boyutu %d; beklenen %d" % (file_path, raw.size, h * w * c)
            )
        raw = raw.reshape(h, w, c)
        values = raw.astype(np.float32) * scale
        gy, gx = np.meshgrid(
            np.arange(h, dtype=np.float32),
            np.arange(w, dtype=np.float32),
            indexing="ij",
        )

        class_id = np.argmax(raw[:, :, 5:], axis=2)
        best_raw = np.take_along_axis(
            raw[:, :, 5:], class_id[:, :, None], axis=2
        )[:, :, 0]
        score = sigmoid(values[:, :, 4]) * sigmoid(
            best_raw.astype(np.float32) * scale
        )
        mask = score >= np.float32(meta["conf"])
        ys, xs = np.nonzero(mask)
        for y, x in zip(ys.tolist(), xs.tolist()):
            cx = (values[y, x, 0] + gx[y, x]) * stride
            cy = (values[y, x, 1] + gy[y, x]) * stride
            bw = np.exp(values[y, x, 2]) * stride
            bh = np.exp(values[y, x, 3]) * stride
            candidates.append({
                "class_id": int(class_id[y, x]),
                "score": float(score[y, x]),
                "x1": float(cx - bw * np.float32(0.5)),
                "y1": float(cy - bh * np.float32(0.5)),
                "x2": float(cx + bw * np.float32(0.5)),
                "y2": float(cy + bh * np.float32(0.5)),
            })

    candidates.sort(key=deterministic_key)
    kept = []
    nms_thr = float(meta["nms"])
    for candidate in candidates:
        if any(
            candidate["class_id"] == previous["class_id"]
            and iou(candidate, previous) > nms_thr
            for previous in kept
        ):
            continue
        kept.append(candidate)

    ratio = float(meta["ratio"])
    frame_w = int(meta["frame_w"])
    frame_h = int(meta["frame_h"])
    result = []
    for det in kept:
        x1 = max(0.0, min(det["x1"] / ratio, frame_w - 1.0))
        y1 = max(0.0, min(det["y1"] / ratio, frame_h - 1.0))
        x2 = max(0.0, min(det["x2"] / ratio, frame_w - 1.0))
        y2 = max(0.0, min(det["y2"] / ratio, frame_h - 1.0))
        if x2 <= x1 or y2 <= y1:
            continue
        det = dict(det, x1=x1, y1=y1, x2=x2, y2=y2)
        det["cx"] = (x1 + x2) * 0.5
        det["cy"] = (y1 + y2) * 0.5
        result.append(det)
    return result


def read_cpp(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    numeric = ("score", "x1", "y1", "x2", "y2", "cx", "cy")
    result = []
    for row in rows:
        det = {"class_id": int(row["class_id"])}
        det.update({key: float(row[key]) for key in numeric})
        result.append(det)
    return result


def verify(reference, cpp, coord_atol, score_atol):
    if len(reference) != len(cpp):
        raise AssertionError(
            "Tespit sayisi farkli: Python=%d, C++=%d" % (len(reference), len(cpp))
        )
    max_coord_error = 0.0
    max_score_error = 0.0
    for index, (expected, actual) in enumerate(zip(reference, cpp)):
        if expected["class_id"] != actual["class_id"]:
            raise AssertionError(
                "%d. tespitte sinif farkli: Python=%d, C++=%d"
                % (index, expected["class_id"], actual["class_id"])
            )
        score_error = abs(expected["score"] - actual["score"])
        max_score_error = max(max_score_error, score_error)
        if score_error > score_atol:
            raise AssertionError(
                "%d. tespitte skor farki %.8f > %.8f"
                % (index, score_error, score_atol)
            )
        for key in ("x1", "y1", "x2", "y2", "cx", "cy"):
            error = abs(expected[key] - actual[key])
            max_coord_error = max(max_coord_error, error)
            if error > coord_atol:
                raise AssertionError(
                    "%d. tespitte %s farki %.6f > %.6f"
                    % (index, key, error, coord_atol)
                )
        if abs(actual["cx"] - (actual["x1"] + actual["x2"]) * 0.5) > coord_atol:
            raise AssertionError("%d. C++ cx degeri kutu merkezi degil" % index)
        if abs(actual["cy"] - (actual["y1"] + actual["y2"]) * 0.5) > coord_atol:
            raise AssertionError("%d. C++ cy degeri kutu merkezi degil" % index)
    return max_coord_error, max_score_error


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", required=True, type=Path)
    parser.add_argument("--coord-atol", type=float, default=0.02)
    parser.add_argument("--score-atol", type=float, default=1e-5)
    return parser.parse_args()


def main():
    args = parse_args()
    meta = read_meta(args.dump_dir / "meta.txt")
    verify_input_quantization(args.dump_dir, meta)
    reference = decode_reference(args.dump_dir, meta)
    cpp = read_cpp(args.dump_dir / "cpp_detections.csv")
    coord_error, score_error = verify(
        reference, cpp, args.coord_atol, args.score_atol
    )
    print(
        "OK: INT8 input + %d tespit; Python/C++ decode-NMS-merkez esdeger "
        "(maks koordinat farki %.6f px, skor farki %.8f)"
        % (len(reference), coord_error, score_error)
    )


if __name__ == "__main__":
    main()
