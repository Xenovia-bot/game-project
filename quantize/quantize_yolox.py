#!/usr/bin/env python3
"""Havadan arac YOLOX-Nano'su icin Vitis AI PTQ kuantalama ve xmodel export.

Vitis AI 3.0 pytorch docker'i icinde calistirilir (conda env: vitis-ai-pytorch).

Beklenen veri duzeni (`tools/build_dataset.py --images-out` ciktisi):

    <data-dir>/annotations/instances_val.json
    <data-dir>/annotations/instances_train.json   # istege bagli, kalibrasyon icin
    <data-dir>/images/<kaynak>/<ad>.jpg           # COCO file_name ile birebir

Ornek akis (docker icinde, /workspace altinda):
  ARGS="--exp-file yolox_nano_visdrone.py --ckpt best_ckpt.pth --data-dir datasets/merged"

  # 0) (istege bagli) DPU uyumluluk raporu
  python quantize_yolox.py --inspect $ARGS

  # 1) float AP@500 dogrulama (Kaggle'daki degerle karsilastirin)
  python quantize_yolox.py --quant-mode float $ARGS

  # 2) kalibrasyon (PTQ)
  python quantize_yolox.py --quant-mode calib --subset-len 300 $ARGS

  # 3) INT8 AP@500 olcumu (tam val seti; accuracy gate burada uretilir)
  python quantize_yolox.py --quant-mode test --float-map <FLOAT_AP500> $ARGS

  # 4) xmodel export (derleme girdisi)
  python quantize_yolox.py --quant-mode test --deploy --subset-len 1 --batch-size 1 $ARGS
"""

import argparse
import hashlib
import importlib.util
import json
import random
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

EXPECTED_YOLOX_COMMIT = "6ddff4824372906469a7fae2dc3206c7aa4bbaee"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_quant_artifacts(root):
    digest = hashlib.sha256()
    files = sorted(
        path for path in Path(root).rglob("*")
        if path.is_file()
        and path.name != "accuracy_gate.json"
        and path.suffix not in {".xmodel", ".onnx"}
    )
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def accuracy_gate_identity(args, ann_file, input_size, quant_info):
    return {
        "checkpoint_sha256": sha256_file(args.ckpt),
        "exp_sha256": sha256_file(args.exp_file),
        "annotations_sha256": sha256_file(ann_file),
        "quant_info_sha256": sha256_file(quant_info),
        "quant_artifacts_sha256": sha256_quant_artifacts(Path(quant_info).parent),
        "input_size": list(input_size),
        "conf": args.conf,
        "nms": args.nms,
        "fast_finetune": args.fast_finetune,
    }


def verify_yolox_version(checkpoint_path):
    """Artifact bildirimi ile kurulu editable YOLOX kaynagini karsilastirir."""
    commit_file = Path(checkpoint_path).resolve().with_name("YOLOX_COMMIT.txt")
    if not commit_file.is_file():
        raise SystemExit(
            "HATA: YOLOX_COMMIT.txt checkpoint ile ayni klasorde olmali."
        )
    declared = commit_file.read_text().strip()
    if declared != EXPECTED_YOLOX_COMMIT:
        raise SystemExit(
            "HATA: artifact YOLOX commit'i beklenen surum degil: %s" % declared
        )

    import yolox

    repo_dir = Path(yolox.__file__).resolve().parent.parent
    try:
        installed = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            "HATA: kurulu YOLOX git commit'i dogrulanamadi: %s" % exc
        )
    if installed != declared:
        raise SystemExit(
            "HATA: artifact/kurulu YOLOX surumu farkli: %s != %s"
            % (declared, installed)
        )
    print("YOLOX commit dogrulandi:", installed)


def safe_torch_load(path):
    """PyTorch 1.12 ve 2.6+ ile ayni checkpoint yukleme davranisi."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Vitis AI 3.0 / PyTorch 1.12
        return torch.load(path, map_location="cpu")


def load_checkpoint_strict(model, path):
    """Yanlis exp/checkpoint birlesimini sessizce kabul etmez."""
    checkpoint = safe_torch_load(path)
    state = (
        checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    if not isinstance(state, dict):
        raise SystemExit("HATA: checkpoint icinde state_dict bulunamadi: %s" % path)
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise SystemExit(
            "HATA: checkpoint ve exp mimarisi birebir uyusmuyor. "
            "Kaggle artifacts paketindeki exp dosyasini ve sabitlenmis YOLOX "
            "commit'ini kullanin.\n%s" % exc
        )
    print("Checkpoint tam eslesti: %d/%d katman" %
          (len(state), len(model.state_dict())))


def load_exp(exp_file):
    spec = importlib.util.spec_from_file_location("visdrone_exp", exp_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Exp()


class DeployModel(nn.Module):
    """YOLOX'un DPU'da kosacak kismi: backbone + FPN + head konvolusyonlari.

    Cikti: her stride (8/16/32) icin ham [B, 4+1+num_classes, H, W] haritasi
    (kanal sirasi: reg(4), obj(1), cls(num_classes)). Sigmoid ve grid cozumu
    DPU disinda yapilir: bu scriptte decode(), kartta C++ uygulamasi.
    """

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head

    def forward(self, x):
        fpn_outs = self.backbone(x)
        outs = []
        for k, feat in enumerate(fpn_outs):
            xk = self.head.stems[k](feat)
            cls_feat = self.head.cls_convs[k](xk)
            reg_feat = self.head.reg_convs[k](xk)
            outs.append(torch.cat([
                self.head.reg_preds[k](reg_feat),
                self.head.obj_preds[k](reg_feat),
                self.head.cls_preds[k](cls_feat),
            ], dim=1))
        return tuple(outs)


def letterbox(img, size):
    """YOLOX ValTransform ile birebir ayni on isleme.

    BGR, 0-255 (normalizasyon yok), sol-ust yerlesim, 114 dolgu, CHW float32.
    """
    h, w = size
    padded = np.full((h, w, 3), 114, dtype=np.uint8)
    r = min(h / img.shape[0], w / img.shape[1])
    nw, nh = int(img.shape[1] * r), int(img.shape[0] * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded[:nh, :nw] = resized
    tensor = padded.transpose(2, 0, 1).astype(np.float32)
    return tensor, r


def decode(outputs, input_size):
    """Ham cikti haritalarini [B, N, 5+nc] kutulara cozer (girdi olceginde).

    Kutu: (cx, cy, w, h); skorlar: sigmoid(obj), sigmoid(cls...).
    """
    flat = []
    for out in outputs:
        b, c, hh, ww = out.shape
        stride = input_size[0] / hh
        try:
            yv, xv = torch.meshgrid(torch.arange(hh), torch.arange(ww), indexing="ij")
        except TypeError:  # eski torch surumleri
            yv, xv = torch.meshgrid(torch.arange(hh), torch.arange(ww))
        grid = torch.stack((xv, yv), 2).reshape(1, hh * ww, 2).float()
        o = out.permute(0, 2, 3, 1).reshape(b, hh * ww, c)
        xy = (o[..., 0:2] + grid) * stride
        wh = torch.exp(o[..., 2:4]) * stride
        scores = torch.sigmoid(o[..., 4:])
        flat.append(torch.cat([xy, wh, scores], dim=-1))
    return torch.cat(flat, dim=1)


def nms_single_class(boxes, scores, iou_thr):
    """Basit NMS (xyxy, numpy). Kucuk aday sayilari icin yeterince hizli."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(area_i + area_r - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def postprocess_frame(pred, conf_thr, nms_thr):
    """[N, 5+nc] tensorunu (x1,y1,x2,y2,score,cls) listesine cevirir."""
    pred = pred.numpy()
    obj = pred[:, 4]
    cls_scores = pred[:, 5:]
    cls_ids = cls_scores.argmax(axis=1)
    scores = obj * cls_scores[np.arange(len(pred)), cls_ids]
    mask = scores >= conf_thr
    if not mask.any():
        return []
    xywh = pred[mask, :4]
    scores = scores[mask]
    cls_ids = cls_ids[mask]
    boxes = np.empty_like(xywh)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0

    dets = []
    for cls in np.unique(cls_ids):
        sel = cls_ids == cls
        for i in nms_single_class(boxes[sel], scores[sel], nms_thr):
            b = boxes[sel][i]
            dets.append((b[0], b[1], b[2], b[3], scores[sel][i], int(cls)))
    return dets


def evaluate(run_model, ann_file, img_dir, input_size, conf_thr, nms_thr,
             num_classes, subset_len=None, tag=""):
    """Resmi VisDrone DET tarzi AP@500; fast_finetune metrigi."""
    from pycocotools.coco import COCO
    from visdrone_eval import evaluate_visdrone

    coco_gt = COCO(str(ann_file))
    img_ids = sorted(coco_gt.getImgIds())
    if subset_len:
        img_ids = img_ids[:subset_len]

    results = []
    t0 = time.time()
    for n, img_id in enumerate(img_ids, 1):
        info = coco_gt.loadImgs(img_id)[0]
        img = cv2.imread(str(Path(img_dir) / info["file_name"]))
        if img is None:
            continue
        tensor, r = letterbox(img, input_size)
        with torch.no_grad():
            outs = run_model(torch.from_numpy(tensor).unsqueeze(0))
            pred = decode([o.float().cpu() for o in outs], input_size)[0]
        for x1, y1, x2, y2, score, cls in postprocess_frame(pred, conf_thr, nms_thr):
            results.append({
                "image_id": img_id,
                "category_id": int(cls) + 1,  # COCO kategori id'leri 1'den baslar
                "bbox": [float(x1 / r), float(y1 / r),
                         float((x2 - x1) / r), float((y2 - y1) / r)],
                "score": float(score),
            })
        if n % 50 == 0:
            print("  [%s] %d/%d goruntu, %.0f sn" % (tag, n, len(img_ids), time.time() - t0))

    if not results:
        print("UYARI: hic tespit uretilmedi (AP@500=0)")
        return 0.0
    if subset_len:
        # Yardimci tum coco image id'lerini gezer; alt-kume testinde GT'yi de daralt.
        coco_gt.dataset["images"] = [
            image for image in coco_gt.dataset["images"] if image["id"] in img_ids
        ]
        coco_gt.dataset["annotations"] = [
            annotation for annotation in coco_gt.dataset["annotations"]
            if annotation["image_id"] in img_ids
        ]
        coco_gt.createIndex()
    metrics = evaluate_visdrone(coco_gt, results, max_dets=500)
    print(
        "[%s] VisDrone AP@500=%.4f  AP50@500=%.4f  AP75@500=%.4f"
        % (tag, metrics["ap"], metrics["ap50"], metrics["ap75"])
    )
    return metrics["ap"]


IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png")


def collect_calib_paths(data_dir, images_dir, calib_dir=None):
    """Kalibrasyon goruntulerini secer ve nereden geldigini yazdirir.

    Birlestirilmis veri seti tek bir `images/<kaynak>/<ad>.jpg` agacinda durur;
    train ve val ayrimi anotasyon dosyalarindadir. Bu yuzden train anotasyonu
    varsa kalibrasyon **yalnizca train** goruntuleriyle yapilir; yoksa tum
    agac kullanilir ve val'in de karistigi acikca uyarilir.
    """
    if calib_dir:
        paths = sorted(p for ext in IMAGE_EXTS
                       for p in Path(calib_dir).rglob(ext))
        print("Kalibrasyon kaynagi: --calib-dir %s" % calib_dir)
        return paths

    train_ann = Path(data_dir) / "annotations" / "instances_train.json"
    if train_ann.is_file():
        names = [image["file_name"]
                 for image in json.loads(train_ann.read_text())["images"]]
        paths = [images_dir / name for name in names
                 if (images_dir / name).is_file()]
        print("Kalibrasyon kaynagi: %s (train bolumu, %d goruntu)"
              % (train_ann, len(paths)))
        return paths

    paths = sorted(p for ext in IMAGE_EXTS for p in Path(images_dir).rglob(ext))
    print("UYARI: instances_train.json yok; kalibrasyon %s agacinin tamamindan "
          "yapiliyor (val goruntuleri de dahil)." % images_dir)
    return paths


def calibrate(run_model, paths, input_size, subset_len, batch_size):
    if not paths:
        raise SystemExit("HATA: kalibrasyon goruntusu bulunamadi")
    random.seed(42)
    paths = sorted(paths)
    random.shuffle(paths)
    paths = paths[:subset_len]
    print("Kalibrasyon: %d goruntu" % len(paths))

    batch = []
    with torch.no_grad():
        for i, p in enumerate(paths, 1):
            img = cv2.imread(str(p))
            if img is None:
                continue
            tensor, _ = letterbox(img, input_size)
            batch.append(torch.from_numpy(tensor))
            if len(batch) == batch_size or i == len(paths):
                run_model(torch.stack(batch))
                batch = []
            if i % 50 == 0:
                print("  %d/%d" % (i, len(paths)))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp-file", required=True, help="yolox_nano_visdrone.py yolu")
    p.add_argument("--ckpt", required=True, help="best_ckpt.pth yolu")
    p.add_argument("--data-dir", required=True,
                   help="birlestirilmis veri seti koku: <yol>/annotations ve "
                        "<yol>/images alt klasorlerini icerir")
    p.add_argument("--output", default="build/quant", help="cikti klasoru")
    p.add_argument("--quant-mode", default="calib", choices=["float", "calib", "test"])
    p.add_argument("--subset-len", type=int, default=None,
                   help="islenecek goruntu sayisi (calib varsayilani 300)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--img-size", type=int, default=None,
                   help="girdi boyutu (varsayilan: exp.test_size)")
    p.add_argument("--conf", type=float, default=0.001, help="AP icin esik")
    p.add_argument("--nms", type=float, default=0.65)
    p.add_argument("--calib-dir", default=None,
                   help="kalibrasyon goruntu klasoru (varsayilan: "
                        "instances_train.json'daki train goruntuleri, "
                        "o yoksa images/ agacinin tamami)")
    p.add_argument("--deploy", action="store_true", help="xmodel export et")
    p.add_argument("--fast-finetune", action="store_true",
                   help="AdaQuant ile gelismis kalibrasyon (yavas ama daha isabetli)")
    p.add_argument("--float-map", type=float, default=None,
                   help="Kaggle/float AP@500; INT8 kayip sinirini denetler")
    p.add_argument("--max-map-drop", type=float, default=0.02,
                   help="izin verilen mutlak AP kaybi (varsayilan: 0.02)")
    p.add_argument("--inspect", action="store_true", help="DPU uyumluluk raporu uret")
    p.add_argument("--target", default="DPUCZDX8G_ISA1_B4096",
                   help="inspector DPU hedefi (KV260)")
    return p.parse_args()


def main():
    args = parse_args()
    if not (0.0 < args.conf < 1.0) or not (0.0 <= args.nms <= 1.0):
        raise SystemExit("HATA: --conf (0,1), --nms [0,1] araliginda olmali")
    if args.batch_size <= 0 or (args.subset_len is not None and args.subset_len <= 0):
        raise SystemExit("HATA: batch-size ve subset-len pozitif olmali")
    if args.max_map_drop < 0:
        raise SystemExit("HATA: --max-map-drop negatif olamaz")
    if args.float_map is not None and not (0.0 <= args.float_map <= 1.0):
        raise SystemExit("HATA: --float-map 0 ile 1 arasinda olmali")
    if args.deploy and args.quant_mode != "test":
        raise SystemExit("HATA: --deploy yalniz --quant-mode test ile kullanilir")

    verify_yolox_version(args.ckpt)
    exp = load_exp(args.exp_file)
    input_size = (args.img_size, args.img_size) if args.img_size else tuple(exp.test_size)
    if len(input_size) != 2 or any(size <= 0 or size % 32 for size in input_size):
        raise SystemExit("HATA: model girdi boyutlari pozitif ve 32'nin kati olmali")

    model = exp.get_model()
    load_checkpoint_strict(model, args.ckpt)
    model.eval()

    deploy_model = DeployModel(model).eval()

    # Birlestirilmis veri seti duzeni (tools/build_dataset.py ciktisi):
    #   <data-dir>/annotations/instances_val.json
    #   <data-dir>/images/<kaynak>/<ad>.jpg      <- COCO file_name ile birebir
    data_dir = Path(args.data_dir)
    ann_val = data_dir / "annotations" / "instances_val.json"
    val_dir = data_dir / "images"
    if not ann_val.is_file():
        raise SystemExit("HATA: val anotasyonu yok: %s" % ann_val)
    if not val_dir.is_dir():
        raise SystemExit(
            "HATA: goruntu klasoru yok: %s\n"
            "  Beklenen duzen: <data-dir>/images/<kaynak>/<ad>.jpg "
            "(build_dataset.py --images-out ciktisi)" % val_dir
        )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    quant_info = out_dir / "quant_info.json"
    gate_file = out_dir / "accuracy_gate.json"
    dummy = torch.randn(args.batch_size, 3, *input_size)

    if args.deploy:
        if not gate_file.is_file():
            raise SystemExit(
                "HATA: accuracy_gate.json yok. Once tam val setinde INT8 "
                "AP testini --float-map ile basariyla tamamlayin."
            )
        if not quant_info.is_file():
            raise SystemExit("HATA: quant_info.json yok; once kalibrasyon yapin.")
        gate = json.loads(gate_file.read_text())
        identity = accuracy_gate_identity(args, ann_val, input_size, quant_info)
        mismatched = [
            key for key, value in identity.items() if gate.get(key) != value
        ]
        if mismatched:
            raise SystemExit(
                "HATA: AP kabul testi mevcut artifactlarla eslesmiyor: %s. "
                "INT8 AP testini yeniden calistirin." % ", ".join(mismatched)
            )

    if args.inspect:
        from pytorch_nndct.apis import Inspector
        inspector = Inspector(args.target)
        inspector.inspect(deploy_model, (dummy,), device=torch.device("cpu"),
                          output_dir=str(out_dir / "inspect"))
        print("Rapor:", out_dir / "inspect")
        return

    if args.quant_mode == "float":
        evaluate(deploy_model, ann_val, val_dir, input_size, args.conf, args.nms,
                 exp.num_classes, args.subset_len, tag="float")
        return

    from pytorch_nndct.apis import torch_quantizer
    quantizer = torch_quantizer(
        args.quant_mode,
        deploy_model,
        (dummy,),
        output_dir=str(out_dir),
        device=torch.device("cpu"),
    )
    qmodel = quantizer.quant_model
    qmodel.eval()

    if args.quant_mode == "calib":
        calib_paths = collect_calib_paths(data_dir, val_dir, args.calib_dir)
        calibrate(qmodel, calib_paths, input_size,
                  args.subset_len or 300, args.batch_size)
        if args.fast_finetune:
            print("fast_finetune (AdaQuant) calisiyor; CPU'da uzun surebilir...")
            quantizer.fast_finetune(
                evaluate,
                (qmodel, ann_val, val_dir, input_size, args.conf, args.nms,
                 exp.num_classes, 100, "ft"),
            )
        quantizer.export_quant_config()
        print("Kalibrasyon tamamlandi:", out_dir)
    else:  # test
        if args.fast_finetune:
            quantizer.load_ft_param()
        if args.deploy:
            if args.batch_size != 1 or (args.subset_len or 1) != 1:
                raise SystemExit("--deploy icin --batch-size 1 --subset-len 1 kullanin")
            # export oncesi bir forward gecisi zorunlu
            evaluate(qmodel, ann_val, val_dir, input_size, args.conf, args.nms,
                     exp.num_classes, subset_len=1, tag="deploy")
            quantizer.export_xmodel(str(out_dir), deploy_check=True)
            print("Kuantalanmis xmodel yazildi:", out_dir)
        else:
            if args.float_map is None:
                raise SystemExit(
                    "HATA: tam INT8 kabul testi icin --float-map zorunludur."
                )
            if args.subset_len is not None:
                raise SystemExit(
                    "HATA: accuracy gate alt-kumeyle olusturulamaz; "
                    "--subset-len kullanmadan tam val setini test edin."
                )
            int8_map = evaluate(
                qmodel, ann_val, val_dir, input_size, args.conf, args.nms,
                exp.num_classes, args.subset_len, tag="int8"
            )
            if args.float_map is not None:
                drop = args.float_map - int8_map
                print("AP kaybi: %.4f (sinir: %.4f)" % (drop, args.max_map_drop))
                if drop > args.max_map_drop:
                    raise SystemExit(
                        "HATA: PTQ AP kaybi siniri asti. Calib/test adimlarini "
                        "--fast-finetune ile tekrarlayin; hala asiyorsa modeli "
                        "kartta kullanmayin ve Vitis AI QAT ortamina gecin."
                    )
                gate = accuracy_gate_identity(
                    args, ann_val, input_size, quant_info
                )
                gate.update({
                    "float_ap500": args.float_map,
                    "int8_ap500": int8_map,
                    "max_ap_drop": args.max_map_drop,
                })
                gate_file.write_text(json.dumps(gate, indent=2) + "\n")
                print("Accuracy gate yazildi:", gate_file)


if __name__ == "__main__":
    main()
