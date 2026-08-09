#!/usr/bin/env python3
"""INT8 AP kaybinin nereden geldigini olcer.

Hipotez: DeployModel reg(4)+obj(1)+cls(2) kanallarini torch.cat ile tek
tensore birlestiriyor. Vitis AI'da concat girdilerinin AYNI fix_point'i
paylasmasini zorunlu kilar (donanimda concat sadece bellek duzenidir).
cls/obj logitleri genis, reg ofsetleri dar aralikta oldugundan ortak olcek
genise gore secilir ve reg kanallari 8 bitin cok az seviyesine sikisir.

Calistirma (docker icinde, /workspace altinda):
    python tani_int8.py
"""
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import torch

EXP = "yolox_nano_visdrone.py"
CKPT = "best_ckpt.pth"
DATA = Path("datasets/merged")
QUANT_INFO = Path("build/quant/quant_info.json")
N_IMAGES = 24
CHANNEL_ROLE = ["reg_x", "reg_y", "reg_w", "reg_h", "obj", "cls_0", "cls_1"]


def load_exp(path):
    spec = importlib.util.spec_from_file_location("exp_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Exp()


def letterbox(img, size):
    h, w = size
    padded = np.full((h, w, 3), 114, dtype=np.uint8)
    r = min(h / img.shape[0], w / img.shape[1])
    nw, nh = int(img.shape[1] * r), int(img.shape[0] * r)
    padded[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return padded.transpose(2, 0, 1).astype(np.float32)


print("=" * 72)
print("1. QUANT_INFO.JSON - SECILEN FIX_POINT DEGERLERI")
print("=" * 72)
if QUANT_INFO.is_file():
    info = json.loads(QUANT_INFO.read_text())

    def walk(node, path=""):
        """Sozlugu gezip fix_point tasiyan girisleri toplar."""
        found = []
        if isinstance(node, dict):
            for key, value in node.items():
                sub = f"{path}/{key}" if path else str(key)
                if key in ("output", "input") and isinstance(value, dict):
                    for name, spec in value.items():
                        if isinstance(spec, list) and len(spec) >= 2:
                            found.append((f"{sub}/{name}", spec))
                found.extend(walk(value, sub))
        elif isinstance(node, list):
            for i, value in enumerate(node):
                found.extend(walk(value, f"{path}[{i}]"))
        return found

    entries = walk(info)
    keys = ("cls_preds", "obj_preds", "reg_preds", "concat", "24084", "24378",
            "24672")
    hits = [(n, s) for n, s in entries
            if any(k in n for k in keys)]
    if hits:
        print(f"{'tensor':<58}{'[bit, fix_point]'}")
        print("-" * 78)
        for name, spec in hits[:40]:
            print(f"{name[-56:]:<58}{spec}")
    else:
        print("  Bas tensorleri isimle bulunamadi; ham yapi ozeti:")
        print("  ust duzey anahtarlar:", list(info)[:10])
        for name, spec in entries[:15]:
            print(f"    {name[-60:]:<62}{spec}")
    print()
    print("  NOT: dequantize adimi = 2^-fix_point. reg kanallari icin bu adim")
    print("       stride ile carpilarak piksel hatasina donusur.")
else:
    print(f"  {QUANT_INFO} yok - once kalibrasyonu calistirin.")

print()
print("=" * 72)
print("2. FLOAT MODELIN GERCEK KANAL ARALIKLARI")
print("=" * 72)
exp = load_exp(EXP)
model = exp.get_model()
raw = torch.load(CKPT, map_location="cpu")
state = raw.get("model", raw)
state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
model.load_state_dict(state, strict=True)
model.eval()


class Deploy(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.backbone, self.head = m.backbone, m.head

    def forward(self, x):
        outs = []
        for k, feat in enumerate(self.backbone(x)):
            xk = self.head.stems[k](feat)
            outs.append((self.head.reg_preds[k](self.head.reg_convs[k](xk)),
                         self.head.obj_preds[k](self.head.reg_convs[k](xk)),
                         self.head.cls_preds[k](self.head.cls_convs[k](xk))))
        return outs


deploy = Deploy(model).eval()

val = json.loads((DATA / "annotations" / "instances_val.json").read_text())
names = [i["file_name"] for i in val["images"]]
step = max(1, len(names) // N_IMAGES)
picks = names[::step][:N_IMAGES]

lo = [np.full(7, np.inf) for _ in range(3)]
hi = [np.full(7, -np.inf) for _ in range(3)]
with torch.no_grad():
    for name in picks:
        img = cv2.imread(str(DATA / "images" / name))
        if img is None:
            continue
        x = torch.from_numpy(letterbox(img, tuple(exp.test_size))).unsqueeze(0)
        for lvl, (reg, obj, cls) in enumerate(deploy(x)):
            block = torch.cat([reg, obj, cls], 1)[0]  # [7, H, W]
            lo[lvl] = np.minimum(lo[lvl], block.amin(dim=(1, 2)).numpy())
            hi[lvl] = np.maximum(hi[lvl], block.amax(dim=(1, 2)).numpy())

for lvl, stride in enumerate((8, 16, 32)):
    span = float(max(abs(lo[lvl]).max(), abs(hi[lvl]).max()))
    fp = int(np.floor(np.log2(127.0 / span))) if span > 0 else 0
    quant = 2.0 ** -fp
    print(f"\n--- seviye {lvl}  (stride {stride}) ---")
    print(f"{'kanal':<8}{'min':>10}{'max':>10}{'genislik':>11}"
          f"{'ortak olcekte seviye':>22}")
    for c in range(7):
        width = float(hi[lvl][c] - lo[lvl][c])
        levels = width / quant
        print(f"{CHANNEL_ROLE[c]:<8}{lo[lvl][c]:>10.3f}{hi[lvl][c]:>10.3f}"
              f"{width:>11.3f}{levels:>22.1f}")
    print(f"  tensorun toplam genisligi : +-{span:.2f}"
          f"  -> ortak fix_point ~{fp}, adim {quant:.4f}")
    print(f"  reg_x/reg_y adimi piksel  : {quant * stride:.2f} px")
    print(f"  reg_w/reg_h boyut hatasi  : ~%{100*(np.exp(quant/2)-1):.1f}")

print()
print("=" * 72)
print("3. YORUM")
print("=" * 72)
print("  'ortak olcekte seviye' sutunu, o kanalin 256 seviyeden kacini")
print("  kullanabildigini gosterir. reg kanallari icin bu sayi 20'nin")
print("  altindaysa hipotez dogrulanmis demektir: concat, dar araliktaki")
print("  regresyon ciktisini genis araliktaki logitlerle ayni olcege")
print("  zorluyor ve konumlandirma hassasiyeti yok oluyor (AP75 cokusu).")
