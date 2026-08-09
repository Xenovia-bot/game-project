#!/usr/bin/env python3
"""Kaggle not defterini yeni (birlestirilmis 2 sinifli) veri akisina gore uretir.

%%writefile hucrelerinin govdesi bos birakilir; tools/_sync_notebook_embeds.py
onlari kaynak dosyalardan doldurur.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.splitlines(keepends=True)}


CELLS = []

CELLS.append(md("""# KV260 icin YOLOX-Nano - Havadan Arac Tespiti (Kaggle)

**Hedef:** 2 sinif - `land_vehicle`, `sea_vehicle`. Her tespitin merkez
noktasi (cx, cy) uretilir; model KV260 DPU'suna (DPUCZDX8G) derlenir.

**Bagli olmasi gereken Kaggle veri setleri** (Add Input):

| Veri seti | Icerik |
| --- | --- |
| VisDrone2019-DET (train + val) | kara araci: car / van / truck / bus |
| aerial-vehicle-sources | VESSELimg + Military Vehicle Recognition + Mendeley UAV Military |

Ayarlar: **Accelerator = GPU**, **Internet = On**.
"""))

CELLS.append(code('''import os
from pathlib import Path

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
%cd {WORK}
!nvidia-smi

# ============================ VARYANT SECIMI ============================
# "nano" : depthwise conv kullanir, 0.9M parametre, hizli.
# "tiny" : depthwise YOK, ~5M parametre. INT8 kuantalamaya cok daha uygun --
#          Nano'da INT8 kaybi 0.2285 olculdu ve kok neden depthwise conv'larin
#          per-tensor kuantalanmasiydi (bkz. HANDOFF). Bedeli: DPU suresi
#          artar, 30 FPS butcesi kartta yeniden olculmelidir.
VARIANT = "nano"        # <-- Tiny egitmek icin "tiny" yapin
# ========================================================================

EXP_FILE = f"yolox_{VARIANT}_visdrone.py"
INIT_URL = ("https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
            f"0.1.1rc0/yolox_{VARIANT}.pth")
print("varyant :", VARIANT)
print("exp     :", EXP_FILE)
print("baslangic agirligi:", INIT_URL.rsplit("/", 1)[-1])
'''))

CELLS.append(code('''# YOLOX kurulumu (Kaggle'daki hazir torch surumune dokunmadan).
# Kaggle ve Vitis AI VM ayni commit'i kullanir; main dali kullanilmaz.
import importlib
import site
import subprocess
import sys

YOLOX_COMMIT = "6ddff4824372906469a7fae2dc3206c7aa4bbaee"
YOLOX_DIR = Path(WORK) / "YOLOX"

if not YOLOX_DIR.is_dir():
    !git clone --filter=blob:none --no-checkout https://github.com/Megvii-BaseDetection/YOLOX.git "{YOLOX_DIR}"
!git -C "{YOLOX_DIR}" fetch --depth 1 origin {YOLOX_COMMIT}
!git -C "{YOLOX_DIR}" checkout --detach {YOLOX_COMMIT}
current = !git -C "{YOLOX_DIR}" rev-parse HEAD
assert current and current[0] == YOLOX_COMMIT, f"Yanlis YOLOX commit'i: {current}"


def _pip(*args):
    """pip'i cagirir. `!pip` kabuk cagrisinin aksine hatayi yutmaz."""
    proc = subprocess.run([sys.executable, "-m", "pip", *args],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:])
    return proc.returncode


# --no-build-isolation sart: YOLOX'un setup.py'si torch'u import eder, pip'in
# izole build ortaminda torch bulunmaz ve kurulum sessizce basarisiz olur.
rc = _pip("install", "--no-deps", "--no-build-isolation", "-e", str(YOLOX_DIR))
if rc != 0:
    print("Editable kurulum basarisiz; editable olmayan kuruluma dusuluyor.")
    rc = _pip("install", "--no-deps", "--no-build-isolation", str(YOLOX_DIR))
assert rc == 0, "YOLOX kurulumu basarisiz (yukaridaki pip ciktisina bakin)."

assert _pip("install", "-q", "loguru", "tabulate", "psutil", "pycocotools",
            "thop", "ninja") == 0, "Yardimci paket kurulumu basarisiz."

# pip'in yazdigi .pth dosyalari yalnizca yorumlayici acilisinda okunur; calisan
# kernel'in sys.path'ini elle tazelemezsek import ayni oturumda basarisiz olur.
for _site_dir in getattr(site, "getsitepackages", list)():
    site.addsitedir(_site_dir)
if str(YOLOX_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOX_DIR))
importlib.invalidate_caches()

# numpy 1.24+ uyumlulugu: kaldirilan eski takma adlar icin shim
import numpy as np
for _alias, _type in (("float", float), ("int", int), ("bool", bool)):
    if _alias not in np.__dict__:
        setattr(np, _alias, _type)

import torch
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
import yolox
print("yolox:", yolox.__version__, "|", yolox.__file__)
'''))

# 3, 4, 5, 6 -> %%writefile (tools/_sync_notebook_embeds.py doldurur)
CELLS.append(code("%%writefile build_dataset.py\n"))
CELLS.append(code("%%writefile visdrone_eval.py\n"))
CELLS.append(code("%%writefile yolox_nano_visdrone.py\n"))
# Tiny, Nano'yu devraldigi icin VARIANT ne olursa olsun ikisi de yazilmali.
CELLS.append(code("%%writefile yolox_tiny_visdrone.py\n"))

CELLS.append(code("""# Bagli veri setlerini bul ve tek bir 2-sinifli COCO'ya birlestir.
#
# Kaynaklar ISARET DOSYALARINDAN bulunur, klasor adindan degil. Onceki
# yaklasim 'once eslesen kazanir' idi ve ust klasorler alt agaclarinda isaret
# dosyasi bulundurdugu icin kok dizini secebiliyordu. Artik once isaret
# dosyasi bulunur, kaynak koku oradan YUKARI yurunerek belirlenir.
import json as _json
import zipfile as _zipfile

INPUT = Path("/kaggle/input")
NEEDED = ("visdrone-train", "visdrone-val", "vesselimg", "milrec", "mendeley")


def _classify_categories(names):
    lowered = {str(n).lower() for n in names}
    if {"container", "tugboat"} & lowered:
        return "vesselimg"
    if {"tank", "armoured personnel carrier"} & lowered:
        return "milrec"
    return None


def _scan_extracted(found):
    # Roboflow COCO: <kok>/<bolum>/_annotations.coco.json
    for js in sorted(INPUT.rglob("_annotations.coco.json")):
        try:
            cats = _json.loads(js.read_text()).get("categories", [])
        except Exception:
            continue
        kind = _classify_categories(c.get("name", "") for c in cats)
        if kind:
            found.setdefault(kind, js.parent.parent)

    # VisDrone: kardes images/ + annotations/ klasorleri
    for ann in sorted(INPUT.rglob("annotations")):
        if not ann.is_dir() or not (ann.parent / "images").is_dir():
            continue
        if not any(ann.glob("*.txt")):
            continue
        root = ann.parent
        lowered = str(root).lower()
        key = "visdrone-val" if "val" in lowered else "visdrone-train"
        found.setdefault(key, root)

    # Mendeley: kardes images/ + labels/ klasorleri
    for lab in sorted(INPUT.rglob("labels")):
        if lab.is_dir() and (lab.parent / "images").is_dir():
            found.setdefault("mendeley", lab.parent.parent)
            break


def _scan_zips(found):
    \"\"\"Kaggle zip'i acmadiysa arsiv iceriginden tani.\"\"\"
    for path in sorted(INPUT.rglob("*.zip")):
        try:
            with _zipfile.ZipFile(path) as z:
                names = z.namelist()
                js = [n for n in names
                      if n.rsplit("/", 1)[-1] == "_annotations.coco.json"]
                if js:
                    cats = _json.loads(z.read(sorted(js)[0])).get("categories", [])
                    kind = _classify_categories(c.get("name", "") for c in cats)
                    if kind:
                        found.setdefault(kind, path)
                    continue
                parts = {p for n in names for p in n.split("/")}
                lowered = path.name.lower()
                if {"images", "annotations"} <= parts:
                    key = "visdrone-val" if "val" in lowered else "visdrone-train"
                    found.setdefault(key, path)
                elif {"images", "labels"} <= parts:
                    found.setdefault("mendeley", path)
        except Exception:
            continue


def discover():
    found = {}
    _scan_extracted(found)
    if set(NEEDED) - set(found):
        _scan_zips(found)
    missing = [k for k in NEEDED if k not in found]
    if missing:
        listing = "\\n".join(f"  {p}" for p in sorted(INPUT.glob("*")))
        raise SystemExit(
            f"Bulunamayan kaynaklar: {missing}\\n"
            f"Bagli veri setleri:\\n{listing}\\n"
            f"Add Input ile eksik olani baglayin."
        )
    return found


SOURCES = discover()
for _name in NEEDED:
    print(f"{_name:<16} {SOURCES[_name]}")

DATASET_DIR = Path(WORK) / "datasets" / "merged"

# Kabuk dizesi yerine argüman listesi: kaynak yollarinda bosluk var
# ("A Multi-Class UAV Military ..."), kabuk alintilamasi kirilgan olurdu.
_cmd = [sys.executable, "build_dataset.py",
        "--out", str(DATASET_DIR),
        "--images-out", str(DATASET_DIR / "images")]
for _key, _path in SOURCES.items():
    _cmd += ["--source", f"{_key}={_path}"]

_proc = subprocess.run(_cmd, text=True)
assert _proc.returncode == 0, (
    "build_dataset.py basarisiz oldu; yukaridaki cikti hatanin sebebini yazar."
)
"""))

CELLS.append(code('''# Baslangic agirligi: YOLOX'un resmi Megvii COCO checkpoint'i.
# COCO'da hem arac hem `boat` siniflari var; bu fine-tune icin dogrudan
# uygun bir baslangic noktasi.
import importlib.util
import urllib.request

WDIR = Path(WORK) / "weights"
WDIR.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location("exp_mod", EXP_FILE)
exp_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp_mod)
ref_state = exp_mod.Exp().get_model().state_dict()

mg = WDIR / INIT_URL.rsplit("/", 1)[-1]
if not mg.exists():
    urllib.request.urlretrieve(INIT_URL, mg)

# PyTorch 2.6+ varsayilani weights_only=True, Vitis AI 3.0/PyTorch 1.12 ise
# bu parametreyi tanimaz. Iki ortamda da acik ve guvenilir yerel ckpt yukle.
try:
    raw = torch.load(mg, map_location="cpu", weights_only=False)
except TypeError:
    raw = torch.load(mg, map_location="cpu")
inner = raw.get("model", raw.get("state_dict", raw)) if isinstance(raw, dict) else raw
inner = {(k[7:] if k.startswith("module.") else k): v for k, v in inner.items()}

matched = {
    k: v for k, v in inner.items()
    if k in ref_state and ref_state[k].shape == v.shape
}
missing = sorted(set(ref_state) - set(matched))
allowed_missing = all(
    k == "backbone.backbone.stem.space_to_depth.weight" or ".cls_preds." in k
    for k in missing
)
ratio = len(matched) / max(len(ref_state), 1)
assert ratio >= 0.95 and allowed_missing, (
    f"Baslangic checkpoint'i mimariyle uyumsuz: eslesme={ratio:.1%}, "
    f"beklenmeyen eksikler={missing}"
)

# 2 sinifli head ve sabit DPUFocus agirligi yeni modelin baslangicindan,
# diger katmanlar Megvii'den gelir.
init_state = dict(ref_state)
init_state.update(matched)
INIT_CKPT = str(WDIR / "init_ckpt.pth")
torch.save({
    "model": init_state,
    "meta": {"source": str(mg), "yolox_commit": YOLOX_COMMIT,
             "matched_ratio": ratio},
}, INIT_CKPT)
print(f"Megvii baslangici dogrulandi: {ratio:.1%} -> {INIT_CKPT}")
print("Yalnizca cls head (2 sinif) ve sabit DPUFocus katmani yeniden baslatildi.")
'''))

CELLS.append(md("## Egitim"))

CELLS.append(code('''# 2 sinif (land_vehicle / sea_vehicle), girdi 896x512 (16:9), 40 epoch.
# 10 sinifli referans kosumda model epoch ~40'ta doymustu; 80 yerine 40 epoch.
BATCH = 16  # OOM olursa 8 yapin

%cd {WORK}
!python YOLOX/tools/train.py -f {EXP_FILE} -d 1 -b {BATCH} --fp16 -c weights/init_ckpt.pth
'''))

CELLS.append(code('''# Devam (resume): onceki oturumun Output'unu bu oturuma input olarak
# bagladiktan sonra asagidaki degiskene latest_ckpt.pth yolunu yazin.
RESUME_CKPT = ""  # or. "/kaggle/input/ONCEKI/YOLOX_outputs/yolox_nano_visdrone/latest_ckpt.pth"

if RESUME_CKPT:
    !python YOLOX/tools/train.py -f {EXP_FILE} -d 1 -b {BATCH} --fp16 --resume -c "{RESUME_CKPT}"
else:
    print("RESUME_CKPT bos; bu hucre yalnizca yarim kalan egitimi surdurmek icin.")
'''))

CELLS.append(code('''# Degerlendirme: resmi VisDrone ignore filtresi + global top-500 + VOC AP,
# ayrica sinif bazli AP ve P/R/F1. Bu sayilari not edin: kuantalama sonrasi
# ayni protokolle karsilastirilacak.


def find_best_ckpt():
    """Bu oturumda egitildiyse yerelden, aksi halde bagli input'tan alir."""
    local = Path(f"YOLOX_outputs/{Path(EXP_FILE).stem}/best_ckpt.pth")
    if local.exists():
        return local
    if INPUT.exists():
        for candidate in sorted(INPUT.glob("**/best_ckpt.pth")):
            return candidate
    raise SystemExit(
        "best_ckpt.pth bulunamadi. Once egitim hucresini calistirin veya "
        "onceki oturumun ciktisini bu oturuma input olarak baglayin."
    )


BEST_CKPT = str(find_best_ckpt())
print("checkpoint:", BEST_CKPT)
!python YOLOX/tools/eval.py -f {EXP_FILE} -c "{BEST_CKPT}" -d 1 -b 16 --conf 0.001
'''))

CELLS.append(code('''# Gorsel kontrol: kutular + MERKEZ NOKTALARI (cx, cy)
# KV260 uygulamasindaki hesabin aynisi: cx=(x1+x2)/2, cy=(y1+y2)/2
import json
import random

import cv2
import matplotlib.pyplot as plt

from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess

spec2 = importlib.util.spec_from_file_location("exp_mod2", EXP_FILE)
exp_mod2 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(exp_mod2)
exp = exp_mod2.Exp()
CLASSES = exp_mod2.TARGET_CLASSES

model = exp.get_model().eval()
try:
    ckpt = torch.load(BEST_CKPT, map_location="cpu", weights_only=False)
except TypeError:
    ckpt = torch.load(BEST_CKPT, map_location="cpu")
model.load_state_dict(ckpt["model"], strict=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

CONF_VIS = 0.15  # kartla ayni dagitim esigi
val_transform = ValTransform(legacy=False)

# Her kaynaktan birer ornek: iki sinifi da gorelim
val_json = json.loads(
    (DATASET_DIR / "annotations" / "instances_val.json").read_text())
by_source = {}
for _image in val_json["images"]:
    by_source.setdefault(_image["source"], []).append(_image["file_name"])
random.seed(0)
sample_names = [random.choice(v) for v in by_source.values()]

fig, axes = plt.subplots(len(sample_names), 1,
                         figsize=(16, 9 * len(sample_names)))
axes = np.atleast_1d(axes)
for ax, name in zip(axes, sample_names):
    img0 = cv2.imread(str(DATASET_DIR / "images" / name))
    h0, w0 = img0.shape[:2]
    ratio = min(exp.test_size[0] / h0, exp.test_size[1] / w0)
    img, _ = val_transform(img0, None, exp.test_size)
    with torch.no_grad():
        out = model(torch.from_numpy(img).unsqueeze(0).float().to(device))
        out = postprocess(out, exp.num_classes, CONF_VIS, exp.nmsthre)[0]
    vis = img0.copy()
    if out is not None:
        for *box, obj_conf, cls_conf, cls_id in out.cpu().numpy():
            x1, y1, x2, y2 = [v / ratio for v in box]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            score = obj_conf * cls_conf
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)
            cv2.circle(vis, (int(cx), int(cy)), 4, (0, 0, 255), -1)
            cv2.putText(
                vis,
                f"{CLASSES[int(cls_id)]} {score:.2f} ({int(cx)},{int(cy)})",
                (int(x1), max(14, int(y1) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title(name)
    ax.axis("off")
plt.tight_layout()
plt.show()
'''))

CELLS.append(code('''# Oracle VM'e tasinacak dosyalari paketle
import shutil

ART = Path(WORK) / "artifacts"
ART.mkdir(exist_ok=True)
shutil.copy(BEST_CKPT, ART / "best_ckpt.pth")
for _name in ("yolox_nano_visdrone.py", "yolox_tiny_visdrone.py",
              "visdrone_eval.py", "build_dataset.py"):
    shutil.copy(_name, ART / _name)
# Kuantalama ayni val setiyle olculmeli: anotasyonu da tasi
shutil.copy(DATASET_DIR / "annotations" / "instances_val.json",
            ART / "instances_val.json")
(ART / "classes.txt").write_text("\\n".join(CLASSES))
(ART / "YOLOX_COMMIT.txt").write_text(YOLOX_COMMIT + "\\n")
# Hangi varyantin egitildigi artifact icinde yazili olsun: VM'de yanlis exp
# dosyasiyla kuantalamak sessiz bir hata olurdu.
(ART / "VARIANT.txt").write_text(VARIANT + "\\n")
(ART / "EXP_FILE.txt").write_text(EXP_FILE + "\\n")
zip_path = shutil.make_archive(str(Path(WORK) / "yolox_aerial_artifacts"),
                               "zip", ART)
print("Hazir:", zip_path)
print("Not defteri kaydedilince Output sekmesinden indirebilirsiniz.")
print("VM'e ayrica val goruntuleri lazim:", DATASET_DIR / "images")
'''))

CELLS.append(md("""## Sonraki adim: Oracle VM'de kuantalama

`yolox_aerial_artifacts.zip` dosyasini ve `datasets/merged/images` klasorunu
VM'e tasiyin, ardindan `quantize/README.md` adimlarini izleyin.
"""))


def main():
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = ROOT / "training" / "kaggle_visdrone_yolox.ipynb"
    target.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(f"yazildi: {target}  ({len(CELLS)} hucre)")
    for i, cell in enumerate(CELLS):
        head = "".join(cell["source"]).split("\n")[0][:58]
        print(f"  {i:>2} {cell['cell_type']:<9} {head}")


if __name__ == "__main__":
    main()
