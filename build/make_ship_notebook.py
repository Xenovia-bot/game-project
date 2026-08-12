#!/usr/bin/env python3
"""Gemi tespiti (tek sinif "ship") icin Kaggle not defterini uretir.

%%writefile hucrelerinin govdesi burada bos birakilir;
tools/_sync_ship_notebook_embeds.py onlari kaynak dosyalardan doldurur.

NOT DEFTERINI ELLE DUZENLEMEYIN: notebook bu betigin ciktisidir, elle
eklenen hucreler bir sonraki uretimde kaybolur. Yeni hucre buraya eklenir.
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

CELLS.append(md("""# KV260 icin YOLOX-Tiny - Gemi Tespiti (Kaggle)

**Hedef:** tek sinif `ship`. Kamera artik havadan degil, gemiyle yaklasik
ayni mesafede/yukseklikte (kiyi, iskele, gemi guvertesi). Girdi termal veya
gri ton olabilir; egitim buna gore alan-saglamligi augmentasyonu icerir
(bkz. `yolox_tiny_ship.py` basligi).

**Bagli olmasi gereken Kaggle veri seti** (Add Input): 6 ham kaynak zip'i
(`tools/prepare_ship_kaggle_upload.py` ile hazirlanir) -- hangi klasor/dataset
adiyla yuklendigi ONEMLI DEGIL, asagidaki kesif hucresi kaynaklari klasor
adindan degil **kategori imzasindan** tanir.

Ayarlar: **Accelerator = GPU**, **Internet = On**.
"""))

CELLS.append(code('''import os
from pathlib import Path

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
%cd {WORK}
!nvidia-smi

EXP_FILE = "yolox_tiny_ship.py"
INIT_URL = ("https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
            "0.1.1rc0/yolox_tiny.pth")
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
            "thop", "ninja", "albumentations") == 0, "Yardimci paket kurulumu basarisiz."

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
import albumentations
print("albumentations:", albumentations.__version__)
'''))

# 3, 4, 5, 6 -> %%writefile (tools/_sync_ship_notebook_embeds.py doldurur).
# build_ship_dataset.py, kardes modulu tam adi ile import eder; eski aerial
# aracinin adi olan build_dataset.py yazilirsa Kaggle'da ModuleNotFoundError
# ile durur.
CELLS.append(code("%%writefile dataset_common.py\n"))
CELLS.append(code("%%writefile build_ship_dataset.py\n"))
CELLS.append(code("%%writefile yolox_tiny_ship.py\n"))
# Metrik modulu: mAP + kart esiginde P/R/F1. Ayni dosya VM'de de kullanilir
# (quantize_yolox.py onu import eder), boylece float/INT8/Kaggle sayilari
# ayni tanimla uretilir.
CELLS.append(code("%%writefile ship_metrics.py\n"))

CELLS.append(code('''# 6 ham kaynagi bul ve tek sinifli COCO'ya birlestir.
#
# Kaynaklar KATEGORI IMZASINDAN taninir, klasor/dosya adindan DEGIL: Kaggle
# yuklenen zip'i acabilir ya da acmayabilir, ve kullanicinin verdigi dataset
# slug'i onceden bilinmiyor (aerial projedeki 'discover()' ile ayni gerekce
# -- bkz. HANDOFF, "isaret dosyasindan yukari yurume").
import json as _json
import zipfile as _zipfile

INPUT = Path("/kaggle/input")
NEEDED = ("vais_smd_marvel", "singapore_maritime", "sea_vessels",
          "ship_model", "ir_thermal", "wutdet")

# tools/build_ship_dataset.py CLASS_MAPS ile BIREBIR ayni tutulmali (test bunu
# zorunlu kilar). Her kumeyi COCO kategorilerinin TAM kumesiyle karsilastirir
# -- alt-kume degil, tam esitlik: "ship_model"in tek kategorisi "ship" iken
# baska hicbir kaynak kategori kumesini {"ship"}'e daraltmiyor.
CATEGORY_SIGNATURES = {
    "vais_smd_marvel": {"vessel", "buoy", "object"},
    "singapore_maritime": {"objects", "boat", "buoy", "ferry",
                           "flying bird-plane", "kayak", "other",
                           "sail boat", "speed boat", "vessel-ship"},
    "sea_vessels": {"sea-vessels", "fishing boat", "merchant ship",
                    "military ship", "patrol boat", "sails boat",
                    "submarine", "tugboat", "yacht"},
    "ship_model": {"ship"},
    "ir_thermal": {"boat", "bulk carrier", "canoe", "container ship",
                   "fishing boat", "liner", "sailboat", "warship"},
}


def _classify(names):
    lowered = frozenset(str(n).strip().lower() for n in names)
    for key, sig in CATEGORY_SIGNATURES.items():
        if lowered == sig:
            return key
    return None


def _scan_extracted(found):
    # 5 COCO kaynagi: <kok>/<bolum>/_annotations.coco.json
    for js in sorted(INPUT.rglob("_annotations.coco.json")):
        try:
            cats = _json.loads(js.read_text()).get("categories", [])
        except Exception:
            continue
        key = _classify(c.get("name", "") for c in cats)
        if key:
            found.setdefault(key, js.parent.parent)

    # WUTDet: voc/Annotations/*.xml + voc/JPEGImages/ kardes klasorleri.
    # Kok, Annotations'in DOGRUDAN ebeveyni -- tum arsivi degil yalnizca
    # WUTDet'e ait dosyalari kapsasin diye (paylasilan ust klasor hatasi,
    # bkz. aerial projede test_parent_directory_is_not_mistaken_for_a_source).
    for ann in sorted(INPUT.rglob("Annotations")):
        if not ann.is_dir() or not (ann.parent / "JPEGImages").is_dir():
            continue
        if not any(ann.glob("*.xml")):
            continue
        found.setdefault("wutdet", ann.parent)


def _scan_zips(found):
    """Kaggle zip'i acmadiysa arsiv iceriginden tani."""
    for path in sorted(INPUT.rglob("*.zip")):
        try:
            with _zipfile.ZipFile(path) as z:
                names = z.namelist()
                js = [n for n in names
                      if n.rsplit("/", 1)[-1] == "_annotations.coco.json"]
                if js:
                    cats = _json.loads(z.read(sorted(js)[0])).get("categories", [])
                    key = _classify(c.get("name", "") for c in cats)
                    if key:
                        found.setdefault(key, path)
                    continue
                parts = {p for n in names for p in n.split("/")}
                has_xml = any(n.lower().endswith(".xml") for n in names)
                if {"Annotations", "JPEGImages"} <= parts and has_xml:
                    found.setdefault("wutdet", path)
        except Exception:
            continue


def discover_ship_sources():
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


SOURCES = discover_ship_sources()
for _name in NEEDED:
    print(f"{_name:<20} {SOURCES[_name]}")

DATASET_DIR = Path(WORK) / "datasets" / "ship_merged"

# Kabuk dizesi yerine argüman listesi: kaynak yollarinda bosluk var
# ("Sea Vessels Dataset...", "ship model..."), kabuk alintilamasi kirilgan olurdu.
_cmd = [sys.executable, "build_ship_dataset.py",
        "--out", str(DATASET_DIR),
        "--images-out", str(DATASET_DIR / "images")]
for _key, _path in SOURCES.items():
    _cmd += ["--source", f"{_key}={_path}"]

_proc = subprocess.run(_cmd, text=True)
assert _proc.returncode == 0, (
    "build_ship_dataset.py basarisiz oldu; yukaridaki cikti hatanin sebebini yazar."
)
'''))

CELLS.append(code('''# Baslangic agirligi: YOLOX'un resmi Megvii COCO checkpoint'i (yolox_tiny).
# COCO'da "boat" sinifi var; tek sinifli bu fine-tune icin dogrudan uygun
# bir baslangic noktasi (aerial projedeki ayni gerekce).
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

# Tek sinifli head (COCO'nun 80'i degil) ve sabit DPUFocus agirligi yeni
# modelin baslangicindan, diger katmanlar Megvii'den gelir.
init_state = dict(ref_state)
init_state.update(matched)
INIT_CKPT = str(WDIR / "init_ckpt.pth")
torch.save({
    "model": init_state,
    "meta": {"source": str(mg), "yolox_commit": YOLOX_COMMIT,
             "matched_ratio": ratio},
}, INIT_CKPT)
print(f"Megvii baslangici dogrulandi: {ratio:.1%} -> {INIT_CKPT}")
print("Yalnizca cls head (1 sinif) ve sabit DPUFocus katmani yeniden baslatildi.")
'''))

CELLS.append(md("## Egitim"))

CELLS.append(code('''# Tek sinif (ship), girdi 512x512 (kare), 30 epoch -- bkz. yolox_tiny_ship.py
# basligindaki olcum: kaynaklarin %62'si kare, kutu boyutu heterojen.
# 30 epoch TEMKINLI bir baslangic (bu veri icin epoch-AP egrisi henuz
# olculmedi) -- TensorBoard/log'daki AP@0.50 egrisine bakip gerekirse
# ayarlayin.
BATCH = 32  # 512x512 daha kucuk oldugu icin aerial'in 16'sindan yuksek
            # baslanabilir; OOM olursa 16, sonra 8 yapin.

%cd {WORK}
!python YOLOX/tools/train.py -f {EXP_FILE} -d 1 -b {BATCH} --fp16 -c weights/init_ckpt.pth
'''))

CELLS.append(code('''# Devam (resume): onceki oturumun Output'unu bu oturuma input olarak
# bagladiktan sonra asagidaki degiskene latest_ckpt.pth yolunu yazin.
RESUME_CKPT = ""  # or. "/kaggle/input/ONCEKI/YOLOX_outputs/yolox_tiny_ship/latest_ckpt.pth"

if RESUME_CKPT:
    !python YOLOX/tools/train.py -f {EXP_FILE} -d 1 -b {BATCH} --fp16 --resume -c "{RESUME_CKPT}"
else:
    print("RESUME_CKPT bos; bu hucre yalnizca yarim kalan egitimi surdurmek icin.")
'''))

CELLS.append(code('''# Degerlendirme: standart COCO mAP (bu veri setinde VisDrone'un ignore-region
# kavrami yok, ozel protokole gerek yok -- bkz. yolox_tiny_ship.py get_evaluator).


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
!python YOLOX/tools/eval.py -f {EXP_FILE} -c "{BEST_CKPT}" -d 1 -b 32 --conf 0.001
'''))

CELLS.append(md("""### Teslim edilecek metrikler

Yukaridaki hucre YOLOX'un standart AP ciktisidir. Asagidaki hucre ayni
checkpoint icin **mAP + kart esiginde P/R/F1 + kaynak bazli tablo** uretir;
ayni `ship_metrics.py` VM'de kuantalama sonrasi da kullanilir, yani float ve
INT8 sayilari ayni tanimla olculur.

Uc set uzerinde birden calisir:

| set | ne icin |
|---|---|
| val | model secimi bu sette yapildi -- iyimser, kiyas icin |
| **test** | **raporlanacak sayi**; egitim/ayar bu seti hic gormedi |
| val (gri) | gri-ton dayanikliligi: AP dususu dayanikliligin olcusu |
"""))

CELLS.append(code('''# mAP + F1 + kaynak bazli AP.
# Kaynak bazli tablo neden: kaynaklar cok farkli (ir_thermal GERCEK termal,
# vais'in %60'i gri ton, digerleri RGB). Tek global sayi, termalin cokmesini
# gizleyebilir -- termal/gri dayaniklilik sarti ancak o satirdan dogrulanir.
import contextlib
import importlib.util
import io as _io

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO

from yolox.data import ValTransform
from yolox.utils import postprocess

from ship_metrics import evaluate_ship, format_metrics

_BGR_LUMA = np.array([0.114, 0.587, 0.299], dtype=np.float32).reshape(3, 1, 1)

spec3 = importlib.util.spec_from_file_location("exp_mod3", EXP_FILE)
_mod3 = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(_mod3)
_exp = _mod3.Exp()
_model = _exp.get_model().eval()
try:
    _ckpt = torch.load(BEST_CKPT, map_location="cpu", weights_only=False)
except TypeError:
    _ckpt = torch.load(BEST_CKPT, map_location="cpu")
_model.load_state_dict(_ckpt["model"], strict=True)
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model.to(_device)
_transform = ValTransform(legacy=False)


def detect_all(ann_name, gray=False, batch=16, conf=0.001):
    """Egitimdeki ValTransform'un aynisi; gray=True ise BT.601 parlakligi
    (GrayThermalTransform ile ayni formul)."""
    with contextlib.redirect_stdout(_io.StringIO()):
        coco = COCO(str(DATASET_DIR / "annotations" / ann_name))
    images = coco.dataset["images"]
    results = []
    for start in range(0, len(images), batch):
        chunk = images[start:start + batch]
        tensors, ratios = [], []
        for meta in chunk:
            raw = cv2.imread(str(DATASET_DIR / "images" / meta["file_name"]))
            img, _ = _transform(raw, None, _exp.test_size)
            if gray:
                img = np.repeat((img * _BGR_LUMA).sum(axis=0, keepdims=True), 3, axis=0)
            tensors.append(torch.from_numpy(np.ascontiguousarray(img)))
            ratios.append(min(_exp.test_size[0] / raw.shape[0],
                              _exp.test_size[1] / raw.shape[1]))
        with torch.no_grad():
            out = _model(torch.stack(tensors).float().to(_device))
            out = postprocess(out, _exp.num_classes, conf, _exp.nmsthre)
        for meta, ratio, dets in zip(chunk, ratios, out):
            if dets is None:
                continue
            for *box, obj_conf, cls_conf, _cls in dets.cpu().numpy():
                x1, y1, x2, y2 = [v / ratio for v in box]
                results.append({"image_id": meta["id"], "category_id": 1,
                                "bbox": [float(x1), float(y1),
                                         float(x2 - x1), float(y2 - y1)],
                                "score": float(obj_conf * cls_conf)})
        if (start // batch) % 20 == 0:
            print(f"  {min(start + batch, len(images))}/{len(images)}", end="\\r")
    return coco, results


def report(ann_name, gray=False):
    coco, dets = detect_all(ann_name, gray=gray)
    if not dets:
        print(f"{ann_name}: hic tespit yok -- checkpoint veya esik yanlis olabilir.")
        return None
    metrics = evaluate_ship(coco, dets, deploy_conf=_exp.deploy_conf)
    label = f"{ann_name}{' [GRI]' if gray else ''}"
    print(format_metrics(metrics, label))
    print()
    return metrics["ap"]


ap_val = report("instances_val.json")
ap_test = report("instances_test.json")          # RAPORLANACAK SAYI
ap_gray = report("instances_val.json", gray=True)

print("=" * 66)
if ap_val and ap_test:
    print(f"val {ap_val:.4f} -> test {ap_test:.4f}  "
          f"(fark {ap_test - ap_val:+.4f}; val model seciminden dolayi iyimser)")
if ap_val and ap_gray:
    print(f"GRI DAYANIKLILIGI: {ap_val:.4f} -> {ap_gray:.4f} "
          f"({(ap_gray - ap_val) / ap_val * 100:+.1f}%)")
print("VM'de kuantalama sonrasi ayni modulle olcun; sayilar kiyaslanabilir.")
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

# Her kaynaktan birer ornek: 6 kaynagin da nasil goruldugunu izleyelim
val_json = json.loads(
    (DATASET_DIR / "annotations" / "instances_val.json").read_text())
by_source = {}
for _image in val_json["images"]:
    by_source.setdefault(_image["source"], []).append(_image["file_name"])
random.seed(0)
sample_names = [random.choice(v) for v in by_source.values()]

fig, axes = plt.subplots(len(sample_names), 1,
                         figsize=(10, 8 * len(sample_names)))
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
for _name in ("yolox_tiny_ship.py", "build_ship_dataset.py",
              "dataset_common.py", "ship_metrics.py"):
    shutil.copy(_name, ART / _name)
# Kuantalama train disindan kalibre edilmemeli: train anotasyonu PTQ
# kalibrasyonunu sinirlar, val anotasyonu ise yalniz accuracy gate icindir.
shutil.copy(DATASET_DIR / "annotations" / "instances_train.json",
            ART / "instances_train.json")
shutil.copy(DATASET_DIR / "annotations" / "instances_val.json",
            ART / "instances_val.json")
# Nihai sayi test setinde raporlanir (quantize_yolox.py --report-only)
shutil.copy(DATASET_DIR / "annotations" / "instances_test.json",
            ART / "instances_test.json")
(ART / "classes.txt").write_text("\\n".join(CLASSES))
(ART / "YOLOX_COMMIT.txt").write_text(YOLOX_COMMIT + "\\n")
(ART / "EXP_FILE.txt").write_text(EXP_FILE + "\\n")
zip_path = shutil.make_archive(str(Path(WORK) / "yolox_ship_artifacts"),
                               "zip", ART)
print("Hazir:", zip_path)
print("Not defteri kaydedilince Output sekmesinden indirebilirsiniz.")
print("VM'e ayrica val goruntuleri lazim:", DATASET_DIR / "images")
'''))

CELLS.append(md("""## Sonraki adim: Oracle VM'de kuantalama

`yolox_ship_artifacts.zip` dosyasini ve `datasets/ship_merged/images` klasorunu
VM'e tasiyin, ardindan `quantize/README.md` adimlarini izleyin (exp dosyasi
olarak `yolox_tiny_ship.py`, sinif sayisi 1).
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
    target = ROOT / "training" / "kaggle_ship_yolox.ipynb"
    target.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(f"yazildi: {target}  ({len(CELLS)} hucre)")
    for i, cell in enumerate(CELLS):
        head = "".join(cell["source"]).split("\n")[0][:58]
        print(f"  {i:>2} {cell['cell_type']:<9} {head}")


if __name__ == "__main__":
    main()
