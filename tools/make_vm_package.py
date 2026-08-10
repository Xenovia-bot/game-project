#!/usr/bin/env python3
"""Kuantalama VM'ine tasinacak minimum paketi hazirlar.

Neden: birlestirilmis goruntu agaci 3,6 GB ama VM'de yalnizca iki sey lazim:
  * INT8 kabul testi icin **tum val seti** (4.483 goruntu) -- accuracy gate
    alt kumeyle uretilemez, bu yuzden hepsi gerekli
  * PTQ kalibrasyonu icin egitim dagilimindan birkac yuz kare

Sonuc ~626 MB, yani 5,9 kat daha kucuk. VirtualBox paylasilan klasoru veya
scp uzerinden tasimak icin fark ediyor.

Kullanim:
    python tools/make_vm_package.py
    python tools/make_vm_package.py --out D:/vm_package --calib 500
"""

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CALIB_SEED = 42


def copy_images(names, images_root, target_root):
    """<kaynak>/<ad>.jpg duzenini koruyarak kopyalar; var olani atlar."""
    written = skipped = missing = 0
    total = 0
    for name in names:
        src = images_root / name
        if not src.is_file():
            missing += 1
            continue
        dst = target_root / name
        total += src.stat().st_size
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        written += 1
    return written, skipped, missing, total


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merged", default=str(ROOT / "datasets" / "merged"),
                        help="birlestirilmis veri seti koku")
    parser.add_argument("--artifacts", default=str(ROOT / "artifacts"),
                        help="Kaggle artifacts klasoru (best_ckpt.pth vs.)")
    parser.add_argument("--artifacts-tiny", default=str(ROOT / "artifacts_tiny"),
                        help="Tiny checkpoint klasoru; varsa vm_package/tiny/ "
                             "olarak kopyalanir (yoksa sessizce atlanir)")
    parser.add_argument("--out", default=str(ROOT / "vm_package"))
    parser.add_argument("--calib", type=int, default=300,
                        help="kalibrasyon icin secilecek train goruntusu sayisi")
    args = parser.parse_args()

    merged = Path(args.merged)
    artifacts = Path(args.artifacts)
    out = Path(args.out)
    images_root = merged / "images"
    ann_dir = merged / "annotations"

    for path in (images_root, ann_dir / "instances_val.json",
                 ann_dir / "instances_train.json"):
        if not path.exists():
            raise SystemExit(f"HATA: bulunamadi: {path}")

    # --- kaynak dosyalar ---
    needed = {
        "best_ckpt.pth": artifacts / "best_ckpt.pth",
        # Exp'ler REPO'dan gelir, artifacts kopyasindan degil. Sebep: Tiny
        # turevi `yolox_nano_visdrone.py`'yi import edip yalnizca
        # `self.depthwise = False` yapar. artifacts kopyasi `get_model()`
        # icinde `depthwise=True` yazan eski surumdur; Tiny onun yanina
        # konursa depthwise'li model kurulur -- tam kacinmaya calistigimiz
        # sey. Nano icin iki dosya mimari olarak ozdestir (fark yalnizca
        # sabit yerine `self.depthwise` kullanmak), ve checkpoint'in
        # mimariye uydugu `load_checkpoint_strict` kapisinda dogrulanir.
        "yolox_nano_visdrone.py": ROOT / "training" / "exps" / "yolox_nano_visdrone.py",
        "yolox_tiny_visdrone.py": ROOT / "training" / "exps" / "yolox_tiny_visdrone.py",
        "YOLOX_COMMIT.txt": artifacts / "YOLOX_COMMIT.txt",
        # visdrone_eval.py artifacts'takinin degil REPO'daki temiz surumu:
        # ikisi sayisal olarak ozdes, tek kaynak olsun diye repo secildi.
        "visdrone_eval.py": ROOT / "training" / "visdrone_eval.py",
        "quantize_yolox.py": ROOT / "quantize" / "quantize_yolox.py",
        "qat_probe.py": ROOT / "quantize" / "qat_probe.py",
        "compile_kv260.sh": ROOT / "quantize" / "compile_kv260.sh",
    }
    eksik = [str(p) for p in needed.values() if not p.is_file()]
    if eksik:
        raise SystemExit("HATA: su dosyalar yok:\n  " + "\n  ".join(eksik))

    out.mkdir(parents=True, exist_ok=True)
    for name, src in needed.items():
        shutil.copy2(src, out / name)
    print(f">> {len(needed)} kaynak dosya kopyalandi")

    # --- Tiny checkpoint: varsa ayri klasorde tasinir ---
    # Nano'nunkinin uzerine YAZILMAZ; ikisi yan yana durup karsilastirilabilsin.
    # YOLOX_COMMIT.txt kopyalanmak zorunda: verify_yolox_version() onu
    # checkpoint ile ayni klasorde arar.
    tiny_src = Path(args.artifacts_tiny)
    tiny_ckpt = tiny_src / "best_ckpt.pth"
    if tiny_ckpt.is_file():
        tiny_out = out / "tiny"
        tiny_out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tiny_ckpt, tiny_out / "best_ckpt.pth")
        shutil.copy2(needed["YOLOX_COMMIT.txt"], tiny_out / "YOLOX_COMMIT.txt")
        print(f">> Tiny checkpoint  : {tiny_ckpt.stat().st_size/1e6:.0f} MB -> {tiny_out}")
    else:
        print(f">> Tiny checkpoint  : yok ({tiny_ckpt}), atlandi")

    # --- val: TAMAMI (accuracy gate alt kume kabul etmiyor) ---
    target_ann = out / "datasets" / "merged" / "annotations"
    target_ann.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ann_dir / "instances_val.json",
                 target_ann / "instances_val.json")

    val = json.loads((ann_dir / "instances_val.json").read_text(encoding="utf-8"))
    val_names = [image["file_name"] for image in val["images"]]
    w, s, m, size = copy_images(
        val_names, images_root, out / "datasets" / "merged" / "images")
    print(f">> val goruntuleri : {len(val_names):>5} "
          f"({w} yazildi, {s} zaten vardi, {m} eksik)  {size/1e6:.0f} MB")
    if m:
        raise SystemExit(f"HATA: {m} val goruntusu diskte yok; paket eksik olur.")

    # --- kalibrasyon: train bolumunden rastgele ornek ---
    train = json.loads((ann_dir / "instances_train.json").read_text(encoding="utf-8"))
    train_names = [image["file_name"] for image in train["images"]]
    rng = random.Random(CALIB_SEED)
    rng.shuffle(train_names)
    calib_names = train_names[:args.calib]
    w, s, m, csize = copy_images(calib_names, images_root, out / "calib_images")
    print(f">> kalibrasyon     : {len(calib_names):>5} "
          f"({w} yazildi, {s} zaten vardi, {m} eksik)  {csize/1e6:.0f} MB")

    dist = Counter(n.split("/")[0] for n in calib_names)
    full = Counter(n.split("/")[0] for n in train_names)
    print("   kaynak dagilimi (train dagilimini yansitmali):")
    for key in sorted(full):
        print(f"     {key:<11} kalib %{100*dist[key]/max(1,len(calib_names)):.0f}"
              f"   train %{100*full[key]/len(train_names):.0f}")

    (out / "KOMUTLAR.md").write_text(KOMUTLAR, encoding="utf-8")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"\nHazir: {out}   toplam {total/1e6:.0f} MB")
    print("Bu klasoru VM'e tasiyin, icindeki KOMUTLAR.md'yi izleyin.")


KOMUTLAR = """# VM'de kuantalama - kopyala/yapistir

Bu klasoru VM'de `~/Vitis-AI/yolox_visdrone/` altina koyun (docker `/workspace`
olarak baglar). Kart kurulumunuz hazir oldugu icin yalniz bu adimlar kaldi.

## 0. Docker (surum kritik: 3.0, `:latest` = 3.5 KULLANMAYIN)

```bash
cd ~/Vitis-AI
./docker_run.sh xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106
conda activate vitis-ai-pytorch
cd /workspace/yolox_visdrone
```

## 1. YOLOX'u kur (Kaggle ile AYNI commit)

```bash
YOLOX_COMMIT=$(tr -d '\\r\\n' < YOLOX_COMMIT.txt)
test -d YOLOX/.git || git clone --filter=blob:none --no-checkout \\
  https://github.com/Megvii-BaseDetection/YOLOX.git
git -C YOLOX fetch --depth 1 origin "$YOLOX_COMMIT"
git -C YOLOX checkout --detach "$YOLOX_COMMIT"
pip install --no-deps --no-build-isolation -e ./YOLOX
pip install loguru tabulate pycocotools
python -c "import yolox; print(yolox.__version__)"
```

## 2. Ortak argumanlar

```bash
ARGS="--exp-file yolox_nano_visdrone.py --ckpt best_ckpt.pth --data-dir datasets/merged"
```

## 3. DPU uyumluluk raporu (once bunu yapin)

```bash
python quantize_yolox.py --inspect $ARGS
```

Tum katmanlar DPU'ya atanmali. CPU'ya dusen op varsa DURUN - derleme zaten
tek subgraph kapisinda basarisiz olur.

## 4. Float AP - saglama

```bash
python quantize_yolox.py --quant-mode float $ARGS
```

**Beklenen: AP@500 = 0.5874.** Tutmuyorsa yanlis checkpoint/exp/anotasyon var,
devam etmeyin.

## 5. Kalibrasyon (PTQ)

```bash
python quantize_yolox.py --quant-mode calib --subset-len 300 \\
    --calib-dir calib_images $ARGS
```

## 6. INT8 AP - kabul kapisi (UZUN SURER, screen/nohup kullanin)

```bash
python quantize_yolox.py --quant-mode test --float-map 0.5874 $ARGS
```

4.483 goruntu, CPU'da saatler. `--subset-len` **yasak** (gate tam val ister).
Mutlak AP kaybi 0.02'yi asarsa script durur; o zaman 5 ve 6'yi
`--fast-finetune` ile tekrarlayin.

## 7. xmodel export

```bash
python quantize_yolox.py --quant-mode test --deploy --subset-len 1 \\
    --batch-size 1 $ARGS
```

## 8. KV260 icin derle

```bash
bash compile_kv260.sh
```

Cikti: `build/compiled/yolox_nano_visdrone.xmodel`. Script tek DPU subgraph,
1 girdi ve **9 cikti** dogrular: her stride (8/16/32) icin ayri reg(4) /
obj(1) / cls(2) tensoru. Concat kaldirildi cunku Vitis AI concat girdilerini
ayni `fix_point`'e zorluyordu ve reg hassasiyetini yok ediyordu.

> Kartinizin DPU arch'i B4096 degilse `compile_kv260.sh` icindeki `ARCH`
> yolunu degistirin. Kartta `xdputil query` ile kontrol edin.

## 9. YOLOX-Tiny (Nano INT8 kapiyi gecemediyse asil yol budur)

Nano'da kayip 0.2306'da kaldi ve kok neden olculdu: **depthwise conv'larin
per-tensor kuantalanmasi**. Tiny `depthwise=False` kullanir, yani kok nedeni
ortadan kaldirir. Kaggle'daki Tiny egitimi bitince:

### 9.1 Dosyalari yerlestirin

Tiny checkpoint'ini Nano'nunkinin **uzerine yazmayin**; ayri klasore koyun.
`YOLOX_COMMIT.txt` checkpoint ile ayni klasorde olmak zorunda (surum kapisi
oraya bakar):

```bash
mkdir -p tiny
cp <indirdiginiz>/best_ckpt.pth tiny/best_ckpt.pth
cp YOLOX_COMMIT.txt tiny/
```

`yolox_tiny_visdrone.py` ve `yolox_nano_visdrone.py` ayni klasorde olmali --
Tiny exp'i Nano exp'ini import eder (ortak girdi boyutu, sinif semasi, ReLU,
DPUFocus; tek degisken mimari olsun diye).

### 9.2 Ayri cikti klasoru

```bash
ARGS_T="--exp-file yolox_tiny_visdrone.py --ckpt tiny/best_ckpt.pth --data-dir datasets/merged --output build/quant_tiny"
```

`--output` ayri oldugu icin Nano'nun kalibrasyonu ve accuracy gate'i
bozulmaz; iki sonucu yan yana karsilastirabilirsiniz.

### 9.3 Ayni merdiven, AdaQuant YOK

```bash
python quantize_yolox.py --inspect $ARGS_T
python quantize_yolox.py --quant-mode float $ARGS_T
python quantize_yolox.py --quant-mode calib --subset-len 300 --calib-dir calib_images $ARGS_T
python quantize_yolox.py --quant-mode test --float-map <TINY_FLOAT_AP> $ARGS_T
```

`--float-map` degerini adim 2'nin (`--quant-mode float`) ciktisindan alin --
Nano'nun 0.5874'unu **kullanmayin**, Tiny'nin kendi float AP'si farklidir.

**AdaQuant (`--fast-finetune`) eklemeyin.** Nano'da olculdu: 2 sa 14 dk
surdu ve hicbir sey kazandirmadi (0.3589 -> 0.3568). Tiny'de belirleyici
olan kalibrasyon degil mimari.

> ⚠️ Tiny ~5M parametre (Nano 0,9M, 5,6 kat). INT8 kapiyi gecse bile
> **30 FPS butcesi kartta yeniden olculmelidir.**

## 10. (opsiyonel) QAT yapilabilir mi? -- ONCE OLCUN

PTQ merdiveni Nano'da tukendi (CLE + bias correction + AdaQuant, kayip
0.2306). Kalan vendor yolu QAT ama QAT bir **egitim** islemi. Bu VM'de GPU
yok (VirtualBox NVIDIA passthrough yapmaz), yani CPU'da koser.

Kac saat surecegini tahmin etmeyin, olcun:

```bash
python qat_probe.py --exp-file yolox_nano_visdrone.py \\
    --ckpt best_ckpt.pth --data-dir datasets/merged --batch-size 4
```

Birkac dakikada biter ve sn/iterasyon + toplam saat projeksiyonu verir.
Cikan sayiyi paylasin; QAT portunu ona gore boyutlandiracagiz.

> QAT'in kendisi icin **etiketli egitim verisi** gerekir; bu pakette yok
> (yalnizca val + etiketsiz kalibrasyon kareleri var). Probe olcumu val
> setinden yapar, ek veri istemez.

## Sonra

`yolox_nano_visdrone.xmodel` dosyasini karta kopyalayin ve
`deploy/README.md` adim 4'ten devam edin (imaj/DPU adimlarini atlayin).
"""


if __name__ == "__main__":
    main()
