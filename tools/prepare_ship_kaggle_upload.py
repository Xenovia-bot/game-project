#!/usr/bin/env python3
"""Gemi veri seti icin Kaggle'a yuklenecek klasoru hazirlar.

build_ship_dataset.py Kaggle'da CALISTIRILACAK (yerelde birlestirilip
yuklenmeyecek): birlestirilmis set ~GB'larca yer tutar, ham zip'ler ise Kaggle
deposunda zaten durur. Bu yuzden burada yalnizca 6 HAM kaynak zip'i kopyalanir;
birlestirmeyi not defterinin kesif hucresi kendisi yapar -- kaynaklari klasor
adindan degil kategori imzasindan bulup `--source` argumanlarini kendi uretir,
yani asagida basilan komutu elle yazmaniz gerekmez.

Kullanim:
    python tools/prepare_ship_kaggle_upload.py
    python tools/prepare_ship_kaggle_upload.py --user BASKA_KULLANICI --out D:/upload
"""

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NEEDED = [
    "VAIS_RGB-SMD-MARITIME-WSODD-MARVEL.v5-rgb_40_grayscale_60.coco.zip",
    "Singapore maritime.v5i.coco.zip",
    "Sea Vessels Dataset.v2-sea_vessels_v2.coco.zip",
    "ship model.v4i.coco.zip",
    "ir.v1i.coco.zip",
    "WUTDet Part A.zip",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(Path.home() / "OneDrive" / "Masaüstü" / "datasets"))
    parser.add_argument("--out", default=str(Path.home() / "kaggle_upload_ship"))
    parser.add_argument("--user", default="burakzorgeen")
    parser.add_argument("--slug", default="ship-detection-sources")
    args = parser.parse_args()

    data_dir, out = Path(args.data_dir), Path(args.out)
    located = {}
    for name in NEEDED:
        matches = sorted(data_dir.rglob(name))
        if matches:
            located[name] = matches[0]
    missing = [n for n in NEEDED if n not in located]
    if missing:
        raise SystemExit(
            "HATA: su dosyalar bulunamadi:\n  " + "\n  ".join(missing) +
            f"\nAranan dizin (ozyinelemeli): {data_dir}"
        )

    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for name in NEEDED:
        src, dst = located[name], out / name
        size = src.stat().st_size
        total += size
        if dst.exists() and dst.stat().st_size == size:
            print(f"  atlandi (zaten var)  {name}")
            continue
        print(f"  kopyalaniyor          {name}  ({size/1e6:.0f} MB)")
        shutil.copy2(src, dst)

    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": args.slug,
        "id": f"{args.user}/{args.slug}",
        "licenses": [{"name": "other"}],
    }, indent=2), encoding="utf-8")

    print(f"\nHazir: {out}  (toplam {total/1e9:.2f} GB)")
    print("\nSimdi calistirin:")
    print(f'    kaggle datasets create -p "{out}"')
    print("\nSonra Kaggle notebook'unda (training/kaggle_ship_yolox.ipynb) "
          "'Add Input' ile bu veri setini baglayin.")
    print("Birlestirme komutunu elle yazmayin: kesif hucresi 6 kaynagi kategori "
          "imzasindan bulup build_ship_dataset.py'yi kendisi calistirir.")


if __name__ == "__main__":
    main()
