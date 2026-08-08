#!/usr/bin/env python3
"""Kaggle'a yuklenecek klasoru hazirlar.

Egitim icin gereken 5 zip'i tek klasore kopyalar ve Kaggle CLI'nin bekledigi
dataset-metadata.json dosyasini yazar. Kullanilmayan VisDrone test bolumleri
bilerek disarida birakilir.

Kullanim:
    python tools/prepare_kaggle_upload.py
    python tools/prepare_kaggle_upload.py --user BASKA_KULLANICI --out D:/upload
"""

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: build_dataset.py'nin kullandigi kaynaklar. Test bolumleri kullanilmiyor.
NEEDED = [
    "VisDrone2019-DET-train.zip",
    "VisDrone2019-DET-val.zip",
    "VESSELimg.v4i.coco.zip",
    "Military Vehicle Recognition.v7i.coco.zip",
    "A Multi-Class UAV Military Object Detection Datase.zip",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(ROOT / "datasets"))
    parser.add_argument("--out", default=str(Path.home() / "kaggle_upload"))
    parser.add_argument("--user", default="burakzorgeen",
                        help="Kaggle kullanici adi (dataset id icin). "
                             "Profil adresindeki kaggle.com/<ad> ile birebir "
                             "ayni olmali, yoksa 'datasets create' reddeder.")
    parser.add_argument("--slug", default="aerial-vehicle-sources")
    args = parser.parse_args()

    data_dir, out = Path(args.data_dir), Path(args.out)
    # Zip'ler duz `datasets/` altinda da, kaynak bazli alt klasorlerde de
    # (datasets/aerial-land/, datasets/milrec/, ...) durabilir; ozyinelemeli ara.
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


if __name__ == "__main__":
    main()
