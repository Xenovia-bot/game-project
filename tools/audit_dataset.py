#!/usr/bin/env python3
"""Birlestirilmis veri setini egitim oncesi bagimsiz olarak denetler.

`build_dataset.py` kendi ic dogrulamasini yapar; bu arac ondan **bagimsiz**
olarak, uretilen JSON'lari ve gercek goruntu dosyalarini kaynaga karsi
yeniden kontrol eder. Amac saatlerce surecek bir egitimi bozacak sessiz
hatalari once yakalamak.

Denetimler
----------
  1. Arsiv butunlugu       zip'ler bozuk mu (CRC)
  2. COCO semasi           kimlik benzersizligi, kategori, zorunlu alanlar
  3. pycocotools yuklemesi egitimin kullandigi kutuphane dosyayi okuyabiliyor mu
  4. Kutu gecerliligi      sinir disi, sifir alan, asiri kucuk/buyuk
  5. Bolme ayrikligi       goruntu ve oturum bazinda sizinti
  6. Goruntu erisimi       ornek goruntuler gercekten acilabiliyor mu
  7. Boyut tutarliligi     JSON'daki genislik/yukseklik gercek dosyayla ayni mi
  8. Kaynak karsilastirma  rastgele orneklerde kutular kaynakla birebir mi
  9. Sinif dagilimi        kaynak basina land/sea/ignore sayilari
 10. Bos kare orani        anotasyonsuz goruntu payi

Kullanim:
    python tools/audit_dataset.py --merged datasets/merged --data-dir datasets
"""

import argparse
import io
import json
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.build_dataset import (  # noqa: E402
    LAND, SEA, Archive, MILREC_MAP, VESSELIMG_MAP, session_key,
)

SAMPLE = 40


class Report:
    def __init__(self):
        self.failures = []
        self.warnings = []

    def check(self, ok, message, detail=""):
        mark = "  OK  " if ok else " HATA "
        print(f"{mark} {message}")
        if detail:
            print(f"        {detail}")
        if not ok:
            self.failures.append(message)

    def warn(self, condition, message):
        if condition:
            print(f" UYARI {message}")
            self.warnings.append(message)

    def summary(self):
        print("\n" + "=" * 74)
        if self.failures:
            print(f"DENETIM BASARISIZ - {len(self.failures)} sorun")
            for f in self.failures:
                print(f"  - {f}")
            return 1
        print("TUM DENETIMLER GECTI")
        if self.warnings:
            print(f"({len(self.warnings)} uyari - engelleyici degil)")
        return 0


def section(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def audit(merged_dir, data_dir, sources):
    rep = Report()
    ann_dir = merged_dir / "annotations"
    images_root = merged_dir / "images"

    # ---------------------------------------------------------------- 1
    section("1. ARSIV BUTUNLUGU")
    for label, path in sources.items():
        if Path(path).suffix.lower() != ".zip":
            print(f"  ATLA  {label}: klasor ({path})")
            continue
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
        rep.check(bad is None, f"{label}: zip saglam",
                  "" if bad is None else f"bozuk giris: {bad}")

    # ---------------------------------------------------------------- 2
    section("2. COCO SEMASI")
    splits = {}
    for name in ("train", "val"):
        path = ann_dir / f"instances_{name}.json"
        rep.check(path.exists(), f"{name}: dosya var ({path.name})")
        if not path.exists():
            return rep.summary()
        splits[name] = json.loads(path.read_text(encoding="utf-8"))

    for name, data in splits.items():
        images, anns = data["images"], data["annotations"]
        rep.check([c["name"] for c in data["categories"]]
                  == ["land_vehicle", "sea_vehicle"],
                  f"{name}: kategoriler dogru")
        rep.check(len({i["id"] for i in images}) == len(images),
                  f"{name}: goruntu kimlikleri benzersiz")
        rep.check(len({a["id"] for a in anns}) == len(anns),
                  f"{name}: anotasyon kimlikleri benzersiz")
        rep.check(len({i["file_name"] for i in images}) == len(images),
                  f"{name}: dosya adlari benzersiz")
        ids = {i["id"] for i in images}
        rep.check(all(a["image_id"] in ids for a in anns),
                  f"{name}: her kutu var olan bir goruntuye bagli")
        rep.check(all(a["category_id"] in (LAND, SEA) for a in anns),
                  f"{name}: kategori kimlikleri {{1,2}}")
        required = {"id", "file_name", "width", "height"}
        rep.check(all(required <= set(i) for i in images),
                  f"{name}: goruntu kayitlarinda zorunlu alanlar var")

    # ---------------------------------------------------------------- 3
    section("3. PYCOCOTOOLS YUKLEMESI (egitimin kullandigi yol)")
    try:
        from pycocotools.coco import COCO
        for name in ("train", "val"):
            coco = COCO(str(ann_dir / f"instances_{name}.json"))
            n_img = len(coco.getImgIds())
            n_ann = len(coco.getAnnIds())
            # YOLOX hedefleri iscrowd=False ile ceker; ignore kutulari dislanmali
            targets = len(coco.getAnnIds(iscrowd=False))
            crowd = len(coco.getAnnIds(iscrowd=True))
            rep.check(n_img > 0 and n_ann > 0,
                      f"{name}: pycocotools yukledi",
                      f"goruntu={n_img} kutu={n_ann} hedef={targets} crowd={crowd}")
            rep.check(targets + crowd == n_ann,
                      f"{name}: iscrowd ayrimi tutarli")
    except ImportError:
        rep.warn(True, "pycocotools kurulu degil, bu denetim atlandi")

    # ---------------------------------------------------------------- 4
    section("4. KUTU GECERLILIGI")
    for name, data in splits.items():
        dims = {i["id"]: (i["width"], i["height"]) for i in data["images"]}
        oob = zero = tiny = huge = 0
        areas = []
        for a in data["annotations"]:
            x, y, w, h = a["bbox"]
            W, H = dims[a["image_id"]]
            if w <= 0 or h <= 0:
                zero += 1
                continue
            if x < -0.01 or y < -0.01 or x + w > W + 0.01 or y + h > H + 0.01:
                oob += 1
            side = (w * h) ** 0.5
            areas.append(side)
            if side < 2:
                tiny += 1
            if w > W * 1.01 or h > H * 1.01:
                huge += 1
        rep.check(zero == 0, f"{name}: sifir/negatif alanli kutu yok", f"{zero} adet")
        rep.check(oob == 0, f"{name}: sinir disi kutu yok", f"{oob} adet")
        rep.check(huge == 0, f"{name}: goruntuden buyuk kutu yok", f"{huge} adet")
        rep.warn(tiny > 0, f"{name}: {tiny} kutu 2 pikselden kucuk")
        a = np.array(areas)
        print(f"        kutu boyutu (sqrt alan): medyan={np.median(a):.1f} "
              f"p1={np.percentile(a,1):.1f} p99={np.percentile(a,99):.1f}")

    # ---------------------------------------------------------------- 5
    section("5. BOLME AYRIKLIGI")
    tr_files = {i["file_name"] for i in splits["train"]["images"]}
    va_files = {i["file_name"] for i in splits["val"]["images"]}
    rep.check(not (tr_files & va_files), "hicbir goruntu iki bolumde degil",
              f"kesisim: {len(tr_files & va_files)}")

    def group_of(file_name):
        src = file_name.split("/")[0]
        return session_key(file_name, src)

    tr_groups = {group_of(f) for f in tr_files}
    va_groups = {group_of(f) for f in va_files}
    leaked = tr_groups & va_groups
    rep.check(not leaked, "hicbir oturum iki bolumde degil",
              f"sizan oturum: {len(leaked)}")

    # ---------------------------------------------------------------- 6/7
    section("6-7. GORUNTU ERISIMI VE BOYUT TUTARLILIGI")
    if not images_root.exists():
        rep.warn(True, f"{images_root} yok - goruntu denetimleri atlandi "
                       "(build_dataset.py --images-out ile uretin)")
    else:
        random.seed(4)
        by_source = defaultdict(list)
        for data in splits.values():
            for im in data["images"]:
                by_source[im["file_name"].split("/")[0]].append(im)
        for source, items in sorted(by_source.items()):
            picks = random.sample(items, min(SAMPLE // 4, len(items)))
            missing = mismatched = unreadable = 0
            for im in picks:
                path = images_root / im["file_name"]
                if not path.exists():
                    missing += 1
                    continue
                try:
                    with Image.open(path) as handle:
                        size = handle.size
                        handle.verify()
                except Exception:
                    unreadable += 1
                    continue
                if size != (im["width"], im["height"]):
                    mismatched += 1
            rep.check(missing == 0, f"{source}: ornek goruntuler diskte var",
                      f"eksik {missing}/{len(picks)}")
            rep.check(unreadable == 0, f"{source}: goruntuler acilabiliyor",
                      f"bozuk {unreadable}/{len(picks)}")
            rep.check(mismatched == 0, f"{source}: JSON boyutu dosyayla ayni",
                      f"uyusmayan {mismatched}/{len(picks)}")

    # ---------------------------------------------------------------- 8
    section("8. KAYNAKLA BIREBIR KARSILASTIRMA (Roboflow COCO)")
    for source, class_map, keep_names in (
            ("vesselimg", VESSELIMG_MAP,
             {k for k, v in VESSELIMG_MAP.items() if v in (LAND, SEA)}),
            ("milrec", MILREC_MAP,
             {k for k, v in MILREC_MAP.items() if v in (LAND, SEA)})):
        path = sources.get(source)
        if path is None:
            continue
        original = {}
        with Archive(path) as archive:
            for split_dir in ("train", "valid", "test"):
                member = f"{split_dir}/_annotations.coco.json"
                if not archive.exists(member):
                    continue
                data = json.loads(archive.read(member))
                cats = {c["id"]: c["name"].strip().lower()
                        for c in data["categories"]}
                per = defaultdict(list)
                for a in data["annotations"]:
                    per[a["image_id"]].append(
                        (cats[a["category_id"]],
                         tuple(round(v, 2) for v in a["bbox"])))
                for im in data["images"]:
                    original[f"{source}/{im['file_name']}"] = per.get(im["id"], [])

        merged = defaultdict(list)
        for data in splits.values():
            index = {i["id"]: i["file_name"] for i in data["images"]}
            for a in data["annotations"]:
                name = index[a["image_id"]]
                if name.startswith(source + "/"):
                    merged[name].append(a)

        random.seed(9)
        picks = random.sample(sorted(merged), min(SAMPLE, len(merged)))
        bad = 0
        for name in picks:
            want = sorted(b for n, b in original.get(name, []) if n in keep_names)
            got = sorted(tuple(round(v, 2) for v in a["bbox"])
                         for a in merged[name] if not a["ignore"])
            if want != got:
                bad += 1
                if bad <= 2:
                    print(f"        FARK {name[:50]}")
                    print(f"          beklenen {want[:3]}")
                    print(f"          alinan   {got[:3]}")
        rep.check(bad == 0,
                  f"{source}: {len(picks)} karede kutular kaynakla ayni",
                  f"uyusmayan {bad}")

    # ---------------------------------------------------------------- 9
    section("9. SINIF DAGILIMI")
    header = f"{'kaynak':<12}{'goruntu':>9}{'land':>9}{'sea':>8}{'ignore':>8}{'bos kare':>10}"
    print(header + "\n" + "-" * len(header))
    grand = Counter()
    for name, data in splits.items():
        per_source = defaultdict(Counter)
        anns_by_image = defaultdict(list)
        for a in data["annotations"]:
            anns_by_image[a["image_id"]].append(a)
        for im in data["images"]:
            src = im["file_name"].split("/")[0]
            counter = per_source[src]
            counter["img"] += 1
            anns = anns_by_image.get(im["id"], [])
            if not anns:
                counter["empty"] += 1
            for a in anns:
                if a["ignore"]:
                    counter["ignore"] += 1
                elif a["category_id"] == LAND:
                    counter["land"] += 1
                else:
                    counter["sea"] += 1
        print(f"  [{name}]")
        for src in sorted(per_source):
            c = per_source[src]
            print(f"{src:<12}{c['img']:>9}{c['land']:>9}{c['sea']:>8}"
                  f"{c['ignore']:>8}{c['empty']:>10}")
            grand[f"{name}:land"] += c["land"]
            grand[f"{name}:sea"] += c["sea"]
            grand[f"{name}:empty"] += c["empty"]
            grand[f"{name}:img"] += c["img"]

    # ---------------------------------------------------------------- 10
    section("10. BOS KARE ORANI VE DENGE")
    for name in ("train", "val"):
        img = grand[f"{name}:img"]
        empty = grand[f"{name}:empty"]
        land, sea = grand[f"{name}:land"], grand[f"{name}:sea"]
        ratio = 100 * empty / max(1, img)
        print(f"  {name}: bos kare {empty}/{img} (%{ratio:.1f}), "
              f"land:sea = {land/max(1,sea):.1f}:1")
        rep.check(ratio < 40, f"{name}: bos kare orani makul (<%40)",
                  f"%{ratio:.1f}")
        rep.check(sea > 0 and land > 0, f"{name}: iki sinif da temsil ediliyor")

    return rep.summary()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merged", default="datasets/merged")
    parser.add_argument("--data-dir", default="datasets")
    parser.add_argument("--source", action="append", default=None,
                        metavar="ETIKET=YOL")
    args = parser.parse_args()

    sources = {}
    for item in (args.source or ()):
        key, _, value = item.partition("=")
        sources[key.strip()] = value.strip()
    if not sources:
        base = Path(args.data_dir)
        for candidate in sorted(base.rglob("*.zip")):
            name = candidate.name.lower()
            if "visdrone" in name and "train" in name:
                sources["visdrone-train"] = str(candidate)
            elif "visdrone" in name and "val" in name:
                sources["visdrone-val"] = str(candidate)
            elif "vesselimg" in name:
                sources["vesselimg"] = str(candidate)
            elif "military vehicle" in name:
                sources["milrec"] = str(candidate)
            elif "multi-class" in name:
                sources["mendeley"] = str(candidate)

    raise SystemExit(audit(Path(args.merged), Path(args.data_dir), sources))


if __name__ == "__main__":
    main()
