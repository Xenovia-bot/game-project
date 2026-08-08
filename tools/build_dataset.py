#!/usr/bin/env python3
"""Dort kaynagi tek bir 2-sinifli COCO veri setinde birlestirir.

Hedef siniflar
--------------
  1 = land_vehicle   2 = sea_vehicle

Kaynaklar ve formatlari
-----------------------
  visdrone   VisDrone-DET yerel txt   (x,y,w,h,score,category,trunc,occ)
  vesselimg  Roboflow COCO JSON
  milrec     Roboflow COCO JSON       (Military Vehicle Recognition)
  mendeley   YOLO txt                 (cls cx cy w h, normalize)

Uc tur karar
------------
  LAND / SEA  -> hedef kutu
  IGNORE      -> iscrowd=1, ignore=1. Ne tespit beklenir ne arka plan sayilir.
  None (at)   -> anotasyon uretilmez; nesne arka plan olur. Hedef disi ve
                 gorsel olarak farkli siniflar icin **istenen** davranis
                 (or. Buoy: "samandira gemi degil" ogretilir).

Bilinen tuzaklar ve alinan onlemler
-----------------------------------
* VESSELimg 23 cekim oturumundan olusuyor ve Roboflow kareleri **rastgele**
  bolmus: 23 oturumun tamami train/valid/test'te birden bulunuyor. Bu haliyle
  dogrulama skoru sahte cikar. Burada bolme **oturum bazli** yeniden yapilir.
* VisDrone'un ignored-region (category 0) ve score=0 crowd semantigi korunur.
* Kutular goruntu sinirlarina kirpilir; sifir/negatif alanlilar atilir.

Kullanim
--------
    python tools/build_dataset.py --data-dir datasets --out datasets/merged
    python tools/build_dataset.py ... --repeat vesselimg=2 --subsample visdrone=0.5
"""

import argparse
import io
import json
import os
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

TARGET_NAMES = ("land_vehicle", "sea_vehicle")
LAND, SEA = 1, 2
IGNORE = "ignore"

#: VisDrone kategori kimligi -> hedef. 0 = ignored-region (ozel islenir),
#: 11 = others (resmi protokolde hedef degil).
VISDRONE_MAP = {
    1: None,   # pedestrian
    2: None,   # people
    3: None,   # bicycle        - iki tekerlekli, siluet arabaya benzemiyor
    4: LAND,   # car
    5: LAND,   # van
    6: LAND,   # truck
    7: None,   # tricycle
    8: None,   # awning-tricycle
    9: LAND,   # bus
    10: None,  # motor
    11: None,  # others
}

#: Roboflow COCO kategori adi -> hedef. Adlar kucuk harfe indirilerek eslenir.
VESSELIMG_MAP = {
    "container": SEA,
    "chemical": SEA,
    "passenger-roro": SEA,
    "tugboat": SEA,
    "pilot": IGNORE,   # kilavuz botu: deniz araci ama medyan 15 px
    "buoy": None,      # araç degil; kasten hard negative
    "boats": None,     # Roboflow ust-kategori artigi, ornegi yok
}

MILREC_MAP = {
    "tank": LAND,
    "armoured personnel carrier": LAND,
    "air-fighter": None,
    "bomber": None,
    "soldier": None,
    "military-tank-plane-soldier": None,  # Roboflow artigi
}

#: Mendeley data.yaml sirasi: ['tank', 'drone', 'people', 'soldier']
MENDELEY_MAP = {0: LAND, 1: None, 2: None, 3: None}

#: Roboflow kaynaklarinda dogrulamaya ayrilacak oturum orani.
#:
#: Her iki Roboflow export'u da bolmeyi **kare/kopya bazinda** yapmis:
#:   * VESSELimg  - 23 cekim oturumunun tamami train/valid/test'te birden
#:   * milrec     - ayni kaynak goruntunun augment kopyalari farkli bolumlerde
#: Ikisinde de bolme oturum (kaynak goruntu) bazinda yeniden yapilir.
ROBOFLOW_VAL_FRACTION = 0.25
SPLIT_SEED = 1337


# --------------------------------------------------------------------------
# zip veya klasor farkini gizleyen ince katman
# --------------------------------------------------------------------------
class Archive:
    """Bir .zip dosyasini veya bir klasoru ayni arayuzle okur."""

    def __init__(self, path):
        self.path = Path(path)
        self._zip = None
        if self.path.suffix.lower() == ".zip":
            self._zip = zipfile.ZipFile(self.path)
            self._names = [n for n in self._zip.namelist() if not n.endswith("/")]
        elif self.path.is_dir():
            self._names = [
                p.relative_to(self.path).as_posix()
                for p in self.path.rglob("*") if p.is_file()
            ]
        else:
            raise SystemExit(f"HATA: bulunamadi veya desteklenmiyor: {path}")
        self._name_set = set(self._names)

    def names(self):
        return self._names

    def read(self, name):
        if self._zip is not None:
            return self._zip.read(name)
        return (self.path / name).read_bytes()

    def open(self, name):
        if self._zip is not None:
            return self._zip.open(name)
        return open(self.path / name, "rb")

    def exists(self, name):
        return name in self._name_set

    def image_size(self, name):
        """Yalnizca basligi okur; tum goruntuyu cozmez."""
        with self.open(name) as handle:
            with Image.open(io.BytesIO(handle.read()) if self._zip else handle) as im:
                return im.size  # (width, height)

    def close(self):
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class Record:
    """Birlestirilmis veri setindeki tek bir goruntu."""

    __slots__ = ("source", "member", "file_name", "width", "height",
                 "split", "anns", "ignore_regions", "group")

    def __init__(self, source, member, file_name, width, height, split, group):
        self.source = source          # kaynak adi (visdrone, vesselimg, ...)
        self.member = member          # arsiv icindeki yol
        self.file_name = file_name    # cikti COCO'daki goreli yol
        self.width = width
        self.height = height
        self.split = split            # "train" | "val"
        self.group = group            # sizinti kontrolu icin oturum anahtari
        self.anns = []                # {bbox, category_id, iscrowd, ignore}
        self.ignore_regions = []


def clip_box(x, y, w, h, width, height):
    """Kutuyu goruntuye kirpar; gecersizse None doner."""
    x1 = max(0.0, float(x))
    y1 = max(0.0, float(y))
    x2 = min(float(width), float(x) + float(w))
    y2 = min(float(height), float(y) + float(h))
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def add_box(record, target, box):
    if target is IGNORE:
        record.anns.append({"bbox": box, "category_id": LAND,
                            "iscrowd": 1, "ignore": 1})
    else:
        record.anns.append({"bbox": box, "category_id": target,
                            "iscrowd": 0, "ignore": 0})


# --------------------------------------------------------------------------
# Kaynak okuyucular
# --------------------------------------------------------------------------
def read_visdrone(archive, split, source="visdrone", stats=None):
    """VisDrone yerel txt formati. category 0 -> ignored-region."""
    names = archive.names()
    images = sorted(n for n in names
                    if "/images/" in n and n.lower().endswith(".jpg"))
    records = []
    for member in images:
        stem = member.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        ann_member = member.replace("/images/", "/annotations/")
        ann_member = ann_member.rsplit(".", 1)[0] + ".txt"
        if not archive.exists(ann_member):
            continue
        width, height = archive.image_size(member)
        rec = Record(source, member, f"{source}/{stem}.jpg",
                     width, height, split, group=f"{source}:{stem}")
        for line in archive.read(ann_member).decode("utf-8", "replace").splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            try:
                x, y, w, h, score, cat = (int(float(v)) for v in parts[:6])
            except ValueError:
                continue
            box = clip_box(x, y, w, h, width, height)
            if box is None:
                continue
            if cat == 0:
                rec.ignore_regions.append(box)
                continue
            target = VISDRONE_MAP.get(cat, None)
            if target is None:
                if stats is not None:
                    stats[f"{source}:drop:cat{cat}"] += 1
                continue
            if score == 0:            # resmi protokol: sinif-ozel crowd
                add_box(rec, IGNORE, box)
                if stats is not None:
                    stats[f"{source}:ignore:score0"] += 1
            else:
                add_box(rec, target, box)
                if stats is not None:
                    stats[f"{source}:keep:cat{cat}"] += 1
        records.append(rec)
    return records


def read_roboflow_coco(archive, class_map, source, split_override=None,
                       stats=None):
    """Roboflow COCO export: her bolum klasorunde _annotations.coco.json."""
    records = []
    for split_dir in ("train", "valid", "test"):
        member = f"{split_dir}/_annotations.coco.json"
        if not archive.exists(member):
            continue
        data = json.loads(archive.read(member))
        cats = {c["id"]: str(c["name"]).strip().lower()
                for c in data["categories"]}
        unknown = set(cats.values()) - set(class_map)
        if unknown:
            raise SystemExit(
                f"HATA: {source} icinde eslenmemis kategori: {sorted(unknown)}. "
                f"build_dataset.py icindeki esleme tablosunu guncelleyin."
            )
        by_image = defaultdict(list)
        for ann in data["annotations"]:
            by_image[ann["image_id"]].append(ann)

        split = split_override or ("val" if split_dir in ("valid", "test") else "train")
        for image in data["images"]:
            member_img = f"{split_dir}/{image['file_name']}"
            if not archive.exists(member_img):
                raise SystemExit(f"HATA: {source} goruntusu eksik: {member_img}")
            rec = Record(source, member_img, f"{source}/{image['file_name']}",
                         int(image["width"]), int(image["height"]), split,
                         group=session_key(image["file_name"], source))
            for ann in by_image.get(image["id"], ()):
                name = cats[ann["category_id"]]
                target = class_map[name]
                box = clip_box(*ann["bbox"], rec.width, rec.height)
                if box is None:
                    continue
                if target is None:
                    if stats is not None:
                        stats[f"{source}:drop:{name}"] += 1
                    continue
                add_box(rec, target, box)
                key = "ignore" if target is IGNORE else "keep"
                if stats is not None:
                    stats[f"{source}:{key}:{name}"] += 1
            records.append(rec)
    return records


def read_mendeley_yolo(archive, source="mendeley", stats=None):
    """YOLO txt: 'cls cx cy w h', hepsi 0-1 normalize."""
    names = archive.names()
    images = sorted(n for n in names
                    if "/images/" in n and n.lower().endswith((".jpg", ".png")))
    records = []
    for member in images:
        label = member.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if not archive.exists(label):
            continue
        # .../dataset/<split>/images/<dosya>
        parts = member.split("/")
        split_dir = parts[-3] if len(parts) >= 3 else "train"
        split = "val" if split_dir in ("valid", "val", "test") else "train"
        width, height = archive.image_size(member)
        base = parts[-1]
        rec = Record(source, member, f"{source}/{base}", width, height,
                     split, group=session_key(base, source))
        for line in archive.read(label).decode("utf-8", "replace").splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            try:
                cls = int(float(p[0]))
                cx, cy, bw, bh = (float(v) for v in p[1:5])
            except ValueError:
                continue
            target = MENDELEY_MAP.get(cls, None)
            box = clip_box((cx - bw / 2) * width, (cy - bh / 2) * height,
                           bw * width, bh * height, width, height)
            if box is None:
                continue
            if target is None:
                if stats is not None:
                    stats[f"{source}:drop:cls{cls}"] += 1
                continue
            add_box(rec, target, box)
            if stats is not None:
                stats[f"{source}:keep:cls{cls}"] += 1
        records.append(rec)
    return records


def session_key(file_name, source=""):
    """Ayni cekimden / ayni kaynak goruntuden gelen kareleri gruplayan anahtar.

    Roboflow dosya adlari '<taban>_<kare>_jpg.rf.<hash>.jpg' bicimindedir;
    taban kisim ya cekim oturumunu (kamera + zaman damgasi) ya da augment
    kopyalarinin turedigi kaynak goruntuyu tasir. Ikisi de ayni bolumde
    kalmalidir.

    Anahtar kaynak adiyla oneklenir: iki veri seti de goruntulerini 1, 2, 3
    diye numaraladigi icin onek olmadan alakasiz goruntuler ayni oturum
    sayilirdi. (Kontrol edildi: milrec ve mendeley'deki ayni adli goruntuler
    gorsel olarak tamamen farkli.)
    """
    base = file_name.rsplit("/", 1)[-1]
    prefix = f"{source}:" if source else ""
    match = re.match(r"(.+?)_\d+_(?:jpg|png)\.rf\.", base)
    if match:
        return prefix + match.group(1)
    match = re.match(r"(.+?)\.rf\.", base)
    if match:
        return prefix + match.group(1)
    return prefix + base


def resplit_by_group(records, val_fraction, seed=SPLIT_SEED):
    """Kareleri degil **oturumlari** boler; bitisik kare sizintisini onler."""
    groups = sorted({r.group for r in records})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_val = max(1, round(len(groups) * val_fraction))
    val_groups = set(groups[:n_val])
    for r in records:
        r.split = "val" if r.group in val_groups else "train"
    return len(groups), len(val_groups)


# --------------------------------------------------------------------------
# Dogrulama kapilari
# --------------------------------------------------------------------------
def validate(records):
    """Bozuk bir sey varsa dosya uretmeden hata verir."""
    problems = []
    seen_files = set()
    groups = defaultdict(set)

    for r in records:
        if r.file_name in seen_files:
            problems.append(f"yinelenen dosya adi: {r.file_name}")
        seen_files.add(r.file_name)
        if r.width <= 0 or r.height <= 0:
            problems.append(f"gecersiz goruntu boyutu: {r.file_name}")
        groups[r.group].add(r.split)
        for ann in r.anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                problems.append(f"sifir/negatif kutu: {r.file_name} {ann['bbox']}")
            if x < -1e-6 or y < -1e-6 or x + w > r.width + 1e-6 \
                    or y + h > r.height + 1e-6:
                problems.append(f"sinir disi kutu: {r.file_name} {ann['bbox']} "
                                f"(goruntu {r.width}x{r.height})")
            if ann["category_id"] not in (LAND, SEA):
                problems.append(f"gecersiz kategori: {ann['category_id']}")

    leaked = [g for g, s in groups.items() if len(s) > 1]
    if leaked:
        problems.append(
            f"{len(leaked)} oturum hem train hem val'de "
            f"(ornek: {sorted(leaked)[:3]})"
        )
    return problems


def build_coco(records, split):
    subset = [r for r in records if r.split == split]
    images, annotations = [], []
    ann_id = 1
    for image_id, rec in enumerate(sorted(subset, key=lambda r: r.file_name), 1):
        images.append({
            "id": image_id,
            "file_name": rec.file_name,
            "width": rec.width,
            "height": rec.height,
            "source": rec.source,
            "ignore_regions": rec.ignore_regions,
        })
        for ann in rec.anns:
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": ann["category_id"],
                "bbox": [round(v, 2) for v in ann["bbox"]],
                "area": round(ann["bbox"][2] * ann["bbox"][3], 2),
                "iscrowd": ann["iscrowd"],
                "ignore": ann["ignore"],
            })
            ann_id += 1
    return {
        "info": {"description": "Birlesik 2-sinifli havadan arac veri seti",
                 "class_scheme": "land_vehicle+sea_vehicle"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": n, "supercategory": "vehicle"}
                       for i, n in enumerate(TARGET_NAMES)],
    }


def parse_kv(values):
    out = {}
    for item in values or ():
        if "=" not in item:
            raise SystemExit(f"HATA: 'kaynak=deger' bekleniyordu: {item}")
        key, value = item.split("=", 1)
        out[key.strip()] = float(value)
    return out


def summarise(records, stats):
    print("\n" + "=" * 74)
    print("KAYNAK BAZLI OZET")
    print("=" * 74)
    per_source = defaultdict(lambda: Counter())
    for r in records:
        c = per_source[r.source]
        c["goruntu"] += 1
        c[f"goruntu_{r.split}"] += 1
        for a in r.anns:
            if a["ignore"]:
                c["ignore"] += 1
            elif a["category_id"] == LAND:
                c["land"] += 1
            else:
                c["sea"] += 1
        c["ignore_region"] += len(r.ignore_regions)
    header = f"{'kaynak':<12}{'goruntu':>9}{'train':>8}{'val':>7}{'land':>9}{'sea':>8}{'ignore':>8}"
    print(header); print("-" * len(header))
    for source in sorted(per_source):
        c = per_source[source]
        print(f"{source:<12}{c['goruntu']:>9}{c['goruntu_train']:>8}"
              f"{c['goruntu_val']:>7}{c['land']:>9}{c['sea']:>8}{c['ignore']:>8}")

    print("\nSINIF ESLEME DOKUMU (kaynak:karar:sinif)")
    for key in sorted(stats):
        print(f"  {key:<40} {stats[key]:>8}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="datasets",
                        help="zip/klasor kaynaklarinin bulundugu dizin")
    parser.add_argument("--out", default="datasets/merged",
                        help="cikti dizini (instances_train.json / instances_val.json)")
    parser.add_argument("--repeat", nargs="*", default=[],
                        help="or. vesselimg=2 (train'de tekrar sayisi)")
    parser.add_argument("--subsample", nargs="*", default=[],
                        help="or. visdrone=0.5 (train goruntulerinin orani)")
    parser.add_argument("--dry-run", action="store_true",
                        help="dosya yazma, yalnizca dogrula ve raporla")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    stats = Counter()
    records = []

    plan = [
        ("visdrone-train", "VisDrone2019-DET-train.zip", "visdrone", "train"),
        ("visdrone-val", "VisDrone2019-DET-val.zip", "visdrone", "val"),
        ("vesselimg", "VESSELimg.v4i.coco.zip", "vesselimg", None),
        ("milrec", "Military Vehicle Recognition.v7i.coco.zip", "milrec", None),
        ("mendeley", "A Multi-Class UAV Military Object Detection Datase.zip",
         "mendeley", None),
    ]

    for label, filename, source, split in plan:
        path = data_dir / filename
        if not path.exists():
            alt = data_dir / Path(filename).stem
            if alt.exists():
                path = alt
            else:
                raise SystemExit(f"HATA: kaynak bulunamadi: {path}")
        print(f">> {label:<16} {path.name}")
        archive = Archive(path)
        try:
            if source == "visdrone":
                got = read_visdrone(archive, split, stats=stats)
            elif source in ("vesselimg", "milrec"):
                class_map = VESSELIMG_MAP if source == "vesselimg" else MILREC_MAP
                got = read_roboflow_coco(archive, class_map, source, stats=stats)
                # Roboflow'un kendi bolmesi guvenilmez (bkz. ROBOFLOW_VAL_FRACTION)
                total, val = resplit_by_group(got, ROBOFLOW_VAL_FRACTION)
                print(f"   oturum bazli yeniden bolme: {total} oturum -> {val} val")
            else:
                got = read_mendeley_yolo(archive, stats=stats)
                # Mendeley de Roboflow'dan gecmis (.rf. hash'i); ayni kaynak
                # goruntunun augment kopyalari train/test'e dagilmis durumda.
                total, val = resplit_by_group(got, ROBOFLOW_VAL_FRACTION)
                print(f"   oturum bazli yeniden bolme: {total} oturum -> {val} val")
        finally:
            archive.close()
        print(f"   {len(got)} goruntu")
        records.extend(got)

    # --- tekrar / seyreltme (yalnizca train) ---
    repeat = parse_kv(args.repeat)
    subsample = parse_kv(args.subsample)
    if subsample:
        rng = random.Random(SPLIT_SEED)
        kept = []
        for r in records:
            frac = subsample.get(r.source)
            if r.split == "train" and frac is not None and rng.random() > frac:
                continue
            kept.append(r)
        print(f"\nseyreltme sonrasi: {len(records)} -> {len(kept)} goruntu")
        records = kept
    if repeat:
        extra = []
        for r in records:
            times = int(repeat.get(r.source, 1))
            if r.split == "train" and times > 1:
                extra.extend([r] * (times - 1))
        if extra:
            print(f"tekrar sonrasi: +{len(extra)} goruntu girisi")
            records.extend(extra)

    problems = validate([r for r in records if True])
    if problems:
        print("\n" + "!" * 74)
        print(f"DOGRULAMA BASARISIZ - {len(problems)} sorun")
        for p in problems[:20]:
            print("  -", p)
        if len(problems) > 20:
            print(f"  ... ve {len(problems)-20} tane daha")
        raise SystemExit(1)
    print("\nDogrulama gecti: kutular sinir icinde, kategori {1,2}, "
          "oturum sizintisi yok.")

    summarise(records, stats)

    train = build_coco(records, "train")
    val = build_coco(records, "val")
    print("\n" + "=" * 74)
    print(f"{'':<8}{'goruntu':>10}{'kutu':>10}{'land':>9}{'sea':>8}{'ignore':>8}")
    for name, coco in (("train", train), ("val", val)):
        real = [a for a in coco["annotations"] if not a["ignore"]]
        land = sum(1 for a in real if a["category_id"] == LAND)
        sea = sum(1 for a in real if a["category_id"] == SEA)
        ign = len(coco["annotations"]) - len(real)
        print(f"{name:<8}{len(coco['images']):>10}{len(real):>10}"
              f"{land:>9}{sea:>8}{ign:>8}")
        if sea and land:
            print(f"{'':8}dengesizlik land:sea = {land/sea:.1f} : 1")

    if args.dry_run:
        print("\n--dry-run: dosya yazilmadi.")
        return

    out = Path(args.out)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    for name, coco in (("train", train), ("val", val)):
        target = out / "annotations" / f"instances_{name}.json"
        target.write_text(json.dumps(coco), encoding="utf-8")
        print(f"yazildi: {target}")

    manifest = out / "image_manifest.json"
    manifest.write_text(json.dumps({
        "sources": {label: filename for label, filename, _, _ in plan},
        "members": [{"source": r.source, "member": r.member,
                     "file_name": r.file_name} for r in records],
    }), encoding="utf-8")
    print(f"yazildi: {manifest}  (goruntuleri cikarmak icin)")


if __name__ == "__main__":
    main()
