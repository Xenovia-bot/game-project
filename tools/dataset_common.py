#!/usr/bin/env python3
"""Veri seti birlestiricilerin paylastigi format-bagimsiz yardimcilar.

Buraya YALNIZCA hicbir kaynagin (VisDrone, Roboflow, VOC, ...) semantigini
bilmeyen kod girer: arsiv okuma, kutu kirpma, oturum bazli bolme, goruntu
cikarma. Kaynaga ozel esleme ve okuyucular cagiran modulde kalir.

Neden ayri modul: gemi hatti (tools/build_ship_dataset.py) bu yardimcilarin
9'unu kullaniyordu ve bunun icin havadan/VisDrone birlestiricisinin tamamini
import etmek zorundaydi -- Kaggle not defterine de o dosyanin tamami
gomuluyordu. Gemi isi drone verisi kullanmiyor; bagimlilik da kullanmamali.

Kullanici: tools/build_ship_dataset.py. (Havadan/VisDrone birlestiricisi bu
projeden cikarildi; oturum bazli bolmenin grup-sayisina gore calisan eski
surumleri -- resplit_by_group, carve_test_split -- onunla birlikte silindi.
Gerekirse git gecmisindeki tools/build_dataset.py icinde bulunur.)
"""

import io
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

#: Bolme rastgeleligi tekrarlanabilir olsun diye sabit.
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
        # Bu modulun bolme/manifest yardimcilari Pillow gerektirmez. Importu
        # burada tutmak, yalnizca mevcut COCO JSON'larini denetleyen veya
        # yeniden bolen araclarin gereksiz goruntu kutuphanesi bagimliligiyla
        # acilista durmasini engeller.
        from PIL import Image
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
                 "split", "anns", "ignore_regions", "group", "origin")

    def __init__(self, source, member, file_name, width, height, split, group):
        self.source = source          # kaynak adi (visdrone, vesselimg, ...)
        self.member = member          # arsiv icindeki yol
        # Goruntunun geldigi arsiv. Kaynak adina gore eslemek yetmez:
        # visdrone train ve val ayni kaynak adini paylasir ama farkli
        # zip'lerde durur.
        self.origin = None
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


def has_part(name, part):
    """Yol bileseni tam eslesme ile aranir.

    Arsiv kokunun nereye isaret ettigine gore ayni dosya
    'VisDrone2019-DET-train/images/x.jpg' veya 'images/x.jpg' olarak
    gorunebilir; '/images/' arayan bir kontrol ikincisini kaciririrdi.
    """
    return part in name.split("/")


def swap_part(name, old, new):
    parts = name.split("/")
    return "/".join(new if p == old else p for p in parts)


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


def split_by_group_targets(records, val_fraction, test_fraction, seed=SPLIT_SEED):
    """Gruplari BOLMEDEN, **goruntu sayisi** hedefine gore train/val/test ayirir.

    resplit_by_group()'tan farki hedefin ne oldugu: o, grup SAYISININ bir
    oranini val'e atar. Gruplar cok esitsizse gerceklesen goruntu orani
    hedeften sapar -- olculdu (2026-08-12): hedef %25 val iken
    vais_smd_marvel'de %10,6, sea_vessels'ta %12,0 cikmisti. Burada kota
    goruntu cinsinden tutulur, o yuzden gerceklesen oran hedefe yakin kalir.

    Hedefi asacak grup ATLANIR, bolunmez: bir oturumun (video, kamera akisi,
    ayni fotografin augment kopyalari) yarisini train'e yarisini val'e koymak
    tam da onlemeye calistigimiz sizintidir. Sonucta cok buyuk bir grup
    (or. tek bir IP kamera akisi) her zaman train'de kalir; val o kamerayi
    hic gormez -- istenen davranis budur, val'in bagimsizligi oran
    hassasiyetinden onemli.

    Donus: {"train": n, "val": n, "test": n} goruntu sayilari.
    """
    groups = defaultdict(list)
    for record in records:
        groups[record.group].append(record)
    names = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(names)

    total = len(records)
    assigned = {}
    for split, fraction in (("val", val_fraction), ("test", test_fraction)):
        target = max(1, round(total * fraction))
        taken = 0
        for name in names:
            if name in assigned or taken >= target:
                continue
            size = len(groups[name])
            if taken + size > target * 1.3:
                continue
            assigned[name] = split
            taken += size
        if taken == 0:
            # Hicbir grup kotaya sigmadi (tek dev grup gibi): bolme yapmamak
            # yerine en kucugunu al -- bos val/test sessiz bir hata olurdu.
            free = [n for n in names if n not in assigned]
            if free:
                assigned[min(free, key=lambda n: len(groups[n]))] = split

    counts = Counter()
    for record in records:
        record.split = assigned.get(record.group, "train")
        counts[record.split] += 1
    return dict(counts)


def extract_images(records, images_out):
    """Goruntuleri COCO `file_name` alanlariyla ortusen duzene cikarir.

    Sonuc: <images_out>/<kaynak>/<ad>.jpg  ->  YOLOX tarafinda
    data_dir=<ust dizin>, name="images" ile dogrudan okunur.
    Var olan dosyalar atlanir; yarim kalan cikarma guvenle tekrarlanabilir.
    """
    # Ayni kayit tekrar katsayisi yuzunden birden fazla kez gelebilir.
    unique = {r.file_name: r for r in records}
    by_origin = defaultdict(list)
    for rec in unique.values():
        by_origin[str(rec.origin)].append(rec)

    written = skipped = 0
    for origin in sorted(by_origin):
        with Archive(origin) as archive:
            for rec in sorted(by_origin[origin], key=lambda r: r.file_name):
                target = images_out / rec.file_name
                if target.exists() and target.stat().st_size > 0:
                    skipped += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(rec.member))
                written += 1
        print(f"   {Path(origin).name}: cikarildi")
    print(f"goruntuler: {written} yazildi, {skipped} zaten vardi -> {images_out}")
