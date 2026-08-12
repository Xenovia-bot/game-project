#!/usr/bin/env python3
"""6 kaynagi tek sinifli ("ship") bir COCO veri setinde birlestirir.

Bu, havadan land+sea 2 sinifli projeden AYRI bir pivot: kamera artik gemiyle
yaklasik ayni mesafede/yukseklikte (kiyi, iskele, gemi guvertesi), yukaridan
degil. Termal/gri ton goruntu de kabul edilir -- kullanicinin acik istegi.

Hedef sinif
-----------
  1 = ship (tum gemi/tekne tipleri tek sinifta)

Kaynaklar (hepsi Roboflow COCO export'u, WUTDet haric)
-------------------------------------------------------
  vais_smd_marvel     VAIS+SMD+MARITIME+WSODD+MARVEL birlesimi (Roboflow)
  singapore_maritime  Singapore Maritime Dataset yeniden-export (Roboflow)
  sea_vessels         Sea Vessels Dataset v2 (Roboflow, kucuk, augment'li)
  ship_model          "ship model" v4 (Roboflow, buyuk, tek sinif)
  ir_thermal          ir.v1i (Roboflow, GERCEK termal, 7 gemi tipi)
  wutdet              WUTDet Part A (Pascal VOC XML, 100K'lik setin bir parcasi)

Kategori eslemesi neden tablo, neden anahtar-kelime degil
-----------------------------------------------------------
Her kaynagin kategorileri elle acilip incelendi ("gemi gibi gorunen" ada
gore degil, gercek kutu kullanimina bakilarak -- bkz. VAIS'teki "object"
adli gercek gemi sinifi, veya kullanilmayan "root" kategoriler). Sonuc
CLASS_MAPS tablosuna yazildi. Okuyucu, tabloda olmayan bir kategoriyle
karsilasirsa **sessizce atlamaz, hata verir** -- yeni bir Roboflow surumu
kategori eklerse fark edilsin diye.

Kaynaklar arasi cakisma (kritik, olculdu)
-------------------------------------------
2026-08-12'de tum arsivler uzerinde yeniden olculdu (12.057 goruntu atiliyor):
  * ship_model  -> vais_smd_marvel   9271  ayni MARVEL fotografi (biri gri
                                           tona cevrilmis). 200 rastgele
                                           ciftin 200'u pikselde ayni cikti.
  * singapore   -> vais_smd_marvel    956  ayni SMD video karesi (40/40 ayni)
  * ship_model  -> ship_model         1830 KAYNAK ICI kopya: ayni fotograf
                                           zip'te iki cozunurlukte (orijinal
                                           + Roboflow'un 640x640 kopyasi);
                                           60/60 ayni cikti.
Bu YALNIZCA bir tekrar sorunu degil: ele alinmazsa ayni fotografin biri
train'e biri val/test'e dusup sizinti yaratabilir. cross_source_identity()
bu kimligi yakalar, dedupe_sources() zengin kaynaklari once isleyip
VAIS-merge'den zaten alinmis olani cikarir.

Bilinen sinir: kimlik, dosya adinin bastaki sayisindan uretiliyor. MARVEL
adlari 6 haneli sifir dolgulu ("000027") oldugu icin guvenli, ama dolgusuz
kisa sayilarda ("10_jpg") cakisabiliyor. Olculdu: 909 kisa kimlikten
orneklemede ~%2 yanlis eslesme, yani ~15 goruntu (verinin %0,02'si) bosuna
atiliyor. Hata yonu guvenli tarafta (fazla atar, sizdirmaz).

Bolme politikasi -- neden kaynaklarin kendi train/valid/test'i kullanilmiyor
-----------------------------------------------------------------------------
Roboflow export'lari kareleri RASTGELE bolmus: ayni video/oturum ucunde de
bulunuyor. Bu haliyle dogrulama skoru sahte cikar, o yuzden bolme burada
sifirdan yapilir: her kaynak AYRI, oturum bazli, hedef 70/15/15 (goruntu
sayisi). Oturum anahtari icin bkz. ship_session_key() -- 2026-08-12'de
olculen sizinti oradaki kalip eksikliginden kaynaklaniyordu.

Lisans
------
  vais_smd_marvel      CC BY 4.0
  singapore_maritime   Public Domain
  sea_vessels          CC BY 4.0
  ship_model           CC BY 4.0
  ir_thermal           CC BY 4.0
  wutdet               Belirsiz -- indirilen Part A arsivinde lisans dosyasi
                        yok; kaynak makale (arXiv:2604.07759) GitHub deposunu
                        MIT gosteriyor ama VOC donusum betiginin kendi COCO
                        sablonu "Attribution License" (x-mol.com/groups/MIPC)
                        yaziyor. Kullanmadan once bizzat dogrulayin.

Kullanim
--------
    python tools/build_ship_dataset.py --data-dir "datasets" --out datasets/ship_merged --dry-run
    python tools/build_ship_dataset.py --data-dir "datasets" --out datasets/ship_merged --images-out datasets/ship_merged/images
"""

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

try:  # `python tools/build_ship_dataset.py` dogrudan calistirilinca (kardes modul)
    from dataset_common import (
        Archive,
        Record,
        clip_box,
        extract_images,
        has_part,
        split_by_group_targets,
        session_key,
        swap_part,
    )
except ImportError:  # `tools.build_ship_dataset` olarak paket ustunden import edilince (testler)
    from tools.dataset_common import (
        Archive,
        Record,
        clip_box,
        extract_images,
        has_part,
        split_by_group_targets,
        session_key,
        swap_part,
    )

TARGET_NAMES = ("ship",)
SHIP = 1

#: Roboflow export'larindaki "root"/supercategory kalinti kategoriler hicbir
#: kutuda kullanilmiyor (id her zaman 0'a yakin, annotations'ta gecmiyor);
#: ama read_ship_coco() TUM bildirilen kategorileri esleme ister, yoksa
#: hata verir. Bu yuzden acikca None'a eslenmis olarak tabloda duruyorlar.
CLASS_MAPS = {
    "vais_smd_marvel": {
        "vessel": None,   # kullanilmayan root (VAIS/SMD/MARITIME/WSODD/MARVEL export'unun ust kategorisi)
        "buoy": None,     # arac degil, kasten hard-negative (VESSELimg'deki ayni karar)
        "object": SHIP,   # ⚠️ GERCEK gemi sinifi budur -- Roboflow kullanicisi boyle adlandirmis
    },
    "singapore_maritime": {
        "objects": None,          # kullanilmayan root
        "boat": SHIP,
        "buoy": None,             # hard-negative
        "ferry": SHIP,
        "flying bird-plane": None,  # alakasiz -- kus/ucak, gemi degil
        "kayak": SHIP,
        "other": None,            # belirsiz; ne oldugu dogrulanamadi, atildi
        "sail boat": SHIP,
        "speed boat": SHIP,
        "vessel-ship": SHIP,
    },
    "sea_vessels": {
        "sea-vessels": None,   # kullanilmayan root
        "fishing boat": SHIP,
        "merchant ship": SHIP,
        "military ship": SHIP,
        "patrol boat": SHIP,
        "sails boat": SHIP,
        "submarine": SHIP,
        "tugboat": SHIP,
        "yacht": SHIP,
    },
    "ship_model": {
        # Roboflow'un kendi "root" kategorisi de leaf kategorisi de "ship"
        # adini tasiyor (id'leri farkli, adlari ayni) -- ad bazli esleme
        # yaptigimiz icin tek satir ikisini de kapsiyor.
        "ship": SHIP,
    },
    "ir_thermal": {
        "boat": None,   # kullanilmayan root (0 kutu, gercek sinif adlari asagida)
        "bulk carrier": SHIP,
        "canoe": SHIP,
        "container ship": SHIP,
        "fishing boat": SHIP,
        "liner": SHIP,
        "sailboat": SHIP,
        "warship": SHIP,
    },
}

#: Kaynak -> (zip icindeki varsayilan dosya adi, okuyucu turu).
SOURCE_FILES = {
    "vais_smd_marvel": "VAIS_RGB-SMD-MARITIME-WSODD-MARVEL.v5-rgb_40_grayscale_60.coco.zip",
    "singapore_maritime": "Singapore maritime.v5i.coco.zip",
    "sea_vessels": "Sea Vessels Dataset.v2-sea_vessels_v2.coco.zip",
    "ship_model": "ship model.v4i.coco.zip",
    "ir_thermal": "ir.v1i.coco.zip",
    "wutdet": "WUTDet Part A.zip",
}

#: dedupe_sources() bu sirayla isler: zengin/adanmis kaynaklar once kimlik
#: "sahiplenir", vais_smd_marvel EN SONDA kalir ki zaten baskasinda olan
#: fotograflari/kareleri disarida biraksin. Sira, zenginlik olcumune dayanir:
#: singapore_maritime 9 alt-sinifli + Public Domain (SMD icin en zengin);
#: ship_model MARVEL-tarzi fotograflarin adanmis/daha buyuk kopyasi (renkli).
DEDUPE_ORDER = ("singapore_maritime", "ship_model", "sea_vessels",
                "ir_thermal", "wutdet", "vais_smd_marvel")

#: Hedef bolme: 70 / 15 / 15 (goruntu sayisi olarak; gruplar bolunmedigi
#: icin gerceklesen oran birkac puan sapabilir -- calisma ciktisi gercek
#: orani yazar). Roboflow'un KENDI train/valid/test bolmesi kullanilmaz:
#: kareleri rastgele bolmus, ayni oturum ucunde birden bulunuyor.
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SPLIT_SEED = 1337


def cross_source_identity(file_name):
    """Farkli zip'lerde ayni fotografi/video karesini yakalayan anahtar.

    2026-08-11'de olculdu (bkz. modul basligi): MVI_XXXX_VIS_frameN video
    kare adlandirmasi (SMD kokenli) ve salt-numarali "<sayi>_jpg..." adlandirma
    (MARVEL kokenli) birden fazla kaynakta BIREBIR ayni ID'yle tekrarlaniyor.
    Baska bir adlandirma kalibina uymayan dosyalar icin None doner --
    o durumda cakisma kontrolu yapilmaz (yanlis pozitif sizintiden daha
    guvenli: bilinmeyen bir orintuyu es gecmek, olmayan bir cakismayi
    uydurmaktan iyidir).
    """
    base = file_name.rsplit("/", 1)[-1]
    # Video ID'si TEK BASINA yetmez -- ayni videonun onlarca farkli karesi
    # var; yalnizca "MVI_XXXX" yakalarsak o videonun TUM kareleri tek
    # kimlige collapse olur (2026-08-11'de olculdu: singapore_maritime
    # 6350 goruntuden 63'e dustu). Kare numarasina kadar eslenmeli.
    m = re.match(r"(MVI_\d+.*?_frame\d+)", base)
    if m:
        return "video:" + m.group(1)
    m = re.match(r"^(\d+)_jpg", base)
    if m:
        return "marvel:" + m.group(1)
    return None


#: SMD (Singapore Maritime Dataset) video karesi:
#: "MVI_1478_VIS_OB_frame90_jpg.rf.<hash>.jpg". Kare numarasi 'frame' ekine
#: YAPISIK oldugu icin genel session_key() bu kalibi goremiyor ve her kareyi
#: ayri oturum sayiyordu. OLCULDU (2026-08-12): 6350 goruntu -> 6350 "oturum",
#: yani grup bazli bolme rastgele bolmeye cokuyordu; 63 videonun 63'u de hem
#: train hem val'de cikti ve val karelerinin %93,6'sinin en yakin train karesi
#: 5 kare (~0,17 s) uzaktaydi. Kaynak oneki YOK: ayni video birden fazla
#: kaynakta bulunabiliyor (SMD hem singapore_maritime'da hem vais icinde).
SMD_VIDEO_RE = re.compile(r"^(MVI_\d+[A-Za-z_]*?)_frame\d+", re.IGNORECASE)

#: WUTDet Part A: 100K'lik setin her 20. karesi ("0000000.jpg", "0000020.jpg";
#: olculdu: 5024 ardisik farkin 5020'si tam 20). Makale kareleri 1-5 saniyede
#: bir ornekledigini soyluyor ve dizi bazli bolmeden hic bahsetmiyor
#: (arXiv:2604.07759). Ardisik ID'ler ayni sahne: olculdu (2026-08-12), 1068
#: val goruntusunun 994'unun ID komsusu train'deydi. Gercek video kimligi
#: dosya adinda YOK, o yuzden ID araligi yapay oturum olarak kullanilir.
WUTDET_ID_RE = re.compile(r"^(\d+)\.jpe?g$", re.IGNORECASE)
WUTDET_BUCKET_IDS = 1000          # 20'lik adimda ~50 goruntu

#: ir.v1i: dosya adlari yalnizca '1_XXXX' / '9_XXXX' iki on-eke sahip
#: (olculdu 2026-08-11: 8398 goruntunun 7398'i '1_', 1000'i '9_'), yani
#: session_key() TUM veriyi 2 deve gruba topluyordu ve bolme ya hepsini ya
#: hicbirini val'e atiyordu. On-ekin video kimligi mi yukleme sirasi mi
#: oldugu DOGRULANAMADI; temkinli taraf (gruplamayi kaldirmak degil,
#: inceltmek) secildi.
IR_THERMAL_RE = re.compile(r"^(\d+)_(\d+)_jpg")
IR_THERMAL_BUCKET = 200


def ship_session_key(file_name, source):
    """Bu 6 kaynagin adlandirmalarini bilen oturum anahtari.

    Genel session_key() Roboflow'un '<taban>_<kare>_jpg.rf.<hash>' kalibini
    tanir; asagidaki uc kaynak o kaliba UYMUYOR ve taninmadiklarinda her
    goruntu kendi oturumu olup bolme rastgeleye cokuyor. Tanimadigi bir ad
    icin genel isleve duser (sessizce yanlis gruplamaz).
    """
    base = file_name.rsplit("/", 1)[-1]

    match = SMD_VIDEO_RE.match(base)
    if match:
        return f"smd:{match.group(1).lower()}"

    if source == "wutdet":
        match = WUTDET_ID_RE.match(base)
        if match:
            return f"wutdet:{int(match.group(1)) // WUTDET_BUCKET_IDS}"

    if source == "ir_thermal":
        match = IR_THERMAL_RE.match(base)
        if match:
            return (f"ir_thermal:{match.group(1)}:"
                    f"{int(match.group(2)) // IR_THERMAL_BUCKET}")

    return session_key(file_name, source)


def dedupe_sources(by_source):
    """DEDUPE_ORDER sirasiyla isler; bir kimlik zaten alinmissa sonraki
    kaynaktaki kopyasini atar. Sozluk mutasyona ugratilir (kayitlar filtrelenir).

    Donus: {kaynak: atilan_kayit_sayisi} -- rapor icin.
    """
    claimed = set()
    dropped = {}
    for source in DEDUPE_ORDER:
        records = by_source.get(source, [])
        kept = []
        n_dropped = 0
        for rec in records:
            identity = cross_source_identity(rec.file_name)
            if identity is not None and identity in claimed:
                n_dropped += 1
                continue
            if identity is not None:
                claimed.add(identity)
            kept.append(rec)
        by_source[source] = kept
        dropped[source] = n_dropped
    return dropped


def read_ship_coco(archive, class_map, source, stats=None):
    """Standart Roboflow COCO export'u: train/valid/test + _annotations.coco.json.

    5 kaynagin 5'i de bu duzeni kullaniyor; tek okuyucu class_map ile
    parametrize edilip hepsine uygulanir (build_dataset.py'deki
    read_roboflow_coco'nun ayni deseni, ama LAND/SEA yerine tek sinif SHIP
    kullandigi ve "target=None -> kutuyu at, goruntuyu SAKLA" davranisi
    ayni kaldigi icin ayri fonksiyon: add_box() oradaki LAND/SEA'ye kilitli).
    """
    records = []
    members = sorted(n for n in archive.names()
                     if n.rsplit("/", 1)[-1] == "_annotations.coco.json")
    if not members:
        raise SystemExit(f"HATA: {source} icinde _annotations.coco.json bulunamadi")
    for member in members:
        base = member.rsplit("/", 1)[0] if "/" in member else ""
        data = json.loads(archive.read(member))
        cats = {c["id"]: str(c["name"]).strip().lower() for c in data["categories"]}
        unknown = set(cats.values()) - set(class_map)
        if unknown:
            raise SystemExit(
                f"HATA: {source} icinde eslenmemis kategori: {sorted(unknown)}. "
                f"tools/build_ship_dataset.py CLASS_MAPS tablosunu guncelleyin."
            )
        by_image = defaultdict(list)
        for ann in data["annotations"]:
            by_image[ann["image_id"]].append(ann)
        for image in data["images"]:
            member_img = f"{base}/{image['file_name']}" if base else image["file_name"]
            if not archive.exists(member_img):
                raise SystemExit(f"HATA: {source} goruntusu eksik: {member_img}")
            rec = Record(source, member_img, f"{source}/{image['file_name']}",
                        int(image["width"]), int(image["height"]), "train",
                        group=ship_session_key(image["file_name"], source))
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
                rec.anns.append({"bbox": box, "category_id": SHIP,
                                 "iscrowd": 0, "ignore": 0})
                if stats is not None:
                    stats[f"{source}:keep:{name}"] += 1
            records.append(rec)
    return records


def read_wutdet_voc(archive, source="wutdet", stats=None):
    """WUTDet Part A: Pascal VOC XML (voc/Annotations/*.xml + voc/JPEGImages/).

    Resmi ImageSets/Main/{train,val}.txt bolmesine guvenilmiyor: bu projede
    incelenen her Roboflow kaynagi kare/kopya bazli sizinti tasiyordu (bkz.
    modul basligi); WUTDet'in kendi bolmesi de ayni riski tasiyabilir, hepsi
    "train" olarak okunup asagida oturum bazli yeniden bolunuyor.
    """
    records = []
    xml_members = sorted(n for n in archive.names()
                         if has_part(n, "Annotations") and n.lower().endswith(".xml"))
    if not xml_members:
        raise SystemExit(f"HATA: {source} icinde Annotations/*.xml bulunamadi")
    for member in xml_members:
        root = ET.fromstring(archive.read(member))
        filename = root.findtext("filename")
        if not filename:
            continue
        img_member = swap_part(member, "Annotations", "JPEGImages")
        img_member = img_member.rsplit("/", 1)[0] + "/" + filename
        if not archive.exists(img_member):
            raise SystemExit(f"HATA: {source} goruntusu eksik: {img_member}")
        size = root.find("size")
        width = int(size.findtext("width"))
        height = int(size.findtext("height"))
        rec = Record(source, img_member, f"{source}/{filename}", width, height,
                    "train", group=ship_session_key(filename, source))
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip().lower()
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            x1 = float(bnd.findtext("xmin"))
            y1 = float(bnd.findtext("ymin"))
            x2 = float(bnd.findtext("xmax"))
            y2 = float(bnd.findtext("ymax"))
            box = clip_box(x1, y1, x2 - x1, y2 - y1, width, height)
            if box is None:
                continue
            if name != "ship":
                if stats is not None:
                    stats[f"{source}:drop:{name}"] += 1
                continue
            rec.anns.append({"bbox": box, "category_id": SHIP,
                             "iscrowd": 0, "ignore": 0})
            if stats is not None:
                stats[f"{source}:keep:{name}"] += 1
        records.append(rec)
    return records


# --------------------------------------------------------------------------
# Dogrulama kapisi
# --------------------------------------------------------------------------
def validate(records):
    """Bozuk bir sey varsa dosya uretmeden hata verir.

    build_dataset.py'nin validate()'inden farki: VisDrone gibi 'resmi
    bolmede zaten sizinti var, kabul edilir' istisnasi yok -- burada hicbir
    kaynagin resmi bolmesine guvenilmiyor, dolayisiyla **hicbir oturum**
    birden fazla bolumde gorulmemeli (train/val/test tumu).
    """
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
            if ann["category_id"] != SHIP:
                problems.append(f"gecersiz kategori: {ann['category_id']}")

    leaked = [g for g, s in groups.items() if len(s) > 1]
    if leaked:
        problems.append(
            f"{len(leaked)} oturum birden fazla bolumde "
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
        })
        for ann in rec.anns:
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": ann["category_id"],
                "bbox": [round(v, 2) for v in ann["bbox"]],
                "area": round(ann["bbox"][2] * ann["bbox"][3], 2),
                "iscrowd": ann["iscrowd"],
            })
            ann_id += 1
    return {
        "info": {"description": "Gemi tespiti - tek sinifli COCO (6 kaynak birlesimi)",
                 "class_scheme": "ship"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": SHIP, "name": "ship", "supercategory": "vessel"}],
    }


def read_existing_ship_coco(out):
    """Daha once uretilmis ship_merged JSON'larini yeniden bolmek icin okur.

    Ham zip'ler yerelde artik bulunmasa bile, cikarilmis goruntuler ile COCO
    JSON'lari guvenli bir yeniden bolme icin yeterlidir. Oturum anahtari her
    zaman dosya adindan yeniden uretilir; eski JSON'un onceki (hatali) split
    bilgisine guvenilmez. Bu yol, *yalnizca* mevcut ciktiyi onarmak icindir;
    normal ilk olusturma yine ham kaynaklardan yapilir.
    """
    records = []
    seen_files = set()
    expected_categories = [{"id": SHIP, "name": "ship",
                            "supercategory": "vessel"}]

    for split in ("train", "val", "test"):
        path = out / "annotations" / f"instances_{split}.json"
        if not path.is_file():
            raise SystemExit(f"HATA: mevcut split bulunamadi: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("categories") != expected_categories:
            raise SystemExit(
                f"HATA: {path} tek sinifli ship COCO semasinda degil; "
                "yeniden bolme reddedildi."
            )

        images = data.get("images", [])
        by_id = {image.get("id"): image for image in images}
        if len(by_id) != len(images):
            raise SystemExit(f"HATA: {path} yinelenen image id iceriyor")
        anns_by_image = defaultdict(list)
        for ann in data.get("annotations", []):
            image_id = ann.get("image_id")
            if image_id not in by_id:
                raise SystemExit(f"HATA: {path} bilinmeyen image_id: {image_id}")
            if ann.get("category_id") != SHIP or len(ann.get("bbox", [])) != 4:
                raise SystemExit(f"HATA: {path} gecersiz ship anotasyonu: {ann}")
            anns_by_image[image_id].append(ann)

        for image in images:
            source = str(image.get("source", "")).strip()
            file_name = str(image.get("file_name", "")).replace("\\", "/")
            if source not in SOURCE_FILES:
                raise SystemExit(
                    f"HATA: {path} bilinmeyen/eksik kaynak etiketi: {source!r}"
                )
            if not file_name.startswith(f"{source}/"):
                raise SystemExit(
                    f"HATA: {path} kaynak ve dosya adi uyusmuyor: "
                    f"{source!r} / {file_name!r}"
                )
            if not file_name or file_name in seen_files:
                raise SystemExit(f"HATA: splitler arasi yinelenen dosya: {file_name}")
            seen_files.add(file_name)
            rec = Record(source, None, file_name, int(image.get("width", 0)),
                         int(image.get("height", 0)), split,
                         group=ship_session_key(file_name, source))
            for ann in anns_by_image[image["id"]]:
                rec.anns.append({
                    "bbox": ann["bbox"], "category_id": SHIP,
                    "iscrowd": int(ann.get("iscrowd", 0)), "ignore": 0,
                })
            records.append(rec)
    return records


def repartition_existing_ship_coco(out, val_fraction, test_fraction, dry_run):
    """Mevcut ship_merged ciktilarini oturum bazli %70/%15/%15 yeniden boler."""
    records = read_existing_ship_coco(out)

    # Bir video/oturum iki kaynakta kalmissa kaynak-bazli kota atamasi onu
    # farkli splitlere koyabilir. Ham kaynaklardan global politika ile yeniden
    # uretmek gerekir; burada sessizce sizinti yaratmak yerine duruyoruz.
    group_sources = defaultdict(set)
    for record in records:
        group_sources[record.group].add(record.source)
    shared = sorted(group for group, sources in group_sources.items()
                    if len(sources) > 1)
    if shared:
        raise SystemExit(
            "HATA: %d oturum birden fazla kaynakta goruluyor "
            "(ornek: %s). Ham zip'lerden global yeniden bolme gerekir."
            % (len(shared), shared[:3])
        )

    by_source = defaultdict(list)
    for record in records:
        by_source[record.source].append(record)
    for source in DEDUPE_ORDER:
        source_records = by_source.get(source, [])
        if not source_records:
            raise SystemExit(f"HATA: mevcut cikti icinde kaynak yok: {source}")
        counts = split_by_group_targets(source_records, val_fraction,
                                        test_fraction, seed=SPLIT_SEED)
        print(f"   {source:<20}{len(source_records):>7} goruntu, "
              f"{len({r.group for r in source_records}):>6} oturum -> "
              f"train {counts.get('train', 0)} / val {counts.get('val', 0)} / "
              f"test {counts.get('test', 0)}")

    problems = validate(records)
    if problems:
        print("\n" + "!" * 74)
        print(f"DOGRULAMA BASARISIZ - {len(problems)} sorun")
        for problem in problems[:20]:
            print("  -", problem)
        raise SystemExit(1)

    splits = [(name, build_coco(records, name))
              for name in ("train", "val", "test")]
    print("\nDogrulama gecti: oturum sizintisi yok, kutular ve tek sinif semasi gecerli.")
    print(f"{'':<8}{'goruntu':>10}{'kutu':>10}{'bos':>8}")
    for name, coco in splits:
        with_box = {ann["image_id"] for ann in coco["annotations"]}
        print(f"{name:<8}{len(coco['images']):>10}{len(coco['annotations']):>10}"
              f"{len(coco['images']) - len(with_box):>8}")

    if dry_run:
        print("\n--dry-run: mevcut JSON'lar degistirilmedi.")
        return

    # Tum JSON'lar once bellekte kuruldu ve denetlendi. Gecici dosyalar
    # tamamlanmadan mevcut manifestin uzerine yazilmaz; yarim bolme kalmaz.
    targets = []
    for name, coco in splits:
        target = out / "annotations" / f"instances_{name}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(coco), encoding="utf-8")
        targets.append((temporary, target))
    for temporary, target in targets:
        temporary.replace(target)
    print(f"\nyazildi: {out / 'annotations'} (oturum-bazli yeniden bolme)")


def summarise(by_source_final, stats, dropped_by_dedupe):
    print("\n" + "=" * 74)
    print("KAYNAK BAZLI OZET (dedupe SONRASI)")
    print("=" * 74)
    header = f"{'kaynak':<20}{'goruntu':>9}{'train':>8}{'val':>7}{'test':>7}{'kutu':>9}{'bos':>7}"
    print(header); print("-" * len(header))
    grand_img = grand_box = 0
    for source in DEDUPE_ORDER:
        recs = by_source_final.get(source, [])
        n_img = len(recs)
        n_train = sum(1 for r in recs if r.split == "train")
        n_val = sum(1 for r in recs if r.split == "val")
        n_test = sum(1 for r in recs if r.split == "test")
        n_box = sum(len(r.anns) for r in recs)
        n_empty = sum(1 for r in recs if not r.anns)
        grand_img += n_img; grand_box += n_box
        print(f"{source:<20}{n_img:>9}{n_train:>8}{n_val:>7}{n_test:>7}{n_box:>9}{n_empty:>7}")
    print("-" * len(header))
    print(f"{'TOPLAM':<20}{grand_img:>9}{'':>8}{'':>7}{'':>7}{grand_box:>9}")

    print("\nCAKISMA NEDENIYLE ATILAN (dedupe_sources)")
    for source in DEDUPE_ORDER:
        n = dropped_by_dedupe.get(source, 0)
        if n:
            print(f"  {source:<20}{n:>6} goruntu (baska kaynakta zaten var)")

    print("\nSINIF ESLEME DOKUMU (kaynak:karar:orijinal-ad)")
    for key in sorted(stats):
        print(f"  {key:<45} {stats[key]:>8}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="datasets")
    parser.add_argument("--out", default="datasets/ship_merged")
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION,
                        help="kaynak-bazli ayrilacak test orani")
    parser.add_argument("--images-out", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repartition-existing", action="store_true",
                        help="mevcut --out/annotations JSON'larini ham zip "
                        "olmadan oturum bazli yeniden bol")
    parser.add_argument("--source", action="append", default=None,
                        metavar="ETIKET=YOL",
                        help="tek bir kaynagin yolunu degistir; birden fazla "
                             "kez verilebilir")
    parser.add_argument("--skip", action="append", default=None,
                        metavar="ETIKET",
                        help="bu kaynagi tamamen atla (lisans netlesene kadar "
                             "orn. --skip wutdet)")
    args = parser.parse_args()

    if not 0 < args.val_fraction < 1 or not 0 <= args.test_fraction < 1 \
            or args.val_fraction + args.test_fraction >= 1:
        raise SystemExit(
            "HATA: val orani pozitif, test orani negatif olmayan olmali; "
            "toplamlari 1'den kucuk olmali"
        )

    if args.repartition_existing:
        if args.test_fraction == 0:
            raise SystemExit("HATA: --repartition-existing icin test orani pozitif olmali")
        repartition_existing_ship_coco(Path(args.out), args.val_fraction,
                                       args.test_fraction, args.dry_run)
        return

    data_dir = Path(args.data_dir)
    skip = set(args.skip or ())
    overrides = {}
    for item in (args.source or ()):
        if "=" not in item:
            raise SystemExit(f"HATA: 'etiket=yol' bekleniyordu: {item}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = Path(value.strip())
    unknown = set(overrides) - set(SOURCE_FILES)
    if unknown:
        raise SystemExit(f"HATA: bilinmeyen kaynak etiketi: {sorted(unknown)}")

    stats = Counter()
    by_source = {}
    for source, filename in SOURCE_FILES.items():
        if source in skip:
            print(f">> {source:<20} ATLANDI (--skip)")
            continue
        path = overrides.get(source, data_dir / filename)
        if not path.exists():
            raise SystemExit(f"HATA: kaynak bulunamadi: {path}")
        print(f">> {source:<20} {path.name}")
        with Archive(path) as archive:
            if source == "wutdet":
                got = read_wutdet_voc(archive, source=source, stats=stats)
            else:
                got = read_ship_coco(archive, CLASS_MAPS[source], source, stats=stats)
        if not got:
            raise SystemExit(f"HATA: '{source}' kaynagindan hic goruntu okunamadi")
        for rec in got:
            rec.origin = path
        by_source[source] = got

    # Kopya eleme BOLMEDEN once: sonra elenirse bolme oranlari bozulur
    # (olculdu 2026-08-12: vais_smd_marvel bolundukten sonra 24.648'den
    # 14.421'e dusunce val payi hedef %25 yerine %10,6 cikmisti).
    dropped = dedupe_sources(by_source)

    # Her kaynak AYRI bolunur ki kucuk kaynaklar (sea_vessels, 736 goruntu)
    # buyuklerin yaninda kaybolmasin; global karistirma boyuta gore
    # agirliklanirdi. Gruplar bolunmez -> oturum sizintisi olmaz.
    for source, recs in by_source.items():
        counts = split_by_group_targets(recs, args.val_fraction,
                                        args.test_fraction, seed=SPLIT_SEED)
        n = len(recs)
        groups = len({r.group for r in recs})
        print(f"   {source:<20}{n:>7} goruntu, {groups:>6} oturum -> "
              f"train {counts.get('train',0)} / val {counts.get('val',0)} / "
              f"test {counts.get('test',0)}")

    records = [r for recs in by_source.values() for r in recs]

    problems = validate(records)
    if problems:
        print("\n" + "!" * 74)
        print(f"DOGRULAMA BASARISIZ - {len(problems)} sorun")
        for p in problems[:20]:
            print("  -", p)
        if len(problems) > 20:
            print(f"  ... ve {len(problems)-20} tane daha")
        raise SystemExit(1)
    print("\nDogrulama gecti: kutular sinir icinde, kategori {1}, "
          "oturum sizintisi yok, kaynaklar arasi kopya yok.")

    summarise(by_source, stats, dropped)

    splits = [(name, build_coco(records, name)) for name in ("train", "val", "test")]
    print("\n" + "=" * 74)
    print(f"{'':<8}{'goruntu':>10}{'kutu':>10}{'bos':>8}")
    for name, coco in splits:
        real = [a for a in coco["annotations"]]
        with_box = {a["image_id"] for a in real}
        empty = len(coco["images"]) - len(with_box)
        print(f"{name:<8}{len(coco['images']):>10}{len(real):>10}{empty:>8}")

    if args.dry_run:
        print("\n--dry-run: dosya yazilmadi.")
        return

    out = Path(args.out)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    for name, coco in splits:
        target = out / "annotations" / f"instances_{name}.json"
        target.write_text(json.dumps(coco), encoding="utf-8")
        print(f"yazildi: {target}")

    manifest = out / "image_manifest.json"
    manifest.write_text(json.dumps({
        "sources": {s: str(SOURCE_FILES[s]) for s in by_source},
        "licenses": {
            "vais_smd_marvel": "CC BY 4.0",
            "singapore_maritime": "Public Domain",
            "sea_vessels": "CC BY 4.0",
            "ship_model": "CC BY 4.0",
            "ir_thermal": "CC BY 4.0",
            "wutdet": "BELIRSIZ - kullanmadan once dogrulayin (bkz. modul basligi)",
        },
        "members": [{"source": r.source, "member": r.member,
                     "file_name": r.file_name} for r in records],
    }), encoding="utf-8")
    print(f"yazildi: {manifest}  (goruntuleri cikarmak icin)")

    if args.images_out:
        extract_images(records, Path(args.images_out))


if __name__ == "__main__":
    main()
