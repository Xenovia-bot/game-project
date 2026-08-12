#!/usr/bin/env python3
"""KV260 DPU'suna uyumlu YOLOX-Tiny -- tek sinif "ship" (gemi tespiti pivotu).

Bu, havadan land+sea 2 sinifli projeden AYRI bir dal: kamera artik gemiyle
yaklasik ayni mesafede/yukseklikte (kiyi, iskele, gemi guvertesi), yukaridan
degil. Veri `tools/build_ship_dataset.py` ile 6 kaynaktan birlestirildi
(bkz. o dosyanin basligi -- kaynaklar arasi 2 buyuk kopya kumesi olculup
elendi). Mimari ve DPU uyarlamalari (ReLU + DPUFocus), aerial projede
kartta dogrulanmis Tiny geometrisinden **degistirilmeden** devralinir --
tek degisken veri ve girdi boyutu olsun diye.

Girdi boyutu -- 512x512, olculerek secildi (2026-08-11)
---------------------------------------------------------
Aerial projenin 896x512 karari VisDrone'un neredeyse tamami 16:9/4:3 oldugu
ve kutularin uniform kucuk oldugu bir veri icindi. Bu veri seti FARKLI:
kaynaklarin **%62'si kare** (sea_vessels, ship_model, vais_smd_marvel hepsi
1:1 kirpilmis/stretch'lenmis), yalnizca %22'si 16:9 (singapore_maritime,
wutdet). Kare kanvas bu yuzden burada letterbox israfini aerial'daki
mantigin TERSINE cevirerek en aza indiriyor.

Kutu boyutu da heterojen (aerial'daki gibi uniform kucuk degil): p10=%2.1,
medyan=%8.0, p90=%80.5 (goruntu genisligine oran). 512x512'de olculdu:

    cand=384   p10=8.1px  p25=13.8px  <8px: %9.6   <16px: %29.6
    cand=448   p10=9.4px  p25=16.1px  <8px: %6.5   <16px: %24.9
    cand=512   p10=10.8px p25=18.4px  <8px: %4.2   <16px: %20.4
    cand=640   p10=13.5px p25=23.0px  <8px: %2.3   <16px: %14.1

512 secildi: FPS bu pivotun ana gerekcesi (bkz. HANDOFF_CLAUDE.md §5 --
kartta olculen darboğaz DPU degil, ARM CPU'daki on-isleme; on-isleme suresi
piksel sayisiyla dogru orantili). 512x512 = 262.144 px, 896x512'nin (aerial)
%57'si -- DPU ve on-isleme sureleri kabaca ayni oranda dusmesi beklenir
(TAHMIN, kartta olculmedi). Karsiliginda kutularin yalnizca %4,2'si 8px
altina dusuyor (aerial'da VisDrone'un tamami zaten kucuktu, secim yoktu;
burada VAR ve FPS lehine kullanildi). 640 FPS'te daha az kazandirir ama
kucuk nesne kaybi da azalir -- dogruluk yetersiz kalirsa once denenecek
adim budur, mimariyi degistirmeye gerek yok.

DPU uyumlulugu (aerial projeden degismeden devralindi)
---------------------------------------------------------
  1. act="relu": DPU, SiLU'yu desteklemez.
  2. DPUFocus: Focus'un strided-slice'i DPU'da calismaz; sabit agirlikli
     conv ayni space-to-depth'i uretir.
  3. depthwise=False (Tiny geometrisi): Nano'nun depthwise conv'lari
     per-tensor INT8'de coktu (bkz. HANDOFF §5); Tiny bu sorunu tasimiyor.

Alan-saglamligi augmentasyonu -- termal/gri ton/parlama, arastirilarak eklendi
---------------------------------------------------------------------------------
Karta gidecek gercek girdi buyuk ihtimalle RGB olmayacak (termal kamera veya
gri ton video akisi) ve deniz yuzeyi gunes yansimasi/parlama gibi bozulmalar
tasiyacak. Egitim verisi bunun bir kismini zaten dogal olarak icin
(ir_thermal: gercek termal; vais_smd_marvel'in %60'i gri tona cevrilmis) ama
RGB kaynaklarin (ship_model, sea_vessels, singapore_maritime'in cogu) modelin
renge fazla guvenmesine yol acma riski var. Cozum, egitim sirasinda RGB
goruntuleri de olasiliklarla ayni bozulmalara maruz birakmak.

Kaynak arastirmasi (2026-08-12):
  * Albumentations (Buslaev ve ark. 2020, Information dergisi, MDPI --
    hakemli, https://www.mdpi.com/2078-2489/11/2/125) bu tur pikselsel
    augmentasyonlar icin standart, gecerliligi olculmus kutuphane.
  * ToGray: YOLOv8 tabanli bir dusme-tespiti calismasinda dusuk olasilikla
    uygulanip modelin rengi degil parlaklik/sekli kullanmayi ogrenmesini
    sagladigi raporlandi -- burada TAM olarak istedigimiz ozellik.
  * CLAHE (Contrast Limited Adaptive Histogram Equalization): BVLOS drone
    engel-tespiti calismasinda test edilen 8 Albumentations tekniginden
    biri; dusuk kontrastli (termal) ve asiri-pozlanmis (parlama) goruntuleri
    ayni ailede ele alan klasik, kanitlanmis bir teknik.
  * RandomBrightnessContrast/parlama: "Enhancing Maritime Object Detection
    ... with Data Augmentation" (arXiv:2510.07346) parlaklik/kontrast
    augmentasyonunun DENIZCILIK goruntulerinde olculmus faydasini raporluyor
    (birlesik pipeline ile mAP@0.5 0.80 -> 0.89, +0.09). Bu makale gunes
    parlamasini/yansimayi acikca "kacinilmasi gereken saptirici" olarak
    tanimliyor -- bizim RandomSunFlare/RandomShadow eklentimizin gerekcesi.
  * GaussNoise/ISONoise: dusuk isikli/termal sensorlerin taninan gurultu
    profili; ir.v1i orneklerinde gozle de gorulen tane/gurultu var.

Her teknik dusuk-orta olasilikla (%15-30) uygulanir: amac goruntuleri hep
bozmak degil, egitim boyunca CESITLI kosullara maruz birakmaktir. Yalnizca
egitimde (get_dataset); dogrulama (get_eval_dataset) bozulmamis kalir ki
AP sayisi augmentasyon sansina bagli olmasin.

Gri/termal icin ikinci katman -- zincirin SONUNDA (GrayThermalTransform)
------------------------------------------------------------------------
Yukaridaki ToGray tek basina yetmiyor: zincirin basinda, YOLOX'un
augment_hsv'sinden ONCE calisiyor ve augment_hsv doygunlugu carpmiyor
TOPLUYOR (S=0'a +30'a kadar ekleyebiliyor). Olculdu (2026-08-12, YOLOX
6ddff48'deki fonksiyonun birebir kopyasiyla, 200 deneme): gri goruntulerin
%13'u yeniden renklendi. Bu yuzden gri'ye cevirme bir kez de TUM zincirin
sonunda yapilir; oraya hicbir sey dokunamaz. Ayni yerde, gri orneklerin bir
kisminin polaritesi ters cevrilir (termal white-hot / black-hot -- kamera
ayari, veri ozelligi degil).

Dayanikliligi OLCMEK icin (varsaymak yerine):
    SHIP_EVAL_GRAY=1 python YOLOX/tools/eval.py -f yolox_tiny_ship.py ...
ayni checkpoint'i gri'ye cevrilmis val uzerinde olcer; AP dususu
dayanikliligin sayisal karsiligidir. Gercek termal icin kaynak bazli AP'ye
bakin (ir_thermal) -- bkz. not defterindeki kaynak-bazli degerlendirme.
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

for _alias, _type in (("float", float), ("int", int), ("bool", bool)):
    if _alias not in np.__dict__:
        setattr(np, _alias, _type)

from yolox.data import COCODataset
from yolox.exp import Exp as MyExp

for _helper_dir in (Path(__file__).resolve().parent,
                    Path(__file__).resolve().parent.parent):
    if str(_helper_dir) not in sys.path:
        sys.path.insert(0, str(_helper_dir))

#: Saha tanimi: tum gemi/tekne tipleri tek sinifta (tools/build_ship_dataset.py).
TARGET_CLASSES = ("ship",)


def assert_class_scheme(coco, expected):
    """Anotasyon semasi modelin sinif sayisiyla uyusmuyorsa hemen durur.

    Eski bir sema (orn. 2 sinifli aerial `instances_*.json`) diskte kalirsa
    model sessizce yanlis etiketlerle egitilir. Ucuz bir kapi, pahali bir
    hatayi onler (aerial exp'inde ayni desen).
    """
    if expected is None:
        return
    found = sorted(coco.cats)
    if found != list(range(1, expected + 1)):
        names = [coco.cats[c].get("name", c) for c in found]
        raise ValueError(
            f"Anotasyon semasi uyusmuyor: {expected} sinif bekleniyordu, "
            f"{len(found)} bulundu ({names}). Veriyi "
            f"'tools/build_ship_dataset.py' ile yeniden uretin."
        )


def _build_robust_augmentor():
    """RGB goruntuleri termal/gri-ton/parlama kosullarina maruz birakan
    pikselsel augmentasyon zinciri. Gerekce ve kaynaklar modul basliginda.

    Hepsi PIKSELSEL (kutu koordinatlarini degistirmez) -- bbox_params
    gerekmiyor. load_resized_img() asamasinda, goruntu 512'ye kucultuldukten
    SONRA ve YOLOX'un kendi TrainTransform'undan ONCE calisip duz bir numpy
    goruntu doner (neden kucultmeden sonra: bkz. RobustShipDataset).
    """
    try:
        import albumentations as A
    except ImportError as exc:
        raise SystemExit(
            "HATA: albumentations kurulu degil. Kaggle imajinda varsayilan "
            "olarak bulunur; yerelde 'pip install albumentations' calistirin."
        ) from exc

    return A.Compose([
        # Renge degil parlaklik/sekle guvenmeyi ogretir -- termal/gri ton
        # girdiyle dogrudan ayni dagilim (bkz. modul basligi, ToGray).
        A.ToGray(p=0.25),
        # Deniz yuzeyi gunes yansimasi/parlama VE dusuk kontrastli termal --
        # ayni ailede, tek seferde birini uygular (OneOf: gercekci kalsin
        # diye ustuste yiginlanmaz).
        A.OneOf([
            A.RandomSunFlare(src_radius=150, p=1.0),
            A.RandomShadow(p=1.0),
            A.CLAHE(clip_limit=3.0, p=1.0),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.3),
        # Dusuk isik/termal sensor gurultusu.
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.ISONoise(p=1.0),
        ], p=0.15),
    ])


#: Aga giren goruntunun GARANTILI gri olma olasiligi. Neden ayri bir adim,
#: neden Albumentations'taki ToGray yetmiyor: YOLOX'un augment_hsv'si
#: (data_augment.py) doygunlugu CARPMIYOR, TOPLUYOR -- S=0 olan gri bir
#: goruntuye +30'a kadar doygunluk ekleyebiliyor. Olculdu (2026-08-12, YOLOX
#: 6ddff48'deki fonksiyonun birebir kopyasiyla, 200 deneme): gri goruntulerin
#: %13'u yeniden renklendi, en buyuk kanal farki 29/255. ToGray load_image'ta
#: (HSV'den ONCE) calistigi icin bu geri alma kacinilmaz; bu yuzden gri'ye
#: cevirme burada, TUM augmentasyon zincirinin SONUNDA bir kez daha yapilir.
TRAIN_GRAY_PROB = 0.40

#: Gri'ye cevrilen orneklerin icinde polaritesi ters cevrilenlerin orani.
#: Termal kameralarda white-hot / black-hot iki ayri gorunum uretir ve bu
#: KAMERA AYARIDIR -- sahada hangisinin secilecegi bilinmiyor. Yalnizca
#: white-hot ile egitilen model digerinde coker. Literaturde termal veri
#: uretiminde "intensity inversion" kullaniliyor (bkz. Liu ve ark. 2021,
#: Mobile Information Systems -- CycleGAN + yogunluk tersleme).
#: DIKKAT: bu bir TAHMIN, bu veri setinde faydasi olculmedi; egitim sonrasi
#: kaynak bazli AP ile (ozellikle ir_thermal) dogrulanmali.
TRAIN_INVERT_PROB = 0.25

#: BT.601 parlaklik agirliklari. YOLOX preproc'u kanal sirasini BGR birakir
#: (RGB'ye cevirmez) ve CHW dondurur -- dogrulandi: preproc() icinde yalnizca
#: transpose(2,0,1) var, [:, :, ::-1] yok.
_BGR_LUMA = np.array([0.114, 0.587, 0.299], dtype=np.float32).reshape(3, 1, 1)


class GrayThermalTransform:
    """Bir YOLOX preproc'unu sarar; CIKTIYI gri (ve istege bagli ters) yapar.

    Zincirin en sonunda durur: MosaicDetection -> TrainTransform -> augment_hsv
    -> flip -> BURASI. Boylece hicbir sonraki adim griligi bozamaz.

    Girdi/cikti sozlesmesi YOLOX'unkiyle ayni: preproc CHW float32, BGR,
    0-255 araliginda goruntu dondurur (normalize etmez).
    """

    def __init__(self, inner, gray_prob, invert_prob=0.0):
        self.inner = inner
        self.gray_prob = gray_prob
        self.invert_prob = invert_prob

    def __call__(self, image, target, input_dim):
        image, target = self.inner(image, target, input_dim)
        if random.random() < self.gray_prob:
            luma = (image * _BGR_LUMA).sum(axis=0, keepdims=True)
            image = np.repeat(luma, 3, axis=0)
            if random.random() < self.invert_prob:
                # Letterbox dolgusu (114) da terslenir -> 141. Kabul edildi:
                # dolgu her iki halde de duz bir alan, ve model her iki
                # degeri de gorerek daha saglam olur.
                image = 255.0 - image
        return np.ascontiguousarray(image, dtype=np.float32), target


class RobustShipDataset(COCODataset):
    """COCODataset + load_image() asamasinda alan-saglamligi augmentasyonu.
    Yalnizca EGITIMDE kullanilir (get_dataset); dogrulama (COCODataset,
    duz) bozulmamis kalir ki AP sayisi augmentasyon sansina bagli olmasin.

    Tek override load_resized_img(); geri kalan COCODataset davranisina
    dokunulmaz.
    """

    _augmentor = None  # ilk ornekte kurulur, sonra tekrar kurulmaz

    def __init__(self, *args, expected_num_classes=None, **kwargs):
        # cache=True COCODataset'in read_img()'ini RAM/disk onbellegine
        # yazar; augmentasyon zincirimiz o cagrinin altinda kaldigi icin ilk
        # epoch'ta DONULUR ve her epoch ayni bozulma gorulur. Reddet.
        if kwargs.get("cache", False):
            raise ValueError(
                "RobustShipDataset ile --cache kullanmayin: augmentasyon "
                "ilk epoch'ta donar, her epoch farkli bozulma gormez."
            )
        super().__init__(*args, **kwargs)
        assert_class_scheme(self.coco, expected_num_classes)

    def load_resized_img(self, index):
        """Augmentasyon KUCULTMEDEN SONRA calisir -- olculerek buraya tasindi.

        load_image()'i sarmak dogal duruyordu ama orada goruntu HAM
        cozunurluktedir. Kaggle'da olculdu (2026-08-11 kosusu, epoch 1-2):
        iter_time 1,33 s'nin 0,99 s'si veri bekleme, yani GPU'nun %73'u bos.
        Zincirin maliyeti piksel sayisiyla dogru orantili:
        1920x1080'de 66 ms, 650x650'de 17 ms, 512x512'de 9 ms -- ve egitim
        setinin %25'i 1920x1080. Mozaik yuzunden ornek basina ~2,5 cagri.

        load_resized_img() ayni goruntuyu 512'ye sigacak sekilde kucultulmus
        halde verir; augmentasyon ayni transformlar ve ayni olasiliklarla
        orada calisinca egitim icerigi degismez, maliyet 2-7 kat duser.
        Yan fayda: RandomSunFlare(src_radius=150) artik gercekten gorunur bir
        parlama uretir; ham 1920px'te uygulanip 512'ye inince ~40px'e
        buzuluyordu.

        Enjeksiyon noktasi olarak read_img() degil bu secildi: read_img
        onbellek dekoratorlu (cache_read_img), oraya yazmak --cache acikken
        augmentasyonu dondururdu. __init__ zaten cache'i reddediyor ama
        savunma tek noktaya yigilmasin.
        """
        img = super().load_resized_img(index)
        cls = type(self)
        if cls._augmentor is None:
            cls._augmentor = _build_robust_augmentor()
        return cls._augmentor(image=img)["image"]


class DPUFocus(nn.Module):
    """YOLOX Focus katmaninin DPU-uyumlu birebir karsiligi.

    (aerial exp'indeki DPUFocus ile birebir ayni; kartta dogrulandi --
     2026-08-10, golden test ~1e-5 px farkla gecti.)
    """

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        from yolox.models.network_blocks import BaseConv

        self.space_to_depth = nn.Conv2d(
            in_channels, in_channels * 4, kernel_size=2, stride=2, bias=False
        )
        w = torch.zeros(in_channels * 4, in_channels, 2, 2)
        for block, (r, c) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
            for ch in range(in_channels):
                w[block * in_channels + ch, ch, r, c] = 1.0
        with torch.no_grad():
            self.space_to_depth.weight.copy_(w)
        self.space_to_depth.weight.requires_grad = False
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        return self.conv(self.space_to_depth(x))


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        # ---- model: YOLOX-Tiny geometrisi (resmi exps/default/yolox_tiny.py) ----
        self.depth = 0.33
        self.width = 0.375
        self.act = "relu"
        self.depthwise = False
        self.num_classes = len(TARGET_CLASSES)

        # ---- girdi boyutu: olculerek secildi, bkz. modul basligi ----
        self.input_size = (512, 512)
        self.test_size = (512, 512)
        # random_resize: 32*s her iki eksende de (kare kanvas), s in [12, 18]
        # -> 384x384 .. 576x576 araligi.
        self.random_size = (12, 18)

        # ---- veri seti ----
        # tools/build_ship_dataset.py ciktisi: annotations/ + images/<kaynak>/<ad>.jpg
        self.data_dir = "datasets/ship_merged"
        self.train_ann = "instances_train.json"
        # Test seti yalnizca EN SONDA, bir kez olculmeli; ayri bir exp
        # dosyasi kopyalamak yerine ortam degiskeniyle secilir:
        #   SHIP_EVAL_ANN=instances_test.json python YOLOX/tools/eval.py ...
        self.val_ann = os.environ.get("SHIP_EVAL_ANN", "instances_val.json")
        # Gri/termal dayanikliligi OLCMEK icin: ayni checkpoint, gri'ye
        # cevrilmis val. AP dususu dayanikliligin sayisal karsiligidir.
        #   SHIP_EVAL_GRAY=1 ...
        self.eval_gray = os.environ.get("SHIP_EVAL_GRAY", "") == "1"
        self.image_folder = "images"
        # 2 iken veri hatti darbogazdi (olculdu: data_time 0,99 s / iter_time
        # 1,33 s). Kaggle GPU imajlari 4 vCPU verir; augmentasyon artik
        # kucultulmus goruntude calistigi icin 4 worker GPU'yu doyurmali.
        self.data_num_workers = 4
        self.dataset = None

        # ---- augmentasyon ----
        # Kutu basi ortalama nesne sayisi dusuk (92920 kutu / 39647 goruntu
        # ~= 2,3), ama kumelenmis sahneler var (bir ir_thermal marina karesi
        # 36 kutu tasiyordu) -- YOLOX varsayilani (50) yetersiz kalabilir,
        # aerial'in asiri degeri (1000/4000) ise burada gereksiz.
        self.mosaic_prob = 0.5
        self.mosaic_scale = (0.5, 1.5)
        self.enable_mixup = False

        # ---- egitim ----
        # Bu veri setinde epoch-vs-AP egrisi HENUZ OLCULMEDI (aerial'da
        # ~epoch 25'te doymustu, ama o farkli bir veri seti/gorevdi).
        # 30 epoch temkinli bir baslangic; ilk egitimde AP@0.50 egrisine
        # bakip gerekirse kisaltin/uzatin -- varsayim degil, olcum.
        self.max_epoch = 30
        # YOLOX varsayilani 5'tir ama o 300 epoch icindir (%1,7). Burada 30
        # epoch var; 5 warmup + 8 no_aug birakinca tam augmentasyonlu pencere
        # 17 epoch'a iniyordu. Ustelik bu bir SIFIRDAN egitim degil, COCO
        # agirliklarindan fine-tune -- katmanlar zaten makul bir noktada,
        # uzun LR rampasina ihtiyac yok. 2 ile pencere 20 epoch'a cikiyor.
        # Ilk epoch'larda loss ziplarsa (onceki kosuda 12,2'den duzgun
        # inmisti) once buraya bakin.
        self.warmup_epochs = 2
        self.no_aug_epochs = 8
        self.eval_interval = 5
        self.print_interval = 50
        self.save_history_ckpt = False  # Kaggle diskini doldurmamak icin

        # ---- kartin calisma noktasi ----
        # Kartta calisan uygulama sabit bir guven esigi kullaniyor
        # (deploy/src/main.cpp:164 -> conf_thr = 0.15f). AP bu esikten
        # bagimsizdir ama saha sorusu ("kac gemiyi kaciriyoruz, kac yanlis
        # alarm") esige baglidir; quantize_yolox.py P/R/F1'i bu degerde
        # raporlar. Buradaki sayi main.cpp'deki ile AYNI kalmali, yoksa
        # olctugumuz calisma noktasi kartinkinden farkli olur.
        self.deploy_conf = 0.15

        # Tekrarlanabilirlik: YOLOX train.py bunu gorurse random/torch
        # seed'lenir ve cudnn.deterministic acilir (YOLOX'un kendi uyarisi:
        # egitimi yavaslatabilir). Seed'siz kosuda ayni kod iki farkli
        # sonuc verir ve "degisiklik ise yaradi mi" sorusu cevaplanamaz.
        # Hiz gerekirse None yapin -- ama o zaman sonuclari kiyaslamayin.
        self.seed = 1337

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_model(self, sublinear=False):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels,
                act=self.act, depthwise=self.depthwise,
            )
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels,
                act=self.act, depthwise=self.depthwise,
            )
            self.model = YOLOX(backbone, head)
            self.model.backbone.backbone.stem = DPUFocus(
                3, int(self.width * 64), ksize=3, act=self.act
            )

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_dataset(self, cache=False, cache_type="ram"):
        from yolox.data import TrainTransform

        return RobustShipDataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name=self.image_folder,
            img_size=self.input_size,
            expected_num_classes=self.num_classes,
            preproc=GrayThermalTransform(
                TrainTransform(max_labels=120, flip_prob=self.flip_prob,
                               hsv_prob=self.hsv_prob),
                gray_prob=TRAIN_GRAY_PROB, invert_prob=TRAIN_INVERT_PROB,
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=None):
        # yolox_base.Exp.get_data_loader kopyasi (aerial exp'teki ayni
        # desen): Mosaic 4 goruntuyu birlestirdigi icin KENDI TrainTransform'unu
        # kurar, self.dataset'e verilen preproc'u KULLANMAZ. max_labels bu
        # yuzden burada ayrica belirtilmeli. Tek nesne yogunlugu dusuk (kutu
        # basi ortalama 2,3) ama en yogun goruntu 36 kutu tasiyordu; 4'lu
        # mozaikte teorik tavan ~144 -- aerial'in 4000'i burada gereksiz,
        # 200 yeterli pay birakiyor.
        import torch.distributed as dist

        from yolox.data import (
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master

        if "dataset" not in self.__dict__ or self.dataset is None:
            with wait_for_the_master():
                assert cache_img is None, (
                    "cache_img must be None if you didn't create self.dataset before launch"
                )
                self.dataset = self.get_dataset(cache=False, cache_type=cache_img)

        self.dataset = MosaicDetection(
            dataset=self.dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=GrayThermalTransform(
                TrainTransform(max_labels=200, flip_prob=self.flip_prob,
                               hsv_prob=self.hsv_prob),
                gray_prob=TRAIN_GRAY_PROB, invert_prob=TRAIN_INVERT_PROB,
            ),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )
        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "batch_sampler": batch_sampler,
            "worker_init_fn": worker_init_reset_seed,
        }
        return DataLoader(self.dataset, **dataloader_kwargs)

    def get_eval_dataset(self, **kwargs):
        from yolox.data import ValTransform

        legacy = kwargs.get("legacy", False)
        dataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name=self.image_folder,
            img_size=self.test_size,
            preproc=(GrayThermalTransform(ValTransform(legacy=legacy), gray_prob=1.0)
                     if self.eval_gray else ValTransform(legacy=legacy)),
        )
        assert_class_scheme(dataset.coco, self.num_classes)
        return dataset

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        # Aerial projedeki VisDroneEvaluator'dan (ignore-region, top-500 ozel
        # mantigi) BILEREK farkli: bu veri setinde ignore_regions yok, standart
        # COCOEvaluator/COCO mAP yeterli. Karta ozel bir calisma noktasi
        # (deploy_conf'ta F1) gerekirse aerial'daki desen buraya tasinir --
        # simdiden eklemek erken soyutlama olur.
        from yolox.evaluators import COCOEvaluator

        return COCOEvaluator(
            dataloader=self.get_eval_loader(
                batch_size, is_distributed, testdev=testdev, legacy=legacy
            ),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
