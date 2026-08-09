#!/usr/bin/env python3
"""KV260 DPU'suna (DPUCZDX8G) uyumlu YOLOX-Nano deneyi.

Veri: dort kaynaktan birlestirilmis 2 sinifli havadan arac seti
(bkz. tools/build_dataset.py). Dosya adi tarihsel sebeple korunmustur.

DPU uyumlulugu icin iki degisiklik yapilir:
  1. act="relu": DPU, YOLOX'un varsayilan SiLU aktivasyonunu desteklemez.
  2. Focus stem'i DPUFocus ile degistirilir: Focus'un strided-slice islemi
     DPU'da calismaz; sabit one-hot agirlikli 2x2/stride-2 conv ayni
     space-to-depth dizilimini birebir uretir. Boylece model tek DPU
     subgraph'i olarak derlenir ve onceden egitilmis agirliklar gecerli kalir.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

for _alias, _type in (("float", float), ("int", int), ("bool", bool)):
    if _alias not in np.__dict__:
        setattr(np, _alias, _type)

from yolox.data import COCODataset
from yolox.evaluators import COCOEvaluator
from yolox.exp import Exp as MyExp
from yolox.utils import is_main_process

for _helper_dir in (Path(__file__).resolve().parent,
                    Path(__file__).resolve().parent.parent):
    if str(_helper_dir) not in sys.path:
        sys.path.insert(0, str(_helper_dir))
from visdrone_eval import evaluate_visdrone, format_metrics  # noqa: E402

#: Saha tanimi: havadan bakista kara ve deniz araclari.
#:   land_vehicle - VisDrone car/van/truck/bus + askeri tank/ZPT
#:   sea_vehicle  - VESSELimg container/chemical/ro-ro/tugboat
#: Veri `tools/build_dataset.py` ile dort kaynaktan birlestirilir. Sinif
#: adlari degerlendirmede COCO kategorilerinden okundugu icin tablolar
#: sema degisse de dogru kalir.
TARGET_CLASSES = ("land_vehicle", "sea_vehicle")


def assert_class_scheme(coco, expected):
    """Anotasyon semasi modelin sinif sayisiyla uyusmuyorsa hemen durur.

    Eski bir sema ile uretilmis `instances_*.json` diskte kalirsa model
    sessizce yanlis etiketlerle egitilir ve bu ancak saatler sonra
    degerlendirmede fark edilir. Ucuz bir kapi, pahali bir hatayi onler.
    """
    if expected is None:
        return
    found = sorted(coco.cats)
    if found != list(range(1, expected + 1)):
        names = [coco.cats[c].get("name", c) for c in found]
        raise ValueError(
            f"Anotasyon semasi uyusmuyor: {expected} sinif bekleniyordu, "
            f"{len(found)} bulundu ({names}). Veriyi "
            f"'tools/build_dataset.py' ile yeniden uretin."
        )


class VisDroneTrainingDataset(COCODataset):
    """Egitimde bilincli olarak ignore edilen alanlari negatif ornek yapmaz.

    Donusturucu genel bolgeleri image.ignore_regions, score=0 kutularini ise
    ``iscrowd=1`` olarak saklar. YOLOX crowd kutularini hedeflerden cikartir;
    burada ek olarak ilgili pikseller 114 ile maskelenir.
    """

    def __init__(self, *args, expected_num_classes=None, **kwargs):
        if kwargs.get("cache", False):
            raise ValueError("Ignore maskeleme ile --cache kullanmayin.")
        super().__init__(*args, **kwargs)
        assert_class_scheme(self.coco, expected_num_classes)
        self._collect_ignore_boxes()

    def _collect_ignore_boxes(self):
        self.ignore_boxes = {}
        for image_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=[int(image_id)], iscrowd=True)
            boxes = {
                tuple(int(round(v)) for v in ann["bbox"])
                for ann in self.coco.loadAnns(ann_ids)
            }
            boxes.update(
                tuple(int(round(v)) for v in bbox)
                for bbox in self.coco.imgs[int(image_id)].get("ignore_regions", ())
            )
            self.ignore_boxes[int(image_id)] = sorted(boxes)

    def load_image(self, index):
        img = super().load_image(index)
        height, width = img.shape[:2]
        for x, y, w, h in self.ignore_boxes.get(int(self.ids[index]), ()):
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x + w), min(height, y + h)
            if x2 > x1 and y2 > y1:
                img[y1:y2, x1:x2] = 114
        return img


class VisDroneEvaluator(COCOEvaluator):
    """Resmi DET toolkit eslestirme/VOC AP mantigiyla VisDrone AP@500."""

    def __init__(self, *args, deploy_conf=0.15, **kwargs):
        # COCOEvaluator dagitim esigini bilmez; F1 tablosunun kartla ayni
        # esikte raporlanmasi icin acikca gecilir.
        super().__init__(*args, **kwargs)
        self.deploy_conf = deploy_conf

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None
        if not data_dict:
            return 0, 0, "VisDrone AP@500: hic tespit yok\n"

        coco_gt = self.dataloader.dataset.coco
        # Sinif adlari COCO kategorilerinden okunur: veri semasi degisirse
        # tablo da kendiliginden dogru kalir.
        metrics = evaluate_visdrone(
            coco_gt, data_dict, max_dets=500,
            num_classes=self.num_classes, score_thr=self.deploy_conf,
        )
        ap, ap50 = metrics["ap"], metrics["ap50"]

        inference_time, nms_time, n_samples = (v.item() for v in statistics)
        batch = self.dataloader.batch_size
        info = (
            "VisDrone DET-style evaluation (ignore, global maxDets=500)\n"
            + format_metrics(metrics, "saha") + "\n"
            + f"Average forward = {1000 * inference_time / (n_samples * batch):.2f} ms, "
            + f"NMS = {1000 * nms_time / (n_samples * batch):.2f} ms\n"
        )
        return ap, ap50, info


class DPUFocus(nn.Module):
    """YOLOX Focus katmaninin DPU-uyumlu birebir karsiligi.

    Focus'un dilimleme + birlestirme sirasi (TL, BL, TR, BR) sabit agirlikli
    bir conv ile ayni matematikle uretilir. `conv` alt modul adi korundugu
    icin onceden egitilmis Focus agirliklari dogrudan yuklenebilir.
    """

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        from yolox.models.network_blocks import BaseConv

        self.space_to_depth = nn.Conv2d(
            in_channels, in_channels * 4, kernel_size=2, stride=2, bias=False
        )
        w = torch.zeros(in_channels * 4, in_channels, 2, 2)
        # Focus birlestirme sirasi: TL=(0,0), BL=(1,0), TR=(0,1), BR=(1,1)
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
        # ---- model: YOLOX-Nano geometrisi, DPU-uyumlu aktivasyon ----
        self.depth = 0.33
        self.width = 0.25
        self.act = "relu"
        self.num_classes = len(TARGET_CLASSES)
        # Nano, depthwise-ayrilabilir conv kullanan tek YOLOX varyantidir.
        # Ayri bir oznitelik olarak duruyor ki Tiny turevi (depthwise=False)
        # bu dosyayi devralip yalnizca iki satiri degistirebilsin.
        self.depthwise = True

        # ---- girdi boyutu: kaynak videonun en-boy oranina uydurulmus ----
        # 1920x1080'i 640x640'a letterbox etmek kanvasin %44'unu gri dolguya
        # harcar ve etkin olcek 0.333'te kalir. 896x512 (16:9) ayni kareyi
        # 0.467 olcekle isler: %12 daha fazla hesapla %40 daha yuksek
        # cozunurluk. Kucuk nesne recall'unun ana kaldiraci budur.
        # (h, w) sirasi YOLOX kuralidir.
        self.input_size = (512, 896)
        self.test_size = (512, 896)
        # random_resize yuksekligi 32*s, genisligi 32*int(s*896/512) yapar;
        # s in [14, 18] -> 448x768 .. 576x992 araligi.
        self.random_size = (14, 18)

        # ---- veri seti ----
        # build_dataset.py ciktisi: annotations/ + images/<kaynak>/<ad>.jpg
        # COCO file_name alanlari kaynak alt klasorunu tasidigi icin hem
        # egitim hem dogrulama ayni "images" kokunu kullanir.
        self.data_dir = "datasets/merged"
        self.train_ann = "instances_train.json"
        self.val_ann = "instances_val.json"
        self.image_folder = "images"
        self.data_num_workers = 2
        self.dataset = None

        # ---- augmentasyon (nano varsayilanlari) ----
        self.mosaic_prob = 0.5
        self.mosaic_scale = (0.5, 1.5)
        self.enable_mixup = False

        # ---- egitim ----
        # 10 sinifli calistirmada model epoch ~40'ta doymustu (epoch 45 -> 80
        # arasi AP@0.50 yalnizca +0.006). 40 epoch ayni sonucu yari surede
        # verir. Son 10 epoch mosaic kapali + L1 loss acik: kutu hassasiyetini
        # iyilestirdigi icin (asil darbogazimiz) bu faz korunuyor.
        self.max_epoch = 40
        self.warmup_epochs = 5
        self.no_aug_epochs = 10
        self.eval_interval = 5

        # ---- dagitim calisma noktasi ----
        # Kartta dusuk esikle calisip false positive'leri tracker'in "N karede
        # gorunmeli" kurali ile eleyecegiz (ByteTrack mantigi). Degerlendirme
        # F1'i bu esikte de raporlar.
        self.deploy_conf = 0.15
        self.print_interval = 50
        self.save_history_ckpt = False  # Kaggle diskini doldurmamak icin

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
            # DPU: slice tabanli Focus stem'i esdeger conv surumuyle degistir
            self.model.backbone.backbone.stem = DPUFocus(
                3, int(self.width * 64), ksize=3, act=self.act
            )

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_dataset(self, cache=False, cache_type="ram"):
        from yolox.data import TrainTransform

        return VisDroneTrainingDataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name=self.image_folder,
            img_size=self.input_size,
            expected_num_classes=self.num_classes,
            # VisDrone karelerinde yuzlerce nesne olabilir; varsayilan 50 cok dusuk
            preproc=TrainTransform(
                max_labels=1000, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=None):
        # yolox_base.Exp.get_data_loader kopyasi; tek fark mozaik donusumunde
        # max_labels=4000 (4 yogun VisDrone karesinin mozaigini kirpmaz).
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

        if self.dataset is None:
            with wait_for_the_master():
                assert cache_img is None, (
                    "cache_img must be None if you didn't create self.dataset before launch"
                )
                self.dataset = self.get_dataset(cache=False, cache_type=cache_img)

        self.dataset = MosaicDetection(
            dataset=self.dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=4000, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob
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
        from yolox.data import COCODataset, ValTransform

        legacy = kwargs.get("legacy", False)
        dataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name=self.image_folder,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )
        assert_class_scheme(dataset.coco, self.num_classes)
        return dataset

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        return VisDroneEvaluator(
            dataloader=self.get_eval_loader(
                batch_size, is_distributed, testdev=testdev, legacy=legacy
            ),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
            deploy_conf=self.deploy_conf,
        )
