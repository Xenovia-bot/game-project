#!/usr/bin/env python3
"""YOLOX-Tiny turevi: depthwise conv YOK -> INT8 kuantalamaya cok daha uygun.

Neden bu dosya var (2026-08-09'da olculdu)
------------------------------------------
Nano surumu KV260'in DPUCZDX8G'sinde INT8'e kuantalanirken AP 0.5874 -> 0.3589
dustu (kayip 0.2285, sinir 0.02). Kok neden olculdu: **depthwise conv'larin
per-tensor kuantalanmasi**. BN katlandiktan sonra kanal buyuklukleri arasindaki
oran depthwise katmanlarda medyan 25x, en kotusunde 388x; en zayif kanallar
8 bitin 1-2 seviyesini kullanabiliyor ve INT8'de yok oluyor.

Bu, literaturde tescilli bir basarisizlik kipi (Nagel ve ark., ICCV 2019:
MobileNetV2'de per-tensor INT8 ImageNet dogrulugunu %0.12'ye dusuruyor).
Cross-layer equalization ve bias correction Vitis AI tarafindan **zaten**
uygulaniyor (`bias_corrected: True`) ve yetmedi. AMD'nin kendi YOLOX
tutorial'i bu yuzden INT8+BF16 karma hassasiyet kullaniyor -- ama o yol
DPUCZDX8G'de kapali, cip INT8-only.

**YOLOX-Nano, depthwise=True kullanan tek YOLOX varyantidir; Tiny kullanmaz.**
Bu yuzden Tiny'ye gecmek kok nedeni ortadan kaldirir.

Parametreler YOLOX'un resmi `exps/default/yolox_tiny.py` dosyasindan alindi
(depth 0.33, width 0.375, depthwise yok); girdi boyutu, sinif semasi, DPU
uyarlamalari (ReLU + DPUFocus) ve egitim takvimi bu projeden devralinir.

Bedeli
------
~0.9M -> ~5M parametre. DPU suresi artar; 30 FPS butcesi kartta yeniden
olculmelidir. Baslangic agirligi: Megvii `yolox_tiny.pth`.
"""

import os
import sys
from pathlib import Path

# Nano exp'i ayni klasorden import edebilmek icin: YOLOX'un get_exp_by_file'i
# bunu kendisi yapiyor ama quantize_yolox.py spec_from_file_location kullaniyor
# ve sys.path'e dokunmuyor. Iki yolda da calissin.
for _helper_dir in (Path(__file__).resolve().parent,
                    Path(__file__).resolve().parent.parent):
    if str(_helper_dir) not in sys.path:
        sys.path.insert(0, str(_helper_dir))

from yolox_nano_visdrone import Exp as DpuAwareExp  # noqa: E402


class Exp(DpuAwareExp):
    def __init__(self):
        super().__init__()
        # --- YOLOX-Tiny geometrisi (resmi exps/default/yolox_tiny.py) ---
        self.width = 0.375
        # Kritik fark: Tiny depthwise-ayrilabilir conv kullanmaz.
        self.depthwise = False

        # Girdi boyutu, sinif sayisi, act="relu", DPUFocus, veri yollari ve
        # egitim takvimi ust siniftan gelir -- bilerek degistirilmiyor ki
        # Nano ile Tiny arasindaki tek degisken mimari olsun.
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
