# Claude Handoff — KV260 YOLOX-Nano · Havadan Araç Tespiti

**Bu dosyayı yeni oturumda ilk okuyun.** Ardından `README.md`.
Workspace: `C:\Users\emrez\proje` · Dil: **Türkçe** · Git: yerel repo var, uzak yok.

---

## 1. Hedef (KİLİTLİ — yeniden açmayın)

KV260 kartında video dosyasından **2 sınıf** tespiti ve her tespitin merkez
noktası:

| Sınıf | Kapsam |
| --- | --- |
| `land_vehicle` | VisDrone car/van/truck/bus + askeri tank/ZPT |
| `sea_vehicle` | container, chemical tanker, Ro-Ro, tugboat |

`cx = (x1+x2)/2`, `cy = (y1+y2)/2` → `centers.csv` + `out.avi` + FPS özeti.

Çalışma noktası: **896×512 girdi, 40 epoch, conf 0.15, IoU tabanlı takip, ≥30 FPS.**

> Taksonomi bu noktaya gelene kadar **beş kez** değişti (10 sınıf → 2 → 4 → 3
> → tek sınıf ship → 2 sınıf). Kullanıcı her turda fikir değiştirdi ve hiç
> eğitim yapılmadı. **Yeni bir taksonomi önerisi gelirse önce "önce bir kez
> uçtan uca çalıştıralım" deyin.**

## 0. ŞU AN NE OLUYOR (2026-08-09 akşamı)

**İki iş paralel koşuyor:**

| nerede | ne | durum |
| --- | --- | --- |
| VM docker | Nano · PTQ merdiveninin tamamı | ✅ **bitti — kapı geçilmedi, kayıp 0.2306** |
| Kaggle | **Tiny** eğitimi (`VARIANT = "tiny"`, 40 epoch) | 🔵 koşuyor, ~5 sa |

**Nano kapandı.** CLE + bias correction + AdaQuant hepsi uygulandı, AdaQuant
hiç kazandırmadı (§5). Sorun kalibrasyonda değil, depthwise conv'ların
per-tensor 8 bitte temsil edilemiyor olmasında.

**Sıradaki:** Tiny eğitimi bitince aynı VM adımlarını Tiny için tekrarla —
`--inspect` → float AP → calib → test. Depthwise olmadığı için kaybın çok
daha düşük olması bekleniyor. Geçerse derle ve karta gönder.

**Tiny de geçmezse** (ya da FPS bütçesine sığmazsa) bilinçli karar: Nano'nun
INT8'ini 0.23 kayıpla kabul edip raporda gerekçesiyle yazmak. Ölçüm zaten
elde — bu bir eksiklik değil, belgelenmiş bir mühendislik sınırı.

**Kod tarafında bekleyen iş yok** — dağıtım zinciri 9 tensörlü çıktıya taşındı
(`main.cpp`, `compile_kv260.sh`, `verify_kv260_golden.py`), 74 test geçiyor.

## 2. Zincir ve nerede olduğumuz

| Aşama | Durum |
| --- | --- |
| Veri birleştirme | ✅ **bitti, doğrulandı** (19.476 train / 4.483 val) |
| Kaggle eğitimi (2 sınıf) | ✅ **bitti** 2026-08-09 · 40/40 epoch · **4,9 saat** (T4) · AP@0.50 = **0.8659** |
| Oracle VM · Vitis AI 3.0 PTQ | 🔵 **başladı** — `--inspect` ✅ geçti (aşağı bkz.), sırada float AP |
| KV260 · C++ VART + golden test | ⬜ — **kart kurulumu ZATEN HAZIR** (aşağı bkz.) |
| Rapor | ⬜ |

## 3. Veri seti (tamamlandı)

Dört kaynak `tools/build_dataset.py` ile tek 2 sınıflı COCO'ya birleşiyor.
Zip'ler `datasets/` altında; **Kaggle'a `aerial-vehicle-sources` adıyla
yüklendi.** Yerel çıktı `datasets/merged/` (gitignore'da).

```
kaynak        goruntu   train    val     land     sea  ignore
mendeley         7985    6469   1516     6364       0       0
milrec           2863    2149    714     3085       0       0
vesselimg        6092    4387   1705        0   11909     397
visdrone         7019    6471    548   205663       0       0
-------------------------------------------------------------
train        19476 goruntu / 205340 kutu   (land 196490, sea 8850, ignore 368)
val           4483 goruntu /  21681 kutu   (land  18622, sea 3059, ignore  29)
```

### Sınıf kararları (gerekçeli, değiştirmeyin)

Ölçüt: [arXiv 2407.00018](https://arxiv.org/abs/2407.00018) — **görsel olarak
benzer** sınıfları birleştirmek mAP'yi yükseltir, **farklı** olanları
birleştirmek düşürür (kanguru türleri 0.968'e çıktı; domuz+keçi 0.897/0.781 →
0.699).

- **Tutulan → land:** car, van, truck, bus, tank (×2 kaynak), APC
- **Tutulan → sea:** container, chemical, passenger-roro, tugboat
- **Atılan (arka plan olur, bu istenen):** insan/asker, bisiklet, motosiklet,
  tricycle, awning-tricycle, hava aracı, drone, **Buoy** (kasten — yukarıdan
  tekneye benzer, hard negative)
- **Ignore (ne hedef ne arka plan):** `Pilot` botu (medyan 15 px), VisDrone
  ignored-region ve score=0 crowd kutuları

### Bölme

- VisDrone: kendi resmi train/val bölmesi
- Diğer üçü: Roboflow bölmesi **atıldı**, oturum bazlı yeniden bölündü
  (`ROBOFLOW_VAL_FRACTION = 0.25`, `SPLIT_SEED = 1337`)
- **Test bölümü yok** — gerçek test kartta. İstenirse eklenebilir.

## 4. ⚠️ Bilinen zayıflıklar (kullanıcıya söylendi)

| Sorun | Ayrıntı |
| --- | --- |
| 🔴 **`sea_vehicle` tek kameradan** | **2026-08-09'da ölçüldü.** "23 oturum" aslında **3 çekim günü** (2023-06-27, 10-13, 10-18), hepsi Valencia Limanı'nda **tek kamera kurulumundan**. Val de aynı kameranın aynı günlerinden. **Sızıntı YOK** — kontrol edildi: val karelerinin yalnızca %1,2'sinin train'de RMS<10/255 yakın komşusu var, oturum bazlı bölme işini yapmış. Sorun sızıntı değil: val skoru "deniz aracı tanıyor mu"yu değil **"bu limanı ezberledi mi"**yi ölçüyor. `sea_vehicle` AP'sini rapora **bu uyarıyla** yazın. Tek gerçek çözüm ikinci bir denizcilik kaynağı. |
| 🟡 Dengesizlik 22:1 | Müdahale edilmedi. `--repeat vesselimg=3` / `--subsample visdrone=0.5` düğmeleri **kapalı**; 2026-08-09'da onarıldı ve testlendi (önceden `--repeat` doğrulama kapısını patlatıyordu) — önce ölçülecek |
| 🟡 Dört farklı alan | VisDrone şehir / VESSELimg liman / milrec savaş / mendeley karışık. 0.9M model için zor |
| 🔴 **Mendeley'in çoğu konu dışı** | **2026-08-09'da ölçüldü ve gözle doğrulandı.** 7.985 görüntünün yalnızca **1.799'u (%23)** gerçek zaman damgalı drone çekimi; %62'si boş. İçinde **112 Fortnite ekran görüntüsü** (gri tonlamaya çevrilmiş, 179 "tank" kutusu var), **masa üstünde plastik oyuncak tank fotoğrafları**, getty/istock stok fotoğrafları, YouTube küçük resimleri var. Görüntülerin %33'ü mendeley'den ama kutuların yalnızca **%3'ü**. Temizlik dosya adı kalıbıyla yapılabilir (`Fortnite`, `gettyimages`, `istockphoto`, `maxresdefault`, `screenshot`, `VID_`, `IMG_`, salt-numara). |
| 🔴 **İki sınıf hiç birlikte görünmüyor** | Her iki sınıfı birden içeren görüntü: **train 0, val 0**. Model "su → sea, şehir → land" sahne kısayolunu öğrenebilir; kıyı şeridi üzerinden uçan gerçek videoda bu kırılır. Kartta ilk bakılacak şeylerden biri. |
| 🟡 İki ayrı boyut rejimi | 896×512'ye ölçekten sonra medyan kutu: visdrone **18 px** (%44'ü <16 px), mendeley 52, milrec 64, vesselimg 68 px. `sea` kolay, `land` zor görünecek — sınıf AP farkı semantikten değil çözünürlükten gelebilir. |
| 🟡 Kaynak etiket kalitesi | Dönüşümün sadık olduğu kanıtlandı, **orijinal etiketlerin doğruluğu değil** |
| 🟡 Boş kare oranı | train %24, val %36 (tipik %10). Mendeley kaynaklı — asker/insan atılınca kare boşalıyor |

## 5. Ölçülmüş gerçekler (yeniden türetmeyin)

### 🎯 HEDEF ÇALIŞTIRMA — 2 sınıf, 896×512, 40 epoch, T4, 2026-08-09

Birleşik val setinde (4.483 görüntü), `eval.py --conf 0.001`, resmi ignore
filtresi + global top-500 + VOC AP:

| | AP@[.50:.95] | AP@0.50 | AP@0.75 | En iyi F1 |
| --- | --- | --- | --- | --- |
| **saha (ortalama)** | **0.5874** | **0.8659** | 0.6578 | 0.8274 @ conf 0.46 |
| `land_vehicle` | 0.4796 | 0.7807 | — | 0.7547 |
| `sea_vehicle` | 0.6952 | 0.9512 | — | 0.9012 |

Dağıtım çalışma noktası `conf=0.15`: **F1 = 0.7669, P = 0.6911, R = 0.8685.**
T4'te forward 3,74 ms + NMS 1,29 ms. Megvii başlangıcı %98,9 eşleşti.

AP@0.50 ilerlemesi: e5 0.747 · e10 0.839 · e15 0.844 · e20 0.854 · e25 0.859 ·
e30 0.860 · e40 0.861. **Model epoch ~25'te doydu** (25→40 arası +0.002).
40 epoch fazlasıyla yeterli; tekrar eğitim gerekirse **30 epoch** aynı sonucu
verir ve ~1,2 saat kazandırır.

> ⚠️ **Süre:** 40 epoch **1,5-2 saat değil, 4,9 saat** sürdü. Kaggle GPU
> kotasını buna göre planlayın.

> ⚠️ **Bu sayıları yayınlanmış VisDrone sonuçlarıyla KARŞILAŞTIRMAYIN.**
> Üç sebep: (1) tek birleşik araç sınıfı, 10 sınıflı ortalamadan çok daha
> kolay — kutuların %77'si `car`; (2) val setinin yalnızca 548 görüntüsü
> VisDrone; (3) `sea_vehicle` 0.95'i tek limanın ezberlenmesi
> (bkz. §4). `land_vehicle` 0.78 nispeten gerçek: val land kutularının
> %91'i VisDrone'dan. Raporda bu üç uyarı mutlaka yer almalı.

### ✅ DPU UYUMLULUK KAPISI GEÇTİ — 2026-08-09, Vitis AI 3.0 Inspector

`python quantize_yolox.py --inspect`, hedef `DPUCZDX8G_ISA1_B4096`:

```
[VAIQ_NOTE]: All the operators are assigned to the DPU
```

Doğrulanan üç iddia:

1. **DPUFocus çalışıyor.** `DPUFocus[stem]/Conv2d[space_to_depth]` →
   `nndct_conv2d`, çıktı `[1, 256, 448, 12]` — **DPU'ya atandı**. Orijinal
   Focus'un strided-slice'ı atanmazdı; projenin temel tasarım kararı buydu.
2. **Üç baş çıktısı da 7 kanallı** (`5 + 2 sınıf`): `[1,64,112,7]`,
   `[1,32,56,7]`, `[1,16,28,7]` → 512/64=8, 512/32=16, 512/16=32, yani
   **stride 8/16/32 doğru**. Kare olmayan 896×512 girdi sorun çıkarmadı.
3. **Checkpoint torch 1.12'de açıldı**: `Checkpoint tam eslesti: 643/643
   katman`. Kaggle torch 2.10 ile kaydetmişti; biçim uyumu tutmuş.

CPU'ya düşen tek bir op yok — maxpool, resize (upsample), concat, elemwise_add
dahil hepsi DPU'da. Ayrıntı: `build/quant/inspect/inspect_DPUCZDX8G_ISA1_B4096.txt`

### ✅ FLOAT AP KAPISI GEÇTİ — VM, aynı gün

`python quantize_yolox.py --quant-mode float` (4.483 val görüntüsü, CPU, ~15 dk):

| | Kaggle (T4, YOLOX evaluator) | VM (CPU, bağımsız decode) |
| --- | --- | --- |
| AP@[.50:.95] | 0.5874 | **0.5874** |
| AP@0.50 | 0.8659 | **0.8659** |
| AP@0.75 | 0.6578 | 0.6577 (yuvarlama) |

**Neden önemli:** `quantize_yolox.py` kendi letterbox/decode/NMS'ini kullanır —
YOLOX'un `COCOEvaluator` yolundan tamamen bağımsız bir uygulamadır ve bugüne
kadar uçtan uca doğrulanmamıştı. İki bağımsız yolun dört basamak aynı sayıyı
vermesi, hem taşınan dosyaların hem de kart tarafında kullanılacak decode
mantığının doğru olduğunu gösterir.

**Hız:** 0,12 sn/görüntü (CPU). Tam val ~9 dk çıkarım + AP hesabı.

> ⚠️ `--subset-len` **baştan almaz, eşit aralıklarla yayar** (2026-08-09'da
> düzeltildi). Baştan alma `mendeley`'in boş karelerine denk gelip AP=0
> veriyordu; bu bir boru hattı hatası değil, temsili olmayan örneklemdi.

### 🔴 INT8 KABUL KAPISI REDDETTİ — kök neden bulundu (2026-08-09)

```
float : AP 0.5874  AP50 0.8659  AP75 0.6578
INT8  : AP 0.2987  AP50 0.6126  AP75 0.2526
kayip : 0.2887  (sinir 0.02 -- 14 kati)
```

Kayıp konumlandırmaya yığılmış: AP50 %71 korunmuş, **AP75 yalnızca %38**.
Model nesneyi buluyor ama kutuyu kaba yerleştiriyor.

**Kök neden — ölçüldü (`vm_package/tani_int8.py`):**

`DeployModel` reg(4)+obj(1)+cls(2) kanallarını `torch.cat` ile tek 7 kanallı
tensöre birleştiriyor. Vitis AI'da concat donanımda sadece bellek düzenidir ve
**tüm girdilerin aynı `fix_point`'i paylaşmasını zorunlu kılar.** Ölçülen
float aralıklar:

| seviye | tensör genişliği | ortak adım | `reg_x` seviye sayısı | merkez hatası | boyut hatası |
| --- | --- | --- | --- | --- | --- |
| stride 8 | ±76.0 | 1.0 | **4** / 256 | **8 px** | %65 |
| stride 16 | ±37.9 | 0.5 | 8 | 8 px | %28 |
| stride 32 | ±22.8 | 0.25 | 17 | 8 px | %13 |

Ölçeği belirleyen **`obj` kanalı**: stride 8'de min `-76.0`. sigmoid(-76)=1e-33,
yani tespit için sıfır bilgi taşıyor — ama reg kanallarının hassasiyetini
tek başına yok ediyor. Medyan kutu 18 px iken 8 px merkez hatası IoU 0.75'i
imkânsız kılar.

**Düzeltme 1 — concat kaldırıldı** (`DeployModel` 9 ayrı tensör döndürüyor).
Ölçüldü: **yardımcı oldu ama yetmedi.**

| | concat'li | ayrı tensör | float |
| --- | --- | --- | --- |
| AP | 0.2987 | 0.3589 | 0.5874 |
| AP50 | 0.6126 | 0.6703 | 0.8659 |
| AP75 | 0.2526 | **0.3398** | 0.6578 |
| kayıp | 0.2887 | **0.2285** | — |

AP75'in orantısız iyileşmesi (%34 göreli) teşhisi doğruluyor, ama açığın
yalnızca %21'i kapandı. Kalan kayıp AP50'de de olduğu için sorun baş
çıktısında değil **özellik katmanlarında.**

**Kök neden 2 — ağırlık dağılımı (yerelde ölçüldü, `bn_fold_analysis`):**

BN katlandıktan sonra (`W·γ/√(var+ε)`) kanal büyüklükleri arasındaki oran:

| grup | medyan oran | en zayıf kanalın seviye sayısı (medyan / p10 / min) |
| --- | --- | --- |
| depthwise (30 adet) | 25× | 5.1 / 2.05 / **0.33** |
| normal (74 adet) | 7.5× | 17.0 / 4.60 / **0.10** |

En kötü katman **`backbone.backbone.stem.conv`** — DPUFocus'un BaseConv'u.
Oran **1327×**, en zayıf kanal 8 bitin **0.10 seviyesini** kullanabiliyor,
yani o kanal INT8'de tamamen yok oluyor. Ardından `dark2.0.dconv` (388×).
104 katmanın 4'ünde en zayıf kanal 2 seviyenin altında; 3'ü depthwise.

Bu, literatürde iyi bilinen bir başarısızlık kipi: derinlemesine-ayrılabilir
ağlarda **per-tensor** INT8 PTQ çöker. Nagel ve ark. (ICCV 2019) MobileNetV2'de
naif per-tensor ile ImageNet doğruluğunun **%0.12**'ye düştüğünü, cross-layer
equalization + bias correction ile **%70.92**'ye çıktığını ölçmüş.

**Merdivenin neresindeyiz:**

| çare | durum |
| --- | --- |
| Cross-layer equalization | ✅ Vitis AI zaten yapıyor (`=>Doing weights equalization...`) |
| Bias correction | ❓ `quant_info.json` içinde `bias_corrected` alanı var, değeri kontrol edilmeli (`--nndct_param_corr`) |
| AdaQuant (`--fast-finetune`) | ✅ **denendi — HİÇ KAZANDIRMADI** (aşağı bkz.) |
| QAT | ⚠️ GPU'lu Vitis AI ortamı ister; mevcut kurulum CPU-only, pratikte erişilemez |
| **YOLOX-Tiny'ye geçmek** | ⬜ **kalan tek yol** — kök nedeni ortadan kaldırır |

### 🔴 PTQ MERDIVENI TÜKENDİ — Nano bu donanımda kuantalanamıyor

`--fast-finetune --ft-images 32`, kalibrasyon 2 sa 14 dk, test 31 dk:

| | AP | AP50 | AP75 | kayıp |
| --- | --- | --- | --- | --- |
| float | 0.5874 | 0.8659 | 0.6578 | — |
| INT8, AdaQuant öncesi | 0.3589 | 0.6703 | 0.3398 | 0.2285 |
| INT8, AdaQuant sonrası | 0.3568 | 0.6639 | 0.3423 | **0.2306** |

**AdaQuant hiçbir şey kazandırmadı** (fark gürültü seviyesinde, hatta hafif
negatif). Üç çare de uygulanmış durumda: CLE ✅, bias correction ✅, AdaQuant ✅.

**Yorum:** AdaQuant'ın işi ağırlıkları oynatarak kuantalama hatasını telafi
etmektir. Hiç kazandırmaması, darboğazın ağırlıkların yerinde değil **8 bitin
kendisinde** olduğunu gösterir. Depthwise conv'larda kanal büyüklük oranı
25-388× ve DPUCZDX8G aktivasyonlarda **per-tensor** ölçek kullanır. Nagel ve
ark.'nın MobileNetV2'yi kurtaran çözümü **per-channel** kuantalamaydı; bu
donanımda mevcut değil.

**Sonuç: kalibrasyonla düzeltilebilir bir sorun değil, mimari bir sınır.**

### 🎯 AMD'NİN KENDİ YOLOX-NANO'SU AYNI KAYBI VERİYOR

Vitis AI 3.0 Model Zoo'da **`pt_yolox-nano_coco_416_416_1G_3.0`** var ve
`zcu102 & zcu104 & kv260` için önceden derlenmiş xmodel içeriyor — yani tam
bizim hedefimiz. Paketin `README.md`'sindeki resmi tablo:

| Metric | Float | Quantized (PTQ) | **QAT** |
| --- | --- | --- | --- |
| AP0.50:0.95 | 0.220 | **0.136** | **0.210** |

| | float → PTQ | göreli kayıp |
| --- | --- | --- |
| AMD (COCO, 416×416) | 0.220 → 0.136 | **%38,2** |
| Bu proje (havadan, 896×512) | 0.5874 → 0.3568 | **%39,3** |

**Bizim sonucumuz hatalı değil — vendor'ın yayınladığı sayıyla birebir aynı
bantta.** YOLOX-Nano'nun DPUCZDX8G'de PTQ ile ~%38-39 kaybetmesi beklenen
davranış.

**AMD'nin çözümü Tiny'ye geçmek değil, QAT olmuş:** 0.136 → 0.210, yani
float'ın %95'i. `code/run_qat.sh` 8 GPU + batch 128 kullanıyor; exp'leri
`code/exps/example/custom/yolox_nano_deploy_relu_{q,qat}.py`. Paket ayrıca
QAT ağırlıklarını (`qat/qat.pth`) ve PTQ sonucunu içeriyor — ama **COCO
80 sınıf**, doğrudan kullanılamaz.

**QAT bu projede yapılabilir mi? Donanım var (2026-08-09'da kontrol edildi):**

| gereksinim | durum |
| --- | --- |
| NVIDIA GPU | ✅ RTX 4050 Laptop, 6 GB (Windows host) |
| Docker Desktop + WSL2 | ✅ zaten kurulu |
| Disk | ⚠️ C:'de 56,8 GB boş; Vitis AI GPU imajı büyük |
| VRAM kısıtı | 6 GB → 896×512'de batch ~4-6 (AMD 8 GPU'da 128 kullandı) |

**Sıralama:** önce Tiny (koşuyor, bedava), yetmezse QAT (vendor'ın kanıtlanmış
yolu). Vitis AI 5.x **kullanılamaz**: kart imajı 3.0, 5.x ile derlenen xmodel
yüklenmez; ayrıca 5.x Versal/NPU hedefliyor, DPUCZDX8G'yi değil.

**Literatür ve vendor taraması (2026-08-09):**

- **AMD'nin kendi YOLOX tutorial'ı karma hassasiyet kullanıyor** (Vitis AI 5.1,
  Versal **VEK280**, INT8 + **BF16**). Yani saf INT8'in YOLOX'ta yetersiz
  kaldığı vendor tarafından da kabul edilmiş. **Bizim DPUCZDX8G INT8-only**,
  o yol kapalı. → https://github.com/Xilinx/Vitis-AI-Tutorials
- **Nagel ve ark., ICCV 2019** — derinlemesine-ayrılabilir ağlarda per-tensor
  INT8 PTQ çöküyor (MobileNetV2 %0.12). Çare merdiveni: CLE → bias correction
  → AdaRound/AdaQuant → QAT.
- **YOLO11**, kuantalama sonrası doğruluğu iyileştirmek için **depthwise conv
  kullanımını azaltmış** (arXiv 2510.09653). Mimari düzeyde aynı teşhis.
- **LogicTronix'in KV260 + Vitis AI 3.0 YOLOv5 referansı** (topluluğun en yakın
  örneği) **saf PTQ yapıyor: `fast_finetune` yok, mAP ölçümü de yok** — sadece
  derlenmesini sağlıyor. Doğruluğu ölçen bir referans bulunamadı.
  → https://github.com/LogicTronix/Vitis-AI-Reference-Tutorials

**Tiny varyantı yerelde doğrulandı (2026-08-09, Kaggle'a gitmeden):**

| | Nano | Tiny |
| --- | --- | --- |
| parametre | 897.093 | **5.033.301** (5,6×) |
| **depthwise katman anahtarı** | **180** | **0** |
| width / depthwise | 0.25 / True | 0.375 / False |
| girdi · sınıf · epoch | 512×896 · 2 · 40 | aynı (devralınıyor) |
| çıktı kanalı | 7 ✓ | 7 ✓ |
| stem | DPUFocus ✓ | DPUFocus ✓ |
| Megvii eşleşme | %98,9 | %98,5 |
| eksik anahtar | 7 (`cls_preds` + `space_to_depth`) | aynı |
| notebook hücre 8 kapısı | GEÇER | **GEÇER** |

Doğrulama yöntemi: YOLOX pinlenmiş commit'ten yerele klonlanıp iki exp de
inşa edildi, Megvii `yolox_nano.pth` / `yolox_tiny.pth` indirilip hücre 8'in
eşleşme kapısı birebir çalıştırıldı. **Kaggle'da bu adımları izlemeye gerek
yok.**

**Kritik çıkarım — YOLOX-Tiny:** kaybın kaynağı depthwise conv'ların per-tensor
kuantalanması. **YOLOX-Nano `depthwise=True` kullanan tek YOLOX varyantı;
YOLOX-Tiny `depthwise=False`.** Yani Tiny'ye geçmek kök nedeni ortadan
kaldırır. Bedeli: ~0.9M → ~5M parametre (DPU süresi artar, 30 FPS bütçesi
yeniden ölçülmeli) ve Kaggle'da yeniden eğitim (~5 saat). AdaQuant yetmezse
**doğru hamle budur** — QAT değil.

**Referans çalıştırma** — 10 sınıf, 640×640, 80 epoch, T4, 2026-08-07
(farklı görev ve farklı veri; yalnızca bağlam için):

| Epoch | AP@[.50:.95] | AP@0.50 | AP@0.75 |
| --- | --- | --- | --- |
| 10 | 0.0644 | 0.1376 | 0.0558 |
| 45 | 0.1056 | 0.2144 | 0.0932 |
| 80 | 0.1087 | 0.2205 | 0.0965 |

- Model **epoch ~40'ta doydu** (45→80 arası +0.006) → `max_epoch = 40`
- **AP75/AP50 = 0.44** → darboğaz sınıflandırma değil **çözünürlük**
- Eğitim hızı: T4'te batch 16, 640², `iter_time ≈ 0.42 s`, `data_time ≈ 0.05 s`
  (veri yükleme darboğaz **değil**; SimOTA baskın)
- **YOLOX'un `ETA` alanı kümülatif ortalama kullanır** — ilk yavaş
  iterasyonları taşır, başta 3-5 kat şişik görünür. Gerçek hız için logdaki
  `iter_time` (son 50 iterasyonun ortalaması) kullanın.

**Girdi boyutu** — birleşik veri üzerinde ölçüldü (varsayım değil):

| Girdi | Hesap | Ort. ölçek | Kutu medyanı |
| --- | --- | --- | --- |
| 640×640 | 1.00× | 0.473 | 16.0 px |
| **896×512** | **1.12×** | **0.548** | **19.7 px** |
| 960×544 | 1.27× | 0.584 | 21.0 px |

Görüntülerin %59'u kare olmasına rağmen 896×512 kazanıyor: kutuların %92'si
VisDrone'dan (16:9 / 4:3) geliyor.

**Bağlam:** VisDrone'da yayınlanmış YOLOv11 sonucu F1 = 0.347; yarışma rekoru
AP@0.50 = 0.653 (ensemble + 1536px + tiling). Kullanıcı bir ara "%80 doğruluk"
istedi — VisDrone'da bu **kimsede yok**; yeni 2 sınıflı kapsamda daha
gerçekçi. Kare bazında değil **iz bazında** ölçülmeli.

## 6. Kod haritası

```
tools/build_dataset.py        4 kaynak -> tek 2 sinifli COCO. Sinif eslemesi,
                              oturum bazli bolme, SERT dogrulama kapilari.
                              --source (Kaggle yollari), --images-out,
                              --repeat / --subsample (varsayilan kapali)
tools/audit_dataset.py        BAGIMSIZ denetleyici: 10 kontrol, kaynakla
                              birebir karsilastirma dahil
tools/shorten_zip_names.py    Kaggle 248 bayt sinirini asan adlari kisaltir
tools/prepare_kaggle_upload.py  Yukleme klasorunu hazirlar
tools/verify_kv260_golden.py  Python <-> C++ esdegerlik testi
tools/_sync_notebook_embeds.py  Notebook %%writefile hucrelerini kaynakla esitler

training/visdrone_eval.py     Resmi VisDrone AP@500 (calcAccuracy.m ile birebir)
                              + P/R/F1 + sinif gruplama senaryolari
training/exps/yolox_nano_visdrone.py
                              2 sinif, 896x512, 40 epoch, ReLU + DPUFocus
training/kaggle_visdrone_yolox.ipynb   15 hucre (asagida)
build/make_notebook.py        Notebook'u ureten script (elle duzenlemeyin)

quantize/quantize_yolox.py    PTQ / INT8 AP / xmodel export / DPU inspector
quantize/compile_kv260.sh     Derleme + subgraph dogrulama (kanal = 5 + sinif)

deploy/src/main.cpp           VART demo: decode -> NMS -> takip -> merkez -> CSV
deploy/src/tracker.hpp        IoU + ByteTrack 2 asamali eslestirme, sabit hiz.
                              VART/OpenCV BAGIMSIZ -> kart disinda test edilir
deploy/tests/test_tracker.cpp 10 C++ testi

tests/                        69 test. Calistirma:
                              python -m unittest discover -s tests
```

`build/make_notebook.py` **git'te takipli** (`.gitignore` icinde `build/*` +
`!build/make_notebook.py` istisnasi). Onceden `build/` tumden yok sayiliyordu
ve notebook'un tek ureticisi surum kontrolu disindaydi.

**Notebook hücreleri:** 0 md · 1 kurulum · 2 YOLOX · **3-4-5 `%%writefile`**
(build_dataset / visdrone_eval / exp — `_sync_notebook_embeds.py` doldurur) ·
**6 veri keşfi + birleştirme** · 7 Megvii init · 8 md · **9 eğitim** ·
10 resume · 11 değerlendirme · 12 görsel kontrol · 13 paketleme · 14 md

Notebook'u **elle düzenlemeyin**: `python build/make_notebook.py` sonra
`python tools/_sync_notebook_embeds.py`. Bir test bu senkronu zorunlu kılar.

## 7. 🪤 Sert öğrenilmiş tuzaklar

**Kaggle ortamı**

1. `pip install -e` Kaggle'da **başarısız olur** → `--no-build-isolation` +
   editable olmayan yedek + `site.addsitedir` ile `sys.path` tazeleme. Üçü de
   gerekli, hücre 2'de var.
2. Kaggle yüklenen zip'i **`<klasör>/<zip adı>/...` diye açar.** Keşif bu
   yüzden klasör adına değil **işaret dosyasına** dayanır ve kökü ondan
   **yukarı yürüyerek** bulur. Aksi halde üst klasör kaynak sanılır.
3. Kaggle **248 baytı aşan arşiv girişini reddeder.**
   `shorten_zip_names.py` düzeltir — kısaltma **taban ad bazında** yapılmalı,
   yoksa görüntü/etiket eşleşmesi kırılır.
4. Kaggle CLI: `kaggle.json` **artık hesaba bağlı hiçbir işlem için yetmiyor**
   (2026-08-09'da ölçüldü — eski not "okuma için yeter" diyordu, **yanlış**).
   Anonim/genel okumalar çalışıyor (`datasets list --user ...`), ama kimliğe
   bağlı komutlar — `kernels list -m`, `kernels files`, `kernels output` —
   `Authentication required` veriyor. Çözüm: `kaggle auth login` (tarayıcı
   OAuth, token kopyalamak yok) ya da `KAGGLE_API_TOKEN` /
   `~/.kaggle/access_token`. **Bu adımı kullanıcı kendisi yapar; token/anahtar
   işlemeyin.** Ayrıca Kaggle'ın indirdiği log dosyasının adı (`notebook…`)
   URL slug'ı ile aynı olmayabilir; doğru slug `kaggle kernels list -m`
   çıktısında veya `kaggle.com/code/<kullanici>/<slug>` adresinde.
5. PowerShell 5.1 `Set-Content -Encoding utf8` **BOM ekler**; `kaggle.json`
   BOM'lu olursa "Missing username" hatası verir.
6. Kaggle kullanıcı adı: **`burakzorgeen`** (e-postadaki gibi `burakzorgecen`
   değil).

**Vitis AI docker (2026-08-09'da ölçüldü)**

6b. Ortam: **Python 3.7.12**, torch **1.12.1**, numpy **1.21.6**, cv2 4.7.0,
    `pytorch_nndct` çalışıyor. Conda ortamları: `base`,
    `vitis-ai-pytorch` ← **bunu aktive edin**, `vitis-ai-wego-torch`
    (varsayılan aktif olan bu, yanlış olan da bu).
6c. **YOLOX pinlenmiş commit'i Python 3.7'de import edilemiyor.**
    `yolox/utils/mlflow_logger.py` → `import importlib.metadata` (3.8+).
    Çözüm, `yolox/utils/__init__.py:14`'teki satırı yorum yapmak:
    ```
    sed -i 's/^from \.mlflow_logger import MlflowLogger/#&/' YOLOX/yolox/utils/__init__.py
    ```
    Güvenli: `MlflowLogger` yalnızca `yolox/core/trainer.py`'de kullanılıyor,
    kuantalama trainer'ı hiç çağırmıyor. `verify_yolox_version` git commit'ine
    baktığı için kimlik kontrolü de bozulmuyor.
6d. Konteyneri `--rm` **olmadan** açın (`docker run -it --name vai ...`),
    yoksa YOLOX kurulumu her çıkışta silinir. Geri giriş: `docker start -ai vai`.
    Kurulum: `pip install --no-deps --no-build-isolation -e ./YOLOX`
    (`--no-deps` şart: yoksa pip torch'u yükseltip ortamı bozar).

**Veri**

7. **Üç Roboflow kaynağının da bölmesi sızdırıyordu** — kare/augment kopyası
   bazında bölünmüşler. Oturum bazlı yeniden bölme şart.
8. **Ultralytics'e çevrilmiş VisDrone kopyaları kullanılamaz** —
   ignored-region ve score=0 crowd bilgisi silinmiş, `annotations/` yerine
   `labels/` var.
9. **`tank` kelimesi geçen her sınıf askeri tank değil**: xView `Tank Car` =
   demiryolu vagonu, `Storage Tank` = yakıt deposu.
10. milrec ve mendeley görüntülerini 1, 2, 3 diye numaralamış — **aynı adlı
    görüntüler farklı** (256 bit hash farkı 91-152). Grup anahtarı kaynak
    önekli olmalı.

**Kod**

11. `argparse` `nargs="*"` ile **tekrarlanan bayraklar birbirini ezer** →
    `action="append"`. (`--source`, `--repeat`, `--subsample` üçü de düzeltildi.)
12. VisDrone train ve val **aynı kaynak adını paylaşır ama farklı zip'lerdedir**
    — arşivi kaynak adına göre eşlemek val görüntülerini train zip'inde aratır.
13. Yol eşleştirmesi `"/images/" in name` ile yapılmamalı; arşiv kökü bir
    seviye aşağıdaysa kaçırır → `has_part()` bileşen bazlı eşler.
14. Kaynak yollarında boşluk var → notebook `build_dataset.py`'yi **kabuk
    dizesiyle değil `subprocess` argüman listesiyle** çağırır.
15. **Sınıf sayısına bağlı sabitler koda gömülmemeli.** Taksonomi 10→2
    olunca `verify_kv260_golden.py` içindeki `c != 15` kontrolü doğru bir
    kart dökümünü reddeder hale gelmişti; artık kanal sayısı dökümden
    türetiliyor. Aynı hata sınıfı: `compile_kv260.sh`, `main.cpp`.
16. **Evaluator, Exp'in alanlarını görmez.** `VisDroneEvaluator` içinde
    `self.deploy_conf` kullanılıyordu ama alan `Exp`'te tanımlıydı;
    `COCOEvaluator.__init__` böyle bir alan set etmiyor (sabitlenmiş
    commit'ten doğrulandı) → ilk değerlendirmede `AttributeError`. Artık
    `get_evaluator` açıkça geçiriyor.
17. **CSV başlığı ile satırları elle senkron tutmayın.** `centers.csv`
    başlığında `track_id` eksikti; sütunlar bir kayıyordu.

## 8. Kritik teknik sabitler

- Vitis AI docker: `xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106`
  (`:latest` = 3.5 **yasak**)
- YOLOX commit: `6ddff4824372906469a7fae2dc3206c7aa4bbaee`
- Model: YOLOX-Nano, ReLU (SiLU DPU'da yok), **DPUFocus** (strided-slice yerine
  sabit one-hot conv; önceden eğitilmiş ağırlıklar geçerli kalır)
- Çıktı kanalı = `5 + sınıf sayısı` = **7**
- Graph **tam olarak 1 DPU subgraph** olmalı; CPU subgraph varsa derleme fail
- INT8 kabul: `--float-map` zorunlu, mutlak AP kaybı > `0.02` → export yok
- Golden: `--dump-first-frame` + `tools/verify_kv260_golden.py`
- `centers.csv`: `frame,track_id,class_id,class_name,score,cx,cy`

**YOLOv8/v11 önerisi gelirse:** DFL kafası softmax kullanır, DPUCZDX8G'de
CPU subgraph'ine düşer (Xilinx/Vitis-AI #904) ve tek-subgraph kapımızı ihlal
eder. Ayrıca AGPL-3.0. Doğruluk yetmezse doğru yükseltme **YOLOX-Tiny**.

## 9. Sıradaki adımlar

1. ~~Kaggle eğitimi~~ ✅ **bitti** 2026-08-09, sonuçlar §5'te. Taban çizgisi
   artık var: mendeley temizliğinin (§4) fayda edip etmediği buna karşı
   ölçülebilir. `yolox_aerial_artifacts.zip` Kaggle Output'tan indirilecek.
2. **VM**: `quantize/README.md`. `--inspect` ile kare olmayan girdinin tek DPU
   subgraph'ine derlendiğini doğrula. VM'e `datasets/merged/images` ağacı ve
   `instances_val.json` lazım (kalibrasyonu train'den yapmak için
   `instances_train.json` da). **Açık soru:** INT8 kabul testi 4.483 görüntü
   üzerinde CPU'da saatler sürer ve `--subset-len` bilerek yasak.
   **Float AP referansı: `--float-map 0.5874`** (§5).
3. **Kart**: `deploy/README.md` + golden test + **gerçek FPS ölçümü**.
4. FPS ölçüldükten sonra: gerekirse çözünürlük/tiling kararı.
5. **Rapor**: `docs/report_template.md` (§5'teki sayılar ve üç uyarı ile).

## 9b. Kullanıcının hazır ortamı (2026-08-09'da öğrenildi)

**Tüm ortam hazır — hiçbir kurulum adımı kalmadı:**

- **VM**: VirtualBox + Ubuntu kurulu, Vitis AI docker'ı da kurulmuş
  (kullanıcı beyanı). `quantize/README.md` adım 1-6 **atlanabilir**.
- **Kart**: KV260'da daha önce YOLOv4-tiny çalıştırılmış → SD imajı yazılı,
  VART çalışıyor. `deploy/README.md` adım 1-3 **atlanabilir**.
- **VM paketi**: `python tools/make_vm_package.py` → `vm_package/` (626 MB).
  Tüm görüntü ağacını (3,6 GB) taşımaya gerek yok; paket val setinin tamamını
  (4.483, accuracy gate alt küme kabul etmiyor) + 300 kalibrasyon karesi +
  checkpoint + scriptleri içerir ve `KOMUTLAR.md` ile gelir.

Kullanıcı **KV260'da daha önce YOLOv4-tiny çalıştırmış.** Yani:

- PetaLinux/Vitis AI SD imajı **karta yazılmış ve çalışıyor** →
  `deploy/README.md` adım 1-3 (imaj yazma, seri port, DPU doğrulama) atlanabilir.
- VART çalışma zamanı kartta mevcut, kart üzerinde bir kez çıkarım yapılmış.
- Vitis AI 3.0 "kurulu" diyor — **nerede olduğu netleştirilmeli**: kartın
  üzerindeki çalışma zamanı mı, yoksa kuantalama/derleme için gereken Linux
  docker'ı mı? YOLOv4-tiny genelde Model Zoo'dan **hazır derlenmiş** xmodel ile
  çalıştırılır; bu durumda kuantalama ortamı **kurulmamış** olabilir.
- **Kart Vitis AI 3.0** — kullanıcı 2026-08-09'da açıkça doğruladı ("kart 3.0
  eminim"). Projedeki 3.0 sabiti doğru. (4 Ağustos'taki bir oturumda "vitis
  3.5" geçiyor; o Model Zoo tarafıyla ilgiliydi, kartla değil.)
- **B4096 hâlâ ÖLÇÜLMEDİ.** Kullanıcının hatırlaması ve KV260 hazır imajının
  varsayılanı bu yönde; geçmiş oturumlarda `xdputil query` çıktısı yok.
  **Ama bu pahalı adımları bloklamaz:** DPU mimarisi yalnızca
  `compile_kv260.sh` içindeki `vai_c_xir -a .../KV260/arch.json` adımına
  giriyor (dakikalar). Kalibrasyon, INT8 AP testi ve `--deploy` export'u
  mimariden bağımsızdır. Yanlış çıkarsa yalnızca son adım tekrarlanır.
  Ayrıca üç yerde sert kapı var: `compile_kv260.sh` subgraph/kanal kontrolü,
  `main.cpp` başlangıç doğrulaması ve kartın `fingerprint mismatch` hatası —
  sessizce yanlış çalışma ihtimali yok.

## 10. Kullanıcıyla çalışma notları

- İletişim dili **Türkçe**.
- **Kısa, numaralı, tek eylemli adımlar** verin. Uzun ve çok seçenekli
  mesajlar işi yavaşlatıyor.
- **Tahminle iş yapılmasın; ölçüm ve doğrulama istenir.** Sessiz hatalar bu
  projede en büyük risk — her iddia ölçülerek desteklenmeli.
- **Overengineering'den kaçının.** Gerekmeyen soyutlama ve ölü kod istenmiyor.
- Başka AI araçlarından gelen öneriler soyut olarak doğru olsa da bu projenin
  kısıtlarında yanlış çıktı (xView sınıfları, YOLOv8/DFL). Öneri gelince
  **kaynağı doğrulayın**, hafızadan cevap vermeyin.
- Projenin değeri model eğitmekte değil **donanım-farkında dağıtımda**:
  DPUFocus, tek-subgraph kapısı, kuantalama kabul kapıları, golden test.
- **API anahtarı/token işlemeyin** — kimlik doğrulama adımlarını kullanıcı
  kendisi yapar.

## 11. Doğrulama komutları

```bash
python -m unittest discover -s tests          # 69 test (g++ varsa C++ dahil)
python tools/audit_dataset.py --merged datasets/merged --data-dir datasets
python tools/_sync_notebook_embeds.py         # notebook hucrelerini esitle
python tools/build_dataset.py --dry-run       # sayilar: 19476/205340, 4483/21681
```

Windows'ta `g++` PATH'te değilse C++ tracker testi atlanır:
`C:\msys64\ucrt64\bin` ekleyin. Python: `C:\Users\emrez\AppData\Local\Python\bin\python.exe`.
