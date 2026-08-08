# Claude Handoff — KV260 YOLOX-Nano × VisDrone

Bu dosya, önceki Cursor sohbetinin bağlamını kayıpsız devam ettirmek içindir.
Yeni Claude oturumunda **önce bunu**, sonra planı ve `C:\Users\emrez\proje` klasörünü okut.

## 1. Proje hedefi

KV260 kartında YOLOX-Nano ile VisDrone nesne tespiti; her kutunun merkez noktası:

`cx = (x1+x2)/2`, `cy = (y1+y2)/2`

Çıktılar: `out.avi` + `centers.csv` + FPS özeti.

Zincir:

1. Kaggle → fine-tune + AP@500
2. Oracle VM (Ubuntu) → Vitis AI **3.0** PTQ + derleme
3. KV260 → C++ VART demo + golden test

## 2. Plan dosyası (dikkat: kısmen eski)

Plan yolu:

`C:\Users\emrez\.cursor\plans\kv260_yolox_visdrone_planı_b9133002.plan.md`

Planın **hedefi doğru**, ama bazı maddeler **sonraki düzeltmelerle geçersiz**.
Kod ve `README.md` planın üstündedir.

### Planda artık yanlış / güncellenmiş olanlar

| Plan ifadesi | Güncel gerçek |
| --- | --- |
| Vitis AI Model Zoo `pt_yolox-nano` tabanı | **Kullanılmıyor.** Megvii `yolox_nano.pth` + DPUFocus/ReLU Exp |
| YOLOX `main` / unpinned clone | Sabit commit: `6ddff4824372906469a7fae2dc3206c7aa4bbaee` |
| ignored/others atılır | `category=0` ignore korunur; `others`(11) hedef değil; `score=0` sınıf-özel crowd |
| 10 sınıf | **2 sınıf**: person / vehicle (`--classes 2`) |
| 640×640 kare girdi | **896×512 (16:9)** — letterbox dolgusu %44'ten ~%0'a iner |
| 80 epoch | **40 epoch** — model 40'ta doyuyor (ölçüldü) |
| COCO mAP / maxDets=100 | Resmi VisDrone tarzı **global top-500 + VOC AP** (`training/visdrone_eval.py`) |
| QAT Kaggle yedek planı | **Uygulanabilir değil** (QuantStub vb. gerekir). PTQ kaybı >0.02 ise export yok |
| “tek DPU subgraph sayısı 1” yeter | Graph **yalnızca 1 DPU subgraph** olmalı; CPU subgraph olursa fail |
| Kod iskeleti eksik | Repo uygulandı; review sonrası güvenlik kapıları eklendi |

## 0. KILITLENMIS HEDEF (2026-08-08)

Sistem **iki sinif** tespit edecek ve her tespitin merkez noktasini uretecek:

| Sinif | Kapsam |
| --- | --- |
| `military_vehicle` | tank, zirhli arac, askeri kamyon |
| `ship` | buyuk deniz araci |

Kisitlar:

- **Bakis acisi drone/ucak olmali** — havadan karaya veya havadan denize.
  Uydu goruntusu (xView, DOTA, DIOR, HRSC) **kullanilmayacak**: 0.3 m GSD'de
  7 m'lik bir tank ~23 piksele duser ve kacindigimiz kucuk-nesne problemine
  geri donulur.
- **Insan tespit edilmeyecek.** Ancak askeri veri setlerinde `soldier`/`people`
  etiketli oldugu icin bunlar **ignore** olarak isaretlenecek; etiketsiz
  birakilirsa model onlari arka plan olarak ogrenir ve bu tespiti bozar.
- Buyuk nesne = kolay problem. VisDrone yayasi 896x512'de 5-14 px iken tank
  19-37 px, buyuk gemi 93-467 px olur. Kullanicinin "%80 recall" hedefi bu
  siniflarda gercekci hale gelir.

### Terminoloji tuzagi (iki kez yasandi)

`tank` kelimesi geçen her sinif askeri tank degildir:

- xView `Tank Car` = demiryolu tanker vagonu
- xView / DOTA `Storage Tank` = yakit deposu

Veri seti secerken sinif listesi **resmi dokumandan** dogrulanmali.

## 3. Mevcut durum (2026-08-08)

### Tamamlanan: 10 sınıflı referans çalıştırma

Kaggle'da 10 sınıf / 640×640 / 80 epoch eğitimi bitti:

| Epoch | AP@[.50:.95] | AP@0.50 | AP@0.75 |
| --- | --- | --- | --- |
| 10 | 0.0644 | 0.1376 | 0.0558 |
| 45 | 0.1056 | 0.2144 | 0.0932 |
| 80 | 0.1087 | 0.2205 | 0.0965 |

İki bulgu sonraki kararları belirledi:

1. Model **epoch ~40'ta doydu** (45→80 arası +0.006) → `max_epoch` 40.
2. **AP75/AP50 = 0.44** → darboğaz sınıflandırma değil çözünürlük →
   girdi 16:9'a (896×512) çevrildi.

### Yapılan pivot: 2 sınıf + 16:9 girdi + takip

Kullanıcının hedefi **yalnızca insan ve taşıtın merkez noktası**, ≥30 FPS,
iz bazında yüksek recall. Buna göre:

- `visdrone2coco.py --classes 2` → `person` / `vehicle` şeması
- `input_size = (512, 896)`, `random_size = (14, 18)`
- `max_epoch = 40`, `no_aug_epochs = 10`
- Dağıtım eşiği `0.30` → `0.15` (recall odaklı; takip temizliyor)
- `main.cpp` sınıf adları 2'ye indi; `compile_kv260.sh` kanal sayısını
  sınıf sayısından türetiyor (5+N; 15 hardcode kalktı)
- Eğitim **ve** val veri setinde kategori sayısı doğrulanıyor — eski
  10 sınıflı JSON kalırsa eğitim başlamadan hata verir

### Eklenen: takip (tracking)

`deploy/src/tracker.hpp` — IoU tabanlı, ByteTrack'in iki aşamalı
eşleştirmesiyle. Kalman ve Macar algoritması yok; hareket modeli sabit hız.
VART/OpenCV bağımsız tutuldu, bu yüzden kart dışında derlenip test edilebiliyor
(`deploy/tests/test_tracker.cpp`, 10 test). `tests/test_cpp_tracker.py` bunu
otomatik derleyip çalıştırır; derleyici yoksa atlar.

`centers.csv` artık `frame,track_id,class_id,class_name,score,cx,cy`.
Yeni bayraklar: `--no-track`, `--track-n-init` (3), `--track-max-age` (30).
Golden dump takipten **önceki** ham tespitleri yazar; eşdeğerlik testi etkilenmez.

### Eklenen ölçüm altyapısı

- P/R/F1 (en iyi nokta + dağıtım eşiği), sınıf bazlı AP
- Sınıf gruplama senaryoları (`GROUP_2`, `GROUP_3`, tek sınıf) — bir modeli
  farklı sınıf tanımlarıyla yeniden eğitmeden ölçmek için
- `tools/tiling.py` + `training/eval_tiled.py` — dilimlenmiş çıkarım.
  **Spekülatif**: 16:9 girdi kararı sonrası muhtemelen gerekmeyecek, yedek plan

- Yerel doğrulama: `python -m unittest discover -s tests -v` → **39/39 OK**
  (g++ PATH'te ise C++ tracker testleri dahil).
- Notebook `%%writefile` hücreleri kaynakla senkron: `python tools/_sync_notebook_embeds.py`

### Sıradaki kullanıcı adımı

1. `training/kaggle_visdrone_yolox.ipynb` → Kaggle'da sıfırdan çalıştır
   (2 sınıf, 896×512, 40 epoch — ~1,5-2 saat)
2. Yeni `yolox_visdrone_artifacts.zip` indir
3. Eski 10 sınıflı COCO JSON / checkpoint / xmodel **kullanma**
4. VM: `quantize/README.md` — `--inspect` ile **kare olmayan girdinin** tek DPU
   subgraph'ine derlendiğini doğrula
5. Kart: `deploy/README.md` + golden test, **gerçek FPS'i ölç**
6. FPS ölçüldükten sonra: gerekirse tiling (takip artık hazır)

### Henüz yapılmayanlar / bilinen sınırlar

- `training/eval_tiled.py` torch gerektirdiği için **yerelde çalıştırılamadı**;
  ilk gerçek koşusu Kaggle'da olacak.
- `main.cpp` VART/OpenCV gerektirdiği için yerelde **derlenemedi**; takip
  entegrasyonu gözle incelendi, ilk gerçek derleme kartta/SDK'da olacak.
- Golden test yalnızca ilk karenin ham tespitlerini karşılaştırır; **takip
  zamansal olduğu için golden testle doğrulanamaz** — derlenen C++ birim
  testleriyle doğrulanıyor.
- **İz bazında recall ölçülmedi.** Bunun için iz kimlikli video veri seti
  (VisDrone-VID / MOT) gerekir. Kullanıcının "100 nesnenin 80'i" hedefi
  kare bazında değil, bu metrikle ölçülmelidir.
- VisDrone'da **gemi/uçak/tank yok**. Gerekirse DIOR ikinci aşama olarak
  eklenebilir; ancak görsel olarak çok farklı sınıfları `vehicle` içine
  katmak zarar verir ([arXiv 2407.00018](https://arxiv.org/abs/2407.00018):
  benzer sınıfları birleştir, farklı olanları ayrı tut).

## 4. Kritik teknik sabitler

- Vitis AI docker: `xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106` (`:latest`=3.5 yasak)
- YOLOX commit: `6ddff4824372906469a7fae2dc3206c7aa4bbaee`
- Exp: ReLU + `DPUFocus`, **2 sınıf** (person/vehicle), **896×512 (16:9)**,
  `max_labels` 1000/4000, 40 epoch
- Çıktı kanalı = 5 + sınıf sayısı = **7** (eskiden 15 hardcode'du)
- Dağıtım eşiği: `--conf 0.15` (recall odaklı)
- Merkez: letterbox ters dönüşümünden **sonra** orijinal video koordinatlarında
- INT8 kabul: `--float-map` zorunlu; drop > `0.02` → export yok; `accuracy_gate.json` şart
- Golden: `--dump-first-frame` + `tools/verify_kv260_golden.py` (INT8 input HALF_UP + decode/NMS/cx,cy)

## 5. Önemli dosyalar

```
C:\Users\emrez\proje\
  README.md
  HANDOFF_CLAUDE.md          ← bu dosya
  tools/visdrone2coco.py         ← --classes 2 destegi
  tools/tiling.py                ← dilimlenmis cikarim (spekulatif, yedek)
  tools/verify_kv260_golden.py
  tools/_sync_notebook_embeds.py
  training/kaggle_visdrone_yolox.ipynb
  training/exps/yolox_nano_visdrone.py
  training/visdrone_eval.py      ← P/R/F1 + sinif gruplama senaryolari
  training/eval_tiled.py         ← tam kare vs tiling karsilastirmasi
  quantize/quantize_yolox.py
  quantize/compile_kv260.sh
  quantize/README.md
  deploy/src/main.cpp
  deploy/src/tracker.hpp         ← IoU + ByteTrack 2 asamali eslestirme
  deploy/tests/test_tracker.cpp  ← 10 C++ testi
  deploy/build.sh
  deploy/CMakeLists.txt
  deploy/README.md
  docs/report_template.md
  tests/                     ← 39 test (C++ tracker testleri dahil)
```

## 6. Sohbetten kalan kararlar / kısıtlar

- Colab yok → **Kaggle**
- WSL2 yok → **Oracle VM Ubuntu + Docker**
- Deploy: **PetaLinux + VART C++**, video dosyası girişi
- Plan dosyasını **kullanıcı izni olmadan düzenleme**
- Commit yalnızca kullanıcı isterse
- Frontend design kuralları bu projeyle ilgili değil

## 7. Claude’a ilk mesaj şablonu

Aşağıyı yeni Claude sohbetine yapıştır:

```text
KV260 + YOLOX-Nano + VisDrone projesine devam ediyoruz.

Önce oku:
1) C:\Users\emrez\proje\HANDOFF_CLAUDE.md
2) C:\Users\emrez\proje\README.md
3) C:\Users\emrez\.cursor\plans\kv260_yolox_visdrone_planı_b9133002.plan.md
   (plan hedefi doğru ama HANDOFF'taki "güncellenmiş olanlar" tablosu planın üstündedir)

Workspace: C:\Users\emrez\proje

Durum: kod/testler hazır; sırada Kaggle eğitiminden başlayarak uçtan uca çalıştırma.
Plan dosyasını iznim olmadan düzenleme.
```
