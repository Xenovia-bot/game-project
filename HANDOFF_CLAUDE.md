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

## 2. Zincir ve nerede olduğumuz

| Aşama | Durum |
| --- | --- |
| Veri birleştirme | ✅ **bitti, doğrulandı** (19.476 train / 4.483 val) |
| Kaggle eğitimi (2 sınıf) | ⬜ **sıradaki iş** — ~1,5-2 saat |
| Oracle VM · Vitis AI 3.0 PTQ | ⬜ |
| KV260 · C++ VART + golden test | ⬜ |
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
| 🔴 **`sea_vehicle` 17 sahneden** | 4.387 görüntü ama Valencia Limanı'nda yalnızca 17 kamera kurulumu. Model sahneyi ezberleyebilir. Val de aynı limandan 6 sahne. **Tek gerçek çözüm ikinci bir denizcilik kaynağı.** |
| 🟡 Dengesizlik 22:1 | Müdahale edilmedi. `--repeat vesselimg=3` / `--subsample visdrone=0.5` düğmeleri hazır ama **kapalı** — önce ölçülecek |
| 🟡 Dört farklı alan | VisDrone şehir / VESSELimg liman / milrec savaş / mendeley karışık. 0.9M model için zor |
| 🟡 Mendeley içeriği | Makalede "sentetik veri artırma" geçiyor; oranı **doğrulanmadı** |
| 🟡 Kaynak etiket kalitesi | Dönüşümün sadık olduğu kanıtlandı, **orijinal etiketlerin doğruluğu değil** |
| 🟡 Boş kare oranı | train %24, val %36 (tipik %10). Mendeley kaynaklı — asker/insan atılınca kare boşalıyor |

## 5. Ölçülmüş gerçekler (yeniden türetmeyin)

**Referans çalıştırma** — 10 sınıf, 640×640, 80 epoch, T4, 2026-08-07:

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
4. Kaggle CLI: `kaggle.json` **okuma** için yeter, **yükleme** yeni token
   ister (`kaggle auth login` veya `KAGGLE_API_TOKEN` / `~/.kaggle/access_token`).
5. PowerShell 5.1 `Set-Content -Encoding utf8` **BOM ekler**; `kaggle.json`
   BOM'lu olursa "Missing username" hatası verir.
6. Kaggle kullanıcı adı: **`burakzorgeen`** (e-postadaki gibi `burakzorgecen`
   değil).

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
    `action="append"`.
12. VisDrone train ve val **aynı kaynak adını paylaşır ama farklı zip'lerdedir**
    — arşivi kaynak adına göre eşlemek val görüntülerini train zip'inde aratır.
13. Yol eşleştirmesi `"/images/" in name` ile yapılmamalı; arşiv kökü bir
    seviye aşağıdaysa kaçırır → `has_part()` bileşen bazlı eşler.
14. Kaynak yollarında boşluk var → notebook `build_dataset.py`'yi **kabuk
    dizesiyle değil `subprocess` argüman listesiyle** çağırır.

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

1. **Kaggle**: notebook'u import et → `aerial-vehicle-sources` bağla → GPU +
   Internet aç → **hücre 1-6 çalıştır, dur**. Hücre 6 şunu basmalı:
   `train 19476 / 205340`, `val 4483 / 21681`. Tutmazsa çıktıyı incele.
2. Hücre 7 → 9 ile **eğitim** (~1,5-2 saat, 40 epoch).
3. Hücre 11 değerlendirme → **sınıf bazlı AP'ye bak**. `sea_vehicle` için
   train-val farkı büyükse 17 sahneye aşırı uyum var demektir.
4. Hücre 13 → `yolox_aerial_artifacts.zip` indir.
5. **VM**: `quantize/README.md`. `--inspect` ile kare olmayan girdinin tek DPU
   subgraph'ine derlendiğini doğrula. VM'e val görüntüleri de lazım.
6. **Kart**: `deploy/README.md` + golden test + **gerçek FPS ölçümü**.
7. FPS ölçüldükten sonra: gerekirse çözünürlük/tiling kararı.
8. **Rapor**: `docs/report_template.md`.

## 10. Kullanıcıyla çalışma notları

- **Türkçe** konuşur, Türkçe cevap bekler.
- **Uzun mesajlarda kayboluyor.** Kısa, numaralı, tek eylemli adımlar verin.
  "Neyi nasıl yapacağım anlamadım" dediyse mesaj uzun demektir.
- Tahminle iş yapılmasından hoşlanmıyor; **ölçüm ve doğrulama** istiyor.
  "tek hatada patlarız" dedi — sessiz hatalar en büyük risk.
- **Overengineering'den kaçının** — açıkça söyledi.
- Başka AI'lardan (ChatGPT, Gemini) öneri getiriyor. Bunlar soyut olarak doğru
  ama bu projenin kısıtlarında yanlış çıktı (xView sınıfları, YOLOv8). Öneri
  gelince **kaynağı doğrulayın**, hafızadan cevap vermeyin.
- Amiri "YOLO lise seviyesi" demiş; projenin değeri model eğitmekte değil
  **donanım-farkında dağıtımda** (DPUFocus, kuantalama kapıları, golden test).
- Kimlik bilgisi paylaşmaya çalıştı — **API anahtarı/token işlemeyin**, kendisi
  yapsın.

## 11. Doğrulama komutları

```bash
python -m unittest discover -s tests          # 69 test (g++ varsa C++ dahil)
python tools/audit_dataset.py --merged datasets/merged --data-dir datasets
python tools/_sync_notebook_embeds.py         # notebook hucrelerini esitle
```

Windows'ta `g++` PATH'te değilse C++ tracker testi atlanır:
`C:\msys64\ucrt64\bin` ekleyin. Python: `C:\Users\emrez\AppData\Local\Python\bin\python.exe`.
