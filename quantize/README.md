# Oracle VM'de Kuantalama ve Derleme (Vitis AI 3.0)

Amaç: Kaggle'da eğitilen float YOLOX-Tiny **gemi tespiti** modelini (tek sınıf
`ship`) INT8'e kuantalamak, doğruluğunu ölçmek ve KV260 DPU'su
(DPUCZDX8G B4096) için `.xmodel`'e derlemek.

Ölçülen metrik: standart COCO mAP (AP@[.50:.95], AP@0.50, AP@0.75) **ve**
kartın çalışma noktasında (conf=0.15) precision/recall/F1. AP tanımı
Kaggle'daki eğitim değerlendirmesiyle aynıdır; float ↔ INT8 ↔ Kaggle
sayıları ancak bu sayede karşılaştırılabilir.

> Sürüm kuralı: KV260 hazır kart imajı Vitis AI **3.0** olduğundan docker da
> **3.0** etiketiyle çekilir. `:latest` etiketi 3.5'e işaret eder, kullanmayın.

## 1. VM hazırlığı (VirtualBox)

- Sistem → Anakart/İşlemci: en az **8 GB RAM**, **4 vCPU**
- Depolama: en az **80 GB boş disk** (docker imajı + modeller + veri)
- Ağ: NAT yeterli
- Windows ↔ VM dosya alışverişi: VirtualBox paylaşılan klasörü veya `scp`

## 2. Docker kurulumu (Ubuntu içinde)

```bash
sudo apt update
sudo apt install -y docker.io git unzip
sudo usermod -aG docker $USER
# Grup üyeliğinin işlemesi için oturumu kapatıp açın (veya: newgrp docker)
docker run hello-world
```

## 3. Vitis AI 3.0 deposu ve docker imajı

```bash
cd ~
git clone -b 3.0 https://github.com/Xilinx/Vitis-AI.git
cd Vitis-AI
docker pull xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106
```

## 4. Çalışma alanını hazırla

`Vitis-AI` klasörü docker'da `/workspace` olarak bağlanır. Şu yapıyı kurun:

```
Vitis-AI/
└── yolox_ship/
    ├── quantize_yolox.py            # bu klasördeki dosya
    ├── compile_kv260.sh             # bu klasördeki dosya
    ├── best_ckpt.pth                # Kaggle artifacts.zip içinden
    ├── yolox_tiny_ship.py           # Kaggle artifacts.zip içinden
    ├── ship_metrics.py              # mAP + F1 modülü (artifacts.zip içinden)
    ├── YOLOX_COMMIT.txt             # iki ortamda aynı kaynak sürümü
    └── datasets/ship_merged/
        ├── annotations/instances_val.json     # artifacts.zip içinden
        ├── annotations/instances_test.json    # nihai rapor için
        ├── annotations/instances_train.json   # isteğe bağlı (kalibrasyon için)
        └── images/<kaynak>/<ad>.jpg           # Kaggle'daki merged/images ağacı
```

> **Anotasyon üretmeyin.** Veri seti Kaggle'da `build_ship_dataset.py` ile bir kez
> üretildi; `instances_val.json` / `instances_test.json` artifacts paketinden gelir. VM'de yeniden
> üretmek bölmeyi değiştirir ve float ↔ INT8 karşılaştırmasını geçersiz kılar.

Görüntüler: `quantize_yolox.py`, COCO `file_name` alanlarını doğrudan
`images/` kökünün altında arar (`images/ir_thermal/xxx.jpg` gibi), bu yüzden
ağacı **düzleştirmeyin**.

Kalibrasyon kaynağı sırayla seçilir:

1. `--calib-dir <klasör>` verilmişse o klasör (özyinelemeli),
2. yoksa `annotations/instances_train.json` varsa **yalnızca train** görüntüleri,
3. o da yoksa `images/` ağacının tamamı — bu durumda val görüntüleri de
   kalibrasyona karışır ve script bunu uyarı olarak yazar.

`instances_train.json` Kaggle çıktısında `datasets/ship_merged/annotations/`
altındadır; 2. yolu kullanmak için onu da kopyalayın.

## 5. Docker'ı başlat

```bash
cd ~/Vitis-AI
./docker_run.sh xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106
# docker içinde:
conda activate vitis-ai-pytorch
cd /workspace/yolox_ship
```

## 6. Docker içinde YOLOX'u kur (hazır torch sürümüne dokunmadan)

```bash
YOLOX_COMMIT=$(tr -d '\r\n' < YOLOX_COMMIT.txt)
test "$YOLOX_COMMIT" = "6ddff4824372906469a7fae2dc3206c7aa4bbaee"
test -d YOLOX/.git || \
  git clone --filter=blob:none --no-checkout https://github.com/Megvii-BaseDetection/YOLOX.git
git -C YOLOX fetch --depth 1 origin "$YOLOX_COMMIT"
git -C YOLOX checkout --detach "$YOLOX_COMMIT"
test "$(git -C YOLOX rev-parse HEAD)" = "$YOLOX_COMMIT"
# --no-build-isolation sart: YOLOX'un setup.py'si torch'u import eder, pip'in
# izole build ortaminda torch bulunmaz ve kurulum basarisiz olur.
pip install --no-deps --no-build-isolation -e ./YOLOX
pip install loguru tabulate pycocotools
python -c "import yolox; print(yolox.__version__, yolox.__file__)"
```

Son satır hata verirse editable kurulum tutmamıştır; editable olmayan kuruluma
düşün:

```bash
pip install --no-deps --no-build-isolation ./YOLOX
```

## 7. Veriyi doğrula

Kuantalamaya girmeden önce dosyaların yerinde olduğunu görün — INT8 AP testi
saatler sürer, yarısında dosya eksiği çıkması pahalıya patlar:

```bash
cd /workspace/yolox_ship
test -f datasets/ship_merged/annotations/instances_val.json
test -f ship_metrics.py   # yoksa ölçüm adımları ImportError ile durur
python -c "import json;d=json.load(open('datasets/ship_merged/annotations/instances_val.json'));print(len(d['images']),'goruntu',len(d['annotations']),'kutu',[c['name'] for c in d['categories']])"
```

Beklenen çıktı: `7622 goruntu 17168 kutu ['ship']` (2026-08-12 tarihli veri
üretiminden; kendi çalıştırmanızın özeti farklıysa oradaki sayıyı esas alın).
Sınıf adı `ship` değilse yanlış anotasyon dosyasıdır; devam etmeyin.

## 8. Kuantalama akışı

Ortak argümanlar her komutta aynıdır:

```bash
ARGS="--exp-file yolox_tiny_ship.py --ckpt best_ckpt.pth --data-dir datasets/ship_merged"
```

1. (İsteğe bağlı) DPU uyumluluk raporu — tüm katmanların DPU'ya atandığını doğrular:

```bash
python quantize_yolox.py --inspect $ARGS
```

2. Float ölçüm — Kaggle'daki AP ile aynı çıkmalı (sağlama). Çıkan
   `AP@[.50:.95]` değerini `FLOAT_AP` değişkenine yazın:

```bash
python quantize_yolox.py --quant-mode float $ARGS
FLOAT_AP=BURAYA_AP_DEGERINI_YAZIN
```

3. Kalibrasyon (PTQ, ~300 görüntü; CPU'da 10-30 dk):

```bash
python quantize_yolox.py --quant-mode calib --subset-len 300 $ARGS
```

4. INT8 ölçümü ve kabul kapısı (**7622 val görüntüsü**; CPU'da saatler sürer — `screen`
   veya `nohup` altında başlatın, oturum kopunca ölçüm de kaybolur):

```bash
python quantize_yolox.py --quant-mode test --float-map "$FLOAT_AP" $ARGS
```

Script mutlak AP kaybı `0.02` değerini aşarsa hata ile durur. Sınırı yalnızca
bilinçli olarak değiştirmek için `--max-map-drop` kullanın. Kayıp yüksekse
AdaQuant fast-finetune ile kalibrasyon/testi tekrarlayın:

```bash
python quantize_yolox.py --quant-mode calib --subset-len 300 --fast-finetune $ARGS
python quantize_yolox.py --quant-mode test --fast-finetune \
    --float-map "$FLOAT_AP" $ARGS
```

Başarılı tam-val testi `build/quant/accuracy_gate.json` üretir. Checkpoint,
exp, anotasyon, `quant_info.json`, eşikler veya fast-finetune seçimi değişirse
bu kapı geçersiz olur ve export reddedilir.

5. xmodel export (derleme girdisi `build/quant/DeployModel_int.xmodel`):

```bash
python quantize_yolox.py --quant-mode test --deploy --subset-len 1 --batch-size 1 $ARGS
```

Fast-finetune kullandıysanız export komutuna da `--fast-finetune` ekleyin.

6. **Raporlanacak nihai sayı — test setinde, yalnızca bir kez.**

Model seçimi (`best_ckpt`) ve kabul kapısı val setinde yapıldı; aynı sette
rapor vermek iyimser bir sayı üretir. `--report-only` kabul kapısına
dokunmaz, bu yüzden 5. adımdaki export geçerli kalır:

```bash
python quantize_yolox.py --quant-mode test --report-only     --ann instances_test.json $ARGS
```

Çıktı: AP@[.50:.95] / AP@0.50 / AP@0.75, kart eşiğinde (conf=0.15) P/R/F1,
en iyi F1 ve onun eşiği, ayrıca **kaynak bazlı AP tablosu**. Termal
dayanıklılığı `ir_thermal` satırından okuyun: tek global sayı, o kaynağın
çökmesini gizleyebilir.

## 9. KV260 için derleme

```bash
bash compile_kv260.sh            # 4. arguman: sinif sayisi, varsayilan 1 (ship)
```

Script, `build/compiled/yolox_tiny_ship.xmodel` üretir ve subgraph dağılımını
yazdırır. Graph'ın tamamı **tam olarak bir DPU subgraph** olmalıdır; ek CPU/USER
subgraph varsa script başarısız olur. Ayrıca 1 giriş ve üç YOLOX çıkışı
doğrulanır; çıkış kanal sayısı `5 + sınıf sayısı` olmalıdır (1 sınıf → **6**).
Farklı bir şema kullanıyorsanız: `bash compile_kv260.sh build/quant build/compiled baska_ad 10`.

> Girdi **512×512 karedir** (aerial projedeki 896×512'den farklı). Derleme
> öncesi `python quantize_yolox.py --inspect $ARGS` ile tüm katmanların
> DPU'ya atandığını doğrulayın.

## 10. Sorun giderme

- **Quantized AP çok düştü** → `--fast-finetune` (8. adım). Yeterli olmazsa
  bu xmodel'i kullanmayın. Vitis AI QAT; `QuantStub`, modül tabanlı add/cat
  işlemleri ve Vitis AI uyumlu ayrı bir eğitim ortamı ister; standart Kaggle
  YOLOX eğitiminin üzerine tek bir bayrakla uygulanamaz ve bu akışın kapsamında
  değildir.
- **`vai_c_xir` bulunamadı** → docker içinde `conda activate vitis-ai-pytorch`
  yapıldığından emin olun.
- **Docker disk doldu** → VirtualBox sanal diskini genişletin;
  `docker system prune` ile eski imajları temizleyin.
- **PTQ çok yavaş** → normaldir (CPU). Kalibrasyonda `--subset-len` değerini
  200'e düşürmek kabul edilebilir. **INT8 AP testinde `--subset-len`
  kullanılamaz**: accuracy gate tam val setiyle üretilmek zorundadır, script
  alt küme verilirse durur.
- **`ModuleNotFoundError: ship_metrics`** → `ship_metrics.py` artifacts
  paketinden `/workspace/yolox_ship/` altına kopyalanmamış. Ölçüm adımları
  (float / test / report-only) bu dosya olmadan çalışmaz.
- **`kalibrasyon goruntusu bulunamadi`** → `images/` ağacını düzleştirmişsiniz
  veya yanlış `--data-dir` vermişsiniz. Beklenen düzen 4. adımda.

## Çıktılar (bir sonraki aşamaya taşınacaklar)

- `build/compiled/yolox_tiny_ship.xmodel` → KV260'a kopyalanacak
- `build/quant/accuracy_gate.json` → kullanılan INT8 artifactların AP kabul kaydı
- Float / INT8 AP ve kart eşiğindeki P/R/F1 → rapor için not edin
- 6. adımın test seti çıktısı → **raporlanacak nihai sayı**
- Karttaki decode/NMS/merkez eşdeğerliği → `deploy/README.md` içindeki golden
  test ile ayrıca doğrulanır

Sonraki adım: [deploy/README.md](../deploy/README.md) — kart kurulumu ve C++ uygulaması.
