# Oracle VM'de Kuantalama ve Derleme (Vitis AI 3.0)

Amaç: Kaggle'da eğitilen float YOLOX-Nano modelini INT8'e kuantalamak, doğruluğunu
ölçmek ve KV260 DPU'su (DPUCZDX8G B4096) için `.xmodel`'e derlemek.

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
└── yolox_visdrone/
    ├── quantize_yolox.py            # bu klasördeki dosya
    ├── compile_kv260.sh             # bu klasördeki dosya
    ├── best_ckpt.pth                # Kaggle artifacts.zip içinden
    ├── yolox_nano_visdrone.py       # Kaggle artifacts.zip içinden
    ├── visdrone_eval.py             # resmi DET tarzı AP@500 değerlendirici
    ├── YOLOX_COMMIT.txt             # iki ortamda aynı kaynak sürümü
    └── datasets/merged/
        ├── annotations/instances_val.json     # artifacts.zip içinden
        ├── annotations/instances_train.json   # isteğe bağlı (kalibrasyon için)
        └── images/<kaynak>/<ad>.jpg           # Kaggle'daki merged/images ağacı
```

> **Anotasyon üretmeyin.** Veri seti Kaggle'da `build_dataset.py` ile bir kez
> üretildi; `instances_val.json` artifacts paketinden gelir. VM'de yeniden
> üretmek bölmeyi değiştirir ve float ↔ INT8 karşılaştırmasını geçersiz kılar.

Görüntüler: `quantize_yolox.py`, COCO `file_name` alanlarını doğrudan
`images/` kökünün altında arar (`images/visdrone/xxx.jpg` gibi), bu yüzden
ağacı **düzleştirmeyin**.

Kalibrasyon kaynağı sırayla seçilir:

1. `--calib-dir <klasör>` verilmişse o klasör (özyinelemeli),
2. yoksa `annotations/instances_train.json` varsa **yalnızca train** görüntüleri,
3. o da yoksa `images/` ağacının tamamı — bu durumda val görüntüleri de
   kalibrasyona karışır ve script bunu uyarı olarak yazar.

`instances_train.json` (~30 MB) Kaggle çıktısında `datasets/merged/annotations/`
altındadır; 2. yolu kullanmak için onu da kopyalayın.

## 5. Docker'ı başlat

```bash
cd ~/Vitis-AI
./docker_run.sh xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106
# docker içinde:
conda activate vitis-ai-pytorch
cd /workspace/yolox_visdrone
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
cd /workspace/yolox_visdrone
test -f datasets/merged/annotations/instances_val.json
python -c "import json;d=json.load(open('datasets/merged/annotations/instances_val.json'));print(len(d['images']),'goruntu',len(d['annotations']),'kutu',[c['name'] for c in d['categories']])"
```

Beklenen çıktı: `4483 goruntu 21681 kutu ['land_vehicle', 'sea_vehicle']`.
Sınıf adları farklı çıkarsa yanlış anotasyon dosyasıdır; devam etmeyin.

## 8. Kuantalama akışı

Ortak argümanlar her komutta aynıdır:

```bash
ARGS="--exp-file yolox_nano_visdrone.py --ckpt best_ckpt.pth --data-dir datasets/merged"
```

1. (İsteğe bağlı) DPU uyumluluk raporu — tüm katmanların DPU'ya atandığını doğrular:

```bash
python quantize_yolox.py --inspect $ARGS
```

2. Float AP@500 — Kaggle'daki değerle aynı çıkmalı (sağlama). Çıkan değeri
   `FLOAT_AP` değişkenine yazın:

```bash
python quantize_yolox.py --quant-mode float $ARGS
FLOAT_AP=BURAYA_AP500_DEGERINI_YAZIN
```

3. Kalibrasyon (PTQ, ~300 görüntü; CPU'da 10-30 dk):

```bash
python quantize_yolox.py --quant-mode calib --subset-len 300 $ARGS
```

4. INT8 AP@500 ölçümü (**4483 val görüntüsü**; CPU'da saatler sürer — `screen`
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

## 9. KV260 için derleme

```bash
bash compile_kv260.sh            # 4. arguman: sinif sayisi, varsayilan 2
```

Script, `build/compiled/yolox_nano_visdrone.xmodel` üretir ve subgraph dağılımını
yazdırır. Graph'ın tamamı **tam olarak bir DPU subgraph** olmalıdır; ek CPU/USER
subgraph varsa script başarısız olur. Ayrıca 1 giriş ve üç YOLOX çıkışı
doğrulanır; çıkış kanal sayısı `5 + sınıf sayısı` olmalıdır (2 sınıf → **7**).
Farklı bir şema kullanıyorsanız: `bash compile_kv260.sh build/quant build/compiled yolox_nano_visdrone 10`.

> Girdi **kare değildir** (896×512). Derleme öncesi
> `python quantize_yolox.py --inspect $ARGS` ile tüm katmanların DPU'ya
> atandığını doğrulayın; kare olmayan girdide bu kontrol daha da önemlidir.

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
- **`kalibrasyon goruntusu bulunamadi`** → `images/` ağacını düzleştirmişsiniz
  veya yanlış `--data-dir` vermişsiniz. Beklenen düzen 4. adımda.

## Çıktılar (bir sonraki aşamaya taşınacaklar)

- `build/compiled/yolox_nano_visdrone.xmodel` → KV260'a kopyalanacak
- `build/quant/accuracy_gate.json` → kullanılan INT8 artifactların AP kabul kaydı
- Float ve INT8 AP@500 değerleri → rapor için not edin
- Karttaki decode/NMS/merkez eşdeğerliği → `deploy/README.md` içindeki golden
  test ile ayrıca doğrulanır

Sonraki adım: [deploy/README.md](../deploy/README.md) — kart kurulumu ve C++ uygulaması.
