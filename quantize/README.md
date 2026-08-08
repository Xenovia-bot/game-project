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
    ├── visdrone2coco.py             # projedeki tools/ klasöründen
    ├── best_ckpt.pth                # Kaggle artifacts.zip içinden
    ├── yolox_nano_visdrone.py       # Kaggle artifacts.zip içinden
    ├── visdrone_eval.py             # resmi DET tarzı AP@500 değerlendirici
    ├── YOLOX_COMMIT.txt             # iki ortamda aynı kaynak sürümü
    └── datasets/visdrone_coco/
        ├── annotations/instances_val.json   # 7. adımda üretilecek
        ├── val_images/              # VisDrone2019-DET-val/images kopyası
        └── train_images/            # isteğe bağlı: ~300 train görüntüsü (kalibrasyon)
```

VisDrone **val** paketini (~70 MB) VM'e indirip açın:

```bash
cd ~/Vitis-AI/yolox_visdrone
unzip VisDrone2019-DET-val.zip
mkdir -p datasets/visdrone_coco
cp -r VisDrone2019-DET-val/images datasets/visdrone_coco/val_images
```

Kalibrasyonun eğitim dağılımıyla yapılması tercih edilir: Kaggle'dan ~300 train
görüntüsünü `datasets/visdrone_coco/train_images/` altına kopyalayabilirsiniz.
Kopyalamazsanız script otomatik olarak val görüntüleriyle kalibre eder (kabul
edilebilir bir yedek yoldur).

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

## 7. `instances_val.json`'u üret

`--classes 2` **zorunlu**: Kaggle'da eğitim bu şemayla yapıldı. Eşleşmezse
`quantize_yolox.py` kategori sayısı kapısında durur.

```bash
python visdrone2coco.py --classes 2 \
    --image-dir VisDrone2019-DET-val/images \
    --anno-dir  VisDrone2019-DET-val/annotations \
    --output    datasets/visdrone_coco/annotations/instances_val.json
```

## 8. Kuantalama akışı

Ortak argümanlar her komutta aynıdır:

```bash
ARGS="--exp-file yolox_nano_visdrone.py --ckpt best_ckpt.pth --data-dir datasets/visdrone_coco"
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

4. INT8 AP@500 ölçümü (548 val görüntüsü; CPU'da uzun sürebilir, sabır):

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
- **PTQ çok yavaş** → normaldir (CPU). `--subset-len` değerini 200'e düşürmek
  kabul edilebilir.

## Çıktılar (bir sonraki aşamaya taşınacaklar)

- `build/compiled/yolox_nano_visdrone.xmodel` → KV260'a kopyalanacak
- `build/quant/accuracy_gate.json` → kullanılan INT8 artifactların AP kabul kaydı
- Float ve INT8 AP@500 değerleri → rapor için not edin
- Karttaki decode/NMS/merkez eşdeğerliği → `deploy/README.md` içindeki golden
  test ile ayrıca doğrulanır

Sonraki adım: [deploy/README.md](../deploy/README.md) — kart kurulumu ve C++ uygulaması.
