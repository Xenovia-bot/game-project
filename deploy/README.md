# KV260 Kurulumu ve C++ Uygulaması (PetaLinux + VART)

Amaç: Vitis AI 3.0 hazır imajını karta kurmak, derlenmiş `.xmodel`'i kopyalamak
ve video üzerinde tespit + **merkez noktası** üreten C++ uygulamasını çalıştırmak.

## 1. SD kart imajını hazırla

1. Vitis AI 3.0 KV260 hazır imajını indirin (AMD hesabı gerektirir):
   - Dosya: `xilinx-kv260-dpu-v2022.2-v3.0.0.img.gz`
   - Link: [KV260 board image (Vitis AI 3.0 Quickstart)](https://xilinx.github.io/Vitis-AI/3.0/html/docs/quickstart/mpsoc.html)
     sayfasındaki *"Download the Vitis AI pre-built SD card image"* bölümünden.
2. [Balena Etcher](https://etcher.balena.io/) ile en az 16 GB'lık SD karta yazın
   (`.img.gz` dosyasını doğrudan seçebilirsiniz).
3. SD kartı KV260'a takın; ethernet ve güç bağlayın.

## 2. Karta bağlan

- **Seri port**: USB-UART üzerinden, ardışık COM portlarının en düşüğü,
  115200 baud / 8N1. Açılışta login: `root`, şifre: `root`.
- **IP öğren/ayarla**:

```bash
ifconfig                       # DHCP varsa IP burada
ifconfig eth0 192.168.1.50     # elle vermek icin
```

- **SSH** (bilgisayardan): `ssh root@<KART_IP>` (şifre `root`).

## 3. DPU'yu doğrula

```bash
xdputil query
```

Çıktıda `DPU Arch: DPUCZDX8G ... B4096` benzeri bir satır görmelisiniz.
Bu, derlemede kullandığımız `KV260/arch.json` ile eşleşir. Görmüyorsanız SD
imajı yanlış demektir.

## 4. Dosyaları karta kopyala

Bilgisayardan (veya VM'den):

```bash
scp -r deploy root@<KART_IP>:~/yolox_visdrone
scp build/compiled/yolox_nano_visdrone.xmodel root@<KART_IP>:~/yolox_visdrone/
scp test_video.mp4 root@<KART_IP>:~/yolox_visdrone/
```

Test videosu notu: elinizde drone videosu yoksa val görüntülerinden hızlıca bir
video üretebilirsiniz (bilgisayarda, ffmpeg ile). Kaynağı tek bir alandan seçin
— karışık kaynaklı bir slayt gösterisi takip katmanını anlamsız kılar:

```bash
ffmpeg -framerate 2 -pattern_type glob -i 'datasets/merged/images/visdrone/*.jpg' \
       -vf "scale=1920:-2" -c:v mjpeg -q:v 5 test_video.avi
```

## 5. Uygulamayı derle

### Seçenek A - Kart üzerinde (önerilen, en kolay)

Hazır imajda g++ ve OpenCV geliştirme dosyaları mevcuttur:

```bash
cd ~/yolox_visdrone
bash build.sh
```

### Seçenek B - Oracle VM'de cross-compile

VM'de (docker dışında) PetaLinux SDK'yı bir kez kurun:

```bash
cd ~/Vitis-AI/board_setup/mpsoc
chmod +x host_cross_compiler_setup.sh
./host_cross_compiler_setup.sh          # ~/petalinux_sdk_2022.2 altina kurulur
source ~/petalinux_sdk_2022.2/environment-setup-cortexa72-cortexa53-xilinx-linux
cd <proje>/deploy && bash build.sh      # CXX otomatik SDK derleyicisi olur
scp yolox_visdrone_demo root@<KART_IP>:~/yolox_visdrone/
```

## 6. Çalıştır

```bash
cd ~/yolox_visdrone
./yolox_visdrone_demo yolox_nano_visdrone.xmodel test_video.avi \
    out.avi centers.csv --conf 0.15 --nms 0.45
```

Hızlı deneme için `--max-frames 100` ekleyebilirsiniz.
Uygulama başlangıçta xmodel'in yalnızca bir DPU subgraph, bir NHWC giriş ve
stride 8/16/32 olan üç çıkış içerdiğini doğrular. Çıkış kanal sayısı
`5 + sınıf sayısı` olmalıdır; 2 sınıflı (`land_vehicle`/`sea_vehicle`) modelde **7**.

Varsayılan `--conf` **0.15**'tir: recall odaklı çalışma noktası. Yanlış
pozitifleri takip katmanı elemek üzere tasarlanmıştır; takip henüz
eklenmediyse görsel kontrol için `--conf 0.30` kullanabilirsiniz.

### Takip (tracking)

Uygulama, tespitleri kareler arasında IoU ile eşleştirip her nesneye kalıcı bir
`track_id` verir. Bunun iki işlevi var:

1. **Recall** — bir hedef 30 karenin 18'inde tespit edilse bile iz kesintisiz
   kalır. İz bazında recall, kare bazındakinden belirgin şekilde yüksektir.
2. **Yanlış pozitif filtresi** — bir iz `--track-n-init` (varsayılan 3) kare
   görülmeden çizilmez/loglanmaz. Bu sayede güven eşiği 0.15'e indirilebilir.

ByteTrack'in iki aşamalı eşleştirmesi kullanılır: önce yüksek skorlu
tespitlerle eşleştirilir, ardından eşleşmemiş izler düşük skorlu tespitlerle
kurtarılmaya çalışılır (kısmi kapanma senaryosu). Düşük skorlu tespitlerden
**yeni iz açılmaz**. Hareket modeli sabit hızdır (Kalman yok).

| Bayrak | Varsayılan | Anlamı |
| --- | --- | --- |
| `--track-n-init` | 3 | İz kaç eşleşmeden sonra onaylanır |
| `--track-max-age` | 30 | İz kaç kare görünmezse silinir |
| `--no-track` | kapalı | Takibi devre dışı bırakır, ham tespitleri yazar |

Takip mantığı [`src/tracker.hpp`](src/tracker.hpp) içinde VART/OpenCV'den
bağımsız tutulmuştur; kart dışında derlenip test edilebilir:

```bash
g++ -std=c++17 -O2 -I deploy/src deploy/tests/test_tracker.cpp -o tracker_test
./tracker_test
```

### Çıktılar

- `out.avi` - kutular, sınıf etiketleri ve **kırmızı merkez noktaları** çizilmiş video
- `centers.csv` - satır formatı:
  `frame,track_id,class_id,class_name,score,cx,cy`
  (merkezler orijinal video çözünürlüğündedir: `cx=(x1+x2)/2`, `cy=(y1+y2)/2`.
  `track_id` aynı nesneyi kareler boyunca izler; `--no-track` ile -1 olur)
- Konsol özeti - ortalama ön-işleme / DPU / son-işleme süreleri ve uçtan uca FPS

Sonuçları bilgisayara geri almak için:

```bash
scp root@<KART_IP>:~/yolox_visdrone/out.avi .
scp root@<KART_IP>:~/yolox_visdrone/centers.csv .
```

## 7. Python ↔ C++ eşdeğerlik testi (zorunlu kabul testi)

Aynı DPU INT8 çıkışlarından Python ve C++ tarafında elde edilen kutu, sınıf,
skor ve merkezleri karşılaştırmak için ilk kareyi ham tensorlarla birlikte dökün:

```bash
# KV260 üzerinde
./yolox_visdrone_demo yolox_nano_visdrone.xmodel test_video.avi \
    golden.avi golden_centers.csv --conf 0.15 --nms 0.45 \
    --max-frames 1 --dump-first-frame golden
```

`golden/` klasörünü proje bilgisayarına alın ve bağımsız Python referansını
çalıştırın:

```bash
scp -r root@<KART_IP>:~/yolox_visdrone/golden .
python tools/verify_kv260_golden.py --dump-dir golden
```

Komut `OK` yazmadan performans ölçümüne geçmeyin. Test, BGR→INT8 giriş
dönüşümünü ve C++ decode + sınıf bazlı NMS + letterbox ters dönüşümü +
`cx/cy` hesabını gerçek kart çıktılarıyla denetler.

## 8. Sorun giderme

- **`fingerprint mismatch` / model yüklenmiyor** → xmodel farklı bir arch ile
  derlenmiş; VM'de `compile_kv260.sh`'nin KV260 arch.json kullandığını doğrulayın.
- **mp4 açılmıyor veya çok yavaş** → mp4 çözme ARM CPU'da yapılır; videoyu
  MJPEG/AVI'ye dönüştürün: `ffmpeg -i video.mp4 -c:v mjpeg -q:v 5 video.avi`
- **FPS düşük** → önce özet satırlarına bakın: DPU süresi düşük ama uçtan uca
  yavaşsa darboğaz video çözme/yazmadır (çözünürlüğü düşürün). DPU süresi
  yüksekse modeli 416×416 ile yeniden eğitip/kuantalayıp derleyin.
- **`libvart-runner.so` bulunamadı** → yanlış imaj; Vitis AI 3.0 KV260 imajını
  kullanın (VART hazır gelir).
- **Ekran gerekmiyor** → uygulama dosyaya yazar; monitör/DISPLAY ayarı gerekmez.
