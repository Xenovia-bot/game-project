# KV260'da Test Çalıştırma — Adım Adım

Elinde bir görüntü veya video var, modeli kartta denemek istiyorsun.
Bu dosya baştan sona ne yapacağını söyler.

**Her komut bloğunun başlığında nerede çalıştıracağın yazıyor.**

| etiket | neresi | prompt nasıl görünür |
| --- | --- | --- |
| `[Windows]` | kendi bilgisayarın, PowerShell | `PS C:\Users\emrez\proje>` |
| `[KART]` | KV260, PuTTY penceresi | `root@xilinx-kv260-starterkit-20222:~#` |

Sabitler: kart **192.168.137.50** · Windows Ethernet **192.168.137.1** · şifre **root**

---

## 0. Kart açık mı, ağ duruyor mu

Kartı yeni açtıysan IP'yi tekrar vermen gerekir — `ifconfig` kalıcı değildir.

### [KART]

```bash
ifconfig eth0
```

`inet 192.168.137.50` satırını görmüyorsan:

```bash
ifconfig eth0 192.168.137.50 netmask 255.255.255.0 up
```

### [Windows]

```bash
ping 192.168.137.50
```

Cevap gelmiyorsa durup ağı çöz; kalan adımlar buna bağlı.

---

## 1. Test dosyanı karta uygun hale getir

Kartın OpenCV'si **mp4 açamıyor** (GStreamer'da H.264 eklentisi yok) ve
takip katmanı bir kutuyu çizmek için **3 kare üst üste** eşleşme istiyor.
İkisini de şu araç halleder:

### [Windows] Tek görüntü için

```bash
python tools/make_board_clip.py test.jpg
```

Görüntüyü 30 kare tekrarlayıp `test_kart.avi` üretir.

### [Windows] Video için

```bash
python tools/make_board_clip.py video.mp4 --frames 300
```

Belirli bir kareden başlatmak istersen:

```bash
python tools/make_board_clip.py video.mp4 --start 307 --frames 40
```

---

## 2. Karta gönder

### [Windows]

```bash
pscp -scp test_kart.avi root@192.168.137.50:/home/root/yolox_visdrone/
```

---

## 3. Çalıştır

### [KART]

```bash
cd ~/yolox_visdrone
```

```bash
./yolox_visdrone_demo yolox_tiny_visdrone.xmodel test_kart.avi out.avi centers.csv
```

Çıktının sonunda özet gelir:

```
ort. on-isleme   : ... ms
ort. DPU         : ... ms      <- donanimin temiz sayisi
ort. son-isleme  : ... ms
uctan uca FPS    : ...         <- gercek urun hizi
toplam tespit    : ...
```

### Faydalı seçenekler

| seçenek | ne yapar | ne zaman |
| --- | --- | --- |
| `--conf 0.45` | güven eşiğini yükseltir | yanlış pozitif çoksa (varsayılan 0.15) |
| `--no-track` | takibi kapatır, ham tespitleri gösterir | "bu kutu modelden mi takipten mi" ayrımı için |
| `--max-frames 50` | ilk N kareyi işler | hızlı deneme |
| `--track-max-age 10` | iz kaç kare kayıpken yaşar | iz çok uzun sürüyorsa |

---

## 4. Sonuçları bilgisayara çek

### [Windows]

```bash
pscp -scp root@192.168.137.50:/home/root/yolox_visdrone/out.avi out\
```

```bash
pscp -scp root@192.168.137.50:/home/root/yolox_visdrone/centers.csv out\
```

Birden fazla dosyayı joker ile çekeceksen `-unsafe` ekle:

```bash
pscp -scp -unsafe root@192.168.137.50:/home/root/yolox_visdrone/out_*.avi out\
```

`out.avi` MJPEG'dir — **VLC** ile aç, Media Player açmayabilir.

`centers.csv` sütunları: `frame,track_id,class_id,class_name,score,cx,cy`
(`cx`,`cy` = merkez noktası, projenin asıl çıktısı).

---

## 5. Sonucu yorumlarken

- **Kare başına kaç kutu var?** `centers.csv`'de aynı `frame` değerini
  sayarak bak. Videoda "çok kutu" hissi genelde zaman içindeki toplamdan
  gelir, aynı anda 1-2 kutu vardır.
- **Takip mi tespit mi?** Aynı `--no-track` ile koştur, satır sayılarını
  karşılaştır. Fark, takibin elediği tutarsız tespittir.
- **Kutu yanlış yerdeyse** bu takip hatası değil model hatasıdır: çizim
  kuralı katı (`confirmed && age == 0`), yani her kutu hem ≥3 karedir
  onaylanmış hem de o karede taze tespitle eşleşmiş demektir.
- **Alan uyumuna dikkat.** Model **RGB havadan** görüntüyle eğitildi.
  Termal, yer seviyesi veya eğik açı görüntülerde sonuç ürün kalitesini
  temsil etmez.

---

## Bilinen tuzaklar

| belirti | sebep | çözüm |
| --- | --- | --- |
| `HATA: video acilamadi` | mp4, kartta H.264 kodeği yok | `make_board_clip.py` ile AVI'ye çevir |
| `: invalid option name: pipefail` | betik CRLF satır sonlarıyla gitti | kartta `sed -i 's/\r$//' build.sh` |
| `sftp-server: No such file` | yeni scp/pscp SFTP deniyor | `scp -O` veya `pscp -scp` |
| `Cannot create file` (joker) | pscp joker güvenliği | `-unsafe` ekle |
| `toplam tespit : 0` tek karede | takip 3 kare istiyor | `--frames` en az 5, tercihen 30 |
| `deploy: No such file` | yanlış dizin | `cd C:\Users\emrez\proje` |

---

## Referans: kartta ölçülmüş değerler (2026-08-10)

| | değer |
| --- | --- |
| DPU | 20,0 ms (**50 FPS**), içerikten bağımsız |
| ön-işleme | 31–36 ms ← **darboğaz burası** |
| son-işleme | 0,2 ms |
| uçtan uca | 17,5–19,2 FPS |
| DPU mimarisi | DPUCZDX8G_ISA1_B4096 @300 MHz |
| golden test | Python ≡ C++, ~1e-5 px |

Ayrıntı ve gerekçeler: `HANDOFF_CLAUDE.md` §5.
