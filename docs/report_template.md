# Proje Raporu: KV260 Üzerinde YOLOX-Nano ile İnsan/Taşıt Tespiti, Takip ve Merkez Noktası Çıkarımı

> Bu şablondaki `...` alanlarını kendi ölçümlerinizle doldurun.
> Ölçümün nereden alınacağı her bölümde belirtilmiştir.

## 1. Özet

VisDrone2019-DET veri setiyle fine-tune edilen DPU-uyumlu YOLOX-Nano modeli,
Vitis AI 3.0 zinciriyle INT8'e kuantalanarak Kria KV260 üzerindeki DPUCZDX8G
(B4096) hızlandırıcısında çalıştırılmıştır. Uygulama video karelerinde
**insan ve taşıt** tespit eder, tespitleri kareler arasında takip eder ve her
nesnenin merkez noktasını (cx = (x1+x2)/2, cy = (y1+y2)/2) hesaplayıp
görüntüye çizer ve CSV'ye loglar.

## 2. Sistem yapılandırması

| Bileşen | Değer |
| --- | --- |
| Kart | Kria KV260 Vision AI Starter Kit |
| DPU | DPUCZDX8G B4096 (Vitis AI 3.0 hazır imajı) |
| Model | YOLOX-Nano (depth 0.33, width 0.25, depthwise) |
| DPU uyarlamaları | SiLU→ReLU, Focus→DPUFocus (sabit conv ile space-to-depth) |
| Giriş boyutu | 896×512 (16:9, letterbox, 114 dolgu) |
| Sınıflar | 2: `person`, `vehicle` |
| Eğitim | Kaggle GPU (`...` epoch, batch `...`, init: Megvii COCO) |
| YOLOX kaynak sürümü | `6ddff4824372906469a7fae2dc3206c7aa4bbaee` |
| Kuantalama | Vitis AI 3.0 PTQ, `...` kalibrasyon görüntüsü |
| Çalışma noktası | `--conf 0.15`, `--nms 0.45`, takip `n_init=3`, `max_age=30` |

### Tasarım gerekçeleri (ölçüme dayalı)

| Karar | Gerekçe |
| --- | --- |
| Girdi 896×512 (640×640 yerine) | 1920×1080 → 640×640 letterbox'ta kanvasın %44'ü gri dolgu; etkin ölçek 0.333. 896×512'de 0.467 — %40 daha yüksek çözünürlük, %12 daha fazla hesap |
| 2 sınıf | Görsel olarak benzer kara taşıtlarını birleştirmek AP'yi artırır; sistem hedefi zaten insan/taşıt merkezleri |
| 40 epoch | Referans çalıştırmada model epoch ~40'ta doydu (45→80 arası AP@0.50 yalnızca +0.006) |
| Düşük güven eşiği + takip | Recall odaklı: bir iz 3 kare görülmeden raporlanmaz, bu yüzden eşik 0.15'e indirilebilir |

## 3. Doğruluk sonuçları

Kaynaklar: float → Kaggle eval hücresi; INT8 → `quantize_yolox.py --quant-mode test`.
Her iki ölçüm de resmi ignore filtresi, görüntü başına global top-500 ve
VisDrone VOC AP hesabını kullanır.

| Model | AP@[.50:.95] | AP@0.50 | AP@0.75 | En iyi F1 | P | R |
| --- | --- | --- | --- | --- | --- | --- |
| Float (Kaggle, val) | ... | ... | ... | ... | ... | ... |
| INT8 PTQ (VM, val) | ... | ... | ... | ... | ... | ... |
| Fark | ... | ... | ... | ... | ... | ... |

Sınıf bazlı (float):

| Sınıf | AP | AP50 | F1 |
| --- | --- | --- | --- |
| person | ... | ... | ... |
| vehicle | ... | ... | ... |

**Kuantalama sadakati**: INT8 modeli float modelin AP'sinin `...` %'sini
korumuştur. Kabul kapısı: mutlak AP kaybı ≤ 0.02.

Yorum: `...` (fast_finetune gerekti mi, hangi sınıf daha çok etkilendi)

## 4. Performans sonuçları (KV260)

Kaynak: `yolox_visdrone_demo` konsol özeti.

| Metrik | Değer |
| --- | --- |
| Ortalama ön-işleme | ... ms |
| Ortalama DPU çıkarımı | ... ms |
| Ortalama son-işleme (decode+NMS+takip+merkez) | ... ms |
| Uçtan uca FPS | ... |
| Test videosu | ... (çözünürlük, süre, kare sayısı) |

Hedef 30 FPS (33 ms/kare). Darboğaz: `...` (DPU mu, video giriş/çıkışı mı?)

## 5. Takip sonuçları

| Metrik | Değer |
| --- | --- |
| Toplam oluşturulan iz | ... |
| Onaylanan iz (`n_init`=3 geçen) | ... |
| Ortalama iz ömrü (kare) | ... |
| Kare bazında ortalama tespit | ... |
| Kare bazında ortalama onaylı iz | ... |

Yorum: takibin yanlış pozitifleri ne kadar elediği ve kaçırılan kareleri ne
kadar doldurduğu: `...`

> Not: iz bazında recall'ın sayısal ölçümü, iz kimlikli bir video veri seti
> (VisDrone-VID / MOT) gerektirir. Bu çalışmada ölçülmemiştir.

## 6. Merkez noktası çıktısı

`centers.csv`'den örnek satırlar:

```csv
frame,track_id,class_id,class_name,score,cx,cy
...
```

Örnek kare görüntüsü (`out.avi`'den): `...`

Golden eşdeğerlik testi:

```text
python tools/verify_kv260_golden.py --dump-dir golden
...
```

Sonuç: `...` tespit; maksimum koordinat farkı `...` px, skor farkı `...`.

## 7. Tasarım uzayı (isteğe bağlı ama güçlü)

Aynı donanımda ölçülen doğruluk/hız takası:

| Yapılandırma | AP50 | F1 | KV260 FPS |
| --- | --- | --- | --- |
| Nano 896×512, tam kare | ... | ... | ... |
| Nano 640×640, tam kare (referans, 10 sınıf) | 0.2205 | ... | ... |
| Nano 896×512 + 2×2 tiling | ... | ... | ... |

Tiling ölçümü: `training/eval_tiled.py`.

## 8. Karşılaşılan sorunlar ve çözümler

- `...`

## 9. Sonuç ve gelecek çalışmalar

- `...` (ör. iz bazında recall ölçümü için VisDrone-VID, daha alçak irtifa
  görüntüsüyle ikinci fine-tune, tiling'in FPS bütçesine sığdırılması)

## Ek: literatür bağlamı

| Referans | Sonuç (VisDrone) |
| --- | --- |
| YOLOv11 baseline / YOLO-Drone (2025) | F1 = 0.347 / 0.352 |
| DBNet, VisDrone-DET2021 kazananı | AP@0.50 = 65.3 (ensemble + 1536px + tiling) |
| SOTA AP@[.50:.95] (2023) | 42.2 |

Bu çalışma 0.9M parametreli, INT8 kuantalanmış, gömülü bir DPU'da gerçek
zamanlı çalışan bir modelle karşılaştırılmalıdır.
