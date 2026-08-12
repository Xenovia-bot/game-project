# KV260 için termal / gri-ton gemi tespiti

Bu depo yalnızca tek sınıflı (`ship`) YOLOX-Tiny fine-tune hattını içerir.
Hedef görüntüler kıyı, iskele veya gemi güvertesi seviyesinden; RGB, gri-ton
ya da termal olabilir. Eski VisDrone/araç veri ve artefaktları kaldırıldı.

## Veri seti durumu

`datasets/ship_merged` altındaki altı kaynak oturum bazında ayrılmıştır:

| Split | Görüntü | Kutu |
| --- | ---: | ---: |
| train | 34.408 | 83.904 |
| val | 7.622 | 17.168 |
| test | 7.600 | 18.987 |

Dağılım yaklaşık **%69,3 / %15,4 / %15,3**'tür. Dosya adı kesişimi ve aynı
oturumun birden fazla split'e düşmesi sıfır olmalıdır. Test seti eğitim,
eşik seçimi ve PTQ kalibrasyonunda kullanılmaz.

Mevcut çıktıların denetimi / gerekirse güvenli yeniden bölünmesi:

```powershell
python tools/build_ship_dataset.py --repartition-existing --out datasets/ship_merged --dry-run
python tools/build_ship_dataset.py --repartition-existing --out datasets/ship_merged
```

Ham altı arşiv mevcutsa ilk üretim için:

```powershell
python tools/build_ship_dataset.py --data-dir <ham-veri-klasoru> --out datasets/ship_merged --images-out datasets/ship_merged/images
```

`WUTDet` kaynağının veri lisansı ayrıca doğrulanmalıdır; lisans netleşmeden
ticari/dağıtılan modelde bu kaynağı `--skip wutdet` ile dışarıda bırakın.

## Fine-tune

Kaggle'a altı ham zip'i hazırlayın, ardından `training/kaggle_ship_yolox.ipynb`
not defterini çalıştırın:

```powershell
python tools/prepare_ship_kaggle_upload.py --data-dir <ham-veri-klasoru>
```

Not defteri veri birleştirmeyi, YOLOX-Tiny eğitimini, COCO değerlendirmesini
ve taşıma paketini üretir. `training/exps/yolox_tiny_ship.py` eğitimde
gri-ton/termal dayanıklılığı için yalnızca train tarafında piksel
augmentasyonları uygular; val/test görüntüleri değiştirilmez.

## Klasörler

- `tools/build_ship_dataset.py` — altı kaynağı tek sınıflı COCO'ya birleştirir;
  kopya ve oturum sızıntısı kapıları içerir.
- `training/exps/yolox_tiny_ship.py` — YOLOX-Tiny deney tanımı.
- `training/kaggle_ship_yolox.ipynb` — Kaggle fine-tune akışı.
- `tests/test_build_ship_dataset.py` — split, oturum ve sınıf eşleme testleri.
- `quantize/`, `deploy/` — sonraki KV260 aşaması için korunmuş eski referanslar.

## KV260 notu

`quantize/` ve `deploy/` içindeki mevcut içerik eski iki sınıflı VisDrone
projesinden kalmadır. Kart aşamasına geçmeden önce bunlar tek sınıf, 512×512
gemi deney dosyası ve standart COCO değerlendirmesi için port edilmelidir.
PTQ kalibrasyonu yalnızca `instances_train.json` ile; model/eşik seçimi val
ile; nihai raporlama ise test ile yapılmalıdır.
