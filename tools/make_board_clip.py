#!/usr/bin/env python3
"""Bir goruntuyu veya videoyu KV260'da oynatilabilir MJPEG AVI'ye cevirir.

Neden gerekli:

1. **Kodek.** Kartin OpenCV'si GStreamer kullaniyor ve H.264/`qtdemux`
   eklentisi kurulu degil; `.mp4` acilmiyor ("HATA: video acilamadi").
   MJPEG AVI, OpenCV'nin kendi kodegiyle cozulur, eklenti istemez.

2. **Takip katmani 3 kare ister.** `tracker.hpp` bir izi ancak `n_init=3`
   eslesmeden sonra "confirmed" yapar ve `main.cpp` yalnizca onaylanmis
   izleri cizer. Tek karelik girdide hicbir kutu cikmaz. Bu yuzden tek
   goruntu verildiginde kare `--frames` kadar tekrarlanir.

3. **Kare aramasi kayabiliyor.** `cv2.CAP_PROP_POS_FRAMES` ile MP4'te
   arama en yakin anahtar kareye kayiyor; bu script kareleri sirayla
   okuyup atlar, boylece `--start` tam istenen kareye denk gelir.

Kullanim:
  python tools/make_board_clip.py foto.jpg
  python tools/make_board_clip.py video.mp4 --start 307 --frames 40
"""

import argparse
from pathlib import Path

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def scaled_size(size, max_width):
    """Genisligi `max_width`e indirir, en-boy oranini korur (cift sayiya yuvarlar)."""
    width, height = size
    if not max_width or width <= max_width:
        return size
    new_h = max(2, round(height * max_width / width / 2) * 2)
    return (max_width, new_h)


def write_clip(frames_iter, out_path, fps, size, quality=None):
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size
    )
    if not writer.isOpened():
        raise SystemExit(f"HATA: yazici acilamadi: {out_path}")
    if quality is not None:
        writer.set(cv2.VIDEOWRITER_PROP_QUALITY, quality)
    written = 0
    for frame in frames_iter:
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        writer.write(frame)
        written += 1
    writer.release()
    return written


def from_image(path, count):
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"HATA: goruntu okunamadi: {path}")
    height, width = image.shape[:2]
    return (image for _ in range(count)), (width, height)


def from_video(path, start, count):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"HATA: video acilamadi: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frames():
        # Sirali okuma: arama anahtar kareye kayabiliyor (bkz. modul basligi).
        index = 0
        taken = 0
        while taken < count:
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
            if index < start:
                continue
            yield frame
            taken += 1
        cap.release()

    return frames(), (width, height)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("girdi", type=Path, help="goruntu veya video dosyasi")
    parser.add_argument("-o", "--out", type=Path, default=None,
                        help="cikti .avi (varsayilan: <girdi adi>_kart.avi)")
    parser.add_argument("--start", type=int, default=1, metavar="KARE",
                        help="videoda baslangic karesi, 1 tabanli (varsayilan 1)")
    parser.add_argument("--frames", type=int, default=30, metavar="N",
                        help="kac kare yazilacak. Goruntu girdisinde kare bu "
                             "kadar tekrarlanir; takip 3 kare istedigi icin "
                             "en az 5 verin (varsayilan 30)")
    parser.add_argument("--fps", type=float, default=25.0)
    # Model girdiyi zaten 896x512'ye olcekliyor; daha genis kaynak gondermek
    # tespite bir sey katmaz, sadece dosyayi ve transferi buyutur.
    parser.add_argument("--max-width", type=int, default=960, metavar="PX",
                        help="cikti genisligini bu degere indirir (en-boy "
                             "korunur). Model 896x512'ye olceklediginden daha "
                             "buyugu bosuna. 0 = olcekleme yok "
                             "(varsayilan 960)")
    parser.add_argument("--quality", type=int, default=None, metavar="0-100",
                        help="MJPEG kalitesi; dusurmek dosyayi kucultur")
    args = parser.parse_args()

    if not args.girdi.is_file():
        raise SystemExit(f"HATA: dosya yok: {args.girdi}")
    if args.frames < 1:
        raise SystemExit("HATA: --frames en az 1 olmali")
    if args.start < 1:
        raise SystemExit("HATA: --start 1 tabanlidir, en az 1 olmali")

    out = args.out or args.girdi.with_name(args.girdi.stem + "_kart.avi")
    is_image = args.girdi.suffix.lower() in IMAGE_EXTS

    if is_image:
        if args.frames < 5:
            print("UYARI: takip 3 kare ustuste eslesme istiyor; %d kare az "
                  "olabilir." % args.frames)
        frames, size = from_image(args.girdi, args.frames)
    else:
        frames, size = from_video(args.girdi, args.start, args.frames)

    out_size = scaled_size(size, args.max_width)
    written = write_clip(frames, out, args.fps, out_size, args.quality)
    if written == 0:
        raise SystemExit(
            "HATA: hic kare yazilmadi. Video icin --start dosyanin kare "
            "sayisini asmis olabilir."
        )
    mb = out.stat().st_size / 1e6
    kind = "goruntu (tekrarlandi)" if is_image else f"video (kare {args.start}'ten)"
    print(f"{out}  <-  {kind}")
    note = "" if out_size == size else f"  (kaynak {size[0]}x{size[1]} kucultuldu)"
    print(f"  {out_size[0]}x{out_size[1]}  {written} kare  {mb:.1f} MB{note}")
    if mb > 25:
        print("  UYARI: 25 MB uzeri transferler dogrudan Ethernet baglantisinda"
              " kopabiliyor; --frames azaltin veya --quality dusurun.")
    print(f"\nKarta gonder:\n  pscp -scp {out} root@192.168.137.50:/home/root/yolox_visdrone/")


if __name__ == "__main__":
    main()
