#!/usr/bin/env python3
"""Zip icindeki asiri uzun dosya adlarini kisaltarak arsivi yeniden paketler.

Neden: Kaggle, arsiv girisi adi 248 baytı asan dosyalari reddediyor
("Archive entry: ... name is longer than 248 bytes"). Roboflow'dan gecmis
veri setlerinde orijinal uzun ad + eklenen hash bu siniri asabiliyor.

Kritik nokta: bir goruntunun adi degistiginde **etiketinin adi da** ayni
sekilde degismelidir, yoksa images/ ve labels/ eslesmesi kirilir ve o ornek
sessizce etiketsiz kalir. Bu yuzden kisaltma dosya bazinda degil **taban ad
(stem) bazinda** yapilir: ayni stem'e sahip tum girisler ayni yeni stem'i alir.

Kullanim:
    python tools/shorten_zip_names.py girdi.zip cikti.zip
    python tools/shorten_zip_names.py girdi.zip --in-place
"""

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

#: Kaggle siniri 248; guvenli pay birakiyoruz.
MAX_ENTRY_BYTES = 200


def split_entry(name):
    """'a/b/c.jpg' -> ('a/b/', 'c', '.jpg')"""
    head, _, base = name.rpartition("/")
    prefix = head + "/" if head else ""
    stem, dot, ext = base.rpartition(".")
    if not dot:
        return prefix, base, ""
    return prefix, stem, "." + ext


def plan_renames(names, max_bytes=MAX_ENTRY_BYTES):
    """Uzun girisler icin stem -> yeni stem eslemesi uretir.

    Ayni stem birden fazla uzantiyla gorunebilir (x.jpg + x.txt); hepsi ayni
    yeni stem'i alir. Yeni ad deterministiktir: kisaltilmis on ek + stem'in
    sha1 ozetinin ilk 10 karakteri.
    """
    longest_by_stem = {}
    for name in names:
        prefix, stem, ext = split_entry(name)
        total = len(name.encode("utf-8"))
        if total <= max_bytes:
            continue
        # Ayni stem birden cok yerde olabilir; en uzun yola gore karar ver.
        current = longest_by_stem.get(stem)
        if current is None or total > current[0]:
            longest_by_stem[stem] = (total, prefix, ext)

    renames = {}
    for stem, (total, prefix, ext) in longest_by_stem.items():
        digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:10]
        # Yeni ada ayrilabilecek bayt: sinir - (prefix + '.' + ext)
        budget = max_bytes - len(prefix.encode("utf-8")) - len(ext.encode("utf-8"))
        keep = max(8, budget - len(digest) - 1)
        head = stem.encode("utf-8")[:keep].decode("utf-8", "ignore")
        renames[stem] = f"{head}_{digest}"
    return renames


def apply_renames(name, renames):
    prefix, stem, ext = split_entry(name)
    if stem not in renames:
        return name
    return f"{prefix}{renames[stem]}{ext}"


def repack(source, target, max_bytes=MAX_ENTRY_BYTES):
    with zipfile.ZipFile(source) as src:
        infos = src.infolist()
        names = [i.filename for i in infos if not i.filename.endswith("/")]
        renames = plan_renames(names, max_bytes)
        if not renames:
            print("Kisaltilacak ad yok; arsiv oldugu gibi birakildi.")
            return 0

        print(f"{len(renames)} taban ad kisaltilacak:")
        for stem, new in list(renames.items())[:5]:
            print(f"  {stem[:70]}...  ->  {new[:50]}...")

        written = 0
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as out:
            for info in infos:
                if info.filename.endswith("/"):
                    continue
                new_name = apply_renames(info.filename, renames)
                if len(new_name.encode("utf-8")) > max_bytes:
                    raise SystemExit(
                        f"HATA: kisaltma yetmedi ({len(new_name.encode())} bayt): "
                        f"{new_name}"
                    )
                out.writestr(new_name, src.read(info.filename))
                written += 1
        print(f"{written} giris yazildi -> {target}")
    return len(renames)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("target", nargs="?", default=None)
    parser.add_argument("--in-place", action="store_true",
                        help="kaynagin uzerine yaz (once .bak yedegi alinir)")
    parser.add_argument("--max-bytes", type=int, default=MAX_ENTRY_BYTES)
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"HATA: bulunamadi: {source}")

    if args.in_place:
        temp = source.with_suffix(".repacked.zip")
        count = repack(source, temp, args.max_bytes)
        if count:
            backup = source.with_suffix(".zip.bak")
            shutil.move(str(source), str(backup))
            shutil.move(str(temp), str(source))
            print(f"yerinde guncellendi; yedek: {backup}")
        else:
            temp.unlink(missing_ok=True)
    else:
        target = Path(args.target or source.with_name(source.stem + "_short.zip"))
        repack(source, target, args.max_bytes)


if __name__ == "__main__":
    main()
