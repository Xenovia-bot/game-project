#!/usr/bin/env python3
"""QAT'in bu makinede kac saat surecegini ONCEDEN olcer.

Neden ayri bir script: QAT'i port etmek birkac yuz satir is. Bu is ancak
egitim adiminin makul surede kostugu makinelerde anlamli. VM'de GPU yok
(VirtualBox NVIDIA passthrough yapmaz), yani QAT CPU'da koser ve tek
belirleyici sayi **saniye/iterasyon**. Bu script onu olcer, tahmin etmez.

Uc kademe olculur; her kademe bir oncekinin ustune ne ekledigini gosterir:

  1. float ileri        -> bilinen referansla karsilastirma (0,12 sn/goruntu)
  2. float ileri+geri   -> egitim adiminin gercek maliyeti (SimOTA dahil)
  3. QAT ileri+geri     -> fake-quant'in getirdigi ek yuk

Kullanim (Vitis AI docker icinde, /workspace/yolox_visdrone altinda):

    python qat_probe.py --exp-file yolox_nano_visdrone.py \\
        --ckpt best_ckpt.pth --data-dir datasets/merged

Not: olcum val setinden yapilir cunku mevcut vm_package'da etiketli egitim
verisi yok. Val'de goruntu basina medyan 4,8 kutu, train'de 10,5 -- SimOTA
maliyeti kutu sayisiyla buyudugu icin gercek QAT bir miktar daha yavas
olacaktir. Rapor bunu acikca yazar.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from quantize_yolox import load_checkpoint_strict, load_exp  # noqa: E402


def build_loader(exp, data_dir, batch_size, num_workers):
    """Egitim formatinda (goruntu, hedef) veren dataloader.

    ValTransform etiket dondurmez; QAT bir egitim islemi oldugu icin
    TrainTransform gerekir. Mozaik kapali: kisaltilmis QAT'te de kapali
    olacak (augmentasyon QAT'in isi degil, kuantalama esiklerini ogrenmek).
    """
    from yolox.data import COCODataset, TrainTransform

    dataset = COCODataset(
        data_dir=data_dir,
        json_file=exp.val_ann,
        name=exp.image_folder,
        img_size=exp.input_size,
        preproc=TrainTransform(max_labels=1000, flip_prob=0.0, hsv_prob=0.0),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )


def timed_loop(step_fn, loader, iters, warmup, label):
    """`iters` adet adim koser, ilk `warmup` tanesini olcume katmaz.

    Ilk iterasyonlar lazy allocation ve cache isinmasi tasir; YOLOX'un kendi
    ETA alani da bu yuzden bastan sise gorunur (bkz. HANDOFF §5).
    """
    times = []
    seen = 0
    for batch in loader:
        start = time.time()
        step_fn(batch)
        elapsed = time.time() - start
        seen += 1
        if seen > warmup:
            times.append(elapsed)
        print("  %-22s iter %2d/%d  %6.2f sn%s"
              % (label, seen, iters, elapsed, "  (isinma)" if seen <= warmup else ""))
        if seen >= iters:
            break
    if not times:
        raise SystemExit("HATA: olculecek iterasyon kalmadi; --iters arttirin.")
    return sum(times) / len(times)


def project(sec_per_iter, batch_size, images, epochs):
    """Olculen hizdan toplam QAT suresini cikarir."""
    iters_per_epoch = max(1, images // batch_size)
    total = sec_per_iter * iters_per_epoch * epochs
    return iters_per_epoch, total / 3600.0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp-file", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="QAT'te kullanilacak batch (CPU'da kucuk tutun)")
    parser.add_argument("--iters", type=int, default=6,
                        help="olculecek toplam iterasyon (isinma dahil)")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--skip-qat", action="store_true",
                        help="yalnizca float kademelerini olc")
    parser.add_argument("--plan-images", type=int, default=4000,
                        help="projeksiyon icin QAT'te kullanilacak goruntu sayisi")
    parser.add_argument("--plan-epochs", type=int, default=5)
    args = parser.parse_args()

    print("=" * 68)
    print("ORTAM")
    print("=" * 68)
    import os
    print("  torch            :", torch.__version__)
    print("  CUDA gorunur mu  :", torch.cuda.is_available())
    print("  CPU cekirdek     :", os.cpu_count())
    print("  torch is parcaci :", torch.get_num_threads())
    if torch.cuda.is_available():
        print("  >> GPU VAR: QAT'i GPU'da kosturun, bu probe CPU senaryosu icindir.")

    exp = load_exp(args.exp_file)
    height, width = exp.input_size
    print("  girdi boyutu     : %dx%d" % (width, height))
    print("  batch            :", args.batch_size)

    model = exp.get_model()
    load_checkpoint_strict(model, args.ckpt)
    model.eval()

    loader = build_loader(exp, args.data_dir, args.batch_size, args.num_workers)

    results = {}

    # --- 1. float ileri ---
    print("\n" + "=" * 68)
    print("1. FLOAT ILERI")
    print("=" * 68)

    def fwd(batch):
        images = batch[0].float()
        with torch.no_grad():
            model(images)

    results["float ileri"] = timed_loop(fwd, loader, args.iters, args.warmup, "float ileri")

    # --- 2. float ileri+geri ---
    print("\n" + "=" * 68)
    print("2. FLOAT ILERI+GERI (SimOTA dahil)")
    print("=" * 68)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    def train_step(batch):
        images, targets = batch[0].float(), batch[1].float()
        optimizer.zero_grad()
        outputs = model(images, targets)
        loss = outputs["total_loss"] if isinstance(outputs, dict) else outputs[0]
        loss.backward()
        optimizer.step()

    results["float ileri+geri"] = timed_loop(
        train_step, loader, args.iters, args.warmup, "float egitim")

    # --- 3. QAT ileri+geri ---
    if not args.skip_qat:
        print("\n" + "=" * 68)
        print("3. QAT ILERI+GERI")
        print("=" * 68)
        try:
            from pytorch_nndct import QatProcessor
        except ImportError as exc:
            print("  pytorch_nndct yok, atlandi (%s)" % exc)
        else:
            model.eval()
            dummy = torch.randn([1, 3, height, width], dtype=torch.float32)
            device = torch.device("cpu")
            try:
                processor = QatProcessor(
                    model, dummy, bitwidth=8, mix_bit=False, device=device)
                qat_model = processor.trainable_model(calib_dir="")
            except Exception as exc:  # noqa: BLE001 - hangi hata olursa olsun rapor
                print("  QatProcessor CPU'da kurulamadi: %r" % (exc,))
                print("  >> QAT bu ortamda kosmaz; GPU'lu Vitis AI imaji gerekir.")
                qat_model = None
            if qat_model is not None:
                qat_model.train()
                qat_optimizer = torch.optim.SGD(
                    qat_model.parameters(), lr=1e-4, momentum=0.9)

                def qat_step(batch):
                    images, targets = batch[0].float(), batch[1].float()
                    qat_optimizer.zero_grad()
                    outputs = qat_model(images, targets)
                    loss = outputs["total_loss"] if isinstance(outputs, dict) else outputs[0]
                    loss.backward()
                    qat_optimizer.step()

                results["QAT ileri+geri"] = timed_loop(
                    qat_step, loader, args.iters, args.warmup, "QAT egitim")

    # --- rapor ---
    print("\n" + "=" * 68)
    print("SONUC  (batch %d, %dx%d)" % (args.batch_size, width, height))
    print("=" * 68)
    base = results.get("float ileri")
    for name, sec in results.items():
        per_image = sec / args.batch_size
        rel = "" if name == "float ileri" else "  (%.1fx float ileri)" % (sec / base)
        print("  %-18s %6.2f sn/iter   %5.3f sn/goruntu%s" % (name, sec, per_image, rel))

    qat_sec = results.get("QAT ileri+geri") or results.get("float ileri+geri")
    if qat_sec:
        print("\nPROJEKSIYON  (%d goruntu x %d epoch)"
              % (args.plan_images, args.plan_epochs))
        iters_per_epoch, hours = project(
            qat_sec, args.batch_size, args.plan_images, args.plan_epochs)
        print("  %d iter/epoch  ->  %.1f saat" % (iters_per_epoch, hours))
        if "QAT ileri+geri" not in results:
            print("  UYARI: QAT olculemedi, float egitim hizi kullanildi.")
            print("         Gercek QAT tipik olarak 1,5-3 kat daha yavastir.")
        print("\n  Not: olcum val setinden (4,8 kutu/goruntu). Train'de 10,5 "
              "kutu/goruntu\n       oldugu icin SimOTA maliyeti bir miktar "
              "daha yuksek olacaktir.")


if __name__ == "__main__":
    main()
