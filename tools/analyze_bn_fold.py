"""Hangi katmanlar per-tensor INT8 kuantalamada tehlikeli?

Vitis AI BN'i conv'a katlar: W_fold = W * gamma / sqrt(var + eps).
Kuantalama AGIRLIKLARI TENSOR BASINA olceklendirir. Bir conv'un cikti
kanallari arasinda buyukluk farki cok yuksekse, kucuk kanallar 8 bitin
cok az seviyesini kullanir ve bilgi kaybolur.

Depthwise conv'lar bu konuda kotu unlu: her kanal bagimsiz bir filtre
oldugu icin buyukluk dagilimi cok genis olabilir.
"""
import numpy as np
import torch
from pathlib import Path

CKPT = Path(r"C:\Users\emrez\proje\artifacts\best_ckpt.pth")
EPS = 1e-3  # exp dosyasi BN eps'i 1e-3 yapiyor

raw = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = raw["model"]
sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}

rows = []
for key in sd:
    if not key.endswith(".conv.weight"):
        continue
    base = key[: -len(".conv.weight")]
    bn_w = sd.get(base + ".bn.weight")
    if bn_w is None:
        continue
    w = sd[key].float()
    gamma = bn_w.float()
    var = sd[base + ".bn.running_var"].float()
    scale = gamma / torch.sqrt(var + EPS)
    folded = w * scale.reshape(-1, 1, 1, 1)

    per_ch = folded.abs().amax(dim=(1, 2, 3)).numpy()
    per_ch = per_ch[per_ch > 0]
    if per_ch.size == 0:
        continue
    depthwise = w.shape[1] == 1
    ratio = float(per_ch.max() / max(per_ch.min(), 1e-12))
    # per-tensor olcekte en zayif kanal kac seviye kullanabiliyor?
    levels_weakest = 127.0 * per_ch.min() / per_ch.max()
    rows.append({
        "name": base,
        "dw": depthwise,
        "ch": int(w.shape[0]),
        "max": float(per_ch.max()),
        "min": float(per_ch.min()),
        "ratio": ratio,
        "levels": float(levels_weakest),
        "scale_max": float(scale.abs().max()),
    })

rows.sort(key=lambda r: -r["ratio"])
dw = [r for r in rows if r["dw"]]
pw = [r for r in rows if not r["dw"]]

print("=" * 78)
print("1. OZET")
print("=" * 78)
print(f"  BN'li conv sayisi        : {len(rows)}")
print(f"    depthwise              : {len(dw)}")
print(f"    normal (pointwise vb.) : {len(pw)}")


def summary(name, group):
    if not group:
        return
    r = np.array([g["ratio"] for g in group])
    lv = np.array([g["levels"] for g in group])
    print(f"\n  {name}")
    print(f"    kanal buyukluk orani (max/min) : medyan {np.median(r):8.1f}"
          f"   p90 {np.percentile(r, 90):8.1f}   maks {r.max():8.1f}")
    print(f"    en zayif kanalin seviye sayisi : medyan {np.median(lv):8.1f}"
          f"   p10 {np.percentile(lv, 10):8.2f}   min {lv.min():8.3f}")


summary("DEPTHWISE", dw)
summary("NORMAL", pw)

print()
print("=" * 78)
print("2. EN TEHLIKELI 15 KATMAN (per-tensor olcekte en cok bilgi kaybi)")
print("=" * 78)
print(f"{'katman':<58}{'tip':>5}{'oran':>10}{'seviye':>9}")
print("-" * 82)
for r in rows[:15]:
    print(f"{r['name'][-56:]:<58}{'DW' if r['dw'] else 'PW':>5}"
          f"{r['ratio']:>10.0f}{r['levels']:>9.2f}")

print()
print("=" * 78)
print("3. YORUM")
print("=" * 78)
bad = [r for r in rows if r["levels"] < 2.0]
print(f"  En zayif kanali 2 SEVIYEDEN AZ kullanan katman : {len(bad)} / {len(rows)}")
bad_dw = [r for r in bad if r["dw"]]
print(f"    bunlarin depthwise olani                     : {len(bad_dw)}")
print()
print("  Bir kanal 2 seviyeden az kullaniyorsa o kanalin ciktisi INT8'de")
print("  pratikte YOK olur. Cok sayida boyle katman varsa PTQ'nun tek basina")
print("  yetmemesi beklenir; cozum fast_finetune (AdaQuant) veya QAT'tir.")
