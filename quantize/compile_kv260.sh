#!/usr/bin/env bash
# Kuantalanmis xmodel'i KV260 DPU'su (DPUCZDX8G B4096) icin derler ve
# subgraph dagilimini dogrular. Vitis AI 3.0 docker'i icinde calistirin.
#
# Kullanim: bash compile_kv260.sh [quant_dir] [out_dir] [model_adi] [sinif_sayisi]
#
# Cikti: 3 stride seviyesi x (reg 4 / obj 1 / cls sinif_sayisi) = 9 tensor.
# Bas ciktilari birlestirilmez; gerekcesi asagida ve deploy/src/main.cpp icinde.
set -euo pipefail

QUANT_DIR=${1:-build/quant}
OUT_DIR=${2:-build/compiled}
NAME=${3:-yolox_tiny_ship}
NUM_CLASSES=${4:-1}   # gemi projesi tek sinif: cls kanali 1, cikti 5+1=6
ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json

if [ ! -f "${ARCH}" ]; then
    echo "HATA: KV260 arch.json bulunamadi: ${ARCH}"
    exit 1
fi

shopt -s nullglob
XMODELS=("${QUANT_DIR}"/*_int.xmodel)
if [ "${#XMODELS[@]}" -ne 1 ]; then
    echo "HATA: ${QUANT_DIR} icinde tam olarak bir *_int.xmodel bekleniyor; bulunan: ${#XMODELS[@]}"
    echo "Once: python quantize_yolox.py --quant-mode test --deploy --subset-len 1 --batch-size 1 ..."
    exit 1
fi
XMODEL=${XMODELS[0]}

echo ">> Derleniyor: ${XMODEL}"
mkdir -p "${OUT_DIR}"
vai_c_xir -x "${XMODEL}" -a "${ARCH}" -o "${OUT_DIR}" -n "${NAME}"

echo ""
echo ">> Subgraph kontrolu:"
python3 - "${OUT_DIR}/${NAME}.xmodel" "${NUM_CLASSES}" <<'EOF'
import sys
import xir


graph = xir.Graph.deserialize(sys.argv[1])
subs = graph.get_root_subgraph().toposort_child_subgraph()
dpu = [s for s in subs if s.has_attr("device") and s.get_attr("device") == "DPU"]
non_dpu = [s for s in subs if not (s.has_attr("device") and s.get_attr("device") == "DPU")]
def op_types(sub):
    """Subgraph'taki op turleri; xir surumleri arasinda API oynayabiliyor."""
    try:
        return ",".join(sorted({op.get_type() for op in sub.get_ops()}))
    except Exception:                                  # pragma: no cover
        return "?"


for s in subs:
    dev = s.get_attr("device") if s.has_attr("device") else "?"
    print("  - %-5s %-70s [%s]" % (dev, s.get_name(), op_types(s)))
print("Toplam subgraph: %d, DPU subgraph: %d" % (len(subs), len(dpu)))

# Kapi **DPU subgraph sayisina** bakar, toplama degil.
#
# Onceki hali `len(subs) != 1` istiyordu ve ilk gercek derlemede patladi
# (2026-08-10): 1 USER (girdi) + 1 DPU + 9 CPU `*_fix_` = 11 subgraph.
# O 9 CPU blogu, 9 ayri cikti tensorune gecmenin dogrudan sonucu; her birinin
# sabit noktali -> float donusumu ayri bir subgraph olarak gorunuyor.
# `main.cpp` bu olceklemeyi zaten host tarafinda kendisi yapiyor, o bloklar
# calisma zamaninda kullanilmiyor.
#
# Kapinin asil amaci **agin ortasinda** CPU'ya dusen op yakalamakti; DPU<->CPU
# gidis gelisi FPS'i oldurur. Tek bir DPU subgraph varken bu yapisal olarak
# imkansiz: her CPU blogu zorunlu olarak ya oncesinde ya sonrasindadir.
# Yani `len(dpu) == 1` tek basina yeterli ve dogru kontroldur.
if len(dpu) != 1:
    print("HATA: tam olarak bir DPU subgraph bekleniyordu, bulunan %d!" % len(dpu))
    print("Cozum: quantize_yolox.py --inspect ciktisindaki CPU'ya dusen op'lari inceleyin.")
    sys.exit(1)

# list(): bu xir surumunde get_*_tensors() **set** donduruyor. Set'te ne `+`
# ne de indeksleme var; ikisi de asagida kullaniliyor (2026-08-10'da ilk
# gercek derlemede ortaya cikti). Siralamaya guvenmiyoruz: seviye/rol
# esleme dims'ten turetiliyor, tipki main.cpp'deki gibi.
inputs = list(dpu[0].get_input_tensors())
outputs = list(dpu[0].get_output_tensors())
# 9 cikti = 3 stride seviyesi x (reg 4, obj 1, cls num_classes).
# Bas ciktilari bilerek BIRLESTIRILMEZ: Vitis AI'da concat tum girdilerin ayni
# fix_point'i paylasmasini zorunlu kilar; obj logitleri -76'ya inerken reg
# +-2.9 araliginda oldugu icin ortak olcek reg'i eziyor ve INT8 AP 0.5874 ->
# 0.2987 dusuyordu (2026-08-09'da olculdu).
if len(inputs) != 1 or len(outputs) != 9:
    print("HATA: beklenen tensor sayisi input=1/output=9 "
          "(3 seviye x reg/obj/cls), bulunan %d/%d" % (len(inputs), len(outputs)))
    sys.exit(1)

for tensor in inputs + outputs:
    print("    tensor %-45s %s" % (tensor.name, list(tensor.dims)))
input_dims = list(inputs[0].dims)
if len(input_dims) != 4 or input_dims[0] != 1 or input_dims[-1] != 3:
    print("HATA: girdi NHWC [1,H,W,3] olmali: %s" % input_dims)
    sys.exit(1)

NUM_CLASSES = int(sys.argv[2])

# Rolu KANAL SAYISINDAN degil tensor ADINDAN taniyoruz. YOLOXHead PyTorch'ta
# ayri ModuleList'ler kullaniyor (reg_preds/obj_preds/cls_preds), bu isimler
# xir tensorlerine degismeden geciyor -- gercek derlemede dogrulandi
# (2026-08-13): "..._reg_preds__ModuleList_2__16918_fix" gibi. Kanal sayisina
# bakan onceki surum NUM_CLASSES == 1 veya 4 oldugunda (tek sinif, ya da
# yanlislikla 4 sinif) roller birbirinden ayirt edilemedigi icin HER ZAMAN
# hata veriyordu; gemi projesi tam olarak NUM_CLASSES=1 oldugu icin bu
# noktada durmustu. Isim tabanli esleme sinif sayisindan bagimsizdir.
def _role_from_name(name):
    if "reg_preds" in name:
        return "reg"
    if "obj_preds" in name:
        return "obj"
    if "cls_preds" in name:
        return "cls"
    return None


_EXPECTED_CHANNELS = {"reg": 4, "obj": 1, "cls": NUM_CLASSES}

# Uzamsal boyuta gore seviyelere grupla, rolu isimden tani, kanal sayisini
# sadece SAGLAMA olarak dogrula (yanlis exp/checkpoint sessizce gecmesin).
levels = {}
for tensor in outputs:
    dims = list(tensor.dims)
    if (len(dims) != 4 or dims[0] != 1 or
            input_dims[1] % dims[1] or input_dims[2] % dims[2] or
            input_dims[1] // dims[1] != input_dims[2] // dims[2]):
        print("HATA: cikti tensoru NHWC ve girdiyle uyumlu olmali: %s" % dims)
        sys.exit(1)
    key = (dims[1], dims[2])
    role = _role_from_name(tensor.name)
    if role is None:
        print("HATA: tensor adindan rol cikarilamadi (reg_preds/obj_preds/"
              "cls_preds bekleniyor): %s" % tensor.name)
        sys.exit(1)
    expected = _EXPECTED_CHANNELS[role]
    if dims[-1] != expected:
        print("HATA: %s tensoru %d kanal olmali, %d bulundu: %s"
              % (role, expected, dims[-1], tensor.name))
        sys.exit(1)
    if role in levels.setdefault(key, {}):
        print("HATA: ayni seviyede iki '%s' tensoru var: %s" % (role, dims))
        sys.exit(1)
    levels[key][role] = dims

if len(levels) != 3:
    print("HATA: 3 stride seviyesi bekleniyordu, bulunan %d" % len(levels))
    sys.exit(1)

strides = set()
for (h, w), roles in sorted(levels.items()):
    if set(roles) != {"reg", "obj", "cls"}:
        print("HATA: %dx%d seviyesinde eksik rol: %s" % (h, w, sorted(roles)))
        sys.exit(1)
    strides.add(input_dims[1] // h)
if strides != {8, 16, 32}:
    print("HATA: cikti stride degerleri 8/16/32 olmali: %s" % sorted(strides))
    sys.exit(1)

print("OK: tek DPU subgraph; 1 girdi ve 3 seviye x (reg 4 / obj 1 / cls %d) "
      "dogrulandi." % NUM_CLASSES)
EOF

echo ""
echo ">> KV260'a kopyalanacak dosya: ${OUT_DIR}/${NAME}.xmodel"
