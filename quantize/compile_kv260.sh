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
NAME=${3:-yolox_nano_visdrone}
NUM_CLASSES=${4:-2}
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
for s in subs:
    dev = s.get_attr("device") if s.has_attr("device") else "?"
    print("  - %-5s %s" % (dev, s.get_name()))
print("Toplam subgraph: %d, DPU subgraph: %d" % (len(subs), len(dpu)))
if len(subs) != 1 or len(dpu) != 1 or non_dpu:
    print("HATA: derlenmis graph yalnizca bir DPU subgraph'tan olusmuyor!")
    print("Cozum: quantize_yolox.py --inspect ciktisindaki CPU'ya dusen op'lari inceleyin.")
    sys.exit(1)

inputs = dpu[0].get_input_tensors()
outputs = dpu[0].get_output_tensors()
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
# Rolu kanal sayisindan taniyoruz; ucu de farkli olmali.
if len({4, 1, NUM_CLASSES}) != 3:
    print("HATA: sinif sayisi %d, roller kanal sayisindan ayirt edilemez "
          "(4=reg, 1=obj, nc=cls cakisiyor)" % NUM_CLASSES)
    sys.exit(1)

# Uzamsal boyuta gore seviyelere grupla, rolu kanal sayisindan tani.
levels = {}
for tensor in outputs:
    dims = list(tensor.dims)
    if (len(dims) != 4 or dims[0] != 1 or
            input_dims[1] % dims[1] or input_dims[2] % dims[2] or
            input_dims[1] // dims[1] != input_dims[2] // dims[2]):
        print("HATA: cikti tensoru NHWC ve girdiyle uyumlu olmali: %s" % dims)
        sys.exit(1)
    key = (dims[1], dims[2])
    role = {4: "reg", 1: "obj", NUM_CLASSES: "cls"}.get(dims[-1])
    if role is None:
        print("HATA: taninmayan cikti kanal sayisi %d (beklenen 4/1/%d): %s"
              % (dims[-1], NUM_CLASSES, dims))
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
