#!/usr/bin/env bash
# Kuantalanmis xmodel'i KV260 DPU'su (DPUCZDX8G B4096) icin derler ve
# subgraph dagilimini dogrular. Vitis AI 3.0 docker'i icinde calistirin.
#
# Kullanim: bash compile_kv260.sh [quant_dir] [out_dir] [model_adi] [sinif_sayisi]
#
# Cikti kanali = 4 (reg) + 1 (obj) + sinif_sayisi. Varsayilan 2 sinif
# (person/vehicle) icin 7 kanal beklenir.
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

# YOLOX ham bas ciktisi: reg(4) + obj(1) + cls(num_classes)
EXPECTED_CH = 5 + int(sys.argv[2])

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
if len(inputs) != 1 or len(outputs) != 3:
    print("HATA: beklenen tensor sayisi input=1/output=3, bulunan %d/%d" %
          (len(inputs), len(outputs)))
    sys.exit(1)

for tensor in inputs + outputs:
    print("    tensor %-45s %s" % (tensor.name, list(tensor.dims)))
input_dims = list(inputs[0].dims)
if len(input_dims) != 4 or input_dims[0] != 1 or input_dims[-1] != 3:
    print("HATA: girdi NHWC [1,H,W,3] olmali: %s" % input_dims)
    sys.exit(1)

strides = set()
for tensor in outputs:
    dims = list(tensor.dims)
    if (len(dims) != 4 or dims[0] != 1 or dims[-1] != EXPECTED_CH or
            input_dims[1] % dims[1] or input_dims[2] % dims[2] or
            input_dims[1] // dims[1] != input_dims[2] // dims[2]):
        print("HATA: YOLOX cikti tensoru NHWC ve %d kanalli olmali: %s"
              % (EXPECTED_CH, dims))
        sys.exit(1)
    strides.add(input_dims[1] // dims[1])
if strides != {8, 16, 32}:
    print("HATA: cikti stride degerleri 8/16/32 olmali: %s" % sorted(strides))
    sys.exit(1)

print("OK: tek DPU subgraph; 1 girdi ve 3 adet %d-kanalli cikti dogrulandi."
      % EXPECTED_CH)
EOF

echo ""
echo ">> KV260'a kopyalanacak dosya: ${OUT_DIR}/${NAME}.xmodel"
