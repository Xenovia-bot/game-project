#!/usr/bin/env bash
# yolox_visdrone_demo derleme scripti.
# - KV260 uzerinde dogrudan calistirilabilir (hazir imajda g++ mevcuttur).
# - Cross-compile icin once PetaLinux SDK ortamini source edin
#   (bkz. deploy/README.md), CXX otomatik olarak SDK derleyicisine doner.
set -euo pipefail

CXX=${CXX:-g++}
OUT=yolox_visdrone_demo

if pkg-config --exists opencv4 2>/dev/null; then
    OPENCV_FLAGS="$(pkg-config --cflags --libs opencv4)"
elif pkg-config --exists opencv 2>/dev/null; then
    OPENCV_FLAGS="$(pkg-config --cflags --libs opencv)"
else
    OPENCV_FLAGS="-I/usr/include/opencv4 -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_imgcodecs"
fi

EXTRA_LIBS=""
# Eski GCC (KV260 imaji dahil) std::filesystem icin libstdc++fs isteyebilir.
FS_CHECK="$(mktemp)"
if ! echo '#include <filesystem>
int main(){return std::filesystem::temp_directory_path().empty()?1:0;}' \
    | ${CXX} -std=c++17 -x c++ - -o "${FS_CHECK}" >/dev/null 2>&1; then
    EXTRA_LIBS="-lstdc++fs"
fi
rm -f "${FS_CHECK}"

set -x
${CXX} -std=c++17 -O2 -o ${OUT} src/main.cpp \
    ${OPENCV_FLAGS} \
    -lvart-runner -lvart-dpu-controller -lxir -lunilog -lglog -lpthread \
    ${EXTRA_LIBS}
set +x

echo "Derlendi: ./${OUT}"
