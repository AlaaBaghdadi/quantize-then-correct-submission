#!/usr/bin/env bash
# [REV-R1] Build OpenPose 1.7.0 (CPU_ONLY, BODY_25) once into $W/openpose on the network volume.
# The original gate ran the Windows OpenPose 1.7.0 CPU binaries (OpenPoseDemo.exe, default BODY_25,
# net_resolution -1x368). This builds the same release and model for Linux; WI-0 checks equivalence.
set -euo pipefail
W="${W:-/workspace}"
OP="$W/openpose"
if [ -x "$OP/build/examples/openpose/openpose.bin" ] && [ -f "$OP/models/pose/body_25/pose_iter_584000.caffemodel" ]; then
  echo "[openpose] already built"; exit 0; fi
apt-get install -y -q --no-install-recommends cmake build-essential libprotobuf-dev protobuf-compiler \
  libgoogle-glog-dev libgflags-dev libhdf5-dev libopenblas-dev libboost-all-dev libopencv-dev \
  libleveldb-dev libsnappy-dev liblmdb-dev libatlas-base-dev >/dev/null
rm -rf "$OP"
git clone -q --depth 1 --branch v1.7.0 https://github.com/CMU-Perceptual-Computing-Lab/openpose.git "$OP"
cd "$OP" && git submodule update --init --recursive --depth 1 -q
mkdir -p models/pose/body_25
curl -fsSL -o models/pose/body_25/pose_iter_584000.caffemodel \
  https://huggingface.co/camenduru/openpose/resolve/f5bb0c0a16060ac8b373472a5456c76bd68eb202/pose_iter_584000.caffemodel
test "$(stat -c%s models/pose/body_25/pose_iter_584000.caffemodel)" -gt 100000000
mkdir -p build && cd build
cmake .. -DGPU_MODE=CPU_ONLY -DBUILD_PYTHON=OFF -DBUILD_EXAMPLES=ON -DBUILD_CAFFE=ON \
  -DDOWNLOAD_BODY_25_MODEL=OFF -DDOWNLOAD_BODY_COCO_MODEL=OFF -DDOWNLOAD_BODY_MPI_MODEL=OFF \
  -DDOWNLOAD_FACE_MODEL=OFF -DDOWNLOAD_HAND_MODEL=OFF -DUSE_MKL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
make -j"$(nproc)" >/dev/null 2>"$W/logs/openpose_build.err" || { tail -50 "$W/logs/openpose_build.err"; exit 1; }
test -x examples/openpose/openpose.bin
echo "[openpose] built $(date -u)"
