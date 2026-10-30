#!/usr/bin/env bash
# Build AIHWKIT 1.1.0 with the CapSynapseDevice.
#
#   VENV=.venv ARCH=90 JOBS=8 ./aihwkit_ext/install.sh
#
#   VENV         virtual environment with torch installed (default: .venv)
#   ARCH         CUDA compute capability without the dot, e.g. 80 (A100), 86 (RTX 30xx),
#                89 (RTX 40xx), 90 (H100), 100 (B200); default 89
#   JOBS         parallel compile jobs (default 4)
#   AIHWKIT_SRC  aihwkit checkout (default: .aihwkit-src, cloned if absent)
#   CUDA_HOME    CUDA toolkit (default /usr/local/cuda)
#   SKIP_PIP_INSTALL  if set, build in place only (no pip install -e into VENV)
set -e
SRC=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$SRC/.." && pwd)
DST=${AIHWKIT_SRC:-$REPO/.aihwkit-src}
VENV=${VENV:-$REPO/.venv}
ARCH=${ARCH:-89}
CUDA=${CUDA_HOME:-/usr/local/cuda}
VENV=$(cd "$VENV" 2>/dev/null && pwd) || { echo "virtual environment not found" >&2; exit 1; }
mkdir -p "$(dirname "$DST")"
DST=$(cd "$(dirname "$DST")" && pwd)/$(basename "$DST")

if [ ! -d "$DST" ]; then
  git clone --branch v1.1.0 --depth 1 https://github.com/IBM/aihwkit.git "$DST"
fi
cp "$SRC"/rpu_capsynapse_device.{h,cpp}    "$DST/src/rpucuda/"
cp "$SRC"/rpucuda_capsynapse_device.{h,cu} "$DST/src/rpucuda/cuda/"

# build dependencies of aihwkit (stubgen comes with mypy)
"$VENV/bin/python" - <<'PY' || PIP_CONSTRAINT= "$VENV/bin/pip" install "scikit-build>=0.18.1" pybind11 ninja wheel mypy cmake
import importlib.util, sys
sys.exit(0 if all(importlib.util.find_spec(m) for m in ("skbuild", "pybind11", "ninja", "wheel", "mypy", "cmake")) else 1)
PY

cd "$DST"
export GIT_CEILING_DIRECTORIES=$(dirname "$DST")
if git apply --check "$SRC/registration.diff" 2>/dev/null; then
  git apply "$SRC/registration.diff"
elif ! grep -q CapSynapse src/rpucuda/rpu_simple_device.h; then
  echo "registration.diff does not apply to $DST" >&2; exit 1
fi

export PATH="$VENV/bin:$CUDA/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CMAKE_ARGS="-GNinja -DRPU_CUDA_ARCHITECTURES=$ARCH -DCMAKE_CUDA_ARCHITECTURES=$ARCH"
export CMAKE_BUILD_PARALLEL_LEVEL=${JOBS:-4}
USE_CUDA=1 "$VENV/bin/python" setup.py build_ext --inplace
if [ -z "${SKIP_PIP_INSTALL:-}" ]; then
  PIP_CONSTRAINT= "$VENV/bin/pip" install -e "$DST" --no-build-isolation --no-deps
fi
