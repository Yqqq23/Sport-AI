#!/usr/bin/env bash
# ==============================================================
# basketball-v2  environment setup  (no sudo required)
# ==============================================================
# Usage:
#   cd ~/basketball-v2
#   bash setup_env.sh
#
# Everything installs into a local virtualenv at ~/basketball-v2/.venv
# Nothing is written outside your home directory.
# ==============================================================

set -e  # stop on first error

echo "=============================================="
echo "STEP 0  -  detect GPU and CUDA"
echo "=============================================="

if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Are you on the GPU node?"
    echo "If this is a login node, submit a job or ssh to a compute node first."
    exit 1
fi

nvidia-smi

# nvidia-smi reports the MAX CUDA version the driver supports
CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
echo ""
echo "Driver supports CUDA up to: ${CUDA_VER}"

# pick the closest available PyTorch wheel index
CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)

if [ "$CUDA_MAJOR" -ge 12 ]; then
    if [ "$CUDA_MINOR" -ge 4 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        TAG="cu124"
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        TAG="cu121"
    fi
elif [ "$CUDA_MAJOR" -eq 11 ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    TAG="cu118"
else
    echo "ERROR: unexpected CUDA version ${CUDA_VER}. Install torch manually."
    exit 1
fi

echo "Will install PyTorch build: ${TAG}"
echo ""


echo "=============================================="
echo "STEP 1  -  create virtualenv"
echo "=============================================="

# use python3.10 / 3.11 if available, else default python3
PYBIN=python3
for c in python3.11 python3.10 python3; do
    if command -v $c &> /dev/null; then PYBIN=$c; break; fi
done
echo "Using interpreter: $($PYBIN --version)"

if [ ! -d ".venv" ]; then
    $PYBIN -m venv .venv
    echo "Created .venv"
else
    echo ".venv already exists, reusing it"
fi

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
echo ""


echo "=============================================="
echo "STEP 2  -  install PyTorch (${TAG})"
echo "=============================================="
pip install torch torchvision --index-url "${TORCH_INDEX}"

echo ""
echo "--- verifying CUDA is actually visible ---"
python - <<'PYEOF'
import torch, sys
ok = torch.cuda.is_available()
print("torch      :", torch.__version__)
print("cuda build :", torch.version.cuda)
print("available  :", ok)
if ok:
    print("device     :", torch.cuda.get_device_name(0))
else:
    print()
    print("!!! CUDA NOT AVAILABLE !!!")
    print("A CPU-only wheel was installed, or you are on a node without a GPU.")
    print("Training would run ~40x slower and appear to work. Stop and fix this.")
    sys.exit(1)
PYEOF
echo ""


echo "=============================================="
echo "STEP 3  -  install ultralytics and deps"
echo "=============================================="
pip install "ultralytics>=8.4.63"
pip install opencv-python-headless numpy pyyaml

# opencv-python-headless is used on purpose: the normal opencv-python
# needs system GUI libraries (libGL etc.) that you cannot apt-get
# without sudo. headless has identical read/write/draw functionality
# and only drops cv2.imshow, which this project never calls.

if [ -f requirements.txt ]; then
    echo ""
    echo "--- installing remaining requirements.txt entries ---"
    # torch is intentionally unpinned in requirements.txt; already installed above
    pip install -r requirements.txt || echo "WARN: some requirements failed, see above"
fi
echo ""


echo "=============================================="
echo "STEP 4  -  OCR backend (optional, NOT needed for training)"
echo "=============================================="
echo "tesseract is a SYSTEM binary and normally needs sudo apt-get."
echo "Trying the no-sudo routes..."
echo ""

TESS_OK=0

# route 1: conda-forge, if conda exists
if command -v conda &> /dev/null; then
    echo "conda found, trying conda-forge tesseract..."
    conda install -y -c conda-forge tesseract && TESS_OK=1 || echo "conda route failed"
fi

# route 2: already on the system?
if [ $TESS_OK -eq 0 ] && command -v tesseract &> /dev/null; then
    echo "tesseract binary already present at: $(command -v tesseract)"
    TESS_OK=1
fi

if [ $TESS_OK -eq 1 ]; then
    pip install pytesseract
    echo "OCR backend ready."
else
    echo ""
    echo "  >> tesseract NOT installed. This is OK for now."
    echo "  >> Detector training and ball/rim/shot detection are unaffected."
    echo "  >> Only jersey NUMBER text will stay unknown."
    echo "  >> See the notes at the bottom of this script for alternatives."
fi
echo ""


echo "=============================================="
echo "STEP 5  -  run the project's own env check"
echo "=============================================="
python scripts/check_env.py || true

echo ""
echo "=============================================="
echo "DONE"
echo "=============================================="
echo "Activate the venv in every new shell session with:"
echo ""
echo "    cd ~/basketball-v2 && source .venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. python scripts/prepare_dataset.py --src dataset --out train/datasets/basketball"
echo "  2. python train/train.py train \\"
echo "       --data train/datasets/basketball/data.yaml \\"
echo "       --model yolo26s.pt --epochs 20"
echo ""

# ==============================================================
# NOTES ON TESSERACT WITHOUT SUDO
# ==============================================================
# Option A - conda-forge (handled above, if conda is installed)
#     conda install -c conda-forge tesseract
#
# Option B - ask the cluster admin to install tesseract-ocr.
#     One line for them, permanent fix for everyone.
#
# Option C - switch OCR backend to a pip-only library.
#     src/jersey.py was written with a JerseyOCRBackend Protocol
#     precisely so the backend can be swapped. easyocr installs
#     entirely through pip with no system binary:
#         pip install easyocr
#     This needs a small new class implementing the same Protocol.
#     Do this AFTER the detector is trained - not before.
# ==============================================================
