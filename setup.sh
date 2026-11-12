#!/bin/bash
# One-time setup on a Linux machine with an NVIDIA GPU (tested on SkyNet A40 nodes, CUDA 12 driver).
# Everything (venv, caches, models, third-party code) is created INSIDE this folder; deleting it removes everything.
# Usage (from the project root, on a compute node or any node with internet):  bash setup.sh
set -e
cd "$(dirname "$0")"
source env.sh

# 1. Third-party code at the exact commits we used, plus our local patches (patches/*.patch).
clone() {  # name url commit
  [ -d third_party/$1 ] || git clone -q $2 third_party/$1
  git -C third_party/$1 checkout -q $3
  [ -f patches/$1.patch ] && { git -C third_party/$1 apply --check ../../patches/$1.patch 2>/dev/null \
    && git -C third_party/$1 apply ../../patches/$1.patch && echo "patched $1" || echo "patch for $1 already applied?"; }
  return 0
}
mkdir -p third_party
clone openvla  https://github.com/openvla/openvla.git                     c8f03f4
clone LIBERO   https://github.com/Lifelong-Robot-Learning/LIBERO.git       8f1084e
clone SpecVLA  https://github.com/PineTreeWss/SpecVLA.git                  56ef298
clone KERV     https://github.com/zhengzihaoPKU/KERV.git                   4abb3df

# 2. Python 3.10 venv with the exact package versions (requirements.lock.txt; editable lines are local repos).
command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/ (put it in this folder if you want)"; exit 1; }
[ -d .venv ] || uv venv -q --python 3.10 .venv
grep -v '^-e ' requirements.lock.txt > $TMPDIR/req.txt
uv pip install -q --python .venv/bin/python -r $TMPDIR/req.txt
uv pip install -q --python .venv/bin/python --no-deps -e third_party/openvla
# LIBERO's editable install is broken (empty finder MAPPING) -> add it to the path with a .pth file instead.
echo "$PROJ/third_party/LIBERO" > .venv/lib/python3.10/site-packages/libero_path.pth

# 3. LIBERO config (it asks interactively on first import; answer "n" = use default paths).
mkdir -p $LIBERO_CONFIG_PATH
echo n | python -c "from libero.libero import benchmark" >/dev/null 2>&1 || true

# 4. Models: fine-tuned OpenVLA checkpoints (~15 GB each) from Hugging Face.
for s in spatial object goal 10; do
  python -c "from huggingface_hub import snapshot_download as d; d('openvla/openvla-7b-finetuned-libero-$s')"
done

# 5. Spec-VLA drafter weights (Google Drive, linked from github.com/PineTreeWss/SpecVLA README) must end up as
#    checkpoints/specvla_drafters/libero_{spatial,object,goal,10}/{config.json,pytorch_model.bin}.
#    Download with gdown using HOME=$PROJ/.home so its cookie file stays in this folder.
[ -d checkpoints/specvla_drafters/libero_goal ] || echo "TODO: download Spec-VLA drafters into checkpoints/specvla_drafters/ (see README)"
echo "setup done"
