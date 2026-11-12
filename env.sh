# Source this in every shell/job. Keeps ALL caches, configs, models inside gdl_proj.
export PROJ=${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}  # project root (dir of this file)
export XDG_CACHE_HOME=$PROJ/.cache
export XDG_CONFIG_HOME=$PROJ/.home/config
export XDG_DATA_HOME=$PROJ/.home/share
export HF_HOME=$PROJ/.cache/hf
unset TRANSFORMERS_CACHE
export UV_CACHE_DIR=$PROJ/.cache/uv
export UV_PYTHON_INSTALL_DIR=$PROJ/.cache/uv-python
export PIP_CACHE_DIR=$PROJ/.cache/pip
export TORCH_HOME=$PROJ/.cache/torch
export TRITON_CACHE_DIR=$PROJ/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=$PROJ/.cache/inductor
export TORCH_EXTENSIONS_DIR=$PROJ/.cache/torch_extensions
export MPLCONFIGDIR=$PROJ/.cache/matplotlib
export WANDB_DIR=$PROJ/.cache/wandb WANDB_CACHE_DIR=$PROJ/.cache/wandb WANDB_CONFIG_DIR=$PROJ/.cache/wandb WANDB_MODE=disabled
export LIBERO_CONFIG_PATH=$PROJ/.home/libero
export TMPDIR=$PROJ/.cache/tmp
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
mkdir -p $TMPDIR $MPLCONFIGDIR
export PATH=$PROJ/.venv/bin:$PATH
