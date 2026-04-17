# Running BEVFusion with Conda (No sudo, CUDA 12.0)

This guide sets up BEVFusion in a user-space conda environment on a machine with CUDA 12.0 drivers. No root/sudo access is required at any step.

---

## Requirements

- CUDA 12.0+ driver already installed system-wide (`nvidia-smi` should work)
- ~20 GB free disk space for conda, packages, and checkpoints
- Python 3.8

---

## 1. Install Miniconda (user-space, no sudo)

Skip this step if `conda` is already available.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda3
rm ~/miniconda.sh

# Add conda to your shell
~/miniconda3/bin/conda init bash   # or: conda init zsh
exec "$SHELL"                      # reload shell
```

---

## 2. Create and activate the conda environment

```bash
conda create -n bevfusion python=3.8 -y
conda activate bevfusion
```

All subsequent commands assume this environment is active.

---

## 3. Install PyTorch with CUDA 12 support

The earliest PyTorch version available in the cu121 index is 2.1.0, which is binary-compatible with a CUDA 12.0 driver.

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.1.0+cu121 True
```

---

## 4. Install system-level build dependencies (no sudo)

OpenMPI and mpi4py are needed by `torchpack`. Install both via conda so the MPI wrapper compiler and its dependencies are resolved automatically — building mpi4py with pip against a conda-forge OpenMPI fails because the conda cross-compiler (`x86_64-conda-linux-gnu-cc`) is not present by default.

```bash
conda install -c conda-forge openmpi mpi4py -y
```

---

## 5. Install Python dependencies

```bash
pip install Pillow==9.5.0 tqdm torchpack numba==0.57.1 nuscenes-devkit
```

### Install mmcv

Use the prebuilt wheel for torch 2.0 + cu121:

```bash
pip install mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
```

### Install mmdetection

```bash
pip install mmdet==3.1.0
```

---

## 6. Clone the repo and build CUDA extensions

```bash
git clone https://github.com/DivyanshJindal26/bevfusion.git
cd bevfusion
git checkout claude/cuda-12-compatibility-eYzfa
```

Build and install the custom CUDA ops in-place (no install prefix, no sudo):

```bash
FORCE_CUDA=1 python setup.py develop
```

> **Tip:** If you only have a specific GPU architecture (e.g., Ada Lovelace RTX 4090 = sm_89), you can speed up compilation by limiting the gencode targets:
> ```bash
> TORCH_CUDA_ARCH_LIST="8.9" FORCE_CUDA=1 python setup.py develop
> ```
> Valid values: `7.0` (V100), `7.5` (T4), `8.0` (A100), `8.6` (RTX 3090), `8.9` (RTX 4090), `9.0` (H100).

---

## 7. Prepare the nuScenes dataset

Download the nuScenes full dataset (v1.0-trainval + map expansion) from the official site and place it under `data/nuscenes`:

```
bevfusion/
└── data/
    └── nuscenes/
        ├── maps/
        ├── samples/
        ├── sweeps/
        ├── v1.0-trainval/
        └── v1.0-test/
```

Then generate the info files used by this codebase (do **not** reuse files from stock mmdetection3d — they use a different coordinate convention):

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes --extra-tag nuscenes
```

---

## 8. Download pretrained checkpoints

```bash
bash tools/download_pretrained.sh
```

This creates a `pretrained/` directory with:

| File | Description |
|------|-------------|
| `bevfusion-det.pth` | Camera+LiDAR detection |
| `bevfusion-seg.pth` | Camera+LiDAR segmentation |
| `lidar-only-det.pth` | LiDAR-only detection (TransFusion-L) |
| `camera-only-det.pth` | Camera-only detection |
| `swint-nuimages-pretrained.pth` | Swin-T backbone pretrain |

---

## 9. Evaluation

Replace `[N]` with the number of GPUs available on your machine.

**Camera+LiDAR detection (BEVFusion):**
```bash
torchpack dist-run -np [N] python tools/test.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    pretrained/bevfusion-det.pth --eval bbox
```

**Camera+LiDAR BEV segmentation:**
```bash
torchpack dist-run -np [N] python tools/test.py \
    configs/nuscenes/seg/fusion-bev256d2-lss.yaml \
    pretrained/bevfusion-seg.pth --eval map
```

**Single-GPU evaluation** (just set `-np 1`):
```bash
torchpack dist-run -np 1 python tools/test.py \
    configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml \
    pretrained/lidar-only-det.pth --eval bbox
```

---

## 10. Training

All training commands use `torchpack dist-run`. Adjust `-np` to your GPU count.

**Camera-only detection:**
```bash
torchpack dist-run -np [N] python tools/train.py \
    configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml \
    --model.encoders.camera.backbone.init_cfg.checkpoint \
    pretrained/swint-nuimages-pretrained.pth
```

**LiDAR-only detection:**
```bash
torchpack dist-run -np [N] python tools/train.py \
    configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml
```

**BEVFusion detection (camera+LiDAR, fine-tune from LiDAR checkpoint):**
```bash
torchpack dist-run -np [N] python tools/train.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --model.encoders.camera.backbone.init_cfg.checkpoint \
    pretrained/swint-nuimages-pretrained.pth \
    --load_from pretrained/lidar-only-det.pth
```

**BEVFusion segmentation:**
```bash
torchpack dist-run -np [N] python tools/train.py \
    configs/nuscenes/seg/fusion-bev256d2-lss.yaml \
    --model.encoders.camera.backbone.init_cfg.checkpoint \
    pretrained/swint-nuimages-pretrained.pth
```

> After training completes, run `tools/test.py` separately to get evaluation metrics — the training script does not report final metrics.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'mmdet3d.ops.xxx'`**
The CUDA extensions were not compiled. Re-run `FORCE_CUDA=1 python setup.py develop` from the repo root with the conda env active.

**`CUDA error: no kernel image is available for execution on the device`**
Your GPU architecture is not covered by the compiled gencode targets. Rebuild with `TORCH_CUDA_ARCH_LIST` set to your GPU's compute capability (see step 6).

**`ImportError: libcuda.so.1: cannot open shared object file`**
The CUDA driver library is not visible. On some clusters you need to load a module first, e.g. `module load cuda/12.0`. The conda env itself does not ship the driver library.

**`mpi4py` build fails — `x86_64-conda-linux-gnu-cc` not found**
The conda-forge OpenMPI wrapper was configured with a conda cross-compiler that isn't in the env. Do not install `mpi4py` with pip against a conda-forge OpenMPI. Instead:
```bash
conda install -c conda-forge openmpi mpi4py -y
```

**`ImportError: undefined symbol: PyObject_GET_WEAKREFS_LISTPTR` (or similar symbol errors loading `libtorch_python.so`)**
The error path points to a *different* conda environment (e.g. `dental_yolo`). This means `LD_LIBRARY_PATH` is polluted with library paths from another env. Fix for the current shell:
```bash
unset LD_LIBRARY_PATH
```
Then find and remove the source of the leak so it doesn't come back:
```bash
grep -rn "LD_LIBRARY_PATH" ~/.bashrc ~/.bash_profile ~/.profile 2>/dev/null
```
Remove or comment out any line that hardcodes a path to another conda environment. Never export `LD_LIBRARY_PATH` globally in `.bashrc` — conda manages library paths automatically when you `conda activate`.
