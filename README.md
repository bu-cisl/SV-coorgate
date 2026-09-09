# SV-CoDe

PyTorch implementation of nine-view image demixing and 2D reconstruction using
coordinate-gated networks with positional encoding.

## Setup

Use an NVIDIA GPU with CUDA-enabled PyTorch. Run commands from the repository root.

```bash
conda create -n SV-CoDe python=3.11 -y
conda activate SV-CoDe
```

Install matching `torch` and `torchvision` packages using the
[PyTorch installation selector](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install numpy pandas tifffile pytorch-msssim tensorboardX
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Dependency versions are not pinned; this setup has not been validated end to end.
Evaluation currently requires CUDA and does not support CPU or Apple MPS.

## Data

Supply a directory containing matching TIFF files for each sample:

| File | Contents / shape |
| --- | --- |
| `meas_N.tif` | 2D measurement containing nine lens views |
| `demix_N.tif` | Ground-truth views: `(9, 2400, 2400)` or `(2400, 2400, 9)` |
| `gt_N.tif` | Reconstruction target: `(2400, 2400)` |

Training selects indices `1, 4, 7, ...`, with dataset length equal to the number
of measurement files divided by three (rounded down). Evaluation reads indices
`1` through `--num_samples`; all three files are required for each index.
Data is supplied separately. Lens centers in `train_amp.py` and `result.py` are
configured for the v16/v17 geometry; update both for other acquisitions.

## Training

Edit the `args.dir_data` path in the `scc` branch of `train_amp.py` first.
Training does not accept a `--dir_data` option.

```bash
python train_amp.py --task_name SV_CoDe --computer scc
```

Defaults: 35 epochs, batch size 3, learning rate `1e-4`, BF16 mixed precision,
and MSE + MS-SSIM loss. Use `--precision fp32` or `--precision amp_fp16` if needed.
Checkpoints, logs, and validation previews are saved under `train/SV_CoDe/`.

## Evaluation

Provide the checkpoint and replace the test data path:

```bash
python result.py --task_name SV_CoDe --epoch 35 \
  --checkpoint checkpoints/model_epoch0035.pth \
  --dir_data /path/to/test_data --num_samples 300
```

Use `--num_samples 1` for an initial check. Reconstructed and demixed TIFFs plus
`metrics.csv` (PSNR, SSIM, and timing) are saved to `test/simulation/SV_CoDe_35/`.
Use `python train_amp.py --help` or `python result.py --help` for available options.

## Cluster jobs

The qsub scripts activate `SV-CoDe`. Adjust their project, resource requests,
and data paths for your cluster, then submit from the repository root:

```bash
qsub -cwd train_SV_CoDe.qsub
qsub -cwd test_SV_Code.qsub
```

These jobs are independent. The test job uses the explicit epoch-35 checkpoint.
The current `.gitignore` excludes `*.qsub`, so include these scripts separately
when sharing the materials.

## Code

- `model.py`: network architecture and Fourier loss.
- `dataset.py`: TIFF loading, cropping, and patch extraction.
- `train_amp.py` / `result.py`: training and evaluation.
- `utils.py`: argument logging and checkpoint saving.
