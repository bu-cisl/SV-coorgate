import argparse
import tifffile
from model import FPNet, CoordGate
from dataset import indexGenerate
import os
from pytorch_msssim import MS_SSIM
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import pandas as pd

def unwrap_model(m):
    return m.module if isinstance(m, nn.DataParallel) else m

lens_centers =  [
        (663, 1189), (662, 2089), (657, 2985),
        (1563, 1196), (1557, 2094), (1551, 2991),
        (2461, 1204), (2456, 2102), (2448, 2999)
    ]

def main():
    parser = argparse.ArgumentParser(description='Test the network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir_data', default='/net/engnas/Research/eng_research_cisl/yqw/simulation_beads/2d/lsv_2d_beads_v17_test')
    parser.add_argument('--task_name', default='FDMSE-1e-3')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='Optional checkpoint path. If omitted, uses train/{task_name}/checkpoints/best_model/model_epoch{epoch:04d}.pth',
    )
    # network structure related
    parser.add_argument("--is_gate", type=bool, default=True, help='whether apply coordinate gate')
    parser.add_argument("--is_pe", type=bool, default=True, help='whether apply positional encoding')
    parser.add_argument("--full_rs_only", type=bool, default=False, help='whether apply pe on only first layer')
    parser.add_argument('--cache_eval', default='off', choices=['off','on'])
    parser.add_argument('--num_samples', type=int, default=300)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--precision', default='amp_fp16', choices=['fp32', 'amp_fp16', 'amp_bf16'])
    args = parser.parse_args()
    
    # Define the directory to save reconstructed images
    dir_result = f'test/simulation/{args.task_name}_{args.epoch}/'
    os.makedirs(dir_result, exist_ok=True)
    
    # Set device and precision
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    use_autocast = (device.type == "cuda" and args.precision != 'fp32')
    amp_dtype = torch.float16 if args.precision == 'amp_fp16' else torch.bfloat16
    print(f'Using device: {device}, precision: {args.precision}')
    
    # Initialize model
    model = FPNet(L=4, is_gate = args.is_gate, is_pe= args.is_pe, full_rs_only=args.full_rs_only).to(device)
    
    # load weights
    ckpt = args.checkpoint or f'train/{args.task_name}/checkpoints/best_model/model_epoch{args.epoch:04d}.pth'
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    elif 'model' in state:
        model.load_state_dict(state['model'])
    else:
        new_state = { (k[7:] if k.startswith('module.') else k): v for k, v in state.items() }
        model.load_state_dict(new_state)
    print(f'Loaded model: {ckpt}')
 

    # Set parameters matching the Crop class in dataset.py
    tmp_pad = 900
    tot_len = 2400  # This is the crop size

    # Precompute crop boxes once
    boxes = []
    for (xc, yc) in lens_centers:
        x0 = xc - (tot_len // 2) + tmp_pad
        x1 = xc + (tot_len // 2) + tmp_pad
        y0 = yc - (tot_len // 2) + tmp_pad
        y1 = yc + (tot_len // 2) + tmp_pad
        boxes.append((x0, x1, y0, y1))

    # Precompute once (same geometry for all images)
    idx_list = []
    for (x0, x1, y0, y1) in boxes:
        # local coords -> [H,W,2] tensor on GPU
        idx_hw2 = indexGenerate(False, x0, y0, tot_len, tot_len)
        idx_list.append(idx_hw2)
    index_list = torch.stack(idx_list, dim=0).unsqueeze(0).to(device)  # [1,9,H,W,2]

    # Utility to unwrap model from DDP if needed
    def set_cache_enabled(model: nn.Module, enabled: bool):
        for m in model.modules():
            if isinstance(m, CoordGate):
                m._cache_enabled = enabled
                if not enabled:
                    m.clear_cache()

    # ---- inference loop ----
    ssim_recon_all, psnr_recon_all = [], []
    ssim_demix_all, psnr_demix_all = [], []
    set_cache_enabled(model, args.cache_eval == 'on') 

    # initialize the loss function
    ssim_loss  = MS_SSIM(data_range=1.0, size_average=True, channel=1).to(device)
    ssim_loss9 = MS_SSIM(data_range=1.0, size_average=True, channel=9).to(device)
    mse_loss  = nn.MSELoss().to(device)  # optional: reuse for recon & demix
    def ms_ssim_fp32(x, y, fn):
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).float().clamp_(0, 1).contiguous()
        y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0).float().clamp_(0, 1).contiguous()
        with torch.autocast('cuda', enabled=False):
            return fn(x, y)

    
    rows = []
    # Evaluation mode
    with torch.no_grad():
        model.eval()
        with torch.inference_mode():
            for idx in range(1, args.num_samples + 1):
                # Define paths
                meas_path = os.path.join(args.dir_data, f'meas_{idx}.tif')
                gt_path = os.path.join(args.dir_data, f'gt_{idx}.tif')
                demix_gt_path = os.path.join(args.dir_data, f'demix_{idx}.tif')
                
                # Load measurement image
                meas = tifffile.imread(meas_path).astype('float32')  # [H, W]
                meas = meas / meas.max() if meas.max() > 0 else meas
                
                # Pad meas
                meas = torch.from_numpy(meas).to(device)
                meas = F.pad(meas, (tmp_pad, tmp_pad, tmp_pad, tmp_pad), 'constant', 0)
                
                # ---- crop 9 views (GPU) ----
                crops = [meas[x0:x1, y0:y1] for (x0, x1, y0, y1) in boxes]
                meas_crops = torch.stack(crops, dim=0).unsqueeze(0)  # [1,9,H,W]

                # ---- forward (cached path: index_list=None) ----
                torch.cuda.synchronize(); t0=time.perf_counter()
                with torch.autocast('cuda', enabled=use_autocast, dtype=amp_dtype):
                    if args.cache_eval == 'on':
                        # Only when enabled and you want speed over VRAM:
                        net = unwrap_model(model)
                        net.clear_cache()
                        net.prepare_cache(index_list)
                        demix_output, recon_output = model(meas_crops, index_list=None)  # use cached masks
                    else:
                        demix_output, recon_output = model(meas_crops, index_list=index_list)  # ephemeral masks
                torch.cuda.synchronize()
                model_ms = (time.perf_counter() - t0) * 1000.0
                print(f"Model-only: {model_ms:.2f} ms")
                
                # Load ground truth image
                gt_image = tifffile.imread(gt_path).astype('float32')  # [H, W]
                gt = gt_image / gt_image.max() if gt_image.max() > 0 else gt_image
                gt = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
                
                # Load ground truth demix images
                demix_gt_stack = tifffile.imread(demix_gt_path).astype('float32')  # [9, H, W] or [H, W, 9]
                if demix_gt_stack.ndim == 3 and demix_gt_stack.shape[0] == 9:
                    demix_gt = demix_gt_stack  # [9, H, W]
                elif demix_gt_stack.ndim == 3 and demix_gt_stack.shape[2] == 9:
                    demix_gt = demix_gt_stack.transpose(2, 0, 1)  # [9, H, W]
                else:
                    raise ValueError(f"Unexpected demix_gt_stack shape: {demix_gt_stack.shape} in file {demix_gt_path}")
                demix_gt = demix_gt / demix_gt.max() if demix_gt.max() > 0 else demix_gt
                demix_gt = torch.from_numpy(demix_gt).float().unsqueeze(0).to(device)  # [1, 9, H, W]
                
                # ---- per-image normalization for recon (GPU) ----
                r = recon_output
                r_min = r.amin(dim=(2,3), keepdim=True)
                r_max = r.amax(dim=(2,3), keepdim=True)
                r_rng = torch.clamp(r_max - r_min, min=1e-8)
                recon_norm = (r - r_min) / r_rng

                g = gt
                g_min = g.amin(dim=(2,3), keepdim=True)
                g_max = g.amax(dim=(2,3), keepdim=True)
                g_rng = torch.clamp(g_max - g_min, min=1e-8)
                gt_norm = (g - g_min) / g_rng

                # ---- per-channel normalization for demix (GPU) ----
                d = demix_output
                d_min = d.amin(dim=(2,3), keepdim=True)         # [1,9,1,1]
                d_max = d.amax(dim=(2,3), keepdim=True)
                d_rng = torch.clamp(d_max - d_min, min=1e-8)
                demix_norm = (d - d_min) / d_rng                # [1,9,H,W]

                dg = demix_gt
                dg_min = dg.amin(dim=(2,3), keepdim=True)
                dg_max = dg.amax(dim=(2,3), keepdim=True)
                dg_rng = torch.clamp(dg_max - dg_min, min=1e-8)
                demix_gt_norm = (dg - dg_min) / dg_rng

                # ---- metrics (GPU) ----
                # Recon SSIM / PSNR
                ssim_recon = ms_ssim_fp32(recon_norm, gt_norm, ssim_loss)
                mse_recon = mse_loss(recon_output.squeeze(1), gt)  # consistent with your train/val
                psnr_recon = 20.0 * torch.log10(1.0 / torch.sqrt(torch.clamp(mse_recon, min=1e-12)))

                # # Demix SSIM: average over channels (loop of 9 calls is cheap)
                ssim_demix = ms_ssim_fp32(demix_norm, demix_gt_norm, ssim_loss9)

                # Demix PSNR: vectorized per-channel MSE
                mse_demix_ch = ((demix_output - demix_gt) ** 2).mean(dim=(0,2,3))     # [9]
                psnr_demix_ch = 20.0 * torch.log10(1.0 / torch.sqrt(torch.clamp(mse_demix_ch, min=1e-12)))
                psnr_demix = psnr_demix_ch.mean()

                # ---- save (CPU) ----
                recon_u16 = (recon_norm.squeeze().clamp(0,1).mul_(65535).to(torch.uint16)).cpu().numpy()
                tifffile.imwrite(os.path.join(dir_result, f'recon_{idx}.tif'), recon_u16)

                demix_u16 = (demix_norm.squeeze(0).clamp(0,1).mul_(65535).to(torch.uint16)).cpu().numpy()  # [9,H,W]
                tifffile.imwrite(os.path.join(dir_result, f'demix_{idx}.tif'), demix_u16)

                # ---- log ----
                ssim_r = float(ssim_recon.detach().cpu())
                psnr_r = float(psnr_recon.detach().cpu())
                ssim_d = float(ssim_demix.detach().cpu())
                psnr_d = float(psnr_demix.detach().cpu())
                ssim_recon_all.append(ssim_r); psnr_recon_all.append(psnr_r)
                ssim_demix_all.append(ssim_d); psnr_demix_all.append(psnr_d)

                rows.append({
                    "index": idx,
                    "recon_psnr": psnr_r,
                    "recon_ssim": ssim_r,
                    "demix_psnr": psnr_d,
                    "demix_ssim": ssim_d,
                    "model_ms": model_ms
                })

                print(f'#{idx:04d}/{args.num_samples} | Recon PSNR {psnr_r:.4f} SSIM {ssim_r:.4f} | '
                    f'Demix PSNR {psnr_d:.4f} SSIM {ssim_d:.4f}')

        df = pd.DataFrame(rows)

        # compute mean across numeric columns
        mean_vals = df.select_dtypes(include=[float, int]).mean(numeric_only=True)
        mean_row = {col: mean_vals.get(col, None) for col in df.columns}
        mean_row["index"] = "MEAN"  # label for the last row

        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

        # save CSV (always works, no extra deps)
        csv_path = os.path.join(dir_result, "metrics.csv")
        df.to_csv(csv_path, index=False, float_format="%.6f")
        print(f"Saved CSV: {csv_path}")

        print(f'Average Recon PSNR: {np.mean(psnr_recon_all):.4f}, Average Recon SSIM: {np.mean(ssim_recon_all):.4f}')
        print(f'Average Demix PSNR: {np.mean(psnr_demix_all):.4f}, Average Demix SSIM: {np.mean(ssim_demix_all):.4f}')

if __name__ == "__main__":
    main()
