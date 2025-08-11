from dataset import CM2Dataset, Crop, Noisecm2, Subset, ToTensorcm2
import os
import torch
from utils import Parser, save
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torchvision import transforms
import tifffile
import pandas as pd
from pytorch_msssim import MS_SSIM
from math import log10, sqrt
import torch.optim.lr_scheduler as lr_scheduler
from model import CoordGate, FFTLoss, FPNet
from tensorboardX import SummaryWriter
import numpy as np

def unwrap_model(m):
    return m.module if isinstance(m, nn.DataParallel) else m


parser = argparse.ArgumentParser(description='Train the network for multi-view FPNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train_continue', default='off',  dest='train_continue')
parser.add_argument('--computer', default='scc',choices=['local', 'scc'], dest='computer')
parser.add_argument('--epoch', type=int, default=13, help="Checkpoint epoch number")
parser.add_argument('--task_name', default='SV-CoDe')
# loss funetuning
parser.add_argument('--fft_loss', type=bool, default=False, help='whether apply Fourier loss')
parser.add_argument('--loss_fn', default='ssim',choices=['bce', 'ssim', 'FDMAE', 'FDMSE'], help='Loss besides MSE loss')
parser.add_argument('--alpha', type=float, default=0.85, help='Weight for MSE loss', dest='alpha') #0.85
parser.add_argument('--beta', type=float, default=0.15, help='Weight for SSIM loss', dest='beta')#0.15
parser.add_argument('--precision', default='amp_bf16',
                    choices=['fp32', 'amp_fp16', 'amp_bf16'],
                    help='Numerics mode: pure FP32, AMP FP16, or AMP BF16')
parser.add_argument('--cache_eval', default='off', choices=['off','on'])
parser.add_argument("--num_gpu", type=int, default=[1], dest='num_gpu')
parser.add_argument('--num_epoch', type=int,  default=150, dest='num_epoch')
parser.add_argument('--batch_size', type=int, default=3, dest='batch_size')
parser.add_argument('--lr', type=float, default=1e-4, dest='lr')
parser.add_argument('--train_ratio', type=float, default=0.9, dest='train_ratio')
parser.add_argument('--num_freq_save', type=int,  default=1, dest='num_freq_save')
parser.add_argument("--local_rank", type=int, default=0, dest='local_rank')
parser.add_argument("--early_stop", type=int, default=10, dest='early_stop', help='cancel=None')

if __name__ == '__main__':
    PARSER = Parser(parser)
    args = PARSER.get_arguments()
    args.dir_log  = f'./train/{args.task_name}/log/'
    PARSER.write_args()
    PARSER.print_args()

    # set up the saving folders
    dir_chck = f'./train/{args.task_name}/checkpoints/'
    dir_save = f'./train/{args.task_name}/save/'
    

    torch.manual_seed(3407)
    torch.cuda.empty_cache()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # centers of microlenses in pixels, consistent with forward model
    # v16 and v17
    lens_centers =  [
        (663, 1189), (662, 2089), (657, 2985),
        (1563, 1196), (1557, 2094), (1551, 2991),
        (2461, 1204), (2456, 2102), (2448, 2999)
    ]


    if args.computer == 'local':
        args.dir_data = 'T:/simulation_beads/2d/debug/'
    elif args.computer == 'scc':
        args.dir_data ='/net/engnas/Research/eng_research_cisl/yqw/simulation_beads/2d/lsv_2d_beads_v17'
    else:
        raise ValueError("Unsupported computer environment")

    # create directories for results
    dir_result_val = os.path.join(dir_save, 'val')
    dir_result_train = os.path.join(dir_save, 'train')
    os.makedirs(dir_result_train, exist_ok=True)
    os.makedirs(dir_result_val, exist_ok=True)
    os.makedirs(dir_chck, exist_ok=True)

    # data transformation
    transform_train = transforms.Compose([Crop(lens_centers=lens_centers), Noisecm2(), ToTensorcm2()])
    transform_val = transforms.Compose([Crop(lens_centers=lens_centers), ToTensorcm2()])

    # create the training and validation dataset
    whole_set = CM2Dataset(args.dir_data, transform=transform_val)
    length = len(whole_set)
    train_size = int(args.train_ratio * length)
    validate_size = length - train_size
    train_set, validate_set = torch.utils.data.random_split(whole_set, [train_size, validate_size])
    train_set = Subset(train_set, isVal=False, patch_size=480, stride=240)
    validate_set = Subset(validate_set, isVal=True)
    print(f"Training set size: {len(train_set)}, Validation set size: {len(validate_set)}")

    # apply data transformation
    train_set.dataset.transform = transform_train
    validate_set.dataset.transform = transform_val

    # data loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4, shuffle=True, drop_last=False, timeout=600)
    val_loader = torch.utils.data.DataLoader(validate_set, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

    # initialize model
    model = FPNet().to(args.device)

    # parallel computing
    if torch.cuda.device_count() > 1 and len(args.num_gpu) > 1:
        print(f"Use {torch.cuda.device_count()} GPU")
        model = nn.DataParallel(model)

    # define loss function and optimizer
    def ms_ssim_fp32(x, y, fn):
        # sanitize + force fp32 + contiguous; keep range in [0,1]
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).float().clamp_(0, 1).contiguous()
        y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0).float().clamp_(0, 1).contiguous()
        # run kernel in full precision regardless of surrounding autocast
        with torch.autocast('cuda', enabled=False):
            return fn(x, y)
        
    ssim_loss  = MS_SSIM(data_range=1.0, size_average=True, channel=1)
    ssim_loss9 = MS_SSIM(data_range=1.0, size_average=True, channel=9)
    l2_loss = nn.MSELoss()
    if args.fft_loss:
        fft_loss_fn = FFTLoss(mode=args.loss_fn, norm='ortho')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    # loading from checkpoint if continuing training
    st_epoch = 0
    losslogger = pd.DataFrame()
    best_ssim = 0
    trigger = 0
    if args.train_continue == 'on':
        checkpoint_path = os.path.join(dir_chck,f'best_model/model_epoch{args.epoch:04d}.pth')  # zero-padded to 4 digits
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                new_state_dict = {}
                for k, v in checkpoint.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                model.load_state_dict(new_state_dict)
            optimizer.load_state_dict(checkpoint['optim'])
            st_epoch = args.epoch + 1
            losslogger = checkpoint['losslogger']
            best_ssim = checkpoint.get('best_ssim', 0)
            print(f"Continue training from epoch: {st_epoch}, path: {checkpoint_path}")
        else:
            print(f"Check point '{checkpoint_path}' not found, start from scratch")

    # set TensorBoard
    writer = SummaryWriter(log_dir=args.dir_log)

    # ---- Precision config ----
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True   # allow TF32 on Ampere+
    torch.backends.cudnn.allow_tf32 = True

    prec = args.precision
    use_autocast = (args.device.type == "cuda" and prec != 'fp32')
    amp_dtype = torch.float16 if prec == 'amp_fp16' else torch.bfloat16
    use_scaler = (prec == 'amp_fp16')  # GradScaler only needed for FP16

    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    
    def set_cache_enabled(model: nn.Module, enabled: bool):
        for m in model.modules():
            if isinstance(m, CoordGate):
                m._cache_enabled = enabled
                if not enabled:
                    m.clear_cache()


    for epoch in range(st_epoch + 1, args.num_epoch + 1):
        model.train()
        loss_demix_mse, loss_demix_extra = [], []
        loss_recon_mse, loss_recon_extra = [], []
        loss_total_train = []
        ssim_train, psnr_train = [], []
        for batch, data in enumerate(train_loader, 1):
            gt         = data['gt'].to(args.device, non_blocking=True)
            meas       = data['meas'].to(args.device, non_blocking=True)
            demix_gt   = data['demix'].to(args.device, non_blocking=True)
            index_list = data['index'].to(args.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Forward + MSE under autocast (if enabled)
            with torch.autocast('cuda', enabled=use_autocast, dtype=amp_dtype):
                demix_output, recon_output = model(meas, index_list)
                demix_mse  = l2_loss(demix_output, demix_gt)
                recon_mse = l2_loss(recon_output,  gt)
                
            ssim_index = ms_ssim_fp32(recon_output, gt, ssim_loss)
            if args.loss_fn == 'bce':
                # BCE in full precision (safe with AMP)
                demix_extra = F.binary_cross_entropy(demix_output.float(), demix_gt.float())
                recon_extra = F.binary_cross_entropy(recon_output.float(), gt.float())
            elif args.loss_fn == 'ssim':
                demix_extra = 1.0 - ms_ssim_fp32(demix_output, demix_gt, ssim_loss9)
                recon_extra = 1.0 - ssim_index
            elif args.fft_loss:
                demix_extra = fft_loss_fn(demix_output.float(), demix_gt.float())
                recon_extra = fft_loss_fn(recon_output.float(), gt.float())
            else:
                raise ValueError(f"Unsupported loss function: {args.loss_fn}")
            loss = args.alpha*(demix_mse + recon_mse) + args.beta*(demix_extra + recon_extra)
            
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Metrics (FP32)
            with torch.no_grad():
                mse = l2_loss(recon_output, gt).item()
                psnr = 20 * log10(1.0 / sqrt(mse)) if mse != 0 else 100

            loss_demix_mse.append(demix_mse.item())
            loss_demix_extra.append(demix_extra.item())
            loss_recon_mse.append(recon_mse.item())
            loss_recon_extra.append(recon_extra.item())
            loss_total_train.append(loss.item())
            ssim_train.append(ssim_index.item())
            psnr_train.append(psnr)

            if args.local_rank == 0 and (batch % 10 == 0 or batch == 1):
                print(f'Training: Epoch {epoch}: batch {batch}/{len(train_loader)}: '
                        f'loss_demix_mse: {np.mean(loss_demix_mse):.4f} loss_demix_extra: {np.mean(loss_demix_extra):.4f} '
                        f'loss_recon_mse: {np.mean(loss_recon_mse):.4f} loss_recon_extra: {np.mean(loss_recon_extra):.4f} '
                        f'loss_total: {np.mean(loss_total_train):.4f} SSIM: {np.mean(ssim_train):.4f} PSNR: {np.mean(psnr_train):.2f} dB')

        scheduler.step()

        if args.local_rank == 0:
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Loss/demix_mse', np.mean(loss_demix_mse), epoch)
            writer.add_scalar('Loss/demix_extra', np.mean(loss_demix_extra), epoch)
            writer.add_scalar('Loss/recon_mse', np.mean(loss_recon_mse), epoch)
            writer.add_scalar('Loss/recon_extra', np.mean(loss_recon_extra), epoch)
            writer.add_scalar('Loss/train', np.mean(loss_total_train), epoch)
            writer.add_scalar('SSIM/train', np.mean(ssim_train), epoch)
            writer.add_scalar('PSNR/train', np.mean(psnr_train), epoch)

        print('Validation')
        with torch.no_grad():
            model.eval()
            set_cache_enabled(model, args.cache_eval == 'on') 
            loss_val = []
            ssim_val = []
            psnr_val = []

            for batch, data in enumerate(val_loader, 1):
                if batch > 32: break
                gt         = data['gt'].to(args.device, non_blocking=True)
                meas       = data['meas'].to(args.device, non_blocking=True)
                demix_gt   = data['demix'].to(args.device, non_blocking=True)
                index_list = data['index'].to(args.device, non_blocking=True)

                with torch.autocast('cuda', enabled=use_autocast, dtype=amp_dtype):
                    if args.cache_eval == 'on':
                        # Only when enabled and you want speed over VRAM:
                        net = unwrap_model(model)
                        net.clear_cache()
                        net.prepare_cache(index_list)
                        demix_output, recon_output = model(meas, index_list=None)  # use cached masks
                    else:
                        demix_output, recon_output = model(meas, index_list=index_list)  # ephemeral masks
                    # compute the mse loss
                    demix_mse  = l2_loss(demix_output, demix_gt)
                    mse_loss_v = l2_loss(recon_output,  gt)
                    
                ssim_index = ms_ssim_fp32(recon_output, gt, ssim_loss)
                
                if args.loss_fn == 'bce':
                    # BCE in full precision (safe with AMP)
                    demix_bce = F.binary_cross_entropy(demix_output.float(), demix_gt.float())
                    mse_bce_v = F.binary_cross_entropy(recon_output.float(), gt.float())
                    # Combine losses
                    loss = args.alpha*(demix_mse + mse_loss_v) + args.beta*(demix_bce + mse_bce_v)

                elif args.loss_fn == 'ssim':
                    demix_ssim = 1.0 - ms_ssim_fp32(demix_output, demix_gt, ssim_loss9)
                    ssim_v     = 1.0 - ssim_index
                    loss = args.alpha*(demix_mse + mse_loss_v) + args.beta*(demix_ssim + ssim_v)

                elif args.fft_loss:
                    demix_fft = fft_loss_fn(demix_output.float(), demix_gt.float())
                    fft_v     = fft_loss_fn(recon_output.float(), gt.float())
                    loss = args.alpha*(demix_mse + mse_loss_v) + args.beta*(demix_fft + fft_v)

                else:
                    raise ValueError(f"Unsupported loss function: {args.loss_fn}")

                mse = mse_loss_v.item()
                psnr = 20 * log10(1.0 / sqrt(mse)) if mse != 0 else 100
                loss_val.append(loss.item())
                ssim_val.append(ssim_index.item())
                psnr_val.append(psnr)

                if args.local_rank == 0:
                    print(f'Validation: Epoch {epoch}: batch {batch}/{len(val_loader)}: loss: {np.mean(loss_val):.4f} SSIM: {np.mean(ssim_val):.4f} PSNR: {np.mean(psnr_val):.2f} dB')

            if args.local_rank == 0:
                writer.add_scalar('Loss/val', np.mean(loss_val), epoch)
                writer.add_scalar('SSIM/val', np.mean(ssim_val), epoch)
                writer.add_scalar('PSNR/val', np.mean(psnr_val), epoch)

                if epoch == 1:
                    gt_np = gt.cpu().numpy()
                    im_gt = (np.clip(gt_np[0, 0, ...], 0, 1) * 255).astype(np.uint8)
                    tifffile.imwrite(os.path.join(dir_result_val, f'{epoch}_gt.tif'), im_gt)

                if (epoch % args.num_freq_save) == 0:
                    recon_output_np = recon_output.detach().to(torch.float32).cpu().numpy()
                    im_recon = (np.clip(recon_output_np[0, 0], 0, 1) * 255).astype(np.uint8)
                    tifffile.imwrite(os.path.join(dir_result_val, f'{epoch}_recon.tif'), im_recon)

                    demix_output_np = demix_output.detach().to(torch.float32).cpu().numpy()  # <-- cast here
                    im_demix = (np.clip(demix_output_np[0, 0], 0, 1) * 255).astype(np.uint8)
                    tifffile.imwrite(os.path.join(dir_result_val, f'{epoch}_demix.tif'), im_demix)

        if args.local_rank == 0:
            df = pd.DataFrame({
                'epoch': [epoch],
                'lr': [optimizer.param_groups[0]['lr']],
                'loss_demix_mse': [np.mean(loss_demix_mse)],
                'loss_demix_extra': [np.mean(loss_demix_extra)],
                'loss_recon_mse': [np.mean(loss_recon_mse)],
                'loss_recon_extra': [np.mean(loss_recon_extra)],
                'loss_train': [np.mean(loss_total_train)],
                'ssim_train': [np.mean(ssim_train)],
                'psnr_train': [np.mean(psnr_train)],
                'loss_val': [np.mean(loss_val)],
                'ssim_val': [np.mean(ssim_val)],
                'psnr_val': [np.mean(psnr_val)]
            })
            losslogger = pd.concat([losslogger, df], ignore_index=True)

            trigger += 1
            current_ssim = np.mean(ssim_val)
            if current_ssim > best_ssim:
                save(dir_chck+ '/best_model/', model, optimizer, epoch, losslogger)
                best_ssim = np.mean(ssim_val)
                print("=>saved best model")
                trigger = 0

            # early stop
            if args.early_stop is not None and trigger >= args.early_stop:
                print("=> Early Stopping")
                break

            # save model at regular intervals
            if (epoch % args.num_freq_save) == 0:
                save(dir_chck, model, optimizer, epoch, losslogger)
