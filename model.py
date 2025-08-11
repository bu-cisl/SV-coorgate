import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# LayerNorm2d remains the same
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        C = x.size(1)
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        C = grad_output.size(1)
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=(0, 2, 3)), grad_output.sum(dim=(0, 2, 3)), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

# PositionEncoding class
class PositionEncoding(nn.Module):
    """
    Positional Encoding using sine and cosine functions.
    Args:
        L: Number of frequency bands.
    Outputs:
        pos_enc: Positional encoding tensor of shape [Batch, 4L, H, W]. 
        2 coordinates (x_global, y_global) each with 2L channels (sin and cos for L frequencies).
        2 * 2L = 4L

    """
    def __init__(self, L):
        super(PositionEncoding, self).__init__()
        self.L = L
        freq_bands = 0.5 * np.pi * torch.arange(1, L + 1).float()
        self.register_buffer('freq_bands_buffer', freq_bands)

    def forward(self, index_list, is_pe = True):
        # index_list: [Batch, H, W, 2]
        batch_size, height, width, _ = index_list.shape
        x_global = index_list[:, :, :, 0]
        y_global = index_list[:, :, :, 1]
        if is_pe:
            freq = self.freq_bands_buffer.view(1, 1, 1, self.L)
            sin_x_global = torch.sin(freq * x_global.unsqueeze(-1))
            cos_x_global = torch.cos(freq * x_global.unsqueeze(-1))
            sin_y_global = torch.sin(freq * y_global.unsqueeze(-1))
            cos_y_global = torch.cos(freq * y_global.unsqueeze(-1))

            # Concatenate all positional encodings
            pos_enc = torch.cat([
                sin_x_global, cos_x_global,
                sin_y_global, cos_y_global
            ], dim=-1) # [Batch, H, W, 4L]
        else:
            # else return regular coordinates without encoding
            batch_size, height, width, _ = index_list.shape
            x_global = index_list[:, :, :, 0]
            y_global = index_list[:, :, :, 1]
            pos_enc = torch.stack([x_global, y_global], dim=-1) # [Batch, H, W, 2]
        
        pos_enc = pos_enc.view(batch_size, height, width, -1)
        pos_enc = pos_enc.permute(0, 3, 1, 2)  # [Batch, C, H, W]

        return pos_enc

# Mask class used in CoordGate
class Mask(nn.Module):
    # num_in: input channels (from positional encoding 2*2L or regular coords 2)
    # num_hidden: hidden units in the mask network: 64
    # num_out: output channels (to match conv channels)
    def __init__(self, num_in, num_hidden, num_out):
        super(Mask, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_in, num_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(num_hidden, num_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, height, width, num_in = x.shape
        x = x.reshape(-1, num_in)
        x = self.net(x)
        x = x.reshape(batch_size, height, width, -1)
        return x

# CoordGate class
class CoordGate(nn.Module):
    """
    CoordGate module that applies a learned mask based on positional encoding.
    Args:
        num_in: Number of input channels for the positional encoding (e.g., 8L).
        num_hidden: Number of hidden units in the mask network.
        CNN_num_out: Number of output channels after applying the mask --> here should be same with CNN_num_out_conv to enable multiplication
        CNN_num_in: Number of input channels to the convolutional layer.
        CNN_num_out_conv: Number of output channels from the convolutional layer.
        kernel_size: Kernel size for the convolution (1 or 3).
        stride: Stride for the convolution (default is 1).
    """
    def __init__(self, num_in, num_hidden, CNN_num_out, CNN_num_in, CNN_num_out_conv, kernel_size, stride=1, is_gate=True):
        super(CoordGate, self).__init__()
        self.is_gate = is_gate
        if self.is_gate:
            self.mask = Mask(num_in, num_hidden, CNN_num_out)
        if CNN_num_in == CNN_num_out_conv:
            if kernel_size == 3:
                self.pre_conv = nn.Conv2d(
                    in_channels=CNN_num_in,
                    out_channels=CNN_num_out_conv,
                    kernel_size=3,
                    padding=1,
                    stride=stride,
                    groups=CNN_num_out_conv,
                    bias=True
                )
            else:
                self.pre_conv = nn.Conv2d(
                    in_channels=CNN_num_in,
                    out_channels=CNN_num_out_conv,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    groups=CNN_num_out_conv,
                    bias=True
                )
        else:
            if kernel_size == 3:
                self.pre_conv = nn.Conv2d(
                    in_channels=CNN_num_in,
                    out_channels=CNN_num_out_conv,
                    kernel_size=3,
                    padding=1,
                    stride=stride,
                    groups=1,
                    bias=True
                )
            else:
                self.pre_conv = nn.Conv2d(
                    in_channels=CNN_num_in,
                    out_channels=CNN_num_out_conv,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    groups=1,
                    bias=True
                )
        
        # ---- cache fields ----
        self._cached_mask = None        # [1, C, H, W] or None
        self._cached_hw = None          # (H, W)
        self._cache_enabled = True
    
    @torch.no_grad()
    def precompute_mask(self, index_list_or_pe):
        """
        index_list_or_pe: [1, H, W, inC]  (PE already computed or raw coords encoded upstream)
        Stores mask as [1, C, H, W] in module dtype on module device.
        """
        if not self._cache_enabled: return
        if not self.is_gate:
            self._cached_mask, self._cached_hw = None, None
            return

        dev = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Compute mask once (batch-agnostic: use batch=1)
        m = self.mask(index_list_or_pe)            # [1, H, W, C]
        m = m.permute(0, 3, 1, 2).contiguous()     # [1, C, H, W]

        # Match channels if depthwise width differs
        in_c = m.shape[1]
        out_c = self.pre_conv.out_channels
        if in_c != out_c:
            repeat_factor = out_c // in_c
            m = m.repeat(1, repeat_factor, 1, 1)

        self._cached_mask = m.to(device=dev, dtype=dtype)
        self._cached_hw = (m.shape[-2], m.shape[-1])

    def clear_cache(self):
        self._cached_mask, self._cached_hw = None, None

    # either apply the coordinate gate or use traditional convolution
    def forward(self, x, index_list_or_pe=None, use_cache=True):
        # conv first
        x = self.pre_conv(x)
        # if not gating, return conv output directly
        if not self.is_gate:
            return x

        if self.training:
            # ---- TRAINING: compute mask with grad; do NOT touch cache ----
            if index_list_or_pe is None:
                raise RuntimeError("CoordGate(training): index/PE required to compute mask.")
            
            # compute the mask with grad
            m = self.mask(index_list_or_pe)                 # [B,H,W,C_inMask]
            m = m.permute(0, 3, 1, 2).contiguous()          # [B,C_inMask,H,W]

            out_c = self.pre_conv.out_channels
            in_c  = m.shape[1]

            # match channels if depthwise width differs
            if in_c != out_c:
                if out_c % in_c != 0:
                    raise RuntimeError(f"Mask channels ({in_c}) must divide conv out ({out_c}).")
                repeat_factor = out_c // in_c
                m = m.repeat(1, repeat_factor, 1, 1)        # still keeps grad

            # extend the mask in batch dimension
            if m.shape[0] != x.shape[0]:
                m = m.expand(x.shape[0], -1, -1, -1)
            return x * m
        
        # ---------------- EVAL ----------------
        H, W = x.shape[-2], x.shape[-1]

        # Eval WITH cache (only if requested + enabled)
        if use_cache and self._cache_enabled:
            if (self._cached_mask is None) or (self._cached_hw != (H, W)):
                if index_list_or_pe is None:
                    raise RuntimeError("CoordGate(eval): need index/PE to build cache.")
                # build & store cache once
                self.precompute_mask(index_list_or_pe[:1])
            mask = self._cached_mask
            if mask.shape[0] != x.shape[0]:
                mask = mask.expand(x.shape[0], -1, -1, -1)
            return x * mask

        # Eval NO-CACHE (ephemeral mask; do not store)
        if index_list_or_pe is None:
            raise RuntimeError("CoordGate(eval no-cache): index/PE required.")
        with torch.no_grad():
            m = self.mask(index_list_or_pe).permute(0,3,1,2).contiguous()
            out_c = self.pre_conv.out_channels
            in_c  = m.shape[1]
            if in_c != out_c:
                if out_c % in_c != 0:
                    raise RuntimeError(f"Mask channels ({in_c}) must divide conv out ({out_c}).")
                m = m.repeat(1, out_c // in_c, 1, 1)
            if m.shape[0] != x.shape[0]:
                m = m.expand(x.shape[0], -1, -1, -1)
        return x * m


# SimpleGate mechanism
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

# Simplified Channel Attention (SCA)
class SCA(nn.Module):
    def __init__(self, channels):
        super(SCA, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=True
        )

    def forward(self, x):
        y = self.global_pool(x)
        y = self.conv(y)
        return x * y

class SEFusion(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEFusion, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [Batch, in_channels, H, W]
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# NAFBlock with CoordGate
class NAFBlock_gate(nn.Module):
    def __init__(self, c, num_in, num_hidden, is_gate=True):
        super(NAFBlock_gate, self).__init__()
        dw_channel = c * 2
        self.is_gate = is_gate
        # Replace conv1 with CoordGate
        self.coord_gate1 = CoordGate(num_in, num_hidden, dw_channel, CNN_num_in=c, CNN_num_out_conv=dw_channel, kernel_size=1, is_gate=self.is_gate)
        # Replace conv2 with CoordGate (Depthwise)
        self.coord_gate2 = CoordGate(num_in, num_hidden, dw_channel, CNN_num_in=dw_channel, CNN_num_out_conv=dw_channel, kernel_size=3, is_gate=self.is_gate)
        # Replace conv3 with CoordGate
        self.coord_gate3 = CoordGate(num_in, num_hidden, c, CNN_num_in=dw_channel // 2, CNN_num_out_conv=c, kernel_size=1, is_gate=self.is_gate)

        # Simplified Channel Attention
        self.sca = SCA(dw_channel // 2)

        # SimpleGate
        self.sg = SimpleGate()

        # FFN Part
        ffn_channel = c * 2
        self.coord_gate4 = CoordGate(num_in, num_hidden, ffn_channel, CNN_num_in=c, CNN_num_out_conv=ffn_channel, kernel_size=1, is_gate=self.is_gate)
        self.coord_gate5 = CoordGate(num_in, num_hidden, c, CNN_num_in=ffn_channel // 2, CNN_num_out_conv=c, kernel_size=1, is_gate=self.is_gate)

        # Using LayerNorm2d
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
    
    @torch.no_grad()
    def prepare_cache(self, pe_hwC):
        """pe_hwC: [1,H,W,Cpe] — precompute masks for all gates at this resolution (eval only)."""
        if not self.is_gate: return
        for g in (self.coord_gate1, self.coord_gate2, self.coord_gate3, self.coord_gate4, self.coord_gate5):
            g.precompute_mask(pe_hwC)

    @torch.no_grad()
    def clear_cache(self):
        if not self.is_gate: return
        for g in (self.coord_gate1, self.coord_gate2, self.coord_gate3, self.coord_gate4, self.coord_gate5):
            g.clear_cache()

    def forward(self, x, index_list):
        """
        pe_hwC: [B,H,W,Cpe] when training or on-the-fly eval; or None to use cached masks in eval.
        """
        use_cache = (index_list is None) and (not self.training)

        inp = x

        x = self.norm1(x)

        x = self.coord_gate1(x, index_list, use_cache)
        x = self.coord_gate2(x, index_list, use_cache)

        x = self.sg(x)
        x = self.sca(x)

        x = self.coord_gate3(x, index_list, use_cache)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.coord_gate4(x, index_list, use_cache)

        x = self.sg(x)

        x = self.coord_gate5(x, index_list, use_cache)

        return y + x * self.gamma

# Sequential module supporting multiple inputs
class SequentialMultiInput(nn.Sequential):
    def forward(self, x, index_list = None):
        for module in self._modules.values():
            x = module(x, index_list)
        return x
    
    @torch.no_grad()
    def prepare_cache(self, index_list):
        for module in self._modules.values():
            if hasattr(module, "prepare_cache"):
                module.prepare_cache(index_list)

    @torch.no_grad()
    def clear_cache(self):
        for module in self._modules.values():
            if hasattr(module, "clear_cache"):
                module.clear_cache()

# NAFNet with CoordGate, PositionEncoding 
class NAFNet_gate(nn.Module):
    def __init__(self, img_channel=1, width=32, middle_blk_num=1,
                 enc_blk_nums=[1, 1, 1, 1], dec_blk_nums=[1, 1, 1, 1],
                 num_in=8 * 4, num_hidden=64, num_views=9, L=4, output_channels=1, is_gate=True, is_pe=True, full_rs_only = False):
        super(NAFNet_gate, self).__init__()

        self.num_views = num_views
        self.is_gate = is_gate
        if is_gate:
            self.is_pe = is_pe
            self.position_encoding = PositionEncoding(L=L)

        # Intro layers for each view
        self.intro = nn.ModuleList([
            CoordGate(num_in, num_hidden, width, CNN_num_in=img_channel, CNN_num_out_conv=width, kernel_size=3, is_gate=is_gate)
            for _ in range(num_views)
        ])

        # Fusion layer
        self.fuse_se = SEFusion(in_channels=width * num_views, reduction=16)
        self.fuse_conv = nn.Conv2d(width * num_views, width, kernel_size=1, stride=1, padding=0)

        # Shared encoders and decoders
        self.encoders, self.decoders = nn.ModuleList(), nn.ModuleList()
        self.middle_blks = SequentialMultiInput()
        self.ups, self.downs = nn.ModuleList(), nn.ModuleList()

        # if full_rs_only is True, disable gating in all but full-res layers (only view dependent first layer MLPs)
        if full_rs_only:
            is_gate = False

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(SequentialMultiInput(*[NAFBlock_gate(chan, num_in, num_hidden, is_gate) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, kernel_size=2, stride=2))
            chan *= 2

        self.middle_blks = SequentialMultiInput(*[NAFBlock_gate(chan, num_in, num_hidden, is_gate) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.ConvTranspose2d(chan, chan // 2, kernel_size=2, stride=2))
            chan //= 2
            self.decoders.append(SequentialMultiInput(*[NAFBlock_gate(chan, num_in, num_hidden, is_gate) for _ in range(num)]))

        # Ending layer
        self.ending = CoordGate(num_in, num_hidden, output_channels, CNN_num_in=chan, CNN_num_out_conv=output_channels, kernel_size=3, is_gate=is_gate)
    
    # ---------- helpers ----------
    @torch.no_grad()
    def _build_pe_scales(self, index_base, device, dtype):
        """
        index_base: [B,H,W,2] (full resolution coordinates: x_global, y_global)
        returns: list of PEs [1,H_s,W_s,Cpe] for s=0..S where S=#downs
        """
        pe_scales = []
        idx = index_base
        pe = self.position_encoding(idx, self.is_pe).permute(0,2,3,1).contiguous()
        pe_scales.append(pe[:1].to(device=device, dtype=dtype))
        for _ in range(len(self.downs)):
            idx = F.interpolate(idx.permute(0,3,1,2), scale_factor=0.5, mode='bilinear', align_corners=False).permute(0,2,3,1)
            pe = self.position_encoding(idx, self.is_pe).permute(0,2,3,1).contiguous()
            pe_scales.append(pe[:1].to(device=device, dtype=dtype))
        return pe_scales

    @torch.no_grad()
    def prepare_cache(self, index_list):
        """
        Cache masks at all gated layers for current geometry/resolution.
        index_list: [B, V, H, W, 4]
        """
        if not self.is_gate: return
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype

        # Intro (per view, full res)
        for i in range(self.num_views):
            idx_i = index_list[:, i, :, :, :]
            pe_i = self.position_encoding(idx_i, self.is_pe).permute(0,2,3,1)
            self.intro[i].precompute_mask(pe_i[:1].to(device=device, dtype=dtype))

        # Pyramid PE (shared across trunk)
        base = index_list.mean(dim=1)  # [B,H,W,4]
        pe_scales = self._build_pe_scales(base, device, dtype)  # s0..sS

        # Encoders (level-aligned)
        for lvl, enc in enumerate(self.encoders):
            enc.prepare_cache(pe_scales[lvl])

        # Middle (lowest)
        self.middle_blks.prepare_cache(pe_scales[-1])

        # Decoders (reverse excluding lowest)
        dec_pes = list(reversed(pe_scales[:-1]))
        for dec, pe in zip(self.decoders, dec_pes):
            dec.prepare_cache(pe)

        # Ending at full res
        self.ending.precompute_mask(pe_scales[0])

    @torch.no_grad()
    def clear_cache(self):
        if not self.is_gate: return
        for i in range(self.num_views):
            self.intro[i].clear_cache()
        for enc in self.encoders: enc.clear_cache()
        self.middle_blks.clear_cache()
        for dec in self.decoders: dec.clear_cache()
        self.ending.clear_cache()

    def forward(self, x, index_list=None):
        """
        Training:
            pass index_list -> PEs are computed with grad; masks learned.
        Inference:
            call prepare_cache(index_list) once, then forward with index_list=None to use caches.
        """
        # x: [Batch, num_views, 1, H, W]
        # index_list: [Batch, num_views, H, W, 4]
        num_views = x.size(1)
        feats = []

        # Intro conv: Process each view and concatenate features along channel dimension
        for i in range(num_views):
            xi = x[:, i, :, :, :]  # [Batch, 1, H, W]
            if self.is_gate:
                # training time calculate PE on-the-fly
                if self.training:
                    index_i = index_list[:, i, :, :, :]  # [Batch, H, W, 4]
                    # Generate positional encoding
                    pos_enc = self.position_encoding(index_i, self.is_pe)  # [Batch, 8L, H, W]
                    pos_enc = pos_enc.permute(0, 2, 3, 1)  # [Batch, H, W, 4L]
                    xi = self.intro[i](xi, pos_enc)
                # inference time use cached masks and don't pass index as input
                else:
                    # when the idex_list is None, meaning during inference
                    if index_list is None:
                        xi = self.intro[i](xi, None, use_cache=True)
                    # don't use cache, compute PE on-the-fly during inference
                    else:
                        idx_i = index_list[:, i, :, :, :]
                        pe_i = self.position_encoding(idx_i, self.is_pe).permute(0,2,3,1)
                        xi = self.intro[i](xi, pe_i, use_cache=False)
            else:
                xi = self.intro[i](xi, None)
            feats.append(xi)  # [Batch, C, H, W] # C: width

        # fuse the features across the views
        x = torch.cat(feats, dim=1)  # [Batch, num_views * C, H, W]
        x = self.fuse_se(x)
        x = self.fuse_conv(x)       # [Batch, C, H, W]

        # Build pos-enc pyramid ONLY if gating
        pos_enc_list  = []
        if self.is_gate and (self.training or index_list is not None):
            # Build index pyramid --> full resolution index
            index = index_list.mean(dim=1)  # [Batch, H, W, 4] --> the same for all views
            pos_enc = self.position_encoding(index, self.is_pe).permute(0, 2, 3, 1)  # [Batch, H, W, 8L]
            pos_enc_list.append(pos_enc)
        else:
            pos_enc = None
            pos_enc_list.append(None)

        enc_features = []

        # Encoder
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x, pos_enc)
            enc_features.append(x)
            x = down(x)
            if self.is_gate and (self.training or index_list is not None):
                # Downsample positional encoding --> downsample to match feature map size
                index = F.interpolate(index.permute(0, 3, 1, 2), scale_factor=0.5, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                pos_enc = self.position_encoding(index, self.is_pe).permute(0, 2, 3, 1)
            else:
                pos_enc = None
            pos_enc_list.append(pos_enc)

        # Middle blocks
        x = self.middle_blks(x, pos_enc_list[-1]) # None in cached eval = use caches

        # Decoder
        for decoder, up, skip, pos_enc in zip(self.decoders, self.ups, reversed(enc_features), reversed(pos_enc_list[:-1])):
            x = up(x)
            x = x + skip
            x = decoder(x, pos_enc) # None in cached eval = use caches

        # Ending
        if self.is_gate:
            if self.training:
                x = self.ending(x, pos_enc_list[0], use_cache=False)   # train: with grad
            else:
                if index_list is None:
                    x = self.ending(x, None, use_cache=True)           # eval + cached
                else:
                    x = self.ending(x, pos_enc_list[0], use_cache=False)  # eval + no-cache (compute)
        else:
            x = self.ending(x, None)

        return x

# FPNet class
class FPNet(nn.Module):
    def __init__(self, L=4, is_gate=True, is_pe=True, full_rs_only = False):
        super(FPNet, self).__init__()
        if is_pe:
            num_in = 4 * L  # Positional encoding channels
        else:
            num_in = 2
        # Demix network, outputs [Batch, 9, H, W]
        self.demix_net = NAFNet_gate(
            img_channel=1,   # Measurement is single-channel
            width=32,
            middle_blk_num=1,
            enc_blk_nums=[1, 1, 1, 1],
            dec_blk_nums=[1, 1, 1, 1],
            num_in=num_in,
            num_hidden=64,
            num_views=9,    # Processing the measurement as multi-view input
            L=L,
            output_channels=9,  # Outputting 9 channels for demixed images
            is_gate=is_gate,
            is_pe=is_pe,
            full_rs_only = full_rs_only
        )
        # Reconstruction network, outputs [Batch, 1, H, W]
        self.recon_net = NAFNet_gate(
            img_channel=1,   # Input is the 9-channel demixed output
            width=32,
            middle_blk_num=1,
            enc_blk_nums=[1, 1, 1],
            dec_blk_nums=[1, 1, 1],
            num_in=num_in,
            num_hidden=64,
            num_views=9,    # Processing the demixed output as multi-view input
            L=L,
            output_channels=1,  # Final output is a single-channel image
            is_gate=is_gate,
            is_pe=is_pe,
            full_rs_only = full_rs_only
        )
        self.activation = nn.Sigmoid()

    @torch.no_grad()
    def prepare_cache(self, index_list):
        self.eval()
        if hasattr(self.demix_net, "prepare_cache"):
            self.demix_net.prepare_cache(index_list)
        if hasattr(self.recon_net, "prepare_cache"):
            self.recon_net.prepare_cache(index_list)

    @torch.no_grad()
    def clear_cache(self):
        if hasattr(self.demix_net, "clear_cache"):
            self.demix_net.clear_cache()
        if hasattr(self.recon_net, "clear_cache"):
            self.recon_net.clear_cache()

    def forward(self, x, index_list):
        # x: [Batch, num_views, H, W]
        # index_list: [Batch, num_views, H, W, 4]
        # View demixing
        x_input = x.unsqueeze(2)  # [Batch, num_views, 1, H, W]
        demix_output = self.demix_net(x_input, index_list)  # [Batch, 9, H, W]
        demix_output = self.activation(demix_output)
        demix_output1 = demix_output.unsqueeze(2)  # [Batch, 9, 1, H, W]
        # Fusion and reconstruction
        recon_output = self.recon_net(demix_output1, index_list)  # [Batch, 1, H, W]
        recon_output = self.activation(recon_output)
        return demix_output, recon_output


class FFTLoss(nn.Module):
    """
    L_FFT = mean_i || FFT(x_i) - FFT(y_i) ||^2   (default: complex-domain MSE)

    - AMP/bf16 safe: FFTs are computed in fp32 under autocast(enabled=False)
    - Works for tensors with shape (..., H, W) (e.g., BxC×H×W, B×H×W, B×V×C×H×W, etc.)
    - mode: 'complex' | 'magnitude' | 'logmag'
    """
    def __init__(self, mode: str = 'complex', norm: str = 'ortho'):
        super().__init__()
        assert mode in ('FDMSE', 'FDMAE', 'logmag')
        self.mode = mode # 'FDMSE' | 'FDMAE' | 'logmag'
        self.norm = norm  # 'backward' | 'ortho' | 'forward'

    @staticmethod
    def _sanitize(a, b):
        # Ensure same spatial size and finite values
        if a.shape[-2:] != b.shape[-2:]:
            raise ValueError(f"Spatial size mismatch: {a.shape[-2:]} vs {b.shape[-2:]}")
        a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
        b = torch.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
        return a, b

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = self._sanitize(pred, target)

        # Compute FFTs in full precision regardless of outer autocast/bf16
        with torch.autocast(device_type=pred.device.type, enabled=False):
            x = pred.float()
            y = target.float()
            # rFFTN for real inputs saves compute/memory; compares the same half-spectrum on both
            X = torch.fft.rfftn(x, dim=(-2, -1), norm=self.norm)
            Y = torch.fft.rfftn(y, dim=(-2, -1), norm=self.norm)

            if self.mode == 'FDMSE':
                # Complex MSE == mean(|X - Y|^2)
                diff = X - Y
                loss  = (diff.real.square() + diff.imag.square()).mean()
            elif self.mode == 'FDMAE':
                diff = X - Y
                # MAE in Fourier domain: mean(|diff|)
                # compute magnitude from real/imag to avoid complex abs kernel
                mag2 = diff.real.square() + diff.imag.square()
                loss = torch.sqrt(mag2 + 1e-12).mean()
            else: #'magnitude only'
                loss = torch.nn.functional.mse_loss(X.abs().float(), Y.abs().float())

        # Return a scalar; detached scalar weight keeps graph clean
        return loss


