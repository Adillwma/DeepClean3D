# %%
# 3 head ( 3 axis view ) 3D Denoising Autoencoder with 6 output projecteions output nativly 

from random import choice, random
import torch
from typing import Tuple, Dict, Optional
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# %%
## Pattern Generator


def make_pattern_batch_V2(
    B: int,
    in_shape,
    pattern: str = "square",
    thickness: int = 1,
    size: float = 0.5,
    density: float = 0.3,
    device: torch.device = torch.device('cpu')
):
    """
    Generate 3D binary patterns (0 or 1) in a batch tensor.

    Args:
        B: Batch size.
        in_shape: Tuple (D, H, W).
        pattern: Pattern type ('cross', 'square', 'cube', 'sphere', 'pyramid',
                 'diagonal', 'hollow_cube', 'random', or 'center').
        thickness: Line or border thickness.
        size: Relative size (0.1–1.0) controlling pattern scale.
        density: For random patterns — proportion of filled voxels.
        device: Torch device.
    """
    D, H, W = in_shape
    x = torch.zeros(B, 1, D, H, W, device=device)
    zc, yc, xc = D // 2, H // 2, W // 2
    t = max(1, int(thickness))

    # Helper: coordinate grids
    z, y, xg = torch.meshgrid(
        torch.arange(D, device=device),
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
)
    dist_center = torch.sqrt((z - zc)**2 + (y - yc)**2 + (xg - xc)**2)

    # ---- Patterns ----
    if pattern == "cross":
        x[:, 0, :, yc, xc] = 1.0
        x[:, 0, zc, :, xc] = 1.0
        x[:, 0, zc, yc, :] = 1.0

    elif pattern == "square":
        side_h = int(H * size)
        side_w = int(W * size)
        y0, y1 = yc - side_h // 2, yc + side_h // 2
        x0, x1 = xc - side_w // 2, xc + side_w // 2
        z = zc
        for k in range(-(t // 2), t // 2 + 1):
            x[:, 0, z, y0 + k:y1 + 1 + k, x0] = 1
            x[:, 0, z, y0 + k:y1 + 1 + k, x1] = 1
            x[:, 0, z, y0, x0 + k:x1 + 1 + k] = 1
            x[:, 0, z, y1, x0 + k:x1 + 1 + k] = 1

    elif pattern == "cube":
        side_d = int(D * size)
        side_h = int(H * size)
        side_w = int(W * size)
        z0, z1 = zc - side_d // 2, zc + side_d // 2
        y0, y1 = yc - side_h // 2, yc + side_h // 2
        x0, x1 = xc - side_w // 2, xc + side_w // 2
        x[:, 0, z0:z1, y0:y1, x0:x1] = 1.0

    elif pattern == "hollow_cube":
        side_d = int(D * size)
        side_h = int(H * size)
        side_w = int(W * size)
        z0, z1 = zc - side_d // 2, zc + side_d // 2
        y0, y1 = yc - side_h // 2, yc + side_h // 2
        x0, x1 = xc - side_w // 2, xc + side_w // 2
        cube = torch.zeros_like(x)
        cube[:, 0, z0:z1, y0:y1, x0:x1] = 1.0
        inner = torch.zeros_like(x)
        inner[:, 0, z0 + t:z1 - t, y0 + t:y1 - t, x0 + t:x1 - t] = 1.0
        x = cube - inner
        x = (x > 0).float()

    elif pattern == "sphere":
        radius = (min(D, H, W) * size) / 2
        mask = (dist_center <= radius).float()
        x[:] = mask.unsqueeze(0).unsqueeze(0)

    elif pattern == "pyramid":
        max_h = int(D * size / 2)
        for i in range(-max_h, max_h):
            layer = max_h - abs(i)
            y0, y1 = yc - layer, yc + layer
            x0, x1 = xc - layer, xc + layer
            z_idx = zc + i
            if 0 <= z_idx < D:
                x[:, 0, z_idx, y0:y1, x0:x1] = 1.0

    elif pattern == "diagonal":
        for i in range(min(D, H, W)):
            xi, yi, zi = i, i, i
            if zi < D and yi < H and xi < W:
                x[:, 0, zi, yi, xi] = 1.0
                for k in range(1, t):
                    if yi + k < H and xi + k < W:
                        x[:, 0, zi, yi + k, xi + k] = 1.0

    elif pattern == "blank":
        x = torch.zeros_like(x)

    else:  # "center" or fallback
        x[:, 0, zc, yc, xc] = 1.0

    return x



# %%
# Trainable 3D Denoising Autoencoder with latent vector
# - Accepts (B, 1, D, H, W) or (B, D, H, W) or (D, H, W)
# - Learns a latent bottleneck of size `latent_dim`
# - Reconstructs the input volume; intended for denoising via self-supervision

def _ensure_bchw3(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Normalize input to shape (B, 1, D, H, W). Returns normalized tensor and original (D,H,W).
    Accepts (D,H,W), (B,D,H,W), or (B,1,D,H,W).
    """
    if x.dim() == 3:
        D, H, W = x.shape
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 4:
        # (B, D, H, W)
        B, D, H, W = x.shape
        x = x.unsqueeze(1)
    elif x.dim() == 5:
        # (B, C, D, H, W)
        assert x.size(1) == 1, f"Expected single-channel input, got C={x.size(1)}"
        B, C, D, H, W = x.shape
    else:
        raise ValueError(f"Unsupported input rank {x.dim()}, expected 3/4/5 dims.")
    return x.to(torch.float32), (D, H, W)


def _center_crop_or_pad(x: torch.Tensor, out_shape: Tuple[int, int, int]) -> torch.Tensor:
    """
    Crop or pad a 5D tensor (B,1,D,H,W) to have spatial dims = out_shape.
    Uses zero-padding and centered cropping when needed.
    """
    assert x.dim() == 5 and x.size(1) == 1
    B, C, D, H, W = x.shape
    tD, tH, tW = out_shape

    # Pad if smaller
    pad_d_front = max(0, (tD - D) // 2)
    pad_d_back = max(0, tD - D - pad_d_front)
    pad_h_front = max(0, (tH - H) // 2)
    pad_h_back = max(0, tH - H - pad_h_front)
    pad_w_front = max(0, (tW - W) // 2)
    pad_w_back = max(0, tW - W - pad_w_front)

    if any(p > 0 for p in (pad_w_front, pad_w_back, pad_h_front, pad_h_back, pad_d_front, pad_d_back)):
        # F.pad order: (W_left, W_right, H_top, H_bottom, D_front, D_back)
        x = F.pad(x, (pad_w_front, pad_w_back, pad_h_front, pad_h_back, pad_d_front, pad_d_back))
        B, C, D, H, W = x.shape

    # Crop if larger
    if D > tD:
        start = (D - tD) // 2
        x = x[:, :, start:start + tD, :, :]
        D = tD
    if H > tH:
        start = (H - tH) // 2
        x = x[:, :, :, start:start + tH, :]
        H = tH
    if W > tW:
        start = (W - tW) // 2
        x = x[:, :, :, :, start:start + tW]
        W = tW
    return x


def _center_crop_or_pad_2d(x: torch.Tensor, out_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Crop or pad a 4D tensor (B,1,H,W) to have spatial dims = out_shape.
    Uses zero-padding and centered cropping when needed.
    """
    assert x.dim() == 4 and x.size(1) == 1
    B, C, H, W = x.shape
    tH, tW = out_shape

    # Pad if smaller
    pad_h_front = max(0, (tH - H) // 2)
    pad_h_back = max(0, tH - H - pad_h_front)
    pad_w_front = max(0, (tW - W) // 2)
    pad_w_back = max(0, tW - W - pad_w_front)

    if any(p > 0 for p in (pad_w_front, pad_w_back, pad_h_front, pad_h_back)):
        # F.pad order for 2D: (W_left, W_right, H_top, H_bottom)
        x = F.pad(x, (pad_w_front, pad_w_back, pad_h_front, pad_h_back))
        B, C, H, W = x.shape

    # Crop if larger
    if H > tH:
        start = (H - tH) // 2
        x = x[:, :, start:start + tH, :]
        H = tH
    if W > tW:
        start = (W - tW) // 2
        x = x[:, :, :, start:start + tW]
        W = tW
    return x


def project_six_2d(x5d: torch.Tensor, return_binary: bool = True, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
    """
    Compute six 2D projections (one per side) from a (B,1,D,H,W) tensor.
    - If return_binary=True, first binarizes with threshold then uses OR (amax) along the view axis,
      returning exact 0/1 tensors.
    - If return_binary=False, uses max along the view axis on probabilities.
    Returns a dict with tensors shaped:
      front/back: (B,1,H,W)   [view along +D / -D]
      top/bottom: (B,1,D,W)   [view along +H / -H]
      left/right: (B,1,D,H)   [view along +W / -W]
    Note: front/back (and other pairs) are identical under OR projection by design.
    """
    assert x5d.dim() == 5 and x5d.size(1) == 1, "Expected (B,1,D,H,W)"
    if return_binary:
        x5d = (x5d >= threshold).to(x5d.dtype)

    # OR-like projection via max along the view axis
    front = torch.amax(x5d, dim=2)   # (B,1,H,W)
    back = front.clone()
    top = torch.amax(x5d, dim=3)     # (B,1,D,W)
    bottom = top.clone()
    left = torch.amax(x5d, dim=4)    # (B,1,D,H)
    right = left.clone()

    if return_binary:
        # Ensure exactly 0/1 float values
        front = (front > 0).to(x5d.dtype)
        back = (back > 0).to(x5d.dtype)
        top = (top > 0).to(x5d.dtype)
        bottom = (bottom > 0).to(x5d.dtype)
        left = (left > 0).to(x5d.dtype)
        right = (right > 0).to(x5d.dtype)

    return {
        "front": front,
        "back": back,
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
    }


class DenoisingAutoencoder3D(nn.Module):
    """
    A trainable 3D denoising autoencoder with a latent vector of size `latent_dim`.

    Contract:
    - Input: tensor shaped (B,1,D,H,W) or (B,D,H,W) or (D,H,W), values in [0,1]
    - forward(...): returns six 2D projections (dict of tensors)
    - reconstruct3d(...): returns the reconstructed 3D volume (B,1,D,H,W)
    - Projections are OR-like (no summing along the view). If return_binary=True, outputs are exact 0/1.
    - Errors: raises if channel != 1
    """
    def __init__(self, in_shape: Tuple[int, int, int], latent_dim: int = 256, base_channels: int = 16):
        super().__init__()
        self.in_shape = in_shape  # (D,H,W)
        self.latent_dim = latent_dim

        C = 1
        self.encoder = nn.Sequential(
            nn.Conv3d(C, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
)

        # Infer flattened size from a dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *in_shape)
            enc_feat = self.encoder(dummy)
            self._enc_out_shape = enc_feat.shape[1:]  # (C', D', H', W')
            self._enc_flat_dim = int(enc_feat.numel())

        self.fc_enc = nn.Linear(self._enc_flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self._enc_flat_dim)

        # 2D decoders for each unique projection plane
        c_enc = base_channels * 4
        # HxW projection branch (front/back)
        self.dec_hw = nn.Sequential(
            nn.ConvTranspose2d(c_enc, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_channels, C, kernel_size=4, stride=2, padding=1),
)

        # DxW projection branch (top/bottom)
        self.dec_dw = nn.Sequential(
            nn.ConvTranspose2d(c_enc, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_channels, C, kernel_size=4, stride=2, padding=1),
)

        # DxH projection branch (left/right)
        self.dec_dh = nn.Sequential(
            nn.ConvTranspose2d(c_enc, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_channels, C, kernel_size=4, stride=2, padding=1),
)

    def reconstruct3d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Approximate 3D reconstruction from 2D projections, returns (B,1,D,H,W) in [0,1].
        This avoids 3D decoding by broadcasting the three unique 2D maps.
        """
        x, (D, H, W) = _ensure_bchw3(x)
        projs = self.forward(x, return_binary=False)
        # Use one map per unique view family
        hw = projs["front"]  # (B,1,H,W)
        dw = projs["top"]    # (B,1,D,W)
        dh = projs["left"]   # (B,1,D,H)

        # Broadcast to (B,1,D,H,W)
        vol_hw = hw.unsqueeze(2).expand(-1, -1, D, -1, -1)
        vol_dw = dw.unsqueeze(3).expand(-1, -1, D, H, -1)
        vol_dh = dh.unsqueeze(4).expand(-1, -1, D, H, W)

        # Intersection-like fusion (conservative)
        vol = torch.minimum(vol_hw, torch.minimum(vol_dw, vol_dh))
        return vol

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = _ensure_bchw3(x)
        B = x.size(0)
        z = self.encoder(x)
        z_flat = z.view(B, -1)
        return self.fc_enc(z_flat)

    def decode(self, latent: torch.Tensor, out_shape: Optional[Tuple[int, int, int]] = None) -> Dict[str, torch.Tensor]:
        """
        Decode from latent vector into six 2D projection maps.
        If out_shape is provided, it is interpreted as (D,H,W) to crop/pad outputs.
        """
        B = latent.size(0)
        z_hat_flat = self.fc_dec(latent)
        z_hat = z_hat_flat.view(B, *self._enc_out_shape)  # (B, C', D', H', W')

        Cprime, Dp, Hp, Wp = self._enc_out_shape
        # Reduce along the view axis with max to emulate OR-like aggregation
        feat_hw = torch.amax(z_hat, dim=2)  # (B, C', H', W')
        feat_dw = torch.amax(z_hat, dim=3)  # (B, C', D', W') - treat D' as H
        feat_dh = torch.amax(z_hat, dim=4)  # (B, C', D', H') - treat D' as H, H' as W

        hw_logits = self.dec_hw(feat_hw)   # -> (B,1,*,*)
        dw_logits = self.dec_dw(feat_dw)
        dh_logits = self.dec_dh(feat_dh)

        if out_shape is not None:
            D, H, W = out_shape
            hw_logits = _center_crop_or_pad_2d(hw_logits, (H, W))
            dw_logits = _center_crop_or_pad_2d(dw_logits, (D, W))
            dh_logits = _center_crop_or_pad_2d(dh_logits, (D, H))

        hw = torch.sigmoid(hw_logits)
        dw = torch.sigmoid(dw_logits)
        dh = torch.sigmoid(dh_logits)

        return {
            "front": hw,   # (B,1,H,W)
            "back": hw.clone(),
            "top": dw,    # (B,1,D,W)
            "bottom": dw.clone(),
            "left": dh,   # (B,1,D,H)
            "right": dh.clone(),
        }

    def forward(self, x: torch.Tensor, return_binary: bool = True, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Returns six 2D projections directly from 2D decoder branches.
        If return_binary=True, outputs are thresholded to exact 0/1.
        """
        x, (D, H, W) = _ensure_bchw3(x)
        B = x.size(0)

        # Encode -> latent -> expand back to feature map space
        z = self.encoder(x)
        z_flat = z.view(B, -1)
        latent = self.fc_enc(z_flat)
        # decode to 2D maps at the encoder feature resolution, then upsample via 2D decoders
        projs = self.decode(latent, out_shape=(D, H, W))

        if return_binary:
            for k in projs:
                projs[k] = (projs[k] >= threshold).to(projs[k].dtype)
        return projs


# ---------- Utilities for denoising/projection training ----------
# Adds the specified number of noise voxels to the input batch with value 1
def add_noise(x: torch.Tensor, num_noise_voxels: int) -> torch.Tensor:
    B, C, D, H, W = x.shape
    x_noisy = x.clone()
    for i in range(B):
        # Randomly select noise voxel positions
        z = torch.randint(0, D, (num_noise_voxels,))
        y = torch.randint(0, H, (num_noise_voxels,))
        x = torch.randint(0, W, (num_noise_voxels,))
        x_noisy[i, :, z, y, x] = 1.0
    return x_noisy

# remove points function will make all points in tensor 0 apart from the specified number of voxels
def remove_points(x: torch.Tensor, num_points_to_keep: int) -> torch.Tensor:
    B, C, D, H, W = x.shape
    x_sparse = torch.zeros_like(x)
    for i in range(B):
        # Get indices of non-zero voxels
        non_zero_indices = torch.nonzero(x[i, 0], as_tuple=False)
        if non_zero_indices.size(0) <= num_points_to_keep:
            x_sparse[i] = x[i]
        else:
            # Randomly select indices to keep
            selected_indices = non_zero_indices[torch.randperm(non_zero_indices.size(0))[:num_points_to_keep]]
            for idx in selected_indices:
                z, y, x_coord = idx
                x_sparse[i, 0, z, y, x_coord] = 1.0
    return x_sparse

def get_first_batch_shape(dataloader) -> Tuple[int, int, int]:
    """Peek one batch to infer (D,H,W). Does not consume the iterator for outer loops."""
    batch = next(iter(dataloader))
    if isinstance(batch, (tuple, list)):
        x = batch[0]
    else:
        x = batch
    x, (D, H, W) = _ensure_bchw3(x)
    return (D, H, W)

def _to_device(*tensors, device: torch.device):
    return [t.to(device) for t in tensors]

print("DenoisingAutoencoder3D is defined.")

# %% ---------- Plotting utilities ----------
def plot_2d_projections(x, x_noisy, projs_bin):
    ## 2D Visualisations

    # Visualize central slice for first item + 6 projections
    b0 = 0
    D = x.shape[2]
    mid = D // 2

    # Show 6 projections (binary)
    projs_show = {k: v[b0, 0].detach().cpu() for k, v in projs_bin.items()}
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    axs = axs.ravel()
    axs[0].imshow(projs_show["front"], cmap='gray', vmin=0, vmax=1); axs[0].set_title("Front (H×W)")
    axs[1].imshow(projs_show["back"], cmap='gray', vmin=0, vmax=1); axs[1].set_title("Back (H×W)")
    axs[2].imshow(projs_show["left"], cmap='gray', vmin=0, vmax=1); axs[2].set_title("Left (D×H)")
    axs[3].imshow(projs_show["right"], cmap='gray', vmin=0, vmax=1); axs[3].set_title("Right (D×H)")
    axs[4].imshow(projs_show["top"], cmap='gray', vmin=0, vmax=1); axs[4].set_title("Top (D×W)")
    axs[5].imshow(projs_show["bottom"], cmap='gray', vmin=0, vmax=1); axs[5].set_title("Bottom (D×W)")
    #for a in axs: a.axis('off')
    plt.tight_layout(); 
    plt.show()

    # Show 6 projections for the input
    input_projs = project_six_2d(x_noisy, return_binary=True)
    input_projs_show = {k: v[b0, 0].detach().cpu() for k, v in input_projs.items()} 
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    axs = axs.ravel()
    axs[0].imshow(input_projs_show["front"], cmap='gray', vmin=0, vmax=1); axs[0].set_title("Input Front (H×W)")
    axs[1].imshow(input_projs_show["back"], cmap='gray', vmin=0, vmax=1); axs[1].set_title("Input Back (H×W)")
    axs[2].imshow(input_projs_show["left"], cmap='gray', vmin=0, vmax=1); axs[2].set_title("Input Left (D×H)")
    axs[3].imshow(input_projs_show["right"], cmap='gray', vmin=0, vmax=1); axs[3].set_title("Input Right (D×H)")        
    axs[4].imshow(input_projs_show["top"], cmap='gray', vmin=0, vmax=1); axs[4].set_title("Input Top (D×W)")
    axs[5].imshow(input_projs_show["bottom"], cmap='gray', vmin=0, vmax=1); axs[5].set_title("Input Bottom (D×W)")
    #for a in axs: a.axis('off')
    plt.tight_layout();
    plt.show()

def plot_3d_reconstruction_with_projections(
    vol_rec, projs_bin, projections_cmap_out, projections_cmap_in,
    projections_cmap_map_out=None, projections_cmap_map_in=None,
    voxel_plot=True, show_2d_projections=True, show_3d_points=True,
):
    # 3D Visualisations

    b0 = 0
    D = x.shape[2]
    mid = D // 2


    # --- Create figure with two 3D subplots ---
    fig = plt.figure(figsize=(14, 6))

    # Common axis limits (based on volume size)
    Dv, Hv, Wv = vol_rec.shape[2], vol_rec.shape[3], vol_rec.shape[4]
    xlim, ylim, zlim = (0, Wv), (0, Hv), (0, Dv)




    # Choose colormaps (allow custom mapping function if provided)
    cmap_out = plt.cm.get_cmap(projections_cmap_map_out if projections_cmap_map_out is not None else projections_cmap_out)
    cmap_in  = plt.cm.get_cmap(projections_cmap_map_in  if projections_cmap_map_in  is not None else projections_cmap_in)

    # -----------------------------
    # OUTPUT (left subplot)
    # -----------------------------
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Output Reconstruction (vol_rec)")
    ax1.set_xlabel("X axis")
    ax1.set_ylabel("Y axis")
    ax1.set_zlabel("Z axis")

    # Plot scatter or voxel version
    if show_3d_points:
        if voxel_plot:
            voxels_out = (vol_rec[b0, 0] >= 0.5).detach().cpu().numpy()
            # Matplotlib voxels expects (X,Y,Z) order; transpose from (D,H,W)->(W,H,D)
            voxels_out = np.transpose(voxels_out, (2, 1, 0))
            ax1.voxels(voxels_out, facecolors='tab:red', edgecolor='k', alpha=0.6)
        else:
            # nonzero returns (z,y,x); map to (x,y,z) for plotting
            z_out, y_out, x_out = (vol_rec[b0, 0] >= 0.5).nonzero(as_tuple=True)
            ax1.scatter(x_out.cpu(), y_out.cpu(), z_out.cpu(), s=5, c='tab:red')

    if show_2d_projections:
        # Projection planes (place back planes at the full extent to match 0..Dv/Hv/Wv limits)
        z_front, z_back = 0, Dv
        y_front, y_back = 0, Hv
        x_front, x_back = 0, Wv

        # Compute projections
        proj_xy_front = vol_rec[b0, 0].max(dim=0)[0]  # (H, W)
        proj_xz_back  = vol_rec[b0, 0].max(dim=1)[0]  # (D, W)
        proj_yz_front = vol_rec[b0, 0].max(dim=2)[0]  # (D, H)

        # Binary-threshold projections at 0.5 (values become 0.0 or 1.0)
        proj_xy_front_bin = (proj_xy_front >= 0.5).to(torch.float32)
        proj_xz_back_bin  = (proj_xz_back  >= 0.5).to(torch.float32)
        proj_yz_front_bin = (proj_yz_front >= 0.5).to(torch.float32)

        # Build coordinate grids using one extra vertex per axis so we get 9 faces (steps) per dimension
        # Faces are (Ny-1, Nx-1) so we make vertices of size (H+1, W+1), (D+1, W+1), (D+1, H+1)
        x_coords = np.linspace(0, Wv, Wv + 1)
        y_coords = np.linspace(0, Hv, Hv + 1)
        z_coords = np.linspace(0, Dv, Dv + 1)

        # XY plane at z = 0 (shapes: (H+1, W+1))
        X_xy, Y_xy = np.meshgrid(x_coords, y_coords, indexing='xy')
        Z_xy = np.full((Hv + 1, Wv + 1), z_front, dtype=float)

        # XZ plane at y = Hv (back) (shapes: (D+1, W+1))
        X_xz, Z_xz = np.meshgrid(x_coords, z_coords, indexing='xy')
        Y_xz = np.full((Dv + 1, Wv + 1), y_back, dtype=float)

        # YZ plane at x = 0 (front) (shapes: (D+1, H+1))
        Y_yz, Z_yz = np.meshgrid(y_coords, z_coords, indexing='xy')
        X_yz = np.full((Dv + 1, Hv + 1), x_front, dtype=float)

        ax1.plot_surface(
            X_xy, Y_xy, Z_xy,
            facecolors=cmap_out(proj_xy_front_bin.detach().cpu().numpy()),  # binary mapped
            rstride=1, cstride=1, shade=False, alpha=projection_alpha
        )
        ax1.plot_surface(
            X_xz, Y_xz, Z_xz,
            facecolors=cmap_out(proj_xz_back_bin.detach().cpu().numpy()),   # binary mapped
            rstride=1, cstride=1, shade=False, alpha=projection_alpha
        )
        ax1.plot_surface(
            X_yz, Y_yz, Z_yz,
            facecolors=cmap_out(proj_yz_front_bin.detach().cpu().numpy()),  # binary mapped
            rstride=1, cstride=1, shade=False, alpha=projection_alpha
        )

    if adjust_axis_labels:
        # Fix the axes tick positions to mid buckets and adjust labels
        #  i.e 0.5 = 1, 1.5 = 2 etc
        ax1.set_xticks(np.arange(xlim[0] + 0.5, xlim[1], 1.0))
        ax1.set_yticks(np.arange(ylim[0] + 0.5, ylim[1], 1.0))
        ax1.set_zticks(np.arange(zlim[0] + 0.5, zlim[1], 1.0))
        ax1.set_xticklabels([str(i) for i in range(int(xlim[0]), int(xlim[1]))])
        ax1.set_yticklabels([str(i) for i in range(int(ylim[0]), int(ylim[1]))])
        ax1.set_zticklabels([str(i) for i in range(int(zlim[0]), int(zlim[1]))])

    # -----------------------------
    # INPUT (right subplot)
    # -----------------------------
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("Input Noisy Volume (x_noisy)")
    ax2.set_xlabel("X axis")
    ax2.set_ylabel("Y axis")
    ax2.set_zlabel("Z axis")

    if show_3d_points:
        if voxel_plot:
            voxels_in = (x_noisy[b0, 0] >= 0.5).detach().cpu().numpy()
            # Transpose to (X,Y,Z) for Matplotlib
            voxels_in = np.transpose(voxels_in, (2, 1, 0))
            ax2.voxels(voxels_in, facecolors='tab:blue', edgecolor='k', alpha=0.6)
        else:
            # nonzero returns (z,y,x); map to (x,y,z) for plotting
            z_data, y_data, x_data = (x_noisy[b0, 0] >= 0.5).nonzero(as_tuple=True)
            ax2.scatter(x_data.cpu(), y_data.cpu(), z_data.cpu(), s=5, c='tab:blue')

    if show_2d_projections:
        # Input projections
        proj_xy_front_in = x_noisy[b0, 0].max(dim=0)[0]
        proj_xz_back_in  = x_noisy[b0, 0].max(dim=1)[0]
        proj_yz_front_in = x_noisy[b0, 0].max(dim=2)[0]

        # Binary-threshold input projections at 0.5
        proj_xy_front_in_bin = (proj_xy_front_in >= 0.5).to(torch.float32)
        proj_xz_back_in_bin  = (proj_xz_back_in  >= 0.5).to(torch.float32)
        proj_yz_front_in_bin = (proj_yz_front_in >= 0.5).to(torch.float32)

        ax2.plot_surface(
            X_xy, Y_xy, Z_xy,
            facecolors=cmap_in(proj_xy_front_in_bin.detach().cpu().numpy()),
            rstride=1, cstride=1, shade=False, alpha=projection_alpha
        )
        ax2.plot_surface(
            X_xz, Y_xz, Z_xz,
            facecolors=cmap_in(proj_xz_back_in_bin.detach().cpu().numpy()),
            rstride=1, cstride=1, shade=False, alpha=projection_alpha
        )
        ax2.plot_surface(
            X_yz, Y_yz, Z_yz,
            facecolors=cmap_in(proj_yz_front_in_bin.detach().cpu().numpy()),
            rstride=1, cstride=1, shade=False, alpha=projection_alpha
        )



    if adjust_axis_labels:
        #  match mid-bucket ticks on the right subplot too for consistency
        ax2.set_xticks(np.arange(xlim[0] + 0.5, xlim[1], 1.0))
        ax2.set_yticks(np.arange(ylim[0] + 0.5, ylim[1], 1.0))
        ax2.set_zticks(np.arange(zlim[0] + 0.5, zlim[1], 1.0))
        ax2.set_xticklabels([str(i) for i in range(int(xlim[0]), int(xlim[1]))])
        ax2.set_yticklabels([str(i) for i in range(int(ylim[0]), int(ylim[1]))])
        ax2.set_zticklabels([str(i) for i in range(int(zlim[0]), int(zlim[1]))])

    # -----------------------------
    #  Shared limits & orientation
    # -----------------------------
    for ax in [ax1, ax2]:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.view_init(elev=30, azim=-60)

    plt.tight_layout()
    plt.show()

# %%
# Training settings

in_shape = (20, 20, 20)
use_real_data = False
latent_dim = 21
pattern = "square" #["cross", "square", "cube", "hollow_cube", "sphere", "pyramid", "diagonal", "random"]
epochs = 2000
B = 10
number_noise_points = 0
signal_points = 15

show_2d_projections = True # Toggle 2D projection plots
show_3d_points = True # Tuns off the scatter/voxel plots if False
voxel_plot = True  # Toggle between scatter and voxel visualization / Change to False for scatter plot mode
projection_alpha = 0.2
adjust_axis_labels = True

projections_cmap_out = 'viridis'
projections_cmap_in  = 'viridis'
projections_cmap_map_out = None  # e.g. plt.cm.inferno
projections_cmap_map_in  = None
patterns = ["cross", "square", "cube", "hollow_cube", "sphere", "pyramid", "diagonal", "blank"]

# %%
# Training Loop


# Pick device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_ae = DenoisingAutoencoder3D(in_shape=in_shape, latent_dim=latent_dim).to(device)
optim_ae = torch.optim.Adam(model_ae.parameters(), lr=1e-3)



# Simple data for test , one point in each corner of the cube, i.e 8 points
# x = torch.zeros(B, 1, *in_shape, device=device)
# x[:, 0, 0, 0, 0] = 1.0
# x[:, 0, 0, 0, in_shape[2]-1] = 1.0
# x[:, 0, 0, in_shape[1]-1, 0] = 1.0
# x[:, 0, 0, in_shape[1]-1, in_shape[2]-1] = 1.0
# x[:, 0, in_shape[0]-1, 0, 0] = 1.0
# x[:, 0, in_shape[0]-1, 0, in_shape[2]-1] = 1.0
# x[:, 0, in_shape[0]-1, in_shape[1]-1, 0] = 1.0
# x[:, 0, in_shape[0]-1, in_shape[1]-1, in_shape[2]-1] = 1.0

loop = tqdm(range(epochs), postfix="Starting...")
for step in loop:
    # random uniform sample from patterns list
    random_pattern = patterns[torch.randint(0, len(patterns), (1,)).item()]
    x = make_pattern_batch_V2(B, in_shape, pattern=random_pattern, thickness=1, device=device)

    x_sparse = remove_points(x, signal_points)

    # Add noise for denoising training
    x_noisy = add_noise(x_sparse, number_noise_points)

    # One training step against 2D projections (use probabilistic projections for gradients)
    model_ae.train()
    optim_ae.zero_grad(set_to_none=True)
    projs_out = model_ae(x_noisy, return_binary=False)  # dict of six; values in [0,1]
    # Targets are binary OR-projections of the clean volume
    targets = project_six_2d(x, return_binary=True)
    loss = (
        F.binary_cross_entropy(projs_out["front"], targets["front"]) +    # cut hese down to just three and label X Y and Z for the axis rather than front back etc 
        F.binary_cross_entropy(projs_out["back"], targets["back"]) +
        F.binary_cross_entropy(projs_out["left"], targets["left"]) +
        F.binary_cross_entropy(projs_out["right"], targets["right"]) +
        F.binary_cross_entropy(projs_out["top"], targets["top"]) +
        F.binary_cross_entropy(projs_out["bottom"], targets["bottom"])
    ) / 6.0
    loss.backward()
    optim_ae.step()

    # update tqdm readout with loss using set postfix
    loop.set_postfix({"loss": f"{loss.item():.4f}"})

# Report
with torch.no_grad():

    for pattern in patterns:
        x = make_pattern_batch_V2(B, in_shape, pattern=pattern, thickness=1, device=device)
        x_sparse = remove_points(x, signal_points)
        x_noisy = add_noise(x_sparse, number_noise_points)
        latent_vec = model_ae.encode(x_noisy)
        vol_rec = model_ae.reconstruct3d(x_noisy)
        projs_bin = model_ae(x_noisy, return_binary=True)
        #vol_rec_output = model_ae.reconstruct3d(x_noisy)
        print(f"Batch: {tuple(x.shape)} | Latent: {tuple(latent_vec.shape)} | Loss (proj): {loss.item():.4f}")
        print(f"Recon volume: {tuple(vol_rec.shape)} | Proj shapes: "f"front/back {tuple(projs_bin['front'].shape)}, left/right {tuple(projs_bin['left'].shape)}, top/bottom {tuple(projs_bin['top'].shape)}")

        # Visualize results
        plot_2d_projections(x, x_noisy, projs_bin)

        plot_3d_reconstruction_with_projections(
            vol_rec, projs_bin, projections_cmap_out, projections_cmap_in,
            projections_cmap_map_out=projections_cmap_map_out, projections_cmap_map_in=projections_cmap_map_in,
            voxel_plot=voxel_plot, show_2d_projections=show_2d_projections, show_3d_points=show_3d_points
        )



