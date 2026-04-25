"""
PointGroup -- classical 3D-UNet (spconv) version.

Backbone  : sparse 3D UNet built with spconv
Heads     : semantic (fg/bg) + offset (3-D centroid shift)
Clustering: DBSCAN on offset-shifted foreground points  (inference only)

pip install spconv-cu118   # or spconv-cu117 / spconv-cpu, match your CUDA
pip install scikit-learn   # for DBSCAN
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import DBSCAN

try:
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseConvTensor
except ImportError:
    raise ImportError("Install spconv: pip install spconv-cu118  (adjust for your CUDA version)")

# ---------------------------------------------------------------------------
# Hyper-parameters you might want to tune
# ---------------------------------------------------------------------------
VOXEL_SIZE   = 0.05   # in normalised coord space
CHANNELS     = [32, 64, 128, 128]   # UNet widths per level

# Clustering (inference)
SCORE_THR    = 0.7    # min fg probability
DBSCAN_EPS   = 0.03    # radius in shifted-coord space
DBSCAN_MIN_PTS = 100   # min cluster size


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def make_block(in_ch, out_ch, indice_key):
    """Two SubM convs with BN+ReLU."""
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_ch, out_ch, 3, padding=1, bias=False, indice_key=indice_key),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
        spconv.SubMConv3d(out_ch, out_ch, 3, padding=1, bias=False, indice_key=indice_key),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


def down_block(in_ch, out_ch, indice_key):
    return spconv.SparseSequential(
        spconv.SparseConv3d(in_ch, out_ch, 2, stride=2, bias=False, indice_key=indice_key),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


def up_block(in_ch, out_ch, indice_key):
    return spconv.SparseSequential(
        spconv.SparseInverseConv3d(in_ch, out_ch, 2, indice_key=indice_key, bias=False),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------------------
# 3D Sparse UNet
# ---------------------------------------------------------------------------

class SparseUNet(nn.Module):
    def __init__(self, in_ch=9):
        super().__init__()
        C = CHANNELS  # [32, 64, 128, 128]

        # Encoder
        self.enc0  = make_block(in_ch, C[0], "enc0")
        self.down0 = down_block(C[0],  C[1], "down0")

        self.enc1  = make_block(C[1],  C[1], "enc1")
        self.down1 = down_block(C[1],  C[2], "down1")

        self.enc2  = make_block(C[2],  C[2], "enc2")
        self.down2 = down_block(C[2],  C[3], "down2")

        # Bottleneck
        self.bottleneck = make_block(C[3], C[3], "btn")

        # Decoder
        self.up2  = up_block(C[3], C[2], "down2")
        self.dec2 = make_block(C[2] + C[2], C[2], "dec2")

        self.up1  = up_block(C[2], C[1], "down1")
        self.dec1 = make_block(C[1] + C[1], C[1], "dec1")

        self.up0  = up_block(C[1], C[0], "down0")
        self.dec0 = make_block(C[0] + C[0], C[0], "dec0")

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.down0(e0))
        e2 = self.enc2(self.down1(e1))
        b  = self.bottleneck(self.down2(e2))

        d2 = self.dec2(self._cat(self.up2(b),  e2))
        d1 = self.dec1(self._cat(self.up1(d2), e1))
        d0 = self.dec0(self._cat(self.up0(d1), e0))
        return d0  # SparseConvTensor

    @staticmethod
    def _cat(a, b):
        return a.replace_feature(torch.cat([a.features, b.features], dim=1))


# ---------------------------------------------------------------------------
# Full PointGroup model
# ---------------------------------------------------------------------------

class PointGroup(nn.Module):
    """
    forward() input  : features [B, 9, N]  (xyz | rgb | normal)
    forward() output : dict
        sem_logits : [N_total, 2]
        offsets    : [N_total, 3]
        batch_idx  : [N_total]
        xyz        : [N_total, 3]   (for loss computation)
    """

    def __init__(self, in_channels=9):
        super().__init__()
        self.in_channels = in_channels
        self.voxel_size  = VOXEL_SIZE
        self.unet        = SparseUNet(in_ch=in_channels)

        feat_ch = CHANNELS[0]
        self.sem_head = nn.Sequential(
            nn.Linear(feat_ch, feat_ch), nn.ReLU(inplace=True),
            nn.Linear(feat_ch, 2),
        )
        self.offset_head = nn.Sequential(
            nn.Linear(feat_ch, feat_ch), nn.ReLU(inplace=True),
            nn.Linear(feat_ch, 3),
        )

    # ------------------------------------------------------------------
    def _voxelize(self, coord_list, feat_list, device):
        """
        coord_list : list[Tensor[N_i, 3]]
        feat_list  : list[Tensor[N_i, C]]
        Returns    : SparseConvTensor, v2p_maps (list of inverse indices)
        """
        all_vox = []
        for xyz in coord_list:
            all_vox.append(torch.floor(xyz / self.voxel_size).long())

        all_cat   = torch.cat(all_vox, dim=0)
        min_coord = all_cat.min(dim=0).values
        spatial_shape = (all_cat.max(dim=0).values - min_coord + 1).tolist()
        def round_up_to_multiple(x, m=8):
            return max(m, ((x + m - 1) // m) * m)

        spatial_shape = [round_up_to_multiple(int(s)) for s in spatial_shape]
        # spatial_shape = [max(int(s), 8) for s in spatial_shape]

        vox_coords_list, vox_feats_list, batch_ids = [], [], []
        v2p_maps = []

        for b, (xyz, feat) in enumerate(zip(coord_list, feat_list)):
            vc   = all_vox[b] - min_coord
            flat = (vc[:, 0] * spatial_shape[1] * spatial_shape[2]
                  + vc[:, 1] * spatial_shape[2]
                  + vc[:, 2])
            uniq, inverse = torch.unique(flat, return_inverse=True)
            v2p_maps.append(inverse)

            num_vox  = uniq.shape[0]
            vox_feat = torch.zeros(num_vox, feat.shape[1], device=device)
            count    = torch.zeros(num_vox, 1, device=device)
            vox_feat.index_add_(0, inverse, feat)
            count.index_add_(0, inverse, torch.ones(xyz.shape[0], 1, device=device))
            vox_feat = vox_feat / count.clamp(min=1)

            c0 = uniq // (spatial_shape[1] * spatial_shape[2])
            c1 = (uniq % (spatial_shape[1] * spatial_shape[2])) // spatial_shape[2]
            c2 = uniq % spatial_shape[2]
            vox_coords_list.append(torch.stack([c0, c1, c2], dim=1))
            vox_feats_list.append(vox_feat)
            batch_ids.append(torch.full((num_vox,), b, dtype=torch.int32, device=device))

        coords_b = torch.cat(batch_ids, dim=0).unsqueeze(1)
        coords_3 = torch.cat(vox_coords_list, dim=0).int()
        feats    = torch.cat(vox_feats_list,  dim=0)

        sparse = SparseConvTensor(
            features     = feats,
            indices      = torch.cat([coords_b, coords_3], dim=1),
            spatial_shape= spatial_shape,
            batch_size   = len(coord_list),
        )
        return sparse, v2p_maps

    # ------------------------------------------------------------------
    def forward(self, features):
        device = features.device
        B = features.shape[0]

        coord_list = [features[b, :3, :].T for b in range(B)]   # [N, 3]
        feat_list  = [features[b].T        for b in range(B)]   # [N, 9]

        sparse, v2p_maps = self._voxelize(coord_list, feat_list, device)
        out_sparse = self.unet(sparse)
        vox_feats  = out_sparse.features  # [V_total, C]

        # Scatter voxel features -> points
        point_feats_list, xyz_list = [], []
        vox_offset = 0
        for b in range(B):
            mask_b  = (sparse.indices[:, 0] == b)
            num_vox = int(mask_b.sum())
            vf = vox_feats[vox_offset: vox_offset + num_vox]
            point_feats_list.append(vf[v2p_maps[b]])
            xyz_list.append(coord_list[b])
            vox_offset += num_vox

        point_feats = torch.cat(point_feats_list, dim=0)   # [N_total, C]
        xyz_all     = torch.cat(xyz_list,         dim=0)   # [N_total, 3]

        N_per = features.shape[2]
        batch_idx = torch.cat([
            torch.full((N_per,), b, dtype=torch.long, device=device)
            for b in range(B)
        ])

        return {
            "sem_logits": self.sem_head(point_feats),
            "offsets"   : self.offset_head(point_feats),
            "batch_idx" : batch_idx,
            "xyz"       : xyz_all,
        }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class PointGroupLoss(nn.Module):
    """
    Semantic CE  +  smooth-L1 offset loss on foreground points.

    Usage in training loop:
        pred = model(features)            # features: [B, 9, N]
        loss, info = criterion(pred, instance_labels)   # labels: [B, N]
        loss.backward()
    """

    def __init__(self, offset_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.offset_weight = offset_weight

    def forward(self, pred, instance_labels):
        sem_logits = pred["sem_logits"]   # [N_total, 2]
        offsets    = pred["offsets"]      # [N_total, 3]
        xyz        = pred["xyz"]          # [N_total, 3]
        batch_idx  = pred["batch_idx"]   # [N_total]

        gt_inst = instance_labels.view(-1)        # [N_total]
        gt_sem  = (gt_inst > 0).long()

        sem_loss = self.ce(sem_logits, gt_sem)

        # Offset loss on foreground only
        fg_mask = gt_sem.bool()
        if fg_mask.any():
            xyz_fg  = xyz[fg_mask]
            inst_fg = gt_inst[fg_mask]
            bidx_fg = batch_idx[fg_mask]
            off_fg  = offsets[fg_mask]

            # Per-instance centroid
            gt_offsets = torch.zeros_like(xyz_fg)
            for b in bidx_fg.unique():
                for iid in inst_fg[bidx_fg == b].unique():
                    pt_mask = (bidx_fg == b) & (inst_fg == iid)
                    centroid = xyz_fg[pt_mask].mean(0)
                    gt_offsets[pt_mask] = centroid - xyz_fg[pt_mask]

            offset_loss = nn.functional.smooth_l1_loss(off_fg, gt_offsets)
        else:
            offset_loss = torch.tensor(0.0, device=sem_logits.device)

        loss = sem_loss + self.offset_weight * offset_loss
        return loss, {
            "sem_loss"   : sem_loss.item(),
            "offset_loss": offset_loss.item(),
        }


# ---------------------------------------------------------------------------
# Clustering  (inference only)
# ---------------------------------------------------------------------------

def cluster_instances(shifted: np.ndarray) -> np.ndarray:
    """DBSCAN on offset-shifted fg coords. Returns 1-based instance ids (0=noise)."""
    if len(shifted) == 0:
        return np.array([], dtype=np.int64)
    db     = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_PTS, n_jobs=-1).fit(shifted)
    labels = (db.labels_ + 1).clip(0).astype(np.int64)  # noise -1 -> 0
    return labels


# ---------------------------------------------------------------------------
# API required by evaluate.py
# ---------------------------------------------------------------------------

def initialize_model(
    ckpt_path: str,
    device: torch.device,
    in_channels: int = 9,
    num_classes: int = 2,
) -> nn.Module:
    model = PointGroup(in_channels=in_channels).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and any(k.startswith("module.") for k in checkpoint):
        checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}

    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


def run_inference(
    model: nn.Module,
    features: torch.Tensor,
    score_thr: float = SCORE_THR,
    **kwargs,
) -> torch.Tensor:
    """
    features : [B, 9, N]
    Returns  : [B, N] long, background=0, instances 1..K
    """
    device = features.device
    B, _, N = features.shape

    with torch.no_grad():
        pred = model(features)

    sem_logits = pred["sem_logits"]
    offsets    = pred["offsets"]
    batch_idx  = pred["batch_idx"]
    fg_prob    = torch.softmax(sem_logits, dim=1)[:, 1]

    result = torch.zeros(B, N, dtype=torch.long, device=device)

    for b in range(B):
        mask_b  = batch_idx == b
        prob_b  = fg_prob[mask_b]
        off_b   = offsets[mask_b]
        xyz_b   = features[b, :3, :].T          # [N, 3]

        fg_idx  = torch.where(prob_b >= score_thr)[0]
        if fg_idx.numel() == 0:
            continue

        shifted = (xyz_b[fg_idx] + off_b[fg_idx]).cpu().numpy()
        inst_np = cluster_instances(shifted)

        result[b].scatter_(0, fg_idx, torch.from_numpy(inst_np).to(device))

    return result