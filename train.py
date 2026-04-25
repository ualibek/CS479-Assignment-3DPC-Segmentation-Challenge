import os
import torch
from torch.utils.data import DataLoader
from dataset import InstancePointCloudDataset
from model import PointGroup, PointGroupLoss
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_DATA_DIR   = "/home/ubuntu/ml3d/segmentation_challenge/data/synth_baseline/train"
VAL_DATA_DIR = "/home/ubuntu/ml3d/segmentation_challenge/data/synth_baseline/val"
CKPT_DIR   = "ckpts"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR         = 1e-3
EPOCHS     = 60
SAVE_EVERY = 10   # save checkpoint every N epochs

os.makedirs(CKPT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def collate_fn(batch):
    return batch[0]

train_dataset = InstancePointCloudDataset(TRAIN_DATA_DIR, split="all")
val_dataset   = InstancePointCloudDataset(VAL_DATA_DIR, split="all")

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,  num_workers=4, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model     = PointGroup(in_channels=9).to(DEVICE)
criterion = PointGroupLoss(offset_weight=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_epoch(loader, training: bool, epoch: int):
    model.train() if training else model.eval()
    total_loss = total_sem = total_off = 0.0
    prefix = "Train" if training else "Val"

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=f"Epoch {epoch:03d} {prefix}", leave=False)
        for i, batch in enumerate(pbar, 1):
            features        = batch["features"].unsqueeze(0).to(DEVICE)
            instance_labels = batch["instance_labels"].unsqueeze(0).to(DEVICE)

            pred = model(features)
            loss, info = criterion(pred, instance_labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_sem  += info["sem_loss"]
            total_off  += info["offset_loss"]

            pbar.set_postfix(
                loss=f"{total_loss/i:.4f}",
                sem=f"{total_sem/i:.4f}",
                off=f"{total_off/i:.4f}",
            )

    n = len(loader)
    return total_loss / n, total_sem / n, total_off / n


def save_checkpoint(epoch, best=False):
    name = "best.pth" if best else f"epoch_{epoch:03d}.pth"
    torch.save({
        "epoch"            : epoch,
        "model_state_dict" : model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, os.path.join(CKPT_DIR, name))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    train_loss, train_sem, train_off = run_epoch(train_loader, training=True, epoch=epoch)
    val_loss,   val_sem,   val_off   = run_epoch(val_loader,   training=False, epoch=epoch)
    scheduler.step()

    print(
        f"Epoch {epoch:03d}/{EPOCHS} | "
        f"Train loss {train_loss:.4f} (sem {train_sem:.4f}, off {train_off:.4f}) | "
        f"Val loss {val_loss:.4f} (sem {val_sem:.4f}, off {val_off:.4f})"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(epoch, best=True)
        print(f"  -> New best saved (val_loss={val_loss:.4f})")

    if epoch % SAVE_EVERY == 0:
        save_checkpoint(epoch)