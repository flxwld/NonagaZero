import torch
import os

CHECKPOINT_DIR = "checkpoints"

def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth"):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, filename)
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": torch.get_rng_state()
    }
    torch.save(ckpt, path)
    print(f"Checkpoint saved to {os.path.abspath(path)}")

def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    path = os.path.join(CHECKPOINT_DIR, filename)
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = ckpt.get("epoch", 0) + 1
    torch.set_rng_state(ckpt.get("rng_state", torch.get_rng_state()))
    print(f"Checkpoint loaded from {os.path.abspath(path)}")
    return start_epoch