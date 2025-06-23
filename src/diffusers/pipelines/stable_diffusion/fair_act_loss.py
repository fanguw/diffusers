import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def cam_suppression_loss(
    cam: torch.Tensor,
    mask_img: Image.Image,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Args:
      cam:       [1, 1, H_cam, W_cam] float in [0,1], on any device
      mask_img:  PIL.Image, size (W_img, H_img). 
                 White (255) = area to suppress (loss), black = ignore.
      reduction: "mean" | "sum" | "none"
    
    Returns:
      loss: scalar (if reduction!="none") or tensor [1,1,H_cam,W_cam] if "none".
    """
    # 1) Convert mask to a binary tensor [1,1,H_img,W_img] on CPU
    mask_np = np.array(mask_img.convert("L"))          # (H_img, W_img), 0–255
    mask_bin = (mask_np > 128).astype(np.float32)      # white→1, else→0
    mask_t = torch.from_numpy(mask_bin)                # [H_img, W_img], CPU

    # 2) Move mask to the same device & dtype as cam, add batch+channel dims
    mask_t = mask_t.to(device=cam.device, dtype=cam.dtype)[None, None]  # [1,1,H_img,W_img]

    # 3) Resize mask to cam’s spatial size
    _, _, Hc, Wc = cam.shape
    mask_resized = F.interpolate(mask_t, size=(Hc, Wc), mode="nearest")  # [1,1,Hc,Wc]

    # 4) Compute penalty: cam values in masked regions
    #    we want cam ≈ 0 where mask=1, so penalize cam * mask
    penalty = cam * mask_resized

    # 5) Reduce
    if reduction == "mean":
        return penalty.mean()
    elif reduction == "sum":
        return penalty.sum()
    elif reduction == "none":
        return penalty
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

