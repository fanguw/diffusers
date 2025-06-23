import cv2
import numpy as np
import torch
from PIL import Image

def get_gradcam_overlay(cam, x: Image.Image, text_emb):
    """
    Given:
      cam: a torch.Tensor CAM of shape [1, 1, H_cam, W_cam], values in [0,1]
      x: a PIL.Image.Image (RGB)
      text_emb: unused here, kept for signature consistency
      
    Returns:
      (overlay_pil, orig_pil): 
        overlay_pil: PIL.Image of original blended with heatmap
        orig_pil: PIL.Image of the original image
    """
    # 1) CAM → uint8 heatmap
    cam_np = cam.squeeze().cpu().numpy()        # (H_cam, W_cam)
    cam_uint8 = np.uint8(255 * cam_np)          # scale to [0,255]
    
    # 2) Original image → numpy
    orig_pil = x.convert("RGB")
    orig_np = np.array(orig_pil)                # (H, W, 3), uint8
    
    H, W = orig_np.shape[:2]
    
    # 3) Resize CAM to image size
    cam_resized = cv2.resize(cam_uint8, (W, H))
    
    # 4) Colorize CAM
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # 5) Blend 50/50
    overlay_np = cv2.addWeighted(orig_np, 0.5, heatmap, 0.5, 0)
    
    # 6) Back to PIL
    overlay_pil = Image.fromarray(overlay_np)
    
    return overlay_pil, orig_pil

