import PIL
import torch
from torchvision import transforms
import diffusers
import transformers
from diffusers import StableDiffusionDepth2ImgPipeline
import os
from PIL import Image
import torch.nn.functional as F

import numpy as np
from diffusers import (
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
)

from diffusers.utils.torch_utils import randn_tensor

print(f'Getting model from {os.environ.get("OUTPUT_DIR")}')

scheduler = LMSDiscreteScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=100
)

def slerp(v0, v1, num, t0=0, t1=1):
    v0 = v0.detach().cpu().numpy()
    v1 = v1.detach().cpu().numpy()

    def interpolation(t, v0, v1, DOT_THRESHOLD=0.9995):
        """helper function to spherically interpolate two arrays v1 v2"""
        dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
        if np.abs(dot) > DOT_THRESHOLD:
            v2 = (1 - t) * v0 + t * v1
        else:
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)
            theta_t = theta_0 * t
            sin_theta_t = np.sin(theta_t)
            s0 = np.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            v2 = s0 * v0 + s1 * v1
        return v2

    t = np.linspace(t0, t1, num)

    v3 = torch.tensor(np.array([interpolation(t[i], v0, v1) for i in range(num)]))

    return v3

pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(os.environ.get('OUTPUT_DIR'), scheduler=scheduler)
pipe = pipe.to("cuda")


image = Image.open("/home/ubuntu/expp/aligned0.png").convert("RGB").resize((512, 512))
depth_map = Image.open("/home/ubuntu//expp/depth2.png").convert("RGB").resize((512, 512))


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

seed = 42
generator = torch.manual_seed(seed)

def get_latents(image):
    init_latents = image
    init_latents = torch.cat([init_latents], dim=0)
    shape = init_latents.shape
    noise = randn_tensor(shape, generator=generator, device=device)

    # get latents
    init_latents = scheduler.add_noise(init_latents, noise, scheduler.timesteps[10:])
    latents = init_latents
    return latents
    

image_transform = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.ToTensor()
    ]
)
#image = image_transform(image)
#image = image[None,:,:,:]
#image = image.to("cuda")
#image = transforms.ToPILImage()(image[0])

depth_map = image_transform(depth_map)
depth_map = depth_map[None,:,:,:]
depth_map = depth_map.to("cuda")


depth_map = pipe.depth_estimator(depth_map).predicted_depth

print(depth_map.shape)

#compare(depth_map, image, cmap="gray", start_mode="horizontal", start_slider_pos=0.73)

#depth_map = Image.open("/home/ubuntu//expp/full2.png").convert("RGB").resize((384, 384))

target_size = (512, 512)

# Interpolate the depth map tensor to the target size
depth_map_tensor_resized = F.interpolate(depth_map.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
depth_map_tensor_resized = depth_map_tensor_resized.squeeze(0)

result = pipe("a photo of a watermelon cross section, vertical, no seeds", image=image, depth_map=depth_map_tensor_resized, strength=0.9, num_inference_steps=100)
#compare(result[0][0], image, cmap="gray", start_mode="horizontal", start_slider_pos=0.73)

result[0][0].save("./dw_result.png")

#image.save("./raw_result.png")

"""
depth_map_normalized = (depth_map_tensor_resized - depth_map_tensor_resized.min()) / (depth_map_tensor_resized.max() - depth_map_tensor_resized.min())
depth_map_rgb = torch.cat([depth_map_normalized] * 3, dim=0)  # Shape will be (3, 384, 384)
# Convert tensor to PIL Image
to_pil = transforms.ToPILImage()
depth_map_pil = to_pil(depth_map_rgb.cpu())  # Convert to CPU if tensor was on GPU
depth_map_pil.save("./depth_result.png")
"""

