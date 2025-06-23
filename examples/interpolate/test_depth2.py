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

pipe = StableDiffusionDepth2ImgPipeline.from_pretrained("/home/ubuntu/diffusersfood/data_t/watermelon_v")
pipe = pipe.to("cuda")

data_dir = "/home/ubuntu/w_res2/w_res2"

image = Image.open("/home/ubuntu/expp/aligned0.png").convert("RGB").resize((512, 512))
#depth_map = Image.open(data_dir + "0/").convert("RGB").resize((512, 512))

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

def get_depth_map(idx):
    dir_path = os.path.join(data_dir, idx) 
    m_path = os.path.join(dir_path, "full0.png")
    depth_map = Image.open(m_path).convert("RGB").resize((512, 512))
    depth_map = image_transform(depth_map)
    depth_map = depth_map[None,:,:,:]
    depth_map = depth_map.to("cuda")
    depth_map = pipe.depth_estimator(depth_map).predicted_depth
    target_size = (512, 512)
    depth_map_tensor_resized = F.interpolate(depth_map.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
    depth_map_tensor_resized = depth_map_tensor_resized.squeeze(0)
    depth_map_normalized = (depth_map_tensor_resized - depth_map_tensor_resized.min()) / (depth_map_tensor_resized.max() - depth_map_tensor_resized.min())
    depth_map_rgb = torch.cat([depth_map_normalized] * 3, dim=0)
    to_pil = transforms.ToPILImage()
    depth_map_pil = to_pil(depth_map_rgb.cpu())  # Convert to CPU if tensor was on GPU
    depth_map_pil.save(os.path.join(dir_path, "depth0.png"))
    return depth_map_tensor_resized

target_size = (512, 512)

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

depth1 = get_depth_map("0")

result, copy_latent0 = pipe("a photo of a watermelon cross section, vertical, no seeds", image=image, depth_map=depth1, strength=0.9, num_inference_steps=100)
#compare(result[0][0], image, cmap="gray", start_mode="horizontal", start_slider_pos=0.73)
result[0][0].save(os.path.join(data_dir, os.path.join("0", "label0.png")))

depth179 = get_depth_map("179")

result, copy_latent179 = pipe("a photo of a watermelon cross section, vertical, no seeds", image=image, depth_map=depth179, strength=0.9, num_inference_steps=100)
result[0][0].save(os.path.join(data_dir, os.path.join("179", "label0.png")))
print(copy_latent0.shape)

interpolated_latents = slerp(copy_latent0, copy_latent179, 180)
print("debug1")
print(interpolated_latents.size())
images = []
print("debug2")
for i, latent_vector in enumerate(interpolated_latents):
    latent_vector = latent_vector.to("cuda")
    print(latent_vector.shape)
    depth_map = get_depth_map(str(i))
    result, _ = pipe("a photo of a watermelon cross section, vertical, no seeds", image=image, depth_map=depth_map, strength=0.9, num_inference_steps=100, latents=latent_vector)
    result[0][0].save(os.path.join(data_dir, os.path.join(str(i), "label0.png")))

"""
depth_map_normalized = (depth_map_tensor_resized - depth_map_tensor_resized.min()) / (depth_map_tensor_resized.max() - depth_map_tensor_resized.min())
depth_map_rgb = torch.cat([depth_map_normalized] * 3, dim=0)  # Shape will be (3, 384, 384)
# Convert tensor to PIL Image
to_pil = transforms.ToPILImage()
depth_map_pil = to_pil(depth_map_rgb.cpu())  # Convert to CPU if tensor was on GPU
depth_map_pil.save("./depth_result.png")
"""

