import PIL
import torch
from torchvision import transforms
import diffusers
import transformers
from diffusers import StableDiffusionDepth2ImgPipeline
import os
from PIL import Image
import torch.nn.functional as F

print(f'Getting model from {os.environ.get("OUTPUT_DIR")}')
pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(os.environ.get('OUTPUT_DIR'))
pipe = pipe.to("cuda")


image = Image.open("/home/ubuntu/diffusersfood/data_t/crossiant/2.png").convert("RGB").resize((512, 512))
depth_map = Image.open("/home/ubuntu/diffusersfood/data_t/crossiant/5.png").convert("RGB").resize((512, 512))

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

result, _ = pipe("the cross section of a chocolate filled croissant", image=image, depth_map=depth_map_tensor_resized, strength=0.9, num_inference_steps=100)
#compare(result[0][0], image, cmap="gray", start_mode="horizontal", start_slider_pos=0.73)

result[0][0].save("/home/ubuntu/c2_res/dw_result.png")

image.save("/home/ubuntu/c2_res/raw_result.png")


depth_map_normalized = (depth_map_tensor_resized - depth_map_tensor_resized.min()) / (depth_map_tensor_resized.max() - depth_map_tensor_resized.min())
depth_map_rgb = torch.cat([depth_map_normalized] * 3, dim=0)  # Shape will be (3, 384, 384)
# Convert tensor to PIL Image
to_pil = transforms.ToPILImage()
depth_map_pil = to_pil(depth_map_rgb.cpu())  # Convert to CPU if tensor was on GPU
depth_map_pil.save("/home/ubuntu/c2_res/depth_result.png")

