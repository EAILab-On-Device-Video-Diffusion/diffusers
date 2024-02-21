import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

from diffusers import DiffusionPipeline
from PIL import Image
from io import BytesIO
import numpy as np
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

# pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid")
# pipe = DiffusionPipeline.from_pretrained("./Hotshot-XL")
pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w")

pipe = pipe.to("cuda")
init_image = Image.open('img.png').convert("RGB")
init_image = init_image.resize((224, 224))
init_image = np.array(init_image)
print(init_image.shape)
init_image = torch.Tensor(init_image).permute(2, 0, 1).unsqueeze(0).to("cuda")
prompt = "Spiderman is surfing"

# video_frames = pipe(prompt).frames[0]
video_frames = pipe(prompt, num_inference_steps=5, height=320, width=576, num_frames=24).frames
print(torch.cuda.max_memory_allocated() / 1024 ** 2,"MB")

video_path = export_to_video(video_frames)

video_path = export_to_video(video_frames)
print(f"Video saved to {video_path}")