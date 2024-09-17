#!pip install diffusers transformers
# you need to install the package first
from diffusers import StableDiffusionPipeline

# Replace with your Stable Diffusion model path
model_id = "CompVis/stable-diffusion-v1-4"

# Load the model
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# Set the device (GPU or CPU)
pipe = pipe.to("cuda")  # If you have a GPU

# Generate an image from a text prompt
prompt =input("enter the prompt:")
image = pipe(prompt, guidance_scale=7.5, num_inference_steps=30).images[0]

# Save the image
display(image)
