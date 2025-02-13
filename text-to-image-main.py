#!pip install diffusers transformers streamlit
# you need to install the packages first

import streamlit as st
from diffusers import StableDiffusionPipeline
from PIL import Image

# Replace with your Stable Diffusion model path
model_id = "CompVis/stable-diffusion-v1-4"

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    return StableDiffusionPipeline.from_pretrained(model_id).to("cuda")

pipe = load_model()

# Streamlit app
st.title("Image Generator")

# Text input for the prompt
prompt = st.text_input("Enter the prompt:")

# Generate image button
if st.button("Generate Image"):
    if prompt:
        with st.spinner("Generating image..."):
            # Generate the image
            image = pipe(prompt, guidance_scale=7.5, num_inference_steps=5).images[0]
            
            # Display the image
            st.image(image, caption="Generated Image", use_column_width=True)
    else:
        st.warning("Please enter a prompt to generate an image.")
