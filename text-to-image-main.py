import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)

pipe = load_model()

# Streamlit app
st.title("Stable Diffusion Image Generator")
prompt = st.text_input("Enter the prompt:")

if st.button("Generate Image"):
    if prompt:
        with st.spinner("Generating image..."):
            image = pipe(prompt, guidance_scale=7.5, num_inference_steps=30).images[0]
            st.image(image, caption="Generated Image", use_column_width=True)
    else:
        st.warning("Please enter a prompt to generate an image.")
