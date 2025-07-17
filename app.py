import streamlit as st
import tensorflow_hub as hub
import tensorflow as tf
from PIL import Image
import numpy as np
import io


st.title("🎨 Neural Style Transfer")

# image load
content_file = st.file_uploader("load content image", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("load style image", type=["jpg", "jpeg", "png"])

@st.cache_resource

#loading model with the help of tf_hub library
def load_model():
    return hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")


def load_img(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB") 
    image = image.resize((512, 512))
    img = np.array(image).astype(np.float32) / 255.0 #normalize
    img = img[np.newaxis, ...] #add new dimension
    return tf.constant(img) #convert tensor

if content_file and style_file: 
    with st.spinner("🖌 Style transferring  it can take time ....."):
        content_image = load_img(content_file)
        style_image = load_img(style_file)
        model = load_model()
        stylized_image = model(content_image, style_image)[0]

        # show result
        st.subheader("🎨 Result:")
        st.image(tf.squeeze(stylized_image).numpy(), use_container_width=True)

        # download button
        result = tf.image.convert_image_dtype(stylized_image[0], dtype=tf.uint8)
        result_pil = Image.fromarray(result.numpy())
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button("📥 download image", byte_im, file_name="style_transfer.png", mime="image/png")