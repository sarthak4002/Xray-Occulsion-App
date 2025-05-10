import streamlit as st
from PIL import Image
import tensorflow as tf
from utils.occlusion import run_occlusion_analysis

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/efficientnetb4_multilabel.h5")

class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
               'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
               'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding']

st.title("🩺 Chest X-ray Disease Predictor with Occlusion Maps")

uploaded_file = st.file_uploader("Upload a Chest X-ray", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded X-ray", use_column_width=True)
    
    st.write("🔍 Analyzing...")
    model = load_model()
    
    results = run_occlusion_analysis(image, model, class_names)
    
    for res in results:
        st.image(res['image_bytes'], caption=f"{res['class_name']} ({res['score']:.2f})", use_column_width=True)
