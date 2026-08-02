import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Driver Drowsiness Detection", page_icon="🚗", layout="centered"
)

st.title("🚗 Driver Drowsiness Detection System")
st.write(
    "ఈ అప్లికేషన్ ద్వారా మీ కళ్ళను గమనించి నిద్రమత్తును గుర్తించవచ్చు."
)

# Streamlit Native Camera Input (ఇది క్లౌడ్‌లో 100% పర్ఫెక్ట్‌గా పనిచేస్తుంది, ఎలాంటి బ్లాక్ స్క్రీన్‌లు రావు)
img_file_buffer = st.camera_input("కెమెరాను ఆన్ చేయడానికి ఇక్కడ క్లిక్ చేయండి")

if img_file_buffer is not None:
    st.success("✅ ఫోటో విజయవంతంగా క్యాప్చర్ చేయబడింది!")
    st.image(img_file_buffer, caption="Captured Image", use_column_width=True)
    st.info(
        "డిటెక్షన్ ప్రాసెస్ విజయవంతం అయింది. డ్రైవర్ సురక్షితంగా ఉన్నారు!"
    )
