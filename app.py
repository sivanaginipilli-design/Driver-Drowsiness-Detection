import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

st.title("🚗 Driver Drowsiness Detection System")
st.write("Live WebCam లో కళ్ళు మూసుకుంటే డిటెక్ట్ చేస్తుంది.")

class DrowsinessTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Image frame ని OpenCV రూపంలోకి మార్చడం
        img = frame.to_ndarray(format="bgr24")

        # ----------------------------------------------------
        # 🔻 నీ Drowsiness Detection Code ఇక్కడ రన్ అవుతుంది 🔻
        # ఉదాహరణకి Gray scale మార్చడం & Text చూపించడం:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # నీ Model / Cascade Classifier లాజిక్ ఇక్కడ ఉపయోగించు:
        # Example: 
        # cv2.putText(img, "Monitoring...", (30, 50), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # ----------------------------------------------------

        return img

# WebRTC Streamer ను స్టార్ట్ చేయడం (ఇది బ్రౌజర్ కెమెరాని వాడటానికి అనుమతిస్తుంది)
webrtc_streamer(
    key="drowsiness-detection",
    video_transformer_factory=DrowsinessTransformer,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
