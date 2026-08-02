import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

st.set_page_config(page_title="Driver Drowsiness Detection", layout="centered")

st.title("🚗 Driver Drowsiness Detection System")
st.write("Live WebCam లో కళ్ళు మూసుకుంటే డిటెక్ట్ చేస్తుంది.")

# STUN సర్వర్ కాన్ఫిగరేషన్ (బ్రౌజర్ కెమెరా కనెక్షన్ కోసం చాలా ముఖ్యం)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class DrowsinessProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # కెమెరా ఫ్రేమ్‌ని OpenCV format (BGR) కి మార్చడం
        img = frame.to_ndarray(format="bgr24")

        # =======================================================
        # 🔻 నీ Drowsiness Detection లాజిక్ ఇక్కడ రన్ అవుతుంది 🔻
        # ఉదాహరణకి స్క్రీన్ మీద గ్రీన్ బాక్స్ / టెక్స్ట్ చూపించడానికి:
        
        cv2.putText(img, "Detection Active", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # నీ కళ్ళ / నిద్ర డిటెక్షన్ లాజిక్ (OpenCV/Cascade/MediaPipe) ఇక్కడ పెట్టుకో
        # =======================================================

        # తిరిగి వీడియో ఫ్రేమ్‌గా మార్చి డిస్ప్లే చేయడం
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC Streamer
webrtc_streamer(
    key="drowsiness-detection",
    video_processor_factory=DrowsinessProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)
