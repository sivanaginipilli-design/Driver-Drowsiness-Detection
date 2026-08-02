import cv2
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="Driver Drowsiness Detection", layout="centered")

st.title("🚗 Driver Drowsiness Detection System")
st.write("Live WebCam లో కళ్ళు మూసుకుంటే డిటెక్ట్ చేస్తుంది.")

# పబ్లిక్ STUN సర్వర్స్ (నెట్‌వర్క్ కనెక్షన్ బ్లాక్ అవ్వకుండా ఉండటానికి)
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}
    ]
})

class DrowsinessProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # 1. ఫ్రేమ్‌ని OpenCV BGR ఫార్మాట్‌లోకి మార్చడం
        img = frame.to_ndarray(format="bgr24")

        # 2. ఇక్కడ నీవు రాసుకున్న డ్రౌవ్‌నెస్ డిటెక్షన్ లాజిక్ (Haar Cascade / MediaPipe) పెట్టుకోవచ్చు
        # ఉదాహరణకి టెక్స్ట్ ప్రింట్ అవ్వడానికి:
        cv2.putText(img, "Webcam Active...", (30, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 3. ప్రాసెస్ అయిన ఇమేజ్‌ని తిరిగి బ్రౌజర్‌కి పంపడం
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC Streamer - ব্লాక్ స్క్రీన్ రాకుండా అన్ని సెట్టింగ్‌లతో
webrtc_streamer(
    key="drowsiness-app",
    video_processor_factory=DrowsinessProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
