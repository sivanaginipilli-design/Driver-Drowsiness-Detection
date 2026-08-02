import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import av

st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")

st.title("🚗 Driver Drowsiness Detection System")
st.write("Kindha unna **START** button meeda click chesi camera access allow cheyandi.")

# Browser camera network connection settings
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Webcam processing frame function
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    # =========================================================
    # MEERU YOLOV8 / OPENCV DROWSINESS DETECTION CODE IKKADA ADDEYANDI
    # Example:
    # img = process_drowsiness(img)
    # =========================================================

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Browser Streamlit WebRTC Component
webrtc_streamer(
    key="drowsiness-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
