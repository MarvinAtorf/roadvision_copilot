import os
from pathlib import Path

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="RoadVision Copilot", page_icon="🚦", layout="wide")

# CSS to remove top padding and vertical gaps completely
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 1rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        div[data-testid="stVerticalBlock"] > div {
            gap: 0rem !important;
        }
        .stImage {
            margin-bottom: -1rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- LOGO PATH (DIRECT RELATIVE PATH FOR GITHUB) ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name == "frontend" else CURRENT_DIR

LOGO_PATH = PROJECT_ROOT / "assets" / "logo.png"

# --- DISPLAY BACKEND STATUS ---
with st.sidebar:
    st.subheader("Backend Status")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        response.raise_for_status()
        health = response.json()

        if health.get("chroma_db") == "ok":
            st.success("API: ok")
            st.success("ChromaDB: ok")
        else:
            st.success("API: ok")
            st.warning("ChromaDB: unreachable")

        st.json(health)

    except requests.exceptions.RequestException:
        st.error("Backend not reachable")

# --- STORE CHAT HISTORY IN SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- TOP SECTION (LOGO AT THE VERY TOP) ---
header_col1, header_col2 = st.columns([4, 3])

with header_col1:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=506)
    else:
        st.warning(f"Logo not found at: {LOGO_PATH}")

# --- BOTTOM SECTION (CONTAINERS PLACED DIRECTLY BELOW LOGO) ---
col1, col2 = st.columns([4, 3], gap="small")

# --- LEFT CONTAINER (VIDEO UPLOAD) ---
with col1, st.container(border=True, height=700):
    st.subheader("Traffic Analysis")
    uploaded_video = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        if st.button("Analyze Video"):
            with st.spinner("Processing video on the backend..."):
                try:
                    # Send video to FastAPI backend
                    files = {
                        "file": (
                            uploaded_video.name,
                            uploaded_video.getvalue(),
                            uploaded_video.type,
                        )
                    }
                    response = requests.post(
                        f"{API_BASE_URL}/analyze/video",
                        files=files,
                        timeout=300,
                    )
                    response.raise_for_status()

                    # Render processed video bytes on screen
                    st.success("Analysis complete!")
                    st.video(response.content)

                except requests.exceptions.RequestException as e:
                    st.error(f"An error occurred while processing the video: {e}")
    else:
        st.info("No video uploaded yet.")

    st.button("Generate report")

# --- RIGHT CONTAINER (CHATBOT) ---
with col2, st.container(border=True, height=700):
    st.subheader("Chatbot")
    chat_box = st.container(height=450)
    with chat_box:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Ask something about the video (signs, vehicles, ...)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_box, st.chat_message("user"):
            st.markdown(prompt)

        try:
            response = requests.post(
                f"{API_BASE_URL}/chat",
                json={"messages": st.session_state.messages},
                timeout=15,
            )
            response.raise_for_status()
            answer = response.json().get("answer", "no answer received.")
        except requests.exceptions.RequestException:
            answer = "⚠️ Backend not reachable."

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with chat_box, st.chat_message("assistant"):
            st.markdown(answer)
