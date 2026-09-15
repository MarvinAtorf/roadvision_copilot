import os
from pathlib import Path

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="RoadVision Copilot", page_icon="🚦", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

    # clear LATEST_ANALYSIS_JSON
    try:
        requests.delete(f"{API_BASE_URL}/analyze/video/analysis", timeout=3)
    except requests.exceptions.RequestException:
        st.warning("Failed to clear analysis")

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

# ============================================================
# SIDEBAR - VIDEO ANALYSIS
# ============================================================

with st.sidebar:
    st.subheader("Video Analysis")

    if "video_analysis" in st.session_state:
        analysis = st.session_state.video_analysis

        data = analysis.get(
            "data",
            {},
        )

        video_metadata = data.get(
            "video_metadata",
            {},
        )

        vehicle_analysis = data.get(
            "vehicle_analysis",
            {},
        )

        traffic_light_analysis = data.get(
            "traffic_light_analysis",
            {},
        )

        vehicle_counts = vehicle_analysis.get(
            "vehicle_counts",
            {},
        )

        # --------------------------------------------------------
        # Duration
        # --------------------------------------------------------

        duration = video_metadata.get(
            "duration_seconds",
            0,
        )

        st.write(f"**Duration:** {duration:.1f} s")

        st.markdown("---")

        # --------------------------------------------------------
        # Vehicles
        # --------------------------------------------------------

        st.markdown("**Vehicles**")

        st.write(f"Cars: {vehicle_counts.get('car', 0)}")

        st.write(f"Trucks: {vehicle_counts.get('truck', 0)}")

        st.write(f"Buses: {vehicle_counts.get('bus', 0)}")

        st.write(f"Motorbikes: {vehicle_counts.get('motorbike', 0)}")

        st.write(f"Bicycles: {vehicle_counts.get('bicycle', 0)}")

        st.markdown("---")

        # --------------------------------------------------------
        # Total vehicles
        # --------------------------------------------------------

        st.markdown(f"**Total vehicles: {vehicle_analysis.get('total_unique_vehicles', 0)}**")

        st.markdown("---")

        # --------------------------------------------------------
        # Traffic
        # --------------------------------------------------------

        st.markdown("**Traffic**")

        st.write(f"Traffic lights: {traffic_light_analysis.get('max_visible_simultaneously', 0)}")

    else:
        st.info("Upload and analyze a video to see the results here")

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

    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "mov", "avi"],
    )

    if uploaded_video is not None:
        if st.button("Analyze Video"):
            with st.spinner("Processing the video..."):
                try:
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
                        timeout=1000,
                    )

                    response.raise_for_status()

                    # ------------------------------------------------
                    # Get analysis JSON
                    # ------------------------------------------------

                    analysis_response = requests.get(
                        f"{API_BASE_URL}/analyze/video/analysis",
                        timeout=15,
                    )

                    analysis_response.raise_for_status()

                    analysis = analysis_response.json()

                    # ------------------------------------------------
                    # Save analysis in Streamlit session
                    # ------------------------------------------------

                    st.session_state.video_analysis = analysis

                    # ------------------------------------------------
                    # Save processed video
                    # ------------------------------------------------

                    st.session_state.processed_video = response.content

                    # ------------------------------------------------
                    # Rerun Streamlit
                    # ------------------------------------------------

                    st.rerun()

                except requests.exceptions.RequestException as e:
                    st.error(f"An error occurred while processing the video: {e}")

    else:
        st.info("No video uploaded yet.")

    # ------------------------------------------------
    # Show processed video after analysis
    # ------------------------------------------------

    if "processed_video" in st.session_state:
        st.success("Analysis complete!")

        st.video(st.session_state.processed_video)

    col_report, col_clear = st.columns([6, 1])
    with col_report:
        if uploaded_video:
            st.button("Generate report")


# --- RIGHT CONTAINER (CHATBOT) ---
with col2, st.container(border=True, height=700):
    chatbot_title_col, chatbot_spinner_col = st.columns([3, 1])
    with chatbot_title_col:
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

        with chatbot_spinner_col, st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/chat",
                    json={"messages": st.session_state.messages},
                    timeout=15,
                )
                response.raise_for_status()
                answer = response.json().get("answer", "No answer received.")
            except requests.exceptions.RequestException:
                answer = "⚠️ Backend not reachable."

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with chat_box, st.chat_message("assistant"):
            st.markdown(answer)

    st.info("this is a chatbot and not a real lawyer")
