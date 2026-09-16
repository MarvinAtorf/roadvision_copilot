import os
import threading
from pathlib import Path

import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="RoadVision Copilot", page_icon="🚦", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

    # clear LATEST_ANALYSIS_JSON
    try:
        requests.delete(f"{API_BASE_URL}/analyze/video/analysis", timeout=3)
    except requests.exceptions.RequestException:
        st.warning("Failed to clear analysis")

if "analysis_state" not in st.session_state:
    # None = no analysis in flight. While running, holds a plain dict shared
    # with the background thread: {"running", "video_bytes", "error"}.
    st.session_state.analysis_state = None

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


def _run_analysis_worker(state: dict, api_base_url: str, file_payload: dict) -> None:
    """
    Run the blocking video analysis request in a background thread.

    Only mutates `state`, a plain dict shared with the main thread — never
    touches st.session_state directly, since Streamlit's session state API
    requires the main script's thread context, which this thread does not have.

    :param state: Shared mutable dict, created and referenced before the
        thread is started.
    :param api_base_url: Base URL of the RoadVision backend API.
    :param file_payload: The `files` dict passed to requests.post, containing
        the uploaded video.
    """
    try:
        response = requests.post(
            f"{api_base_url}/analyze/video",
            files=file_payload,
            timeout=1000,
        )
        response.raise_for_status()
        state["video_bytes"] = response.content
        state["error"] = None
    except requests.exceptions.RequestException as exc:
        state["error"] = str(exc)
    finally:
        state["running"] = False


# ============================================================
# HANDLE COMPLETED BACKGROUND ANALYSIS
# ============================================================
# Runs once, right after the background thread has finished, on whichever
# rerun (triggered by autorefresh) first notices running == False.

if st.session_state.analysis_state is not None and not st.session_state.analysis_state["running"]:
    finished_state = st.session_state.analysis_state

    if finished_state["error"]:
        st.session_state.analysis_error = finished_state["error"]
    else:
        try:
            analysis_response = requests.get(
                f"{API_BASE_URL}/analyze/video/analysis",
                timeout=10,
            )
            analysis_response.raise_for_status()

            st.session_state.video_analysis = analysis_response.json()
            st.session_state.processed_video = finished_state["video_bytes"]
            st.session_state.analysis_error = None
        except requests.exceptions.RequestException as exc:
            st.session_state.analysis_error = str(exc)

    st.session_state.analysis_state = None

# ============================================================
# SIDEBAR - VIDEO ANALYSIS
# ============================================================

with st.sidebar:
    st.subheader("Video Analysis")

    analysis_running = (
        st.session_state.analysis_state is not None and st.session_state.analysis_state["running"]
    )

    if analysis_running:
        # Keep rerunning while the background thread is still working, so
        # the live counts below stay fresh.
        st_autorefresh(interval=750, key="live_status_refresh")

        try:
            status_response = requests.get(
                f"{API_BASE_URL}/analyze/video/status",
                timeout=3,
            )
            live_status = status_response.json() if status_response.status_code == 200 else None
        except requests.exceptions.RequestException:
            live_status = None

        if live_status:
            progress = live_status.get("progress_percent", 0)

            st.write(f"**Processing... {progress:.0f}%**")
            st.progress(min(progress / 100, 1.0))

            st.markdown("---")

            st.markdown("**Vehicles**")

            live_vehicle_counts = live_status.get("vehicle_counts", {})

            st.write(f"Cars: {live_vehicle_counts.get('car', 0)}")
            st.write(f"Trucks: {live_vehicle_counts.get('truck', 0)}")
            st.write(f"Buses: {live_vehicle_counts.get('bus', 0)}")
            st.write(f"Motorbikes: {live_vehicle_counts.get('motorbike', 0)}")
            st.write(f"Bicycles: {live_vehicle_counts.get('bicycle', 0)}")

            st.markdown("---")

            st.markdown(f"**Total vehicles: {live_status.get('total_unique_vehicles', 0)}**")

            st.markdown("---")

            st.markdown("**Traffic**")
            st.write(f"Traffic lights: {live_status.get('traffic_light_count', 0)}")
            st.markdown("---")
            st.markdown("**Traffic Signs**")
            live_sign_counts = live_status.get("sign_counts", {})
            st.write(f"Unique signs detected: {live_status.get('total_unique_signs', 0)}")
            top_live_signs = sorted(live_sign_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for sign_name, count in top_live_signs:
                st.write(f"{sign_name}: {count}")

        else:
            st.info("Starting analysis...")

    elif "video_analysis" in st.session_state:
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

        st.markdown("---")

        traffic_sign_analysis = data.get("traffic_sign_analysis", {})
        sign_counts = traffic_sign_analysis.get("sign_counts", {})

        st.markdown("**Traffic signs**")
        st.write(f"Unique signs detected: {traffic_sign_analysis.get('total_unique_signs', 0)}")

        top_signs = sorted(sign_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        for sign_name, count in top_signs:
            st.write(f"{sign_name}: {count}")

        if len(sign_counts) > 8:
            st.caption(f"...and {len(sign_counts) - 8} more sign types")

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

    if st.session_state.get("analysis_error"):
        st.error(f"An error occurred while processing the video: {st.session_state.analysis_error}")
        st.session_state.analysis_error = None

    analysis_running = (
        st.session_state.analysis_state is not None and st.session_state.analysis_state["running"]
    )

    if analysis_running:
        st.info("Processing the video — live counts are shown in the sidebar.")
    elif uploaded_video is not None:
        if st.button("Analyze Video"):
            files = {
                "file": (
                    uploaded_video.name,
                    uploaded_video.getvalue(),
                    uploaded_video.type,
                )
            }

            st.session_state.analysis_state = {
                "running": True,
                "video_bytes": None,
                "error": None,
            }

            worker_thread = threading.Thread(
                target=_run_analysis_worker,
                args=(st.session_state.analysis_state, API_BASE_URL, files),
                daemon=True,
            )
            worker_thread.start()

            st.rerun()
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
