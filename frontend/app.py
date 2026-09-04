import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="RoadVision Copilot", page_icon="🚦", layout="wide")

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Backend-Status anzeigen ---
with st.sidebar:
    st.subheader("Backend Status")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        response.raise_for_status()
        health = response.json()

        api_ok = health.get("api") == "ok"
        chroma_ok = health.get("chroma_db") == "healthy"

        if api_ok:
            st.success("API: ok")
        else:
            st.warning(f"API: {health.get('api', 'unknown')}")

        if chroma_ok:
            st.success("ChromaDB: ok")
        else:
            st.warning("ChromaDB: degraded")

        st.json(health)

    except requests.exceptions.RequestException:
        st.error("Backend not reachable")

st.title("🚦 Roadvision - Copilot")

# --- Chat-Verlauf im Session State halten ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Zwei Container nebeneinander: 80/20 ---
col1, col2 = st.columns([4, 2])

with col1, st.container(border=True, height=770):
    uploaded_video = st.file_uploader("Video hochladen", type=["mp4", "mov", "avi"])
    if uploaded_video is not None:
        st.video(uploaded_video)
    else:
        st.info("Noch kein Video hochgeladen.")
    st.button("Generate report")

with col2, st.container(border=True, height=770):
    st.subheader("Chatbot")
    chat_box = st.container(height=540)
    with chat_box:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Frag zum erkannten Schild..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_box, st.chat_message("user"):
            st.markdown(prompt)

        try:
            response = requests.post(
                f"{API_BASE_URL}/chat",
                json={"message": prompt},
                timeout=15,
            )
            response.raise_for_status()
            answer = response.json().get("answer", "Keine Antwort erhalten.")
        except requests.exceptions.RequestException:
            answer = "⚠️ Backend nicht erreichbar."

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with chat_box, st.chat_message("assistant"):
            st.markdown(answer)
