import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="RoadVision Copilot", page_icon="🚦")

st.title("🚦 RoadVision Copilot")
st.caption("Traffic sign detection + StVO lookup + report generation")

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

    except requests.exceptions.RequestException:
        st.error("Backend not reachable")
