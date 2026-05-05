import os
import streamlit as st
from modes.quick_mode import run_quick
from modes.deep_mode import run_deep
from utils.graph import plot_confidence

st.title("Deepfake Detection System")

mode = st.selectbox("Select Mode", ["Quick Analysis", "Deep Analysis"])
file = st.file_uploader("Upload Video", type=["mp4", "mov"])

if file:
    os.makedirs("temp", exist_ok=True)
    video_path = os.path.abspath("temp/input.mp4")

    with open(video_path, "wb") as f:
        f.write(file.read())

    st.video(video_path)

    if st.button("Run Analysis"):

        if mode == "Quick Analysis":
            result = run_quick(video_path)
        else:
            result = run_deep(video_path)

        st.write("### Result:", result["label"])
        st.write("Confidence:", result["confidence"])
        st.write("Reasons:", result["reasons"])

        # Timestamps
        if "timestamps" in result:
            st.write("### Suspicious Segments:")
            for t in result["timestamps"]:
                st.write(f"{t['time']} sec → {t['issue']}")

        # Graph
        if "graph" in result:
            fig = plot_confidence(result["graph"]["time"], result["graph"]["scores"])
            st.pyplot(fig)