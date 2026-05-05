# TrendHunters-DeepFakeDetect

# Deepfake Detector

An AI-powered Deepfake Detection system that analyzes audio/video inputs and classifies them as **Real** or **Fake** using a dual-mode pipeline (Quick & Deep analysis). The system also provides interpretability through confidence metrics, temporal analysis, and attribution insights.

---

## Features

### Dual Mode Analysis
- **Quick Mode**: Fast inference for real-time or lightweight checks
- **Deep Mode**: High-accuracy analysis with detailed temporal breakdown

### Classification Output
- Outputs final prediction:
  - ✅ Real
  - ❌ Fake

### Confidentiality Score
- Measures reliability and certainty of the prediction
- Helps assess trust level of the result

### Attribution Analysis
- Provides classification reasoning (e.g., facial inconsistencies, audio mismatch indicators)

### Suspicious Segment Detection
- Identifies specific time intervals in the input that appear manipulated
- Displays timestamps for forensic review

### Confidence vs Time Graph
- Visual representation of model confidence across timeline
- Helps understand where anomalies occur in the media

---

## System Overview

The system follows a modular ML pipeline:

1. **Input Processing**
   - Video/Audio extraction
2. **Feature Extraction**
   - Facial cues / audio artifacts / temporal signals
3. **Model Inference**
   - Quick or Deep model selection
4. **Post Processing**
   - Confidence scoring + attribution + segment analysis
5. **Visualization**
   - Graphs and results dashboard

---

## Tech Stack

- Python 
- OpenCV (video processing)
- NumPy / Pandas (data handling)
- PyTorch / TensorFlow (ML models)
- Streamlit (UI dashboard)
- Matplotlib / Plotly (visualizations)
