from utils.audio_utils import analyze_audio
from utils.fusion import fuse
from modes.quick_mode import run_quick

def run_deep(video_path):

    # Run quick visual model first
    quick_result = run_quick(video_path)

    video_score = quick_result["confidence"]
    audio_score = analyze_audio(video_path)

    # Final multimodal fusion
    final_score = fuse(video_score, audio_score)

    label = "Fake" if final_score > 0.5 else "Real"

    reasons = quick_result["reasons"]

    # Add audio explanation if needed
    if audio_score > 0.6:
        reasons.append("Audio anomaly detected (possible lip-sync mismatch or synthetic speech)")

    return {
        "label": label,
        "confidence": round(final_score, 2),
        "reasons": reasons,
        "timestamps": quick_result["timestamps"],
        "graph": quick_result["graph"]
    }