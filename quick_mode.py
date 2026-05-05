import os
from utils.video_utils import extract_frames
from utils.blink import detect_blink_anomaly
from utils.lipsync import lip_sync_score


def run_quick(video_path):
    os.makedirs("temp", exist_ok=True)
    frames, timestamps = extract_frames(video_path)

    reasons = []
    anomaly_times = []
    confidence = 0.3

    blink_flag, _ = detect_blink_anomaly(frames)
    lip_score = lip_sync_score()

    if blink_flag:
        reasons.append("Abnormal blinking pattern")
        confidence += 0.3

        for i in range(min(5, len(timestamps))):
            anomaly_times.append({
                "time": round(timestamps[i], 2),
                "issue": "Blink anomaly"
            })

    if lip_score > 0.6:
        reasons.append("Lip-sync mismatch")
        confidence += 0.3

        for i in range(min(5, len(timestamps))):
            anomaly_times.append({
                "time": round(timestamps[i], 2),
                "issue": "Lip-sync mismatch"
            })

    scores_over_time = []
    for _ in timestamps:
        score = 0.3
        if blink_flag:
            score += 0.2
        if lip_score > 0.6:
            score += 0.2
        scores_over_time.append(score)

    label = "Fake" if confidence > 0.5 else "Real"

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "reasons": reasons if reasons else ["No strong anomaly"],
        "timestamps": anomaly_times,
        "graph": {
            "time": timestamps,
            "scores": scores_over_time
        }
    }