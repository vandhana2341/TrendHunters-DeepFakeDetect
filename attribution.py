def attribute(reasons):
    if "Lip-sync mismatch" in reasons:
        return "Likely Face Swap / Audio-Video Mismatch"

    if "Abnormal blinking pattern" in reasons:
        return "Likely GAN-based generation"

    if "Audio anomaly detected" in reasons:
        return "Likely Voice Cloning"

    return "Unknown / Real"