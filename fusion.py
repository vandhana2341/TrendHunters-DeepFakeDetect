def fuse(video_score, audio_score):
    """
    Fuse video and audio scores to get final deepfake confidence.
    Weighted average: 60% video, 40% audio
    """
    final_score = (video_score * 0.6) + (audio_score * 0.4)
    return min(final_score, 1.0)