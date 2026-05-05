from moviepy import VideoFileClip
import librosa
import numpy as np

def extract_audio(video_path):
    audio_path = "temp/audio.wav"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path,  logger=None)
    return audio_path

def analyze_audio(video_path):
    audio_path = extract_audio(video_path)
    y, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    score = np.mean(np.abs(mfcc))
    return min(score / 100, 1.0)