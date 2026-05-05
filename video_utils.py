import cv2

def extract_frames(video_path, step=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    timestamps = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if i % step == 0:
            frames.append(frame)
            timestamps.append(i / fps)

        i += 1

    cap.release()
    return frames, timestamps