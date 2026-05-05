import mediapipe.tasks as tasks
import mediapipe
import numpy as np
import cv2
import os

def detect_blink_anomaly(frames):
    """
    Detect blink anomalies in video frames.
    Returns (is_anomaly, blink_rate)
    """
    if not frames or len(frames) == 0:
        return False, 0.0
    
    blink_count = 0
    total = len(frames)
    timestamp_ms = 0
    
    try:
        vision_module = getattr(tasks, 'vision', None)
        if vision_module is None:
            raise ImportError("mediapipe.tasks.vision not available")
        
        FaceLandmarker = getattr(vision_module, 'FaceLandmarker', None)
        FaceLandmarkerOptions = getattr(vision_module, 'FaceLandmarkerOptions', None)
        RunningMode = getattr(vision_module, 'RunningMode', None)
        
        if FaceLandmarker is None or FaceLandmarkerOptions is None:
            raise ImportError("FaceLandmarker or FaceLandmarkerOptions not found")
        
        # Get model path - ensure it exists
        model_path = "face_landmarker.task"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create face landmark detector with proper configuration
        options = FaceLandmarkerOptions(
            base_options=tasks.BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO
        )
        detector = FaceLandmarker.create_from_options(options)
        
        # Get Image class
        Image = getattr(mediapipe, 'Image', None)
        if Image is None:
            raise ImportError("mediapipe.Image not found")
        
        previous_state = None
        
        for idx, frame in enumerate(frames):
            try:
                # Validate frame
                if frame is None or len(frame.shape) != 3:
                    continue
                
                # Convert BGR to RGB if needed (OpenCV uses BGR by default)
                if frame.shape[2] == 3:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = frame
                
                # Convert to MediaPipe Image format
                image = Image(
                    image_format=Image.ImageFormat.SRGB,
                    data=frame_rgb
                )
                
                # Detect with timestamp for video mode
                timestamp_ms = idx * 33  # Assuming ~30fps (1000ms / 30fps ≈ 33ms)
                result = detector.detect_for_video(image, timestamp_ms)
                
                if result and result.face_landmarks and len(result.face_landmarks) > 0:
                    # Check for actual blink by analyzing eye landmark closure
                    face_landmarks = result.face_landmarks[0]
                    
                    # Eye landmarks indices (left and right eye)
                    # Left eye: 33, 160, 158, 133, 153, 144
                    # Right eye: 263, 387, 385, 362, 380, 374
                    left_eye_closed = _is_eye_closed(face_landmarks, [33, 160, 158, 133, 153, 144])
                    right_eye_closed = _is_eye_closed(face_landmarks, [263, 387, 385, 362, 380, 374])
                    
                    current_state = left_eye_closed or right_eye_closed
                    
                    # Detect blink transition (open to closed)
                    if previous_state is False and current_state is True:
                        blink_count += 1
                    
                    previous_state = current_state
                else:
                    previous_state = None
                    
            except Exception as e:
                print(f"Error processing frame {idx}: {e}")
                pass
        
        del detector
        
    except (ImportError, FileNotFoundError) as e:
        print(f"Fallback mode: {e}")
        # Fallback: estimate blink rate based on frame variations
        blink_count = _estimate_blinks_fallback(frames)

    blink_rate = blink_count / max(total, 1)
    
    # Thresholds for anomaly detection
    # Normal blink rate: 15-20 blinks per minute ≈ 0.25-0.33 per second
    # For video: multiply by frame_rate to get appropriate threshold
    if blink_rate < 0.05 or blink_rate > 0.8:
        return True, blink_rate
    
    return False, blink_rate


def _is_eye_closed(face_landmarks, eye_indices):
    """
    Determine if eye is closed based on landmark positions.
    Uses vertical distance between upper and lower eyelids.
    """
    if len(face_landmarks) < max(eye_indices):
        return False
    
    try:
        # Get eye corner landmarks
        top = face_landmarks[eye_indices[1]].y
        bottom = face_landmarks[eye_indices[4]].y
        
        # Calculate eye aspect ratio approximation
        eye_opening = abs(top - bottom)
        
        # If vertical opening is very small, eye is closed
        return eye_opening < 0.02
    except (IndexError, AttributeError):
        return False


def _estimate_blinks_fallback(frames):
    """
    Fallback method: estimate blinks by detecting frame intensity changes.
    """
    if len(frames) < 2:
        return 0
    
    blink_count = 0
    threshold = 20  # Intensity difference threshold
    
    for i in range(1, len(frames)):
        try:
            # Compare frame intensities
            frame1_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            frame2_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            diff = cv2.absdiff(frame1_gray, frame2_gray)
            mean_diff = np.mean(diff)
            
            # Sudden brightness drop might indicate blink
            if mean_diff > threshold:
                blink_count += 1
        except Exception:
            pass
    
    return max(0, blink_count // 5)  # Normalize to reduce false positives