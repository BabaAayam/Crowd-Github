import cv2

def get_video_source(source_type, path):
    """Handle both video files and camera streams"""
    if source_type == "cctv":
        return cv2.VideoCapture(path)  # RTSP stream
    elif source_type == "video":
        return cv2.VideoCapture(path)  # Video file
    elif source_type == "camera":
        return cv2.VideoCapture(0)     # Default camera
    else:
        raise ValueError("Invalid source type")