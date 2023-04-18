import cv2

# Mở video
cap = cv2.VideoCapture(r"C:\Users\plado\Downloads\3.mp4")

# Lấy số lượng frame của video
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Lấy tốc độ khung hình của video
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Tính thời gian của video
video_time = num_frames / fps

# In ra thời gian của video
print("Thời gian của video: ", fps, " giây")

# Giải phóng video
cap.release()