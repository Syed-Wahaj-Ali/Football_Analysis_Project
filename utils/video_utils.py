import cv2 as cv

# Reading the Input Video
def read_video(input_path):
    cap = cv.VideoCapture(input_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            frames.append(frame)
    cap.release()
    return frames

# Saving the Video
def save_video(output_video_frames, output_video_path):
    if len(output_video_frames) == 0:
        print("Error: No frames to write!")
        return
    
    # Dynamically get the size from the first frame
    height = output_video_frames[0].shape[0] 
    width = output_video_frames[0].shape[1]

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(output_video_path, fourcc, 24.0, (width, height))

    for frame in output_video_frames:
        out.write(frame)
    
    out.release()