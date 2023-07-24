import cv2

def decompose_by_frames(orig_path, output_directory):
    orig_path = str(orig_path)
    output_directory = str(output_directory)
    video = cv2.VideoCapture(orig_path)
    frame_count = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            output_path = f"{output_directory}/frame_{frame_count:04d}.jpg"
            cv2.imwrite(output_path, frame)
            frame_count += 1
        else:
            break

    video.release()
