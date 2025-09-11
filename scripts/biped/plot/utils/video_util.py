import cv2
import argparse

"""
Add timestamp to the given video file.
"""

def add_timestamp_to_video(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore # or "avc1" if available
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # --- figure out font scale to get ~100px text height ---
    font        = cv2.FONT_HERSHEY_SIMPLEX
    thickness   = 10
    target_h_px = 100
    # height for scale=1
    base_size   = cv2.getTextSize("0.00", font, 1.0, thickness)[0]
    font_scale  = target_h_px / max(base_size[1], 1)

    # position (x,y) is the *baseline*, so add margin
    x, y = 1200, int(60 + target_h_px)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        seconds = frame_idx / fps
        text = f"t={seconds:.2f}s"
        text_color = (255, 255, 255)  # white
        # text_color = (0, 0, 0)
        cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print("Done:", output_path)

def extract_frame(video_path:str, frame_number:int, output_image_path:str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number >= total_frames:
        raise ValueError(f"Frame number {frame_number} exceeds total frames {total_frames}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_number} from {video_path}")

    cv2.imwrite(output_image_path, frame)
    cap.release()
    print(f"Extracted frame {frame_number} to {output_image_path}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Add timestamps to a video.")
    # parser.add_argument("--input", type=str, required=True, help="Path to the input video file.")
    # parser.add_argument("--output", type=str, required=True, help="Path to save the output video file with timestamps.")
    # args = parser.parse_args()
    
    # # Use provided paths or default values
    # input_path  = args.input
    # output_path = args.output
    # add_timestamp_to_video(input_path, output_path)

    video_path = "/home/jkamohara3/hector_ws/isaac_45/IsaacLab4.5/logs/rl_games/manager_sac_rl_games_slip_mlp/2025-08-18_09-00-00/videos/play_rl/rl_slip_2_timestamp.mp4"
    frame_number = 150
    output_path = f"/home/jkamohara3/hector_ws/isaac_45/IsaacLab4.5/logs/rl_games/manager_sac_rl_games_slip_mlp/2025-08-18_09-00-00/videos/play_rl/rl_slip_2_frame{frame_number}.png"
    extract_frame(video_path, frame_number, output_path)