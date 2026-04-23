import cv2
import matplotlib.pyplot as plt
import argparse
import os
import math

def extract_frames(video_path, interval_sec=2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30 # Default to 30 if cannot detect
    
    frames = []
    frame_count = 0
    # Calculate frame step based on interval
    frame_step = int(fps * interval_sec)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Capture frame at every interval
        if frame_count % frame_step == 0:
            # Convert BGR (OpenCV) to RGB (Matplotlib)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        frame_count += 1
    
    cap.release()
    return frames

def main():
    parser = argparse.ArgumentParser(description="Plot frames from two videos at a fixed interval side-by-side.")
    parser.add_argument("--video1", help="Path to the first video (top row)")
    parser.add_argument("--video2", help="Path to the second video (bottom row)")
    parser.add_argument("--interval", type=int, default=2, help="Interval in seconds between frames (default: 2)")
    parser.add_argument("--output", default="video_comparison.png", help="Path to save the output plot")
    parser.add_argument("--title1", default="Video 1", help="Title for the first video")
    parser.add_argument("--title2", default="Video 2", help="Title for the second video")
    
    args = parser.parse_args()
    
    frames1 = extract_frames(args.video1, args.interval)
    frames2 = extract_frames(args.video2, args.interval)
    
    if not frames1 or not frames2:
        print("Error: Could not extract frames from one or both videos.")
        return

    num_frames = max(len(frames1), len(frames2))
    
    fig, axes = plt.subplots(2, num_frames, figsize=(num_frames * 3, 6))
    
    # If there's only one frame, axes is a 1D array or just an object depending on subplots
    if num_frames == 1:
        axes = axes.reshape(2, 1)

    for i in range(num_frames):
        # Top Row
        frame1_idx = min(i, len(frames1) - 1)
        axes[0, i].imshow(frames1[frame1_idx])
        if i == 0:
            axes[0, i].set_ylabel(args.title1, fontsize=12, fontweight='bold')
        axes[0, i].set_title(f"{i * args.interval}s")
        
        # Bottom Row
        frame2_idx = min(i, len(frames2) - 1)
        axes[1, i].imshow(frames2[frame2_idx])
        if i == 0:
            axes[1, i].set_ylabel(args.title2, fontsize=12, fontweight='bold')

        # Clean up both rows: hide ticks and spines but keep labels
        for r in range(2):
            axes[r, i].set_xticks([])
            axes[r, i].set_yticks([])
            for spine in axes[r, i].spines.values():
                spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight', dpi=150)
    print(f"Saved comparison plot to {args.output}")

if __name__ == "__main__":
    main()
