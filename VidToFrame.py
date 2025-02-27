import cv2
import os
import argparse
from datetime import datetime

#if the video aint encrypted right u have to manually convert it: 
#   use the ffmpeg commands to do it ex:
#   ffmpeg -i ~/path/to/video.mp4 -c:v libx264 -crf 23 -preset fast video_converted.mp4

def process_video(input_path, output_format='jpg', quality=95):
    #create parent dir for the extracted files
    parent_dir = "extracted_frames"
    os.makedirs(parent_dir, exist_ok=True)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(parent_dir, f"video_frames_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize video capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    # Calculate frame sampling rate
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        print("Warning: Invalid frame rate, assuming 30 FPS")
        original_fps = 30
    
    desired_fps = 30
    step = 1
    
    if original_fps > desired_fps:
        step = round(original_fps / desired_fps)
        print(f"Original FPS: {original_fps:.2f}, sampling every {step} frames")

    frame_count = 0
    saved_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process frames that match our sampling rate
            if frame_count % step == 0:
                frame_file = os.path.join(output_dir, f"frame_{saved_count:06d}.{output_format}")
                save_params = []
                
                if output_format == 'jpg':
                    save_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                elif output_format == 'png':
                    save_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 9 - round(quality/10)]
                
                cv2.imwrite(frame_file, frame, save_params)
                saved_count += 1
                
                # Progress feedback every 100 saved frames
                if saved_count % 100 == 0:
                    print(f"Processed {saved_count} frames...")
            
            frame_count += 1
    finally:
        cap.release()
    
    print(f"\nSuccess! {saved_count} frames saved to: {output_dir}/")
    print(f"Effective output FPS: {original_fps/step:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract video frames with FPS adjustment')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--format', choices=['jpg', 'png'], default='jpg',
                       help='Output image format (default: jpg)')
    parser.add_argument('--quality', type=int, default=95,
                       help='Image quality (1-100, default: 95)')
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.video_path):
        raise FileNotFoundError(f"Video file not found: {args.video_path}")
    
    process_video(args.video_path, args.format, args.quality)