# Video to Frame Extraction Script

This Python script extracts individual frames from a video file and saves them as image files. It is designed to effectively output 30 frames per second by skipping frames if the original video has a higher frame rate.
### Features

    Frame Extraction:
    Reads the video using OpenCV and extracts frames at an effective rate of 30 FPS. If the video’s original FPS is higher, it automatically skips frames to achieve the desired rate.

    Organized Output:
    Creates a parent folder named extracted_frames (if it doesn’t already exist). For each run, it generates a new timestamped subfolder (e.g., video_frames_20250227_101409) where all extracted frames are saved. This makes it easy to ignore the output directory in version control (add extracted_frames/ to your .gitignore).

    Customizable Image Format and Quality:
    Supports output in either JPEG or PNG format. You can specify the desired image quality via command-line arguments.

### Usage

Run the script from the command line as follows:
##
    python3 VidToFrame.py path/to/video.mp4 --format jpg --quality 95

video_path: Path to the input video file.
    --format: Output image format (choose between jpg or png; default is jpg).
    --quality: Image quality (an integer between 1 and 100, default is 95).

### Converting Videos with FFmpeg

If your video is not encoded in a compatible format (for example, if it’s not using the H.264 codec), you may need to convert it before processing. You can use FFmpeg for this purpose. For example, to convert a video to H.264-encoded MP4, run:
## 
    ffmpeg -i ~/path/to/video.mp4 -c:v libx264 -crf 23 -preset fast video_converted.mp4

After conversion, you can process the new file with the script:
## 
    python3 VidToFrame.py video_converted.mp4 --format jpg --quality 95

### Dependencies

    Python 3.x
    OpenCV (opencv-python)
    FFmpeg (for video conversion if needed)