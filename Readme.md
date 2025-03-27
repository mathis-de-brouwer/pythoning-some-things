# Video to Frame Extraction Script

This Python script extracts individual frames from a video file and saves them as image files. It is designed to effectively output 30 frames per second by skipping frames if the original video has a higher frame rate.

### Features

#### Frame Extraction:
Reads the video using OpenCV and extracts frames at an effective rate of 30 FPS. If the video’s original FPS is higher, it automatically skips frames to achieve the desired rate.

#### Organized Output:
Creates a parent folder named `extracted_frames` (if it doesn’t already exist). For each run, it generates a new timestamped subfolder (e.g., `video_frames_20250227_101409`) where all extracted frames are saved. This makes it easy to ignore the output directory in version control (add `extracted_frames/` to your `.gitignore`).

#### Customizable Image Format and Quality:
Supports output in either JPEG or PNG format. You can specify the desired image quality via command-line arguments.

### Usage

Run the script from the command line as follows:
```bash
python3 VidToFrame.py path/to/video.mp4 --format jpg --quality 95
```

#### Parameters:
- `video_path`: Path to the input video file.
- `--format`: Output image format (choose between `jpg` or `png`; default is `jpg`).
- `--quality`: Image quality (an integer between `1` and `100`, default is `95`).

### Converting Videos with FFmpeg

If your video is not encoded in a compatible format (for example, if it’s not using the H.264 codec), you may need to convert it before processing. You can use FFmpeg for this purpose. For example, to convert a video to H.264-encoded MP4, run:
```bash
ffmpeg -i ~/path/to/video.mp4 -c:v libx264 -crf 23 -preset fast video_converted.mp4
```

After conversion, you can process the new file with the script:
```bash
python3 VidToFrame.py video_converted.mp4 --format jpg --quality 95
```

### Dependencies
- Python 3.x
- OpenCV (`opencv-python`)
- FFmpeg (for video conversion if needed)

# Encrypted Video Streaming System

This project includes a secure, real-time video streaming system that can stream from either a webcam or video file over a WebSocket connection with AES-256 encryption.

## Components

- **Signaling Server**: Acts as the communication relay between sender and receiver
- **Video Sender**: Captures frames from webcam or video file, compresses, encrypts, and sends them
- **Video Receiver**: Receives encrypted frames, decrypts, and displays them with performance metrics

## Features

- **End-to-end Encryption**: All video frames are encrypted using AES-256 in CBC mode
- **Adaptive Quality**: Automatically adjusts compression quality based on framerate
- **Multi-source Support**: Stream from either webcam or video file
- **Performance Metrics**: Real-time display of compression/encryption times, FPS, and more
- **Frame Saving**: Option to save received frames or take snapshots

## Usage

### 1. Start the Signaling Server

```bash
python signalingServer.py
```

This starts the WebSocket server on port 9000.

### 2. Start the Receiver

```bash
python testing\ some\ things/recievertest1.py
```

Options:
```bash
python testing\ some\ things/recievertest1.py --target-fps 15 --max-quality 80 --save-frames
```

### 3. Start the Sender

#### For webcam streaming:
```bash
python testing\ some\ things/sendertest1.py --source 0
```

#### For video file streaming:
```bash
python testing\ some\ things/sendertest1.py --source /path/to/video.mp4
```

#### With custom settings:
```bash
python testing\ some\ things/sendertest1.py --source 1 --quality 70 --fps 20 --width 1280 --height 720
```

### Advanced Options

#### Sender Options
- `--source`: Webcam index (`0, 1`) or path to video file
- `--server`: WebSocket server URL (default: `ws://heliwi.duckdns.org:9000`)
- `--quality`: Initial JPEG compression quality (`1-100`)
- `--fps`: Target streaming framerate
- `--width, --height`: Frame resolution

#### Receiver Options
- `--server`: WebSocket server URL
- `--target-fps`: Target framerate for quality adjustment
- `--max-quality`: Maximum quality to request from sender
- `--save-frames`: Enable saving frames to disk
- `--save-dir`: Directory to save received frames

### Keyboard Controls (Receiver)
- `ESC`: Close the video stream window
- `s`: Save the current frame as a snapshot

### System Requirements
- Python 3.6+
- OpenCV
- Pillow
- websockets
- cryptography
- numpy

## Installation

To use the video streaming system, you'll need to install several Python packages:

```bash
# Install all required dependencies
pip install opencv-python pillow websockets cryptography numpy
