import sys
import asyncio
import json
import websockets
import time
import cv2
import base64
import io
import os
import argparse
from PIL import Image
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

# Default settings
SIGNALING_SERVER = "ws://heliwi.duckdns.org:9000"
JPEG_QUALITY = 50
TARGET_FPS = 15  # Target framerate for sending

# Parse command line arguments
parser = argparse.ArgumentParser(description='Stream video/camera over encrypted websocket')
parser.add_argument('--source', default="0", help='Video source (0 for webcam, path for video file)')
parser.add_argument('--server', default=SIGNALING_SERVER, help='WebSocket server URL')
parser.add_argument('--quality', type=int, default=JPEG_QUALITY, help='Initial JPEG quality (1-100)')
parser.add_argument('--fps', type=int, default=TARGET_FPS, help='Target streaming FPS')
parser.add_argument('--width', type=int, default=640, help='Frame width')
parser.add_argument('--height', type=int, default=480, help='Frame height')
args = parser.parse_args()

SIGNALING_SERVER = args.server
JPEG_QUALITY = args.quality
TARGET_FPS = args.fps
WIDTH = args.width
HEIGHT = args.height

# AES-256 key (must be 32 bytes)
AES_KEY = b'C\x03\xb6\xd2\xc5\t.Brp\x1ce\x0e\xa4\xf6\x8b\xd2\xf6\xb0\x8a\x9c\xd5D\x1e\xf4\xeb\x1d\xe6\x0c\x1d\xff '

def encrypt_data(plain_text):
    """Encrypts data with AES-256 CBC and base64-encodes the output."""
    encrypt_start_time = time.time()

    iv = os.urandom(16)  # AES requires a 16-byte IV
    cipher = Cipher(algorithms.AES(AES_KEY), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()

    # Apply PKCS7 padding
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(plain_text) + padder.finalize()

    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

    encrypt_end_time = time.time()
    encryption_time_ms = (encrypt_end_time - encrypt_start_time) * 1000

    return base64.b64encode(iv + encrypted_data).decode('utf-8'), encryption_time_ms

def open_video_source(source):
    # Check if source is a webcam index (0, 1, etc.) or a file path
    try:
        source_index = int(source)
        print(f"Opening webcam at index {source_index}")
        cap = cv2.VideoCapture(source_index)
        if not cap.isOpened():
            print(f"⚠️ Warning: Could not open webcam at index {source_index}")
            return None, "webcam", 0
        return cap, "webcam", 0
    except ValueError:
        # Not an integer, treat as a file path
        if os.path.isfile(source):
            print(f"Opening video file: {source}")
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"❌ Error: Could not open video file: {source}")
                return None, "file", 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return cap, "file", total_frames
        else:
            print(f"❌ Error: Source is not a valid webcam index or file path: {source}")
            return None, "unknown", 0

async def send_frames(websocket):
    global JPEG_QUALITY
    
    # Open the video source (webcam or file)
    cap, source_type, total_frames = open_video_source(args.source)
    if cap is None:
        print("Failed to open video source. Exiting...")
        return
    
    # Set resolution for webcam
    if source_type == "webcam":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0 and source_type == "file":
        original_fps = 30
        print(f"⚠️ Warning: Invalid FPS in video file, assuming {original_fps} FPS")
    
    print(f"Video source: {width}x{height} @ {original_fps} FPS")
    if source_type == "file":
        print(f"Total frames: {total_frames}")
    
    # Calculate frame delay to match target FPS
    frame_delay = 1.0 / TARGET_FPS
    
    frame_number = 0
    start_time = time.time()
    
    try:
        while True:
            loop_start = time.time()
            ret, frame = cap.read()
            
            if not ret:
                if source_type == "webcam":
                    print("⚠️ Failed to grab frame from webcam. Trying again...")
                    await asyncio.sleep(0.1)
                    continue
                else:  # File mode - loop the video
                    print("End of video file reached, restarting...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_number = 0
                    continue
            
            # Resize frame if needed
            current_h, current_w = frame.shape[:2]
            if current_w != width or current_h != height:
                frame = cv2.resize(frame, (width, height))
            
            # Process and send the frame
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            compressed_image_io = io.BytesIO()
            compression_start_time = time.time()
            pil_image.save(compressed_image_io, format="JPEG", quality=JPEG_QUALITY)
            compression_end_time = time.time()

            compressed_bytes = compressed_image_io.getvalue()
            compressed_size_kb = len(compressed_bytes) / 1024
            compression_time_ms = (compression_end_time - compression_start_time) * 1000

            encrypted_data, encryption_time_ms = encrypt_data(compressed_bytes)

            message = {
                "type": "video_frame",
                "data": encrypted_data,
                "timestamp": timestamp,
                "resolution": f"{width}x{height}",
                "size_kb": round(compressed_size_kb, 2),
                "compression_time_ms": round(compression_time_ms, 2),
                "encryption_time_ms": round(encryption_time_ms, 2),
                "frame_number": frame_number,
                "source_type": source_type
            }

            await websocket.send(json.dumps(message))
            frame_number += 1
            
            # Print status periodically
            if frame_number % 30 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_number / max(elapsed, 0.001)
                if source_type == "file" and total_frames > 0:
                    progress = (frame_number % total_frames) / total_frames * 100
                    print(f"Progress: {progress:.1f}%, FPS: {current_fps:.1f}, Quality: {JPEG_QUALITY}")
                else:
                    print(f"Frames sent: {frame_number}, FPS: {current_fps:.1f}, Quality: {JPEG_QUALITY}")
            
            # Control frame rate
            process_time = time.time() - loop_start
            if process_time < frame_delay:
                await asyncio.sleep(frame_delay - process_time)
    finally:
        cap.release()

async def receive_quality_updates(websocket):
    global JPEG_QUALITY
    while True:
        try:
            message = await websocket.recv()
            message_json = json.loads(message)
            if 'quality' in message_json:
                new_quality = message_json['quality']
                if 1 <= new_quality <= 100:
                    JPEG_QUALITY = new_quality
                    print(f"✅ Quality adjusted to: {JPEG_QUALITY}")
        except websockets.exceptions.ConnectionClosed:
            print("🚫 Connection to server closed")
            break
        except Exception as e:
            print(f"⚠️ Error receiving message: {e}")

async def main():
    print(f"Signaling Server: {SIGNALING_SERVER}")
    print(f"Initial Quality: {JPEG_QUALITY}")
    print(f"Target FPS: {TARGET_FPS}")
    print(f"Source: {args.source}")
    
    try:
        async with websockets.connect(SIGNALING_SERVER) as websocket:
            print(f"✅ Connected to Signaling Server: {SIGNALING_SERVER}")
            await asyncio.gather(
                send_frames(websocket),
                receive_quality_updates(websocket)
            )
    except Exception as e:
        print(f"❌ Connection error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("⏹️ Stopping...")
    except Exception as e:
        print(f"❌ Error: {e}")