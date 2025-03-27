import sys
import asyncio
import json
import websockets
import time
import cv2
import numpy as np
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from collections import deque
import argparse
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description='Receive video stream over encrypted websocket')
parser.add_argument('--server', default="ws://heliwi.duckdns.org:9000", help='WebSocket server URL')
parser.add_argument('--target-fps', type=int, default=15, help='Target FPS for quality adjustment')
parser.add_argument('--max-quality', type=int, default=60, help='Maximum JPEG quality to request')
parser.add_argument('--save-frames', action='store_true', help='Save received frames to disk')
parser.add_argument('--save-dir', default='received_frames', help='Directory to save frames')
args = parser.parse_args()

SIGNALING_SERVER = args.server
wantedFramerate = args.target_fps
maxQuality = args.max_quality
SAVE_FRAMES = args.save_frames
SAVE_DIR = args.save_dir

# Create save directory if needed
if SAVE_FRAMES:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    SAVE_DIR = os.path.join(SAVE_DIR, f"stream_{timestamp}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Frames will be saved to: {SAVE_DIR}")

frame_times = deque(maxlen=100)  # Sliding window of timestamps
latency_times = deque(maxlen=30)  # Track message latency

TARGET_WIDTH, TARGET_HEIGHT = 640, 480
AES_KEY = b'C\x03\xb6\xd2\xc5\t.Brp\x1ce\x0e\xa4\xf6\x8b\xd2\xf6\xb0\x8a\x9c\xd5D\x1e\xf4\xeb\x1d\xe6\x0c\x1d\xff '

def decrypt_data(encrypted_base64):
    decrypt_start = time.time()
    encrypted_data = base64.b64decode(encrypted_base64)
    iv = encrypted_data[:16]
    encrypted_bytes = encrypted_data[16:]

    cipher = Cipher(algorithms.AES(AES_KEY), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    
    decrypted_padded = decryptor.update(encrypted_bytes) + decryptor.finalize()

    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    decrypted_bytes = unpadder.update(decrypted_padded) + unpadder.finalize()
    
    decrypt_time = (time.time() - decrypt_start) * 1000  # ms
    return decrypted_bytes, decrypt_time

async def receive_messages():
    quality = 50
    lastQuality = quality
    async with websockets.connect(SIGNALING_SERVER) as websocket:
        print(f"✅ Connected to Signaling Server: {SIGNALING_SERVER}")
        print(f"Target FPS: {wantedFramerate}, Max Quality: {maxQuality}")

        # Init FPS tracking
        message_count = 0
        last_time = time.time()
        last_executed_q = time.time()
        fps_display = 0
        frameCounter = 0
        total_data_received = 0
        source_type = "unknown"
        
        # Create window with resizable property
        cv2.namedWindow("Video Stream", cv2.WINDOW_NORMAL)

        while True:
            try:
                receive_start = time.time()
                message = await websocket.recv()
                message_json = json.loads(message)
                frameCounter += 1

                # Get source type if available
                if "source_type" in message_json:
                    source_type = message_json["source_type"]

                # Calculate message size and track total data
                message_size_kb = message_json.get("size_kb", 0)
                total_data_received += message_size_kb

                # ✅ Decrypt image
                decrypted_data, decrypt_time_ms = decrypt_data(message_json["data"])
                np_arr = np.frombuffer(decrypted_data, np.uint8)
                decode_start = time.time()
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                decode_time_ms = (time.time() - decode_start) * 1000

                # Save frame if enabled
                if SAVE_FRAMES and frame is not None and frameCounter % 10 == 0:  # Save every 10th frame
                    frame_filename = os.path.join(SAVE_DIR, f"frame_{frameCounter:06d}.jpg")
                    cv2.imwrite(frame_filename, frame)

                # ✅ FPS and latency calculation
                current_time = time.time()
                frame_times.append(current_time)
                receive_time = current_time - receive_start
                latency_times.append(receive_time * 1000)  # Store in ms

                # Remove old timestamps (> 1 second old)
                while frame_times and current_time - frame_times[0] > 1.0:
                    frame_times.popleft()

                fps_display = len(frame_times)  # Number of frames in the last second
                avg_latency = sum(latency_times) / max(len(latency_times), 1)
                
                # ✅ Automatic quality adjustment - increase
                if fps_display > wantedFramerate and frameCounter > 200:
                    if current_time - last_executed_q >= 0.2:
                        quality += 10
                        if quality > maxQuality:
                            quality = maxQuality
                        if quality != lastQuality:
                            print(f"📈 Increasing quality to {quality}")
                            await websocket.send(json.dumps({"quality": quality}))
                        last_executed_q = current_time
                        lastQuality = quality

                # ✅ Automatic quality adjustment - decrease
                if fps_display < wantedFramerate - 2 and frameCounter > 200:
                    if current_time - last_executed_q >= 0.2:
                        quality -= 10
                        if quality < 1:
                            quality = 1
                        if quality != lastQuality:
                            await websocket.send(json.dumps({"quality": quality}))
                            print(f"📉 Decreasing quality to {quality}")
                        last_executed_q = current_time
                        lastQuality = quality

                # ✅ Display info overlay
                if frame is not None:
                    # Get frame metadata
                    frame_num = message_json.get("frame_number", "N/A")
                    data_rate = round(total_data_received / (current_time - last_time), 2) if current_time > last_time else 0
                    
                    # Build multi-line text for comprehensive display
                    info_text = [
                        f"Source: {source_type.upper()}",
                        f"Time: {message_json['timestamp']}",
                        f"Resolution: {message_json['resolution']}",
                        f"Quality: {quality}% | Size: {round(message_json['size_kb'], 2)} KB",
                        f"Comp: {round(message_json['compression_time_ms'], 2)} ms",
                        f"Encrypt: {round(message_json['encryption_time_ms'], 2)} ms",
                        f"Decrypt: {round(decrypt_time_ms, 2)} ms",
                        f"FPS: {fps_display} | Frame: {frame_num}",
                        f"Data rate: {data_rate} KB/s"
                    ]
                    
                    # Draw semi-transparent background for text
                    h, w = frame.shape[:2]
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (5, 5), (400, 230), (0, 0, 0), -1)
                    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
                    
                    # Draw text lines
                    for i, line in enumerate(info_text):
                        y_pos = 30 + (i * 25)
                        # Choose text color - green for good FPS, red for bad
                        color = (0, 255, 0)  # Green by default
                        if i == 7 and fps_display < wantedFramerate:  # FPS line
                            color = (0, 165, 255)  # Orange
                        
                        cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # Show stream source more prominently
                    if source_type == "webcam":
                        cv2.putText(frame, "LIVE CAMERA", (w-220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    elif source_type == "file":
                        cv2.putText(frame, "VIDEO FILE", (w-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                    cv2.imshow("Video Stream", frame)
                    key = cv2.waitKey(1)
                    if key == 27:  # ESC key
                        break
                    elif key == ord('s'):  # Save current frame with 's' key
                        snapshot_dir = os.path.join(os.path.dirname(SAVE_DIR), "snapshots")
                        os.makedirs(snapshot_dir, exist_ok=True)
                        snapshot_file = os.path.join(snapshot_dir, f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
                        cv2.imwrite(snapshot_file, frame)
                        print(f"📸 Snapshot saved to {snapshot_file}")

            except websockets.exceptions.ConnectionClosed:
                print("🚫 Connection to server closed")
                break
            except json.JSONDecodeError:
                print("⚠️ Received invalid JSON message")
                continue
            except Exception as e:
                print(f"⚠️ Error: {e}")
                
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        asyncio.run(receive_messages())
    except KeyboardInterrupt:
        print("⏹️ Stopping...")
        cv2.destroyAllWindows()