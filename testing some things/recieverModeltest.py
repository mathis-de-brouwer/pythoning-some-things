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
import tensorflow as tf
from tensorflow import keras

# Parse command line arguments
parser = argparse.ArgumentParser(description='Receive video stream and perform segmentation')
parser.add_argument('--server', default="ws://heliwi.duckdns.org:9000", help='WebSocket server URL')
parser.add_argument('--target-fps', type=int, default=15, help='Target FPS for quality adjustment')
parser.add_argument('--max-quality', type=int, default=60, help='Maximum JPEG quality to request')
parser.add_argument('--save-frames', action='store_true', help='Save received frames to disk')
parser.add_argument('--save-dir', default='received_frames', help='Directory to save frames')
parser.add_argument('--model-path', default='../model/model_final.keras', help='Path to Keras segmentation model')
parser.add_argument('--no-segmentation', action='store_true', help='Disable segmentation (for testing)')
args = parser.parse_args()

SIGNALING_SERVER = args.server
wantedFramerate = args.target_fps
maxQuality = args.max_quality
SAVE_FRAMES = args.save_frames
SAVE_DIR = args.save_dir
MODEL_PATH = args.model_path
DISABLE_SEGMENTATION = args.no_segmentation

# Create save directories
if SAVE_FRAMES:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    SAVE_DIR = os.path.join(SAVE_DIR, f"stream_{timestamp}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    # Create subdirectories for original and segmented frames
    ORIGINAL_DIR = os.path.join(SAVE_DIR, "original")
    SEGMENTED_DIR = os.path.join(SAVE_DIR, "segmented")
    os.makedirs(ORIGINAL_DIR, exist_ok=True)
    os.makedirs(SEGMENTED_DIR, exist_ok=True)
    print(f"Frames will be saved to: {SAVE_DIR}")

# Performance tracking
frame_times = deque(maxlen=100)  # Sliding window of timestamps
latency_times = deque(maxlen=30)  # Track message latency
segmentation_times = deque(maxlen=30)  # Track segmentation times

TARGET_WIDTH, TARGET_HEIGHT = 640, 480
AES_KEY = b'C\x03\xb6\xd2\xc5\t.Brp\x1ce\x0e\xa4\xf6\x8b\xd2\xf6\xb0\x8a\x9c\xd5D\x1e\xf4\xeb\x1d\xe6\x0c\x1d\xff '

# Load the segmentation model
def load_model(model_path):
    if DISABLE_SEGMENTATION:
        print("⚠️ Segmentation disabled")
        return None
        
    print(f"Loading segmentation model from: {model_path}")
    try:
        model = keras.models.load_model(model_path)
        # Warm up the model with a dummy prediction
        dummy_input = np.zeros((1, TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.float32)
        _ = model.predict(dummy_input)
        print("✅ Model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# Perform segmentation on a frame
def segment_frame(model, frame):
    if model is None or DISABLE_SEGMENTATION:
        return frame, 0  # Return original frame if model not loaded or segmentation disabled
    
    start_time = time.time()
    
    # Preprocess the frame
    input_tensor = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
    input_tensor = np.expand_dims(input_tensor, axis=0) / 255.0  # Normalize to [0,1]
    
    # Run inference
    segmentation_mask = model.predict(input_tensor)[0]  # Get the first prediction
    
    # Post-process the mask
    if segmentation_mask.shape[-1] > 1:  # Multi-class segmentation
        mask = np.argmax(segmentation_mask, axis=-1)
        # Create a colormap for visualization
        colormap = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for class_idx in range(segmentation_mask.shape[-1]):
            color = np.random.randint(0, 255, size=3)
            colormap[mask == class_idx] = color
        
        # Overlay the mask on the original image
        alpha = 0.5
        segmented_frame = cv2.addWeighted(frame, 1-alpha, cv2.resize(colormap, (frame.shape[1], frame.shape[0])), alpha, 0)
    else:  # Binary segmentation
        mask = (segmentation_mask > 0.5).astype(np.uint8) * 255
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_rgb = cv2.resize(mask_rgb, (frame.shape[1], frame.shape[0]))
        
        # Create a green overlay for the mask
        overlay = np.zeros_like(frame)
        overlay[:,:,1] = mask_rgb[:,:,0]  # Green channel
        
        # Blend the original image with the overlay
        alpha = 0.5
        segmented_frame = cv2.addWeighted(frame, 1.0, overlay, alpha, 0)
    
    segmentation_time = (time.time() - start_time) * 1000  # ms
    return segmented_frame, segmentation_time

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
    # Load segmentation model
    model = load_model(MODEL_PATH)
    
    quality = 50
    lastQuality = quality
    async with websockets.connect(SIGNALING_SERVER) as websocket:
        print(f"✅ Connected to Signaling Server: {SIGNALING_SERVER}")
        print(f"Target FPS: {wantedFramerate}, Max Quality: {maxQuality}")

        # Init tracking
        message_count = 0
        last_time = time.time()
        last_executed_q = time.time()
        fps_display = 0
        frameCounter = 0
        total_data_received = 0
        source_type = "unknown"
        
        # Create windows with resizable property
        cv2.namedWindow("Original Stream", cv2.WINDOW_NORMAL)
        if not DISABLE_SEGMENTATION:
            cv2.namedWindow("Segmented Stream", cv2.WINDOW_NORMAL)

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

                if frame is not None:
                    # Perform segmentation
                    segmented_frame, segmentation_time_ms = segment_frame(model, frame)
                    segmentation_times.append(segmentation_time_ms)
                    avg_segmentation_time = sum(segmentation_times) / len(segmentation_times)

                    # Save frames if enabled
                    if SAVE_FRAMES and frameCounter % 10 == 0:  # Save every 10th frame
                        # Save original frame
                        original_file = os.path.join(ORIGINAL_DIR, f"frame_{frameCounter:06d}.jpg")
                        cv2.imwrite(original_file, frame)
                        
                        # Save segmented frame
                        if not DISABLE_SEGMENTATION:
                            segmented_file = os.path.join(SEGMENTED_DIR, f"frame_{frameCounter:06d}.jpg")
                            cv2.imwrite(segmented_file, segmented_frame)

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

                    # ✅ Display info overlay for original frame
                    frame_num = message_json.get("frame_number", "N/A")
                    data_rate = round(total_data_received / (current_time - last_time), 2) if current_time > last_time else 0
                    
                    # Build multi-line text for original frame
                    info_text = [
                        f"Source: {source_type.upper()}",
                        f"Time: {message_json['timestamp']}",
                        f"Resolution: {message_json['resolution']}",
                        f"Quality: {quality}% | Size: {round(message_json['size_kb'], 2)} KB",
                        f"Comp: {round(message_json['compression_time_ms'], 2)} ms",
                        f"Encrypt: {round(message_json['encryption_time_ms'], 2)} ms",
                        f"Decrypt: {round(decrypt_time_ms, 2)} ms",
                        f"FPS: {fps_display} | Frame: {frame_num}",
                        f"Segmentation: {round(avg_segmentation_time, 2)} ms",
                        f"Data rate: {data_rate} KB/s"
                    ]
                    
                    # Draw info on original frame
                    h, w = frame.shape[:2]
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (5, 5), (400, 255), (0, 0, 0), -1)
                    frame_with_info = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
                    
                    # Draw text lines
                    for i, line in enumerate(info_text):
                        y_pos = 30 + (i * 25)
                        color = (0, 255, 0)  # Green by default
                        if i == 7 and fps_display < wantedFramerate:  # FPS line
                            color = (0, 165, 255)  # Orange
                        
                        cv2.putText(frame_with_info, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # Show source indicator
                    if source_type == "webcam":
                        cv2.putText(frame_with_info, "LIVE CAMERA", (w-220, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    elif source_type == "file":
                        cv2.putText(frame_with_info, "VIDEO FILE", (w-200, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                    # Display frames
                    cv2.imshow("Original Stream", frame_with_info)
                    if not DISABLE_SEGMENTATION:
                        # Add title to segmented frame
                        cv2.putText(segmented_frame, "SEGMENTATION", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                        cv2.imshow("Segmented Stream", segmented_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1)
                    if key == 27:  # ESC key
                        break
                    elif key == ord('s'):  # Save snapshot
                        snapshot_dir = os.path.join(os.path.dirname(SAVE_DIR), "snapshots")
                        os.makedirs(snapshot_dir, exist_ok=True)
                        timestamp = time.strftime('%Y%m%d_%H%M%S')
                        
                        # Save original snapshot
                        orig_file = os.path.join(snapshot_dir, f"original_{timestamp}.jpg")
                        cv2.imwrite(orig_file, frame)
                        
                        # Save segmented snapshot if available
                        if not DISABLE_SEGMENTATION:
                            seg_file = os.path.join(snapshot_dir, f"segmented_{timestamp}.jpg")
                            cv2.imwrite(seg_file, segmented_frame)
                            
                        print(f"📸 Snapshots saved to {snapshot_dir}")

            except websockets.exceptions.ConnectionClosed:
                print("🚫 Connection to server closed")
                break
            except json.JSONDecodeError:
                print("⚠️ Received invalid JSON message")
                continue
            except Exception as e:
                print(f"⚠️ Error: {e}")
                import traceback
                traceback.print_exc()
                
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        asyncio.run(receive_messages())
    except KeyboardInterrupt:
        print("⏹️ Stopping...")
        cv2.destroyAllWindows()