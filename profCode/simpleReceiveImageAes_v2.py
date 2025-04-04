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

frame_times = deque(maxlen=100)  # Schuivend venster van tijdstempels

SIGNALING_SERVER = "ws://heliwi.duckdns.org:9000"

if len(sys.argv) > 1:
    SIGNALING_SERVER = sys.argv[1]

wantedFramerate = 15
maxQuality = 60



TARGET_WIDTH, TARGET_HEIGHT = 640, 480
AES_KEY = b'C\x03\xb6\xd2\xc5\t.Brp\x1ce\x0e\xa4\xf6\x8b\xd2\xf6\xb0\x8a\x9c\xd5D\x1e\xf4\xeb\x1d\xe6\x0c\x1d\xff '

def decrypt_data(encrypted_base64):
    encrypted_data = base64.b64decode(encrypted_base64)
    iv = encrypted_data[:16]
    encrypted_bytes = encrypted_data[16:]

    cipher = Cipher(algorithms.AES(AES_KEY), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    
    decrypted_padded = decryptor.update(encrypted_bytes) + decryptor.finalize()

    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    decrypted_bytes = unpadder.update(decrypted_padded) + unpadder.finalize()
    
    return decrypted_bytes

async def receive_messages():
    quality = 50
    async with websockets.connect(SIGNALING_SERVER) as websocket:
        print(f"✅ Verbonden met Signaling Server: {SIGNALING_SERVER}")

        # Init FPS tracking
        message_count = 0
        last_time = time.time()
        last_executed_q = time.time()
        fps_display = 0
        frameCounter = 0

        while True:
            try:
                message = await websocket.recv()
                message_json = json.loads(message)
                frameCounter +=1

                # ✅ Decrypt afbeelding
                decrypted_data = decrypt_data(message_json["data"])
                np_arr = np.frombuffer(decrypted_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                #if frame is not None:
                #    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

                # ✅ FPS berekening
                message_count += 1
                current_time = time.time()
                elapsed_time = current_time - last_time
                lastQuality = quality

                current_time = time.time()
                frame_times.append(current_time)

                # Verwijder oude timestamps (> 1 seconde oud)
                while frame_times and current_time - frame_times[0] > 1.0:
                    frame_times.popleft()

                fps_display = len(frame_times)  # Aantal frames in de laatste seconde
                # ✅ Automatisch kwaliteitsaanpassing sturen
                if (fps_display > wantedFramerate and frameCounter > 200): 
                    if current_time - last_executed_q >= 0.2:
                        quality += 10
                        if (quality>maxQuality):
                            quality = maxQuality
                        if (quality != lastQuality):
                            print(f"📉 verhoog kwaliteit naar {quality}")
                            await websocket.send(json.dumps({"quality": quality}))
                        last_executed_q = current_time
                        lastQuality = quality

                # ✅ Automatisch kwaliteitsaanpassing sturen
                if (fps_display < wantedFramerate - 2 and frameCounter > 200):
                    if current_time - last_executed_q >= 0.2:
                        quality -= 10
                        if (quality<1):
                            quality = 1
                        if (quality != lastQuality):
                            await websocket.send(json.dumps({"quality": quality}))
                            print(f"📉 verlaag kwaliteit naar {quality}")

                        last_executed_q = current_time
                        lastQuality = quality


                # ✅ Overlay info op beeld
                if frame is not None:
                    cv2.putText(frame, f"Time: {message_json['timestamp']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Resolution: {message_json['resolution']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Size: {round(message_json['size_kb'], 2)} KB", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Comp. Time({quality})%: {round(message_json['compression_time_ms'], 2)} ms", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Encryption: {round(message_json['encryption_time_ms'], 2)} ms", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"FPS: {fps_display}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    cv2.putText(frame, f"Framecoounter: {frameCounter}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                    cv2.imshow("Ontvangen Afbeelding", frame)
                    cv2.waitKey(1)

            except websockets.exceptions.ConnectionClosed:
                print("🚫 Verbinding met server gesloten")
                break

asyncio.run(receive_messages())
