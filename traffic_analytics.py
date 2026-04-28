from ultralytics import YOLO
import cv2
import math
import csv
import matplotlib.pyplot as plt
from collections import deque
import requests

import os
import requests

SNAP_DIR = "snapshots"
os.makedirs(SNAP_DIR, exist_ok=True)

def send_telegram_photo(image_path, caption=""):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

        response = requests.post(
            url,
            data={"chat_id": CHAT_ID, "caption": caption},
            files={"photo": open(image_path, "rb")},
            timeout=3
        )

        print("Photo sent:", response.status_code)

    except Exception as e:
        print("Photo alert failed:", e)

# -----------------------------
# Telegram Config (OPTIONAL)
# -----------------------------
BOT_TOKEN = "8659715037:AAHDgp7Nv1lVx3oFyJ_VHAKpknRSH_-lN0A"
CHAT_ID = "8585439164"

def send_telegram_alert(message):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

        requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": message
            },
            timeout=3
        )

    except Exception as e:
        print("Telegram alert failed:", e)

# -----------------------------
# Stable speed storage
# -----------------------------
speed_history_dict = {}

# -----------------------------
# 1️⃣ Load YOLO Model
# -----------------------------
model = YOLO("yolo11n.pt")

# -----------------------------
# 2️⃣ Load Video
# -----------------------------
video_path = "data/videos/IMG_4323.MOV"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS:", fps)

output_width = 640
output_height = 360

# -----------------------------
# 3️⃣ Output Video
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("traffic_output.mp4", fourcc, fps, (output_width, output_height))

# -----------------------------
# 4️⃣ CSV Files
# -----------------------------
csv_file = open("vehicle_speed_data.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame", "Vehicle_ID", "Speed_kmph"])

alert_file = open("alerts.csv", "w", newline="")
alert_writer = csv.writer(alert_file)
alert_writer.writerow(["Frame", "Alert"])

# -----------------------------
# 5️⃣ Variables
# -----------------------------
previous_positions = {}
pixel_to_meter = 0.05

congestion_threshold = 15
speed_limit = 60

frame_count = 0
frame_skip = 3

unique_ids = set()

# 🚨 Overspeed control
overspeed_triggered = {}
last_alert_frame = {}
ALERT_COOLDOWN = 120

# -----------------------------
# Stable Speed Function
# -----------------------------
def get_stable_speed(vehicle_id, current_speed):
    if vehicle_id not in speed_history_dict:
        speed_history_dict[vehicle_id] = deque(maxlen=10)

    speed_history_dict[vehicle_id].append(current_speed)
    return sum(speed_history_dict[vehicle_id]) / len(speed_history_dict[vehicle_id])

# -----------------------------
# Graph Data
# -----------------------------
frame_history = deque(maxlen=200)
vehicle_history = deque(maxlen=200)
avg_speed_history = deque(maxlen=200)



def update_graph():
    plt.clf()

    plt.subplot(2, 1, 1)
    plt.title("Vehicle Count Over Time")
    plt.plot(frame_history, vehicle_history)

    plt.subplot(2, 1, 2)
    plt.title("Average Speed")
    plt.plot(frame_history, avg_speed_history)

    plt.tight_layout()
    plt.savefig("traffic_graph.png")   # ✅ SAVE

# -----------------------------
# Alert Display
# -----------------------------
def draw_alert_box(frame, text, color=(0, 0, 255)):
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (w, 50), color, -1)
    cv2.putText(frame, text, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)

# -----------------------------
# -----------------------------
# MAIN LOOP (FIXED)
# -----------------------------
while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ No frame read. Video ended or path issue.")
        break
    else:
        print("✅ Frame read:", frame_count)

    frame_count += 1

    if frame_count % frame_skip != 0:
        continue

    frame = cv2.resize(frame, (output_width, output_height))

    results = model.track(frame, persist=True, verbose=False)

    vehicle_count_frame = 0
    alerts = []
    main_alert = None
    frame_speeds = []

    if results and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, obj_id in zip(boxes, ids):

            obj_id = int(obj_id)
            unique_ids.add(obj_id)
            vehicle_count_frame += 1

            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            speed_kmph = 0

            if obj_id in previous_positions:
                px, py = previous_positions[obj_id]
                distance_pixels = math.sqrt((cx - px)**2 + (cy - py)**2)
                speed_pixels_per_sec = distance_pixels * fps
                speed_mps = speed_pixels_per_sec * pixel_to_meter
                speed_kmph = speed_mps * 3.6

            stable_speed = get_stable_speed(obj_id, speed_kmph)

            previous_positions[obj_id] = (cx, cy)
            frame_speeds.append(stable_speed)

            csv_writer.writerow([frame_count, obj_id, round(stable_speed, 2)])

            # 🚨 OVERSPEED ALERT + PHOTO
            if stable_speed > speed_limit:

                if (obj_id not in overspeed_triggered) or \
                   (frame_count - last_alert_frame.get(obj_id, 0) > ALERT_COOLDOWN):

                    overspeed_triggered[obj_id] = True
                    last_alert_frame[obj_id] = frame_count

                    alert_msg = f"🚨 Vehicle {obj_id} Overspeeding ({stable_speed:.1f} km/h)"
                    alerts.append(alert_msg)
                    main_alert = alert_msg

                    send_telegram_alert(alert_msg)

                    # Draw warning
                    cv2.putText(frame, "OVERSPEED!",
                                (int(x1), int(y1) - 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 255), 2)

                    # 📸 SNAPSHOT
                    crop = frame[int(y1):int(y2), int(x1):int(x2)]

                    if crop.size != 0:
                        img_path = f"{SNAP_DIR}/vehicle_{obj_id}_{frame_count}.jpg"
                        cv2.imwrite(img_path, crop)
                        send_telegram_photo(img_path, alert_msg)

            # Draw box
            cv2.rectangle(frame,
                          (int(x1), int(y1)),
                          (int(x2), int(y2)),
                          (0, 255, 0), 2)

            cv2.putText(frame,
                        f"ID {obj_id} | {stable_speed:.1f} km/h",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2)

    # 🚨 Congestion Detection
    if vehicle_count_frame > congestion_threshold:
        status = "HIGH TRAFFIC"
        color = (0, 0, 255)

        if frame_count % 100 == 0:
            send_telegram_alert("⚠️ HIGH TRAFFIC CONGESTION")

    else:
        status = "NORMAL"
        color = (0, 255, 0)

    # Display
    cv2.putText(frame, f"Vehicles: {vehicle_count_frame}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.putText(frame, f"Traffic: {status}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if main_alert:
        draw_alert_box(frame, main_alert)

    out.write(frame)

    # -----------------------------
    # Display + Output (INSIDE LOOP)
    # -----------------------------
    cv2.putText(frame, f"Vehicles: {vehicle_count_frame}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.putText(frame, f"Total Vehicles: {len(unique_ids)}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(frame, f"Traffic: {status}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if main_alert:
        draw_alert_box(frame, main_alert)

    for alert in alerts:
        alert_writer.writerow([frame_count, alert])

    out.write(frame)

    if frame_count % 60 == 0:
        print("Processed frames:", frame_count)

        # Crop vehicle region
crop = frame[int(y1):int(y2), int(x1):int(x2)]

img_path = f"{SNAP_DIR}/vehicle_{obj_id}_{frame_count}.jpg"
cv2.imwrite(img_path, crop)

# Send image
send_telegram_photo(img_path, alert_msg)

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
out.release()
csv_file.close()
alert_file.close()
cv2.destroyAllWindows()

print("Processing Complete.")