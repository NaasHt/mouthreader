import os
import cv2
import json
import mediapipe as mp
import numpy as np
from pathlib import Path

input_dir = './CK+48'
output_dir = './cropped_mouth'
landmark_dir = './mouth_landmarks_json'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(landmark_dir, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1)

MOUTH_LANDMARKS = list(set([
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 78,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80,
    81, 82, 13, 312, 311, 310, 415, 308
]))

def crop_mouth(img_path, emotion_label):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[!] Failed to read: {img_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        print(f"[!] No face detected: {img_path}")
        return

    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = img.shape

    mouth_points = []
    for idx in MOUTH_LANDMARKS:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        mouth_points.append((x, y))

    mouth_np = np.array(mouth_points)
    x, y, w_box, h_box = cv2.boundingRect(mouth_np)

    pad_x = 10
    pad_y = 10
    x1 = max(x - pad_x, 0)
    y1 = max(y - pad_y, 0)
    x2 = min(x + w_box + pad_x, w)
    y2 = min(y + h_box + pad_y, h)

    mouth_img = img[y1:y2, x1:x2]

    filename = Path(img_path).stem
    out_path = os.path.join(output_dir, emotion_label)
    os.makedirs(out_path, exist_ok=True)
    out_file = os.path.join(out_path, f"{filename}_mouth.jpg")
    cv2.imwrite(out_file, mouth_img)

    landmark_path = os.path.join(landmark_dir, emotion_label)
    os.makedirs(landmark_path, exist_ok=True)
    landmark_file = os.path.join(landmark_path, f"{filename}_landmarks.json")
    with open(landmark_file, 'w') as f:
        json.dump({
            'filename': f"{filename}_mouth.jpg",
            'emotion': emotion_label,
            'mouth_landmarks': mouth_points
        }, f)

    print(f"Saved: {emotion_label}/{filename}_mouth.jpg")

for emotion in os.listdir(input_dir):
    emotion_path = os.path.join(input_dir, emotion)
    if not os.path.isdir(emotion_path):
        continue

    for img_name in os.listdir(emotion_path):
        if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(emotion_path, img_name)
            crop_mouth(img_path, emotion)
