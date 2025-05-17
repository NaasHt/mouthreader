import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import f1_score

input_folder = './testMix(full)'
output_base = './sorted_output'
class_names = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
non_classified_folder = os.path.join(output_base, 'non_classified')

model = tf.keras.models.load_model('face_model.h5')

os.makedirs(output_base, exist_ok=True)
for class_name in class_names:
    os.makedirs(os.path.join(output_base, class_name), exist_ok=True)
os.makedirs(non_classified_folder, exist_ok=True)

results = []
used_filenames = set()

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    if filename in used_filenames:
        continue

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"[!] Error in finding img: {filename}")
        continue

    try:
        img_resized = cv2.resize(img, (48, 48)) 

        img_norm = img_resized.astype('float32') / 255.0
        img_input = np.expand_dims(img_norm, axis=-1) 
        img_input = np.expand_dims(img_input, axis=0) 

        preds = model.predict(img_input, verbose=0)
        max_confidence = np.max(preds)

        if max_confidence < 0.5:
            save_path = os.path.join(non_classified_folder, filename)
            cv2.imwrite(save_path, img_resized)
            print(f"[{filename}] -> non_classified (low confidence: {max_confidence:.2f})")
            continue

        pred_class_idx = np.argmax(preds)
        pred_class = class_names[pred_class_idx]

        save_path = os.path.join(output_base, pred_class, filename)
        cv2.imwrite(save_path, img_resized)

        results.append({'filename': filename, 'predicted_class': pred_class})
        used_filenames.add(filename)
        print(f"[{filename}] -> {pred_class} (confidence: {max_confidence:.2f})")

    except Exception as e:
        print(f"[!] error {filename}: {e}")
        save_path = os.path.join(non_classified_folder, filename)
        cv2.imwrite(save_path, img_resized)
        print(f"[{filename}] saved to non_classified")

print("\nStatistic:")
for class_name in class_names + ['non_classified']:
    class_folder = os.path.join(output_base, class_name)
    count = len([
        f for f in os.listdir(class_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    print(f"{class_name}: {count} imgs")
