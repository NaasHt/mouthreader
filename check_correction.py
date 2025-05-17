import os
from sklearn.metrics import classification_report, f1_score
import numpy as np

# === Папка с предсказанными папками
sorted_dir = 'sorted_output_cut_MIX'
class_names = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
non_classified_folder = os.path.join(sorted_dir, 'non_classified')

# === Задание ground truth: диапазоны imgX
# ground_truth_ranges = {
#     'anger': range(1, 4),
#     'disgust': range(4, 7),
#     'fear': range(7, 10),
#     'happy': range(10, 13),
#     'sadness': range(13, 16),
#     'surprise': range(16, 19),
# }
ground_truth_ranges = {
    'anger': range(1, 11),
    'disgust': range(11, 21),
    'fear': range(21, 31),
    'happy': range(30, 41),
    'sadness': range(40, 51),
    'surprise': range(50, 60),
}
# === Генерация словаря filename → true_class
ground_truth = {}
for emotion, r in ground_truth_ranges.items():
    for i in r:
        ground_truth[f"im{i}"] = emotion

# === Сбор предсказаний и настоящих меток
y_true = []
y_pred = []

for pred_emotion in class_names:
    folder = os.path.join(sorted_dir, pred_emotion)
    if not os.path.exists(folder):
        print(f"[!] Folder '{folder}' does not exist.")
        continue

    for fname in os.listdir(folder):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        base_name = os.path.splitext(fname)[0]
        base_name = base_name.lower().replace('_mouth', '')
        if base_name in ground_truth:
            y_true.append(ground_truth[base_name])
            y_pred.append(pred_emotion)
        else:
            print(f"[!] Unknown image: {fname}")

# === Подсчёт изображений в non_classified
non_classified_count = 0
if os.path.exists(non_classified_folder):
    non_classified_count = len([
        f for f in os.listdir(non_classified_folder)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ])

# === Отчёт по метрикам
print("\nClassification report:")
report = classification_report(y_true, y_pred, labels=class_names, output_dict=True)
for cls in class_names:
    print(f"{cls:10} "
          f"precision: {report[cls]['precision']:.2f}  "
          f"recall: {report[cls]['recall']:.2f}  "
          f"f1-score: {report[cls]['f1-score']:.2f}  "
          f"support: {report[cls]['support']}")

# === Общий F1-score (микро-среднее)
overall_f1 = f1_score(y_true, y_pred, average='micro')
print(f"\nOverall F1-score (micro average): {overall_f1:.2f}")

# === Информация по non_classified
print(f"\nImages in 'non_classified': {non_classified_count}")
