import cv2
import numpy as np
import os
from pathlib import Path

# Настройки
input_dir = r"C:\Users\kek13\Downloads\dataset_change"  # Папка с исходными изображениями
output_dir = r"C:\Users\kek13\Downloads\feature_analysis"  # Папка для результатов

# Создаем подпапки для каждого типа обработки
os.makedirs(os.path.join(output_dir, "canny_edges"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "harris_corners"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "shi_tomasi_corners"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "orb_keypoints"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "feature_matching"), exist_ok=True)

# Получаем список изображений
image_paths = []
for ext in ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"]:
    image_paths.extend(Path(input_dir).glob(f"**/{ext}"))

# Обрабатываем каждое изображение
for img_path in image_paths:
    # Загрузка изображения
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    name = img_path.stem

    # 1. Выделение контуров с помощью алгоритма Canny
    edges_low = cv2.Canny(gray, 50, 150)  # Низкие пороги
    edges_medium = cv2.Canny(gray, 100, 200)  # Средние пороги
    edges_high = cv2.Canny(gray, 150, 250)  # Высокие пороги

    # Сохранение результатов контурного анализа
    cv2.imwrite(os.path.join(output_dir, "canny_edges", f"{name}_low.jpg"), edges_low)
    cv2.imwrite(os.path.join(output_dir, "canny_edges", f"{name}_medium.jpg"), edges_medium)
    cv2.imwrite(os.path.join(output_dir, "canny_edges", f"{name}_high.jpg"), edges_high)

    # 2. Детекция углов: метод Харриса
    harris_img = img.copy()
    gray_float = np.float32(gray)
    harris_response = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
    harris_response = cv2.dilate(harris_response, None)
    harris_img[harris_response > 0.01 * harris_response.max()] = [0, 0, 255]  # Отмечаем углы красным
    cv2.imwrite(os.path.join(output_dir, "harris_corners", f"{name}.jpg"), harris_img)

    # Детекция углов: метод Shi-Tomasi
    shi_tomasi_img = img.copy()
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=50, qualityLevel=0.01, minDistance=10)
    if corners is not None:
        # Заменяем устаревший np.int0 на astype(int)
        corners = np.array(corners, dtype=int)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(shi_tomasi_img, (x, y), 3, (0, 255, 0), -1)  # Отмечаем углы зеленым
    cv2.imwrite(os.path.join(output_dir, "shi_tomasi_corners", f"{name}.jpg"), shi_tomasi_img)

    # 3. Извлечение локальных признаков с помощью ORB
    orb = cv2.ORB_create(nfeatures=100)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Визуализация ключевых точек
    orb_img = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)
    cv2.imwrite(os.path.join(output_dir, "orb_keypoints", f"{name}.jpg"), orb_img)

# 4. Сопоставление признаков на двух похожих изображениях
if len(image_paths) >= 2:
    img1 = cv2.imread(str(image_paths[0]))
    img2 = cv2.imread(str(image_paths[1]))

    if img1 is not None and img2 is not None:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Извлекаем ORB признаки для обоих изображений
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
            # Сопоставляем признаки
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)

            # Сортируем matches по расстоянию
            matches = sorted(matches, key=lambda x: x.distance)

            # Рисуем первые 30 matches
            match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None, flags=2)

            # Сохраняем результат
            cv2.imwrite(os.path.join(output_dir, "feature_matching", "feature_matching.jpg"), match_img)

        else:
            print("Не удалось извлечь дескрипторы для сопоставления")
    else:
        print("Не удалось загрузить изображения для сопоставления")

print("Анализ признаков завершен! Результаты сохранены в папке:", output_dir)