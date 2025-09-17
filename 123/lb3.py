import os
import cv2
import numpy as np
from pathlib import Path

# Пути к папкам
input_dir = r"C:\Users\kek13\Downloads\dataset_change\hearts\train"  # Папка с исходными изображениями
output_base_dir = r"C:\Users\kek13\Downloads\dataset_change\hearts\change"  # Базовая папка для результатов

# Создаем папки для каждого метода обработки
methods = [
    "orig", "gray", "blur3", "blur7", "sharp",
    "bin127", "bin_otsu"
]

for method in methods:
    os.makedirs(os.path.join(output_base_dir, method), exist_ok=True)

# Получаем список всех изображений
image_paths = []
for ext in ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"]:
    image_paths.extend(Path(input_dir).glob(f"**/{ext}"))


# Обрабатываем каждое изображение
for img_path in image_paths:
    try:
        # Читаем изображение
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Не удалось прочитать: {img_path}")
            continue

        # Получаем имя файла без расширения
        name = img_path.stem

        # 1) Оригинал (BGR)
        cv2.imwrite(os.path.join(output_base_dir, "orig", f"{name}.jpg"), img)

        # 2) Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(output_base_dir, "gray", f"{name}.jpg"), gray)

        # 3) Фильтрация: Gaussian Blur 3x3 и 7x7
        blur3 = cv2.GaussianBlur(gray, (3, 3), 0)
        cv2.imwrite(os.path.join(output_base_dir, "blur3", f"{name}.jpg"), blur3)

        blur7 = cv2.GaussianBlur(gray, (7, 7), 0)
        cv2.imwrite(os.path.join(output_base_dir, "blur7", f"{name}.jpg"), blur7)

        # 4) Резкость (ядро 3x3)
        kernel_sharp = np.array([[0, -1, 0],
                                 [-1, 5, -1],
                                 [0, -1, 0]], dtype=np.float32)
        sharp = cv2.filter2D(gray, -1, kernel_sharp)
        cv2.imwrite(os.path.join(output_base_dir, "sharp", f"{name}.jpg"), sharp)

        # 5) Бинаризация: фиксированный порог и Otsu
        _, bin127 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(output_base_dir, "bin127", f"{name}.jpg"), bin127)

        otsu_t, bin_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(os.path.join(output_base_dir, "bin_otsu", f"{name}.jpg"), bin_otsu)

        # Создаем сводное изображение для сравнения
        tmp_paths = [
            os.path.join(output_base_dir, "orig", f"{name}.jpg"),
            os.path.join(output_base_dir, "gray", f"{name}.jpg"),
            os.path.join(output_base_dir, "blur3", f"{name}.jpg"),
            os.path.join(output_base_dir, "blur7", f"{name}.jpg"),
            os.path.join(output_base_dir, "sharp", f"{name}.jpg"),
            os.path.join(output_base_dir, "bin127", f"{name}.jpg"),
            os.path.join(output_base_dir, "bin_otsu", f"{name}.jpg"),
        ]

    except Exception as e:
        print(f"Ошибка при обработке {img_path}: {str(e)}")

print("Обработка всех изображений завершена")
