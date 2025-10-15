import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ========== 1. Параметры HOG ==========
# Убедитесь, что используется правильное написание параметра для вашей версии scikit-image
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'transform_sqrt': False
    # Параметр 'visualize' удален из словаря
}


# ========== 2. Исправленная функция извлечения признаков ==========
def extract_hog_features(images, **hog_params):
    """
    Извлекает HOG-признаки для всех изображений
    """
    features = []
    hog_images = []

    for image in images:
        # Параметр 'visualize' передается только здесь, как аргумент функции
        fd, hog_image = hog(image, **hog_params, visualize=True)  # Американское написание 'visualize'
        # Для старых версий scikit-image может потребоваться британское написание 'visualise'
        # fd, hog_image = hog(image, **hog_params, visualise=True)
        features.append(fd)
        hog_images.append(hog_image)

    return np.array(features), np.array(hog_images)


# ========== 3. Загрузка данных (остается без изменений) ==========
def load_images_from_folders(base_path, classes, subfolders, img_size=(128, 128)):
    images = []
    labels = []

    for class_idx, class_name in enumerate(classes):
        for subfolder in subfolders:
            folder_path = os.path.join(base_path, class_name, subfolder)

            if not os.path.exists(folder_path):
                print(f"Предупреждение: Папка {folder_path} не существует")
                continue

            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(folder_path, filename)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, img_size)
                    images.append(gray)
                    labels.append(class_idx)

    return np.array(images), np.array(labels)


# ========== 4. Основной поток выполнения ==========
DATASET_PATH = r"C:\Users\kek13\Downloads\dataset_change"
CLASSES = ['hearts', 'diamonds', 'spades', 'clubs']
SUBFOLDERS = ['change', 'valid', 'aug', 'test', 'train']

print("Загрузка изображений...")
images, labels = load_images_from_folders(DATASET_PATH, CLASSES, SUBFOLDERS)
print(f"Загружено {len(images)} изображений")

print("Извлечение HOG-признаков...")
X, hog_imgs = extract_hog_features(images, **HOG_PARAMS)  # Передаем параметры без 'visualize'
y = labels

# ... остальная часть кода (разделение данных, обучение моделей, оценка) остается без изменений

print(f"Размерность матрицы признаков: {X.shape}")

# ========== 4. Разделение на обучающую и тестовую выборки ==========
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, range(len(y)), test_size=0.3, random_state=42, stratify=y
)

print(f"Обучающая выборка: {X_train.shape[0]} изображений")
print(f"Тестовая выборка: {X_test.shape[0]} изображений")

# ========== 5. Обучение классификаторов ==========
print("Обучение моделей...")

# SVM классификатор
svm_clf = SVC(kernel='linear', C=1.0, random_state=42)
svm_clf.fit(X_train, y_train)

# k-NN классификатор
knn_clf = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
knn_clf.fit(X_train, y_train)

# ========== 6. Предсказание и оценка качества ==========
# Предсказания
y_pred_svm = svm_clf.predict(X_test)
y_pred_knn = knn_clf.predict(X_test)

# Точность классификации
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print(f"\nРезультаты классификации:")
print(f"SVM Accuracy: {accuracy_svm:.4f} ({accuracy_svm * 100:.2f}%)")
print(f"k-NN Accuracy: {accuracy_knn:.4f} ({accuracy_knn * 100:.2f}%)")


# ========== 7. Визуализация результатов ==========
def plot_confusion_matrix(y_true, y_pred, classes, model_name, ax):
    """
    Строит и отображает матрицу ошибок
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'Confusion Matrix - {model_name}')


def plot_hog_examples(images, hog_images, labels, class_names, indices, title):
    """
    Визуализирует оригинальные изображения и их HOG-представление
    """
    fig, axes = plt.subplots(2, len(indices), figsize=(15, 6))
    if len(indices) == 1:
        axes = axes.reshape(2, 1)

    for i, idx in enumerate(indices):
        # Оригинальное изображение
        axes[0, i].imshow(images[idx], cmap='gray')
        axes[0, i].set_title(f'Original: {class_names[labels[idx]]}')
        axes[0, i].axis('off')

        # HOG-представление
        axes[1, i].imshow(hog_images[idx], cmap='gray')
        axes[1, i].set_title('HOG Representation')
        axes[1, i].axis('off')

    plt.suptitle(title)
    plt.tight_layout()


# Создание фигуры для визуализации
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Матрицы ошибок
plot_confusion_matrix(y_test, y_pred_svm, CLASSES, 'SVM', ax1)
plot_confusion_matrix(y_test, y_pred_knn, CLASSES, 'k-NN', ax2)

# Примеры HOG для тестовой выборки
test_indices = idx_test[:4]  # Первые 4 тестовых изображения
plot_hog_examples(images, hog_imgs, labels, CLASSES, test_indices, "HOG Examples from Test Set")

plt.tight_layout()
plt.show()

# ========== 8. Детальный отчет ==========
print("\n" + "=" * 50)
print("ДЕТАЛЬНЫЙ ОТЧЕТ")
print("=" * 50)

# Детализация по классам для SVM
print(f"\nДетализация для SVM (общая точность: {accuracy_svm:.4f}):")
svm_cm = confusion_matrix(y_test, y_pred_svm)
for i, class_name in enumerate(CLASSES):
    correct = svm_cm[i, i]
    total = np.sum(svm_cm[i, :])
    class_accuracy = correct / total if total > 0 else 0
    print(f"  {class_name}: {correct}/{total} ({class_accuracy:.4f})")

# Детализация по классам для k-NN
print(f"\nДетализация для k-NN (общая точность: {accuracy_knn:.4f}):")
knn_cm = confusion_matrix(y_test, y_pred_knn)
for i, class_name in enumerate(CLASSES):
    correct = knn_cm[i, i]
    total = np.sum(knn_cm[i, :])
    class_accuracy = correct / total if total > 0 else 0
    print(f"  {class_name}: {correct}/{total} ({class_accuracy:.4f})")

# ========== 9. Сравнительный анализ ==========
print(f"\nСРАВНИТЕЛЬНЫЙ АНАЛИЗ:")
print(f"Разница в точности: {(accuracy_svm - accuracy_knn):.4f}")
if accuracy_svm > accuracy_knn:
    print("SVM показал лучшую точность на данном датасете")
elif accuracy_svm < accuracy_knn:
    print("k-NN показал лучшую точность на данном датасете")
else:
    print("Оба классификатора показали одинаковую точность")