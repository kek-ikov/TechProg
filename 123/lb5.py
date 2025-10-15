import cv2
import numpy as np
import os
from pathlib import Path
from skimage import morphology, segmentation, measure, filters

# ============ Настройки ============
input_image = r"C:\Users\kek13\Downloads\dataset_change\clubs\train\clubs_train_002.jpg"  # одно изображение
output_dir = r"C:\Users\kek13\Downloads\segmentation_results"    # папка результатов

# Подпапки
subdirs = {
    "threshold_fixed": "threshold_fixed",
    "threshold_otsu": "threshold_otsu",
    "kmeans": "kmeans",
    "watershed": "watershed",
    "comparison": "comparison"
}
for sd in subdirs.values():
    os.makedirs(os.path.join(output_dir, sd), exist_ok=True)



# ============ Вспомогательные ============
def imread_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Не удалось открыть {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def to_gray(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

def save(path, img_rgb_or_gray):
    # принимает RGB или Gray, сохраняет в BGR/Gray для корректного цвета
    if img_rgb_or_gray.ndim == 3:
        img_bgr = cv2.cvtColor(img_rgb_or_gray, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img_bgr)
    else:
        cv2.imwrite(path, img_rgb_or_gray)

def postprocess_binary(mask, min_area=64):
    m = morphology.remove_small_holes(mask.astype(bool), area_threshold=128)
    m = morphology.remove_small_objects(m, min_size=min_area)
    return (m.astype(np.uint8)) * 255

# ============ Загрузка ============
img = imread_rgb(input_image)
gray = to_gray(img)
name = Path(input_image).stem

# ============ 1) Пороговая сегментация ============
# Фиксированный порог 127
_, bin_fixed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
bin_fixed_pp = postprocess_binary(bin_fixed > 0)
cv2.imwrite(os.path.join(output_dir, subdirs["threshold_fixed"], f"{name}_thr127.png"), bin_fixed)
cv2.imwrite(os.path.join(output_dir, subdirs["threshold_fixed"], f"{name}_thr127_clean.png"), bin_fixed_pp)

# Otsu
otsu_val, bin_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
bin_otsu_pp = postprocess_binary(bin_otsu > 0)
cv2.imwrite(os.path.join(output_dir, subdirs["threshold_otsu"], f"{name}_otsu_{int(otsu_val)}.png"), bin_otsu)
cv2.imwrite(os.path.join(output_dir, subdirs["threshold_otsu"], f"{name}_otsu_clean.png"), bin_otsu_pp)

# ============ 2) K-means (K=2,3,4) ============
def kmeans_segment(img_rgb, k=2, random_state=42):
    h, w, _ = img_rgb.shape
    Z = img_rgb.reshape((-1, 3)).astype(np.float32)
    # OpenCV kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ret, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = centers.astype(np.uint8)
    seg = centers[labels.flatten()].reshape((h, w, 3))
    # Индексация кластеров по яркости центров
    order = np.argsort(centers.mean(axis=1))
    ordered_labels = order[labels.flatten()].reshape(h, w).astype(np.uint8)
    return seg, ordered_labels, centers

for K in [2, 3, 4]:
    segK, ordK, centersK = kmeans_segment(img, k=K)
    save(os.path.join(output_dir, subdirs["kmeans"], f"{name}_k{K}_seg.png"), segK)
    # Визуальная маска по яркости кластера: показываем метки как градации
    ord_vis = (255 * (ordK / ordK.max())).astype(np.uint8) if ordK.max() > 0 else ordK
    cv2.imwrite(os.path.join(output_dir, subdirs["kmeans"], f"{name}_k{K}_labels.png"), ord_vis)
    # Для K=2 можно считать самый светлый кластер как объект
    if K == 2:
        obj_mask = (ordK == ordK.max()).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(output_dir, subdirs["kmeans"], f"{name}_k2_obj_mask.png"), obj_mask)

# ============ 3) Watershed ============
def watershed_segment(img_rgb):
    g = to_gray(img_rgb)
    blur = cv2.GaussianBlur(g, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(th, kernel, iterations=3) > 0

    dist = cv2.distanceTransform(th, distanceType=cv2.DIST_L2, maskSize=5)
    sure_fg = dist > (0.5 * dist.max())

    unknown = (sure_bg.astype(np.uint8) - sure_fg.astype(np.uint8)) > 0

    markers = measure.label(sure_fg)
    markers = markers + 1
    markers[unknown] = 0

    gradient = filters.sobel(g)
    labels_ws = segmentation.watershed(gradient, markers, mask=None)
    # Контуры
    boundaries = segmentation.find_boundaries(labels_ws, mode='outer')
    overlay = img_rgb.copy()
    overlay[boundaries] = [255, 0, 0]
    return labels_ws.astype(np.int32), overlay, gradient, dist

labels_ws, overlay_ws, grad, dist = watershed_segment(img)
# Сохранения
save(os.path.join(output_dir, subdirs["watershed"], f"{name}_overlay.png"), overlay_ws)

# Нормируем grad/dist для просмотра
grad_norm = (255 * (grad / (grad.max() + 1e-6))).astype(np.uint8)
dist_norm = (255 * (dist / (dist.max() + 1e-6))).astype(np.uint8)
cv2.imwrite(os.path.join(output_dir, subdirs["watershed"], f"{name}_sobel_grad.png"), grad_norm)
cv2.imwrite(os.path.join(output_dir, subdirs["watershed"], f"{name}_distance.png"), dist_norm)

# ============ 4) Быстрое сравнение (коллаж 2x2) ============
def stack2x2(a, b, c, d):
    # ожидает одиночные каналы или 3-канальные изображения в формате RGB/Gray
    def to_bgr(x):
        if x.ndim == 2:
            return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
        else:
            return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    A = to_bgr(a); B = to_bgr(b); C = to_bgr(c); D = to_bgr(d)
    h = max(A.shape[0], B.shape[0], C.shape[0], D.shape[0])
    w = max(A.shape[1], B.shape[1], C.shape[1], D.shape[1])
    def pad(img):
        top = h - img.shape[0]
        left = w - img.shape[1]
        return cv2.copyMakeBorder(img, 0, top, 0, left, cv2.BORDER_CONSTANT, value=(0,0,0))
    A = pad(A); B = pad(B); C = pad(C); D = pad(D)
    top_row = np.hstack([A, B])
    bot_row = np.hstack([C, D])
    return np.vstack([top_row, bot_row])

comparison = stack2x2(img, bin_fixed, bin_otsu, overlay_ws)
cv2.imwrite(os.path.join(output_dir, subdirs["comparison"], f"{name}_comparison.png"), comparison)

print("Сегментация завершена! Результаты сохранены в:", output_dir)
