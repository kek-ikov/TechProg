import albumentations as A
import cv2
import os

dataset_folder = r"C:\Users\kek13\Downloads\dataset_change"
output_folder = r"C:\Users\kek13\Downloads\dataset_aug"
classes = ['clubs', 'diamonds', 'hearts', 'spades']

augmentations = [
    A.Compose([
        A.HorizontalFlip(p=1.0),
        A.RandomBrightnessContrast(p=0.5),
    ]),
    A.Compose([
        A.Rotate(limit=30, p=1.0),
        A.GaussianBlur(blur_limit=3, p=0.5),
    ]),
    A.Compose([
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=1.0),
    ])
]

os.makedirs(output_folder, exist_ok=True)

num_augmentations = 3

for cls in classes:
    input_train_path = os.path.join(dataset_folder, cls, 'train')
    output_train_path = os.path.join(output_folder, cls, 'train')
    os.makedirs(output_train_path, exist_ok=True)

    for filename in os.listdir(input_train_path):
        input_file = os.path.join(input_train_path, filename)
        image = cv2.imread(input_file)
        if image is None:
            continue

        for i in range(num_augmentations):
            # Выбираем i-ую аугментацию из списка циклично
            transform = augmentations[i % len(augmentations)]
            augmented = transform(image=image)
            augmented_image = augmented['image']
            output_file = os.path.join(output_train_path, f'aug_{i+1}_{filename}')
            cv2.imwrite(output_file, augmented_image)

print("Аугментация завершена")
