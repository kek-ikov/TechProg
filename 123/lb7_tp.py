import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
from pathlib import Path

# 1. Подготовка датасета с вашей структурой
base_dir = r'C:\Users\kek13\Downloads\dataset_change'  # Укажите путь к папке с hearts, diamonds, spades, clubs

# Параметры
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 10

# Создаем списки путей для обучения и тестирования
train_images = []
train_labels = []
test_images = []
test_labels = []

class_names = ['hearts', 'diamonds', 'spades', 'clubs']
class_indices = {name: idx for idx, name in enumerate(class_names)}

# Собираем данные из вашей структуры
for class_name in class_names:
    class_dir = os.path.join(base_dir, class_name)

    # Train images
    train_dir = os.path.join(class_dir, 'train')
    if os.path.exists(train_dir):
        for img_file in os.listdir(train_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                train_images.append(os.path.join(train_dir, img_file))
                train_labels.append(class_indices[class_name])

    # Test images
    test_dir = os.path.join(class_dir, 'test')
    if os.path.exists(test_dir):
        for img_file in os.listdir(test_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append(os.path.join(test_dir, img_file))
                test_labels.append(class_indices[class_name])

print(f"Found {len(train_images)} training images")
print(f"Found {len(test_images)} test images")


# Создаем кастомный генератор данных
class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size=32, img_size=(128, 128), shuffle=True, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

        # Аугментация для тренировочных данных
        if self.augment:
            self.datagen = ImageDataGenerator(
                rescale=1. / 255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2
            )
        else:
            self.datagen = ImageDataGenerator(rescale=1. / 255)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]

        batch_images = []
        for path in batch_paths:
            img = tf.keras.preprocessing.image.load_img(path, target_size=self.img_size)
            img = tf.keras.preprocessing.image.img_to_array(img)
            batch_images.append(img)

        batch_images = np.array(batch_images)
        batch_labels = tf.keras.utils.to_categorical(batch_labels, num_classes=NUM_CLASSES)

        # Применяем аугментацию только если это тренировочные данные
        if self.augment:
            batch_images = next(self.datagen.flow(batch_images, batch_size=len(batch_images), shuffle=False))
        else:
            batch_images = self.datagen.standardize(batch_images)

        return batch_images, batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            indices = np.arange(len(self.image_paths))
            np.random.shuffle(indices)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]


# Создаем генераторы
train_generator = CustomDataGenerator(train_images, train_labels, batch_size=BATCH_SIZE,
                                      img_size=IMG_SIZE, shuffle=True, augment=True)
test_generator = CustomDataGenerator(test_images, test_labels, batch_size=BATCH_SIZE,
                                     img_size=IMG_SIZE, shuffle=False, augment=False)

# 2. Построение CNN модели
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 3. Обучение модели
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    verbose=1
)

# 4. Построение графиков обучения
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 5. Оценка модели
test_loss, test_acc = model.evaluate(test_generator)
print(f'\nTest accuracy: {test_acc:.4f}')
print(f'Test loss: {test_loss:.4f}')

# Прогноз для тестовой выборки
test_predictions = model.predict(test_generator)
predicted_classes = np.argmax(test_predictions, axis=1)
true_classes = test_labels

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification Report
print('\nClassification Report:')
print(classification_report(true_classes, predicted_classes, target_names=class_names))

# Сравнение с классическими методами
print(f"\nCNN Results:")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Дополнительная информация
print(f"\nDataset Information:")
print(f"Training samples: {len(train_images)}")
print(f"Test samples: {len(test_images)}")
print(f"Number of classes: {NUM_CLASSES}")
print(f"Classes: {class_names}")