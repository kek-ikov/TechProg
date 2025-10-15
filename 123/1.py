import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from PIL import Image, ImageDraw
import pandas as pd
import cv2


def load_dataset():
    """Загружает dataset из папок"""
    base_path = r'C:\Users\kek13\Downloads\dataset_change'
    classes = ['hearts', 'diamonds', 'spades', 'clubs']
    subs = ['train', 'valid', 'test']

    images = []
    labels = []

    class_to_idx = {weapon: i for i, weapon in enumerate(classes)}

    for weapon in classes:
        for subset in subs:
            folder_path = os.path.join(base_path, weapon, subset)

            if not os.path.exists(folder_path):
                print(f"Предупреждение: Папка {folder_path} не найдена")
                continue

            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(folder_path, img_file)

                    try:
                        img = Image.open(img_path)
                        img = img.convert('RGB')
                        img = img.resize((128, 128))
                        img_array = np.array(img) / 255.0

                        images.append(img_array)
                        labels.append(class_to_idx[weapon])

                    except Exception as e:
                        print(f"Ошибка загрузки {img_path}: {e}")

    images = np.array(images)
    labels = np.array(labels)

    labels_one_hot = keras.utils.to_categorical(labels, num_classes=len(classes))

    print(f"Загружено изображений: {len(images)}")
    print(f"Классы: {classes}")
    print(f"Размер изображений: {images.shape}")

    return images, labels_one_hot, labels, classes


def create_custom_cnn(input_shape, num_classes):
    """Создает собственную CNN модель"""
    model = keras.Sequential([
        # Первый сверточный блок
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        # Второй сверточный блок
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        # Третий сверточный блок
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        # Полносвязные слои
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def create_resnet_model(input_shape, num_classes):
    """Создает модель на основе ResNet50 с transfer learning"""
    base_model = keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Замораживаем базовые слои
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def create_mobilenet_model(input_shape, num_classes):
    """Создает модель на основе MobileNetV2 с transfer learning"""
    base_model = keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def plot_training_history(history, model_name):
    """Строит графики обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'{model_name} - Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title(f'{model_name} - Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """Строит матрицу ошибок"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

    return cm


def evaluate_model(model, X_test, y_test, class_names, model_name):
    """Полная оценка модели"""
    # Базовые метрики
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Предсказания
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Матрица ошибок
    cm = plot_confusion_matrix(y_true_classes, y_pred_classes, class_names, model_name)

    # Classification report
    report = classification_report(y_true_classes, y_pred_classes,
                                   target_names=class_names, output_dict=True)

    print(f"\n{model_name} Results:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Loss: {test_loss:.4f}")

    # Анализ ошибок
    error_analysis = {}
    for i, class_name in enumerate(class_names):
        class_indices = np.where(y_true_classes == i)[0]
        if len(class_indices) > 0:
            class_accuracy = np.mean(y_pred_classes[class_indices] == i)
            error_analysis[class_name] = {
                'accuracy': class_accuracy,
                'samples': len(class_indices),
                'errors': np.sum(y_pred_classes[class_indices] != i)
            }

    return {
        'accuracy': test_accuracy,
        'loss': test_loss,
        'confusion_matrix': cm,
        'classification_report': report,
        'error_analysis': error_analysis
    }


def compare_models(results_dict, class_names):
    """Сравнивает результаты всех моделей"""
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 60)

    # Создаем таблицу сравнения
    comparison_data = []
    for model_name, results in results_dict.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': results['accuracy'],
            'Loss': results['loss']
        })

    df_comparison = pd.DataFrame(comparison_data)
    print("\nСравнение метрик:")
    print(df_comparison.to_string(index=False))

    # Визуализация сравнения accuracy
    plt.figure(figsize=(10, 6))
    models = list(results_dict.keys())
    accuracies = [results_dict[model]['accuracy'] for model in models]

    bars = plt.bar(models, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.title('Сравнение Accuracy моделей')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    # Добавляем значения на столбцы
    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{accuracy:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('models_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Анализ ошибок по классам
    print("\n" + "=" * 60)
    print("АНАЛИЗ ОШИБОК ПО КЛАССАМ")
    print("=" * 60)

    for model_name, results in results_dict.items():
        print(f"\n{model_name}:")
        error_analysis = results['error_analysis']
        for class_name, metrics in error_analysis.items():
            print(f"  {class_name}: accuracy={metrics['accuracy']:.3f}, "
                  f"errors={metrics['errors']}/{metrics['samples']}")


def train_and_evaluate_models(X_train, y_train, X_test, y_test, class_names):
    """Обучает и оценивает все модели"""
    input_shape = (128, 128, 3)
    num_classes = len(class_names)

    results = {}

    # 1. Собственная CNN
    print("Создание и обучение собственной CNN...")
    custom_cnn = create_custom_cnn(input_shape, num_classes)
    custom_cnn.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history_custom = custom_cnn.fit(
        X_train, y_train,
        epochs=15,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=1
    )

    plot_training_history(history_custom, "Custom_CNN")
    results['Custom_CNN'] = evaluate_model(custom_cnn, X_test, y_test, class_names, "Custom_CNN")

    # 2. ResNet Transfer Learning
    print("\nСоздание и обучение ResNet50...")
    resnet_model = create_resnet_model(input_shape, num_classes)
    resnet_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history_resnet = resnet_model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=1
    )

    plot_training_history(history_resnet, "ResNet50")
    results['ResNet50'] = evaluate_model(resnet_model, X_test, y_test, class_names, "ResNet50")

    # 3. MobileNet Transfer Learning
    print("\nСоздание и обучение MobileNetV2...")
    mobilenet_model = create_mobilenet_model(input_shape, num_classes)
    mobilenet_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history_mobilenet = mobilenet_model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=1
    )

    plot_training_history(history_mobilenet, "MobileNetV2")
    results['MobileNetV2'] = evaluate_model(mobilenet_model, X_test, y_test, class_names, "MobileNetV2")

    return results, custom_cnn, resnet_model, mobilenet_model


def detect_and_draw_bbox(image_path, model, model_name, class_names):
    """Обнаруживает объект и рисует bounding box на изображении"""
    # Загружаем и обрабатываем изображение
    img = Image.open(image_path)
    original_img = img.copy()
    img = img.convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Получаем предсказание
    prediction = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]

    # Создаем копию изображения для рисования
    draw_img = original_img.copy()
    draw = ImageDraw.Draw(draw_img)

    # Получаем размеры изображения
    width, height = draw_img.size

    # Рисуем bounding box (занимает большую часть изображения)
    margin = 10
    bbox = [margin, margin, width - margin, height - margin]

    # Выбираем цвет в зависимости от уверенности модели
    if confidence > 0.8:
        color = "green"
    elif confidence > 0.5:
        color = "orange"
    else:
        color = "red"

    # Рисуем прямоугольник
    draw.rectangle(bbox, outline=color, width=3)

    # Добавляем текст с предсказанием
    label = f"{class_names[predicted_class]}: {confidence:.2f}"
    text_bbox = draw.textbbox((0, 0), label)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Рисуем фон для текста
    text_background = [margin, margin, margin + text_width + 10, margin + text_height + 10]
    draw.rectangle(text_background, fill="black")

    # Добавляем текст
    draw.text((margin + 5, margin + 5), label, fill="white")

    # Добавляем название модели
    model_text = f"Model: {model_name}"
    model_bbox = draw.textbbox((0, 0), model_text)
    model_text_width = model_bbox[2] - model_bbox[0]

    model_text_background = [width - margin - model_text_width - 10, margin,
                             width - margin, margin + text_height + 10]
    draw.rectangle(model_text_background, fill="blue")
    draw.text((width - margin - model_text_width - 5, margin + 5), model_text, fill="white")

    return draw_img, class_names[predicted_class], confidence


def visualize_model_predictions(image_path, models_dict, class_names):
    """Визуализирует предсказания всех моделей на одном изображении"""
    print(f"\nАнализ изображения: {os.path.basename(image_path)}")
    print("=" * 50)

    # Создаем фигуру для отображения результатов
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    # Отображаем оригинальное изображение
    original_img = Image.open(image_path)
    axes[0].imshow(original_img)
    axes[0].set_title("Оригинальное изображение")
    axes[0].axis('off')

    # Обрабатываем изображение каждой моделью
    results = []
    for i, (model_name, model) in enumerate(models_dict.items(), 1):
        result_img, predicted_class, confidence = detect_and_draw_bbox(
            image_path, model, model_name, class_names
        )

        axes[i].imshow(result_img)
        axes[i].set_title(f"{model_name}\nПредсказание: {predicted_class} ({confidence:.2f})")
        axes[i].axis('off')

        results.append({
            'model': model_name,
            'predicted_class': predicted_class,
            'confidence': confidence
        })

    plt.tight_layout()
    plt.savefig('model_predictions_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Выводим результаты в консоль
    print("\nРезультаты предсказаний:")
    for result in results:
        print(f"{result['model']}: {result['predicted_class']} (уверенность: {result['confidence']:.3f})")

    return results


def main():
    print("Сравнение собственной CNN и Transfer Learning моделей")
    print("=" * 70)

    # Загружаем данные
    X, y, y_labels, class_names = load_dataset()

    if len(X) == 0:
        print("Ошибка: Не загружено ни одного изображения!")
        return

    # Разделяем на тренировочную и тестовую выборки
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y_labels
    )

    print(f"\nРазделение данных:")
    print(f"Training set: {X_train.shape[0]} изображений")
    print(f"Test set: {X_test.shape[0]} изображений")

    # Обучаем и оцениваем все модели
    results, custom_cnn, resnet_model, mobilenet_model = train_and_evaluate_models(
        X_train, y_train, X_test, y_test, class_names
    )

    # Сравниваем модели
    compare_models(results, class_names)

    # Сохраняем лучшую модель
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    print(f"\nЛучшая модель: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.4f})")

    # Демонстрация работы моделей на примере одного изображения
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ РАБОТЫ МОДЕЛЕЙ")
    print("=" * 70)

    # Создаем словарь моделей
    models_dict = {
        'Custom_CNN': custom_cnn,
        'ResNet50': resnet_model,
        'MobileNetV2': mobilenet_model
    }

    # Находим тестовое изображение для демонстрации
    base_path = r'C:\Users\kek13\Downloads\dataset_change'
    demo_image_path = None

    # Ищем первое подходящее изображение в тестовой выборке
    for class_name in class_names:
        test_folder = os.path.join(base_path, class_name, 'test')
        if os.path.exists(test_folder):
            for img_file in os.listdir(test_folder):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    demo_image_path = os.path.join(test_folder, img_file)
                    break
            if demo_image_path:
                break

    if demo_image_path and os.path.exists(demo_image_path):
        # Визуализируем предсказания всех моделей
        prediction_results = visualize_model_predictions(
            demo_image_path, models_dict, class_names
        )
    else:
        print("Не удалось найти тестовое изображение для демонстрации")

        # Альтернатива: используем первое изображение из тестовой выборки
        if len(X_test) > 0:
            print("Используем изображение из тестовой выборки...")
            # Сохраняем временное изображение для демонстрации
            demo_img = (X_test[0] * 255).astype(np.uint8)
            demo_image = Image.fromarray(demo_img)
            temp_path = "temp_demo_image.jpg"
            demo_image.save(temp_path)

            # Визуализируем предсказания всех моделей
            prediction_results = visualize_model_predictions(
                temp_path, models_dict, class_names
            )

            # Удаляем временный файл
            os.remove(temp_path)


if __name__ == "__main__":
    # Установите необходимые библиотеки:
    # pip install tensorflow matplotlib scikit-learn seaborn pillow pandas opencv-python

    main()