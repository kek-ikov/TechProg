import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import Adam

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Parameters
IMG_SIZE = (224, 224)  # ResNet50 input size
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 4
CLASSES = ['hearts', 'diamonds', 'spades', 'clubs']

# Assume dataset root path; adjust as needed
DATASET_ROOT = r'C:\Users\kek13\Downloads\dataset_change'  # e.g., './card_suits/'

# Step 1: Prepare dataset
# We'll use train, valid (as validation), test folders.
# Augmentation will be applied on-the-fly for training.

# Data generators for custom CNN (with augmentation and preprocessing)
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize to [0,1]
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest',
    validation_split=0.0  # No split here since we have separate folders
)

val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load train data
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_ROOT, 'train'),  # Assuming train is under root with class subdirs
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES,
    shuffle=True
)

# Load validation data (using 'valid' folder)
val_generator = val_datagen.flow_from_directory(
    os.path.join(DATASET_ROOT, 'valid'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES,
    shuffle=False
)

# Load test data
test_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_ROOT, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES,
    shuffle=False
)

# If 'aug' folders are separate augmented data, you can combine them into train by copying files,
# but here we apply augmentation on-the-fly. For 'change', assume it's part of train if needed.

# Step 2: Build and train custom CNN
def create_custom_cnn(input_shape, num_classes):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

custom_model = create_custom_cnn((IMG_SIZE[0], IMG_SIZE[1], 3), NUM_CLASSES)
custom_model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

# Train custom model
history_custom = custom_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE
)

# Save history
custom_history = history_custom.history

# Step 3: Evaluate custom model
test_loss, test_acc = custom_model.evaluate(test_generator)
print(f'Custom CNN Test Accuracy: {test_acc:.4f}')  # [web:11]

y_pred_custom = custom_model.predict(test_generator)
y_pred_classes_custom = np.argmax(y_pred_custom, axis=1)
y_true = test_generator.classes

# Confusion matrix
cm_custom = confusion_matrix(y_true, y_pred_classes_custom)
disp_custom = ConfusionMatrixDisplay(confusion_matrix=cm_custom, display_labels=CLASSES)
disp_custom.plot(cmap=plt.cm.Blues)
plt.show()

# Metrics
report_custom = classification_report(y_true, y_pred_classes_custom, target_names=CLASSES)
print('Custom CNN Classification Report:\n', report_custom)  # [web:14]

# Identify error-prone classes: look at diagonal vs off-diagonal in cm_custom

# Step 4: Transfer Learning with ResNet50
# Freeze base model, add custom head
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base_model.trainable = False  # Freeze

# For transfer learning, use preprocess_input in generators
train_datagen_tl = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # ResNet specific preprocessing
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    zoom_range=0.2
)

val_datagen_tl = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen_tl = ImageDataGenerator(preprocessing_function=preprocess_input)

# Reload generators with TL preprocessing
train_generator_tl = train_datagen_tl.flow_from_directory(
    os.path.join(DATASET_ROOT, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES,
    shuffle=True
)

val_generator_tl = val_datagen_tl.flow_from_directory(
    os.path.join(DATASET_ROOT, 'valid'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES,
    shuffle=False
)

test_generator_tl = test_datagen_tl.flow_from_directory(
    os.path.join(DATASET_ROOT, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES,
    shuffle=False
)

# Build TL model
inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
tl_model = keras.Model(inputs, outputs)

tl_model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

# Train TL model
history_tl = tl_model.fit(
    train_generator_tl,
    steps_per_epoch=train_generator_tl.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_generator_tl,
    validation_steps=val_generator_tl.samples // BATCH_SIZE
)

# Evaluate TL model
test_loss_tl, test_acc_tl = tl_model.evaluate(test_generator_tl)
print(f'ResNet50 TL Test Accuracy: {test_acc_tl:.4f}')  # [web:17]

y_pred_tl = tl_model.predict(test_generator_tl)
y_pred_classes_tl = np.argmax(y_pred_tl, axis=1)

# Confusion matrix for TL
cm_tl = confusion_matrix(y_true, y_pred_classes_tl)
disp_tl = ConfusionMatrixDisplay(confusion_matrix=cm_tl, display_labels=CLASSES)
disp_tl.plot(cmap=plt.cm.Blues)
plt.show()

# Metrics for TL
report_tl = classification_report(y_true, y_pred_classes_tl, target_names=CLASSES)
print('ResNet50 TL Classification Report:\n', report_tl)

# Comparison: Compare accuracies, F1-scores from reports. Typically TL outperforms custom on small datasets.  # [web:7][web:16]

# Plot training history for both
def plot_history(history, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['accuracy'], label='Train Acc')
    ax1.plot(history['val_accuracy'], label='Val Acc')
    ax1.set_title(f'{title} Accuracy')
    ax1.legend()
    ax2.plot(history['loss'], label='Train Loss')
    ax2.plot(history['val_loss'], label='Val Loss')
    ax2.set_title(f'{title} Loss')
    ax2.legend()
    plt.show()

plot_history(custom_history, 'Custom CNN')
plot_history(history_tl.history, 'ResNet50 TL')

# Optional: Fine-tune by unfreezing some layers
# base_model.trainable = True
# for layer in base_model.layers[:-10]:  # Unfreeze last 10 layers
#     layer.trainable = False
# tl_model.compile(optimizer=Adam(learning_rate=1e-5),  # Lower LR
#                  loss='categorical_crossentropy',
#                  metrics=['accuracy'])
# history_fine = tl_model.fit(...)  # Retrain
