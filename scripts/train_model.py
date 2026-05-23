import os
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report
import numpy as np

# Add project root to path for config imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import BASE_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS, MODELS_DIR

# -------------------------------------------------
# Paths & Hyper-parameters
# -------------------------------------------------
import argparse
parser = argparse.ArgumentParser(description="Train deepfake detection model")
parser.add_argument("--dataset", type=str, default=os.path.join(BASE_DIR, "dataset_split"),
                    help="Path to the dataset split directory")
args = parser.parse_args()
DATASET_SPLIT = args.dataset

# -------------------------------------------------
# Data Augmentation
# -------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# -------------------------------------------------
# Data Generators
# -------------------------------------------------
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_SPLIT, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True
)

val_generator = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_SPLIT, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False          # keep order for evaluation
)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_SPLIT, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False          # keep order for evaluation
)

# -------------------------------------------------
# Model Definition
# -------------------------------------------------
base_model = EfficientNetB0(weights="imagenet", include_top=False,
                            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=x)

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

# -------------------------------------------------
# Training
# -------------------------------------------------
model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# -------------------------------------------------
# Evaluation on Test Set (loss + accuracy)
# -------------------------------------------------
test_loss, test_acc = model.evaluate(test_generator, verbose=0)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# -------------------------------------------------
# ----------  PRECISION / RECALL / F1  ----------
# -------------------------------------------------
# 1. Get true labels
y_true = test_generator.classes                 # 0/1 array (same order as generator)

# 2. Predict probabilities
y_pred_prob = model.predict(test_generator, verbose=0).ravel()

# 3. Convert probabilities → binary predictions (threshold = 0.5)
y_pred = (y_pred_prob >= 0.5).astype(int)

# 4. Classification report (precision, recall, f1 per class)
report = classification_report(y_true, y_pred,
                               target_names=test_generator.class_indices.keys(),
                               digits=4)
print("\n=== Classification Report (Test) ===")
print(report)

# -------------------------------------------------
# Optional: Same metrics on the validation set
# -------------------------------------------------
val_pred_prob = model.predict(val_generator, verbose=0).ravel()
val_pred = (val_pred_prob >= 0.5).astype(int)
val_report = classification_report(val_generator.classes, val_pred,
                                   target_names=val_generator.class_indices.keys(),
                                   digits=4)
print("\n=== Classification Report (Validation) ===")
print(val_report)

# -------------------------------------------------
# Save Model
# -------------------------------------------------
save_path = os.path.join(MODELS_DIR, "deepfake_effnet.h5")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
model.save(save_path)
print(f"Model saved to: {save_path}")