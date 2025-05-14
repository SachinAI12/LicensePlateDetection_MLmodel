#Installing Tensorflow Library
!pip install tensorflow

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab.patches import cv2_imshow



# Step 1: Data Preparation
def load_and_preprocess_images(folder, label, img_size=(128, 128)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, img_size)
            img = img / 255.0  # Normalize the image
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)




def prepare_data(positive_folder, negative_folder, test_size=0.2):
    positive_images, positive_labels = load_and_preprocess_images(positive_folder, 1)
    negative_images, negative_labels = load_and_preprocess_images(negative_folder, 0)
    X = np.concatenate((positive_images, negative_images), axis=0)
    y = np.concatenate((positive_labels, negative_labels), axis=0)
    return train_test_split(X, y, test_size=test_size, random_state=42)


# Step 2: Model Development
def create_cnn_model(input_shape=(128, 128, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 3: Data Augmentation
def create_data_generator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True):
    return ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        horizontal_flip=horizontal_flip
    )

# Step 4: Model Training
def train_model(model, train_data, val_data, datagen, batch_size=32, epochs=10):
    X_train, y_train = train_data
    X_val, y_val = val_data
    history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                        validation_data=(X_val, y_val),
                        epochs=epochs)
    return history



# Step 5: Plotting Training History
def plot_training_history(history, val_data, model):
    X_val, y_val = val_data

    # Plot Accuracy over epochs
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Calculate precision, recall, F1 score over epochs
    precision_scores = []
    recall_scores = []
    f1_scores = []
    for epoch in range(len(history.history['accuracy'])):
        model.fit(X_val, y_val, epochs=1, verbose=0)
        y_pred = model.predict(X_val)
        y_pred = (y_pred > 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    # Plot precision, recall, and F1 score over epochs
    plt.subplot(1, 2, 2)
    plt.plot(precision_scores, label='Precision')
    plt.plot(recall_scores, label='Recall')
    plt.plot(f1_scores, label='F1 Score')
    plt.title('Precision, Recall, and F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    plt.show()

# Step 6: Model Evaluation
def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_pred = (y_pred > 0.5).astype(int)
    report = classification_report(y_val, y_pred)
    print(report)
    return report


# Step 7: Model Testing
def test_model_on_image(model, image_path, img_size=(128, 128)):
    img = cv2.imread(image_path)

      # Check if the image was loaded successfully
    if img is None:
        print(f"Error: Unable to load image at {image_path}. Please check the file path.")
        return

    img_resized = cv2.resize(img, img_size)
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    prediction = model.predict(img_resized)
    prediction = prediction[0][0]

   # Display the image using cv2_imshow
    cv2_imshow(cv2.imread(image_path))

    if prediction > 0.8:
        print("License Plate Detected")
    else:
        print("No License Plate Detected")


# Main Workflow
if __name__ == "__main__":
    # Define paths to the datasets
    positive_folder = '/content/PositiveImages_Dataset/images/'
    negative_folder = '/content/NegativeImages_Dataset/images/'

    # Step 1: Data Preparation
    X_train, X_val, y_train, y_val = prepare_data(positive_folder, negative_folder)

    # Step 2: Model Development
    model = create_cnn_model()

    # Step 3: Data Augmentation
    datagen = create_data_generator()

    # Step 4: Model Training
    history = train_model(model, (X_train, y_train), (X_val, y_val), datagen)

    # Step 5: Plot Training History
    plot_training_history(history, (X_val, y_val), model)

    # Step 6: Model Evaluation
    evaluate_model(model, X_val, y_val)

    # Step 7: Save the Model
    model.save('license_plate_detection_model.h5'),




 