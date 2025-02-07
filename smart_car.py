# Import required libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import gdown
import os
import zipfile

# Define model creation function
def create_model():
    # Use MobileNetV2 as base model (smaller and faster than other models)
    base_model = MobileNetV2(weights='imagenet',
                            include_top=False,
                            input_shape=(224, 224, 3))

    # Freeze the base model layers
    base_model.trainable = False

    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(3, activation='softmax')(x)  # 3 classes: left, forward, right

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile model
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

    return model

# Download and prepare dataset
def prepare_dataset():
    # Download Comma.ai dataset (small subset)
    # Note: This is just an example. You might want to use a different dataset
    url = 'https://drive.google.com/uc?id=1Ue4XohCOV5YXy57S_5tDfCVqzLr101M7'
    output = 'driving_dataset.zip'
    gdown.download(url, output, quiet=False)

    # Extract dataset
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall('driving_data')

    # Create directories for our simplified classes
    os.makedirs('processed_data/train/left', exist_ok=True)
    os.makedirs('processed_data/train/forward', exist_ok=True)
    os.makedirs('processed_data/train/right', exist_ok=True)
    os.makedirs('processed_data/val/left', exist_ok=True)
    os.makedirs('processed_data/val/forward', exist_ok=True)
    os.makedirs('processed_data/val/right', exist_ok=True)

# Train the model
def train_model():
    # Create data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    # Setup data generators
    train_generator = train_datagen.flow_from_directory(
        'processed_data/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        'processed_data/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    # Create and train model
    model = create_model()

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // 32,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // 32
    )

    # Save the model
    model.save('small_car_model.h5')

    return history

# Test model prediction
def test_model(model, image_path):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(224, 224)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    directions = ['left', 'forward', 'right']
    predicted_direction = directions[np.argmax(predictions[0])]

    return predicted_direction, predictions[0]

# Main execution
if __name__ == "__main__":
    # Prepare dataset
    prepare_dataset()

    # Train model
    history = train_model()

    # Plot training results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
