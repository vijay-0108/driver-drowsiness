import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D
def generator(
    dir_path, 
    gen=ImageDataGenerator(rescale=1./255), 
    shuffle=True, 
    batch_size=32, 
    target_size=(24, 24), 
    class_mode='categorical'
):

    return gen.flow_from_directory(
        directory=dir_path,
        batch_size=batch_size,
        shuffle=shuffle,
        color_mode='grayscale',
        class_mode=class_mode,
        target_size=target_size
    )
#This is the path for the data sets
train_dir = r'C:\Users\saich\Downloads\Drowsiness detection\dataset_new\train'
valid_dir = r'C:\Users\saich\Downloads\Drowsiness detection\dataset_new\test'

# Batch size and image size
BS = 32
TS = (24, 24)

# Generating training and validation batches
train_batch = generator(train_dir, batch_size=BS, target_size=TS)
valid_batch = generator(valid_dir, batch_size=BS, target_size=TS)

# Compute steps per epoch and validation steps
SPE = len(train_batch.classes) // BS
VS = len(valid_batch.classes) // BS
print(f"Steps per epoch: {SPE}, Validation steps: {VS}")

# Build the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # Adjusted to match the number of classes (4)
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_batch,
    validation_data=valid_batch,
    epochs=15,
    steps_per_epoch=SPE,
    validation_steps=VS
)

# Save the trained model
os.makedirs('models', exist_ok=True)
model.save('models/cnnCat4.h5', overwrite=True)
print("Model saved successfully!")

# Plot training accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Over Epochs')

plt.show()
