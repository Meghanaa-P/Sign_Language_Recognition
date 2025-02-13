import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator

# Step 1: Data Augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Rescale images to [0, 1] range
    rotation_range=20,  # Random rotations for data augmentation
    width_shift_range=0.2,  # Horizontal shifting
    height_shift_range=0.2,  # Vertical shifting
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    validation_split=0.2  # Split the data into training and validation sets
)

# Load data from the updated folder structure
train_data = datagen.flow_from_directory(
    r"D:\DS",  # The root folder with subfolders for each class (RH_A, LH_A, etc.)
    target_size=(224, 224),  # Resize images to fit the model input
    batch_size=32,
    class_mode='categorical',  # Multi-class classification
    subset='training',  # Training subset
    shuffle=True
)

val_data = datagen.flow_from_directory(
    r"D:\DS",  # Same root folder for validation
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',  # Validation subset
    shuffle=True
)

# Step 2: Model Building
# Use MobileNetV2 as the base model with pre-trained weights from ImageNet
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model layers to prevent training them

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # Global average pooling layer
    layers.Dense(256, activation='relu'),  # Fully connected layer
    layers.Dropout(0.5),  # Dropout layer for regularization
    layers.Dense(train_data.num_classes, activation='softmax')  # Output layer with the number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 3: Train the Model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs= 80 ,  # Number of epochs to train
    steps_per_epoch=train_data.samples // train_data.batch_size,
    validation_steps=val_data.samples // val_data.batch_size
)

# Step 4: Save the Trained Model
model.save("dataset.h5")  # Save the model after training

# Optionally: You can evaluate or predict with the trained model here=