#!/usr/bin/env python3
import sys
import json
import os
import argparse
import zipfile
import math
import re
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy

default_label_file = 'label/findings_fixed.json'
default_input_file = 'CheXpert-v1.0 batch 2 (train 1).zip'
default_validate_file = 'CheXpert-v1.0 batch 1 (validate & csv).zip'
epochs = 10  # Number of epochs for training
batch_size = 2  # Adjust based on your system's memory
bounding_square = 1440
image_ext = '.jpg'

labels_in_order = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
                   "Edema", "Consolidation", "Pneumonia", "Atelectasis",
                   "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
                   "Support Devices", "No Finding"]

def normalize_path(name):
    match = re.match(r'^.*\(train [^)]*\)/', name)
    if match:
        return 'train/' + name[match.end():]
    match = re.match(r'^.*\(validate & csv\)/', name)
    if match:
        return name[match.end():]
    raise ValueError(f"Invalid path format: {name}")
    

def count_jpg_in_zip(zip_path, labels):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        return sum(1 for name in zf.namelist() if name.endswith(image_ext) and normalize_path(name) in labels)

def jpg_files_from_zip(zip_path, labels):
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith(image_ext):
                if normalize_path(name) not in labels:
                    continue
                yield name


def pad_to_fixed_size(image, target_size=(bounding_square, bounding_square)):
    # Get original dimensions
    current_height = tf.shape(image)[0]
    current_width = tf.shape(image)[1]

    scale = tf.minimum(
        target_size[0] / tf.cast(current_height, tf.float32),
        target_size[1] / tf.cast(current_width, tf.float32)
    )

    # scale down if it is too large
    def resize_needed():
        new_height = tf.cast(tf.cast(current_height, tf.float32) * scale, tf.int32)
        new_width = tf.cast(tf.cast(current_width, tf.float32) * scale, tf.int32)
        return tf.image.resize(image, [new_height, new_width], method='bilinear')

    def no_resize_needed():
        return image

    # Only resize if the image is larger
    image = tf.cond(
        tf.logical_or(current_height > target_size[0], current_width > target_size[1]),
        resize_needed,
        no_resize_needed
    )

    # Get dimensions after possible resize
    current_height = tf.shape(image)[0]
    current_width = tf.shape(image)[1]

    # Compute padding
    pad_height = target_size[0] - current_height
    pad_width = target_size[1] - current_width

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    padded = tf.pad(
        image,
        paddings=[[pad_top, pad_bottom], [pad_left, pad_right], [0,0]],
        mode='CONSTANT',
        constant_values=0
    )
    # The model needs to know the shape of the input tensor, so we set it explicitly here
    padded.set_shape([target_size[0], target_size[1], 1])
    # tf.print("Padded image from (", current_height, ",", current_width, ") to", tf.shape(padded))
    return padded


def find_file(image_path):
    return image_path[:image_path.find('/')] + '.zip'

def dataset_from_zips(zip_paths, labels):
    filenames = []
    for path in zip_paths:
        filenames.extend(jpg_files_from_zip(path, labels))
    file_names = tf.constant(filenames)
    file_names = tf.data.Dataset.from_tensor_slices(file_names)
    def jpg_from_zip_list(filename):
        def _read_image_and_label(filename_str):
            try:
                filename_str = filename_str.numpy().decode('utf-8')
                data = None
                try:
                    with zipfile.ZipFile(find_file(filename_str), 'r') as zpfa:
                        with zpfa.open(filename_str) as zf:
                            data = zf.read()
                except Exception as e:
                    with zipfile.ZipFile(find_file(filename_str), 'r') as zpfa:
                        with zpfa.open(filename_str) as zf:
                            data = zf.read()
                if data is None:
                    return None, None
                image = tf.image.convert_image_dtype(tf.io.decode_jpeg(data, channels=1), dtype=tf.float32)
                return pad_to_fixed_size(image), labels[normalize_path(filename_str)]
            except Exception as e:
                print(f"Warning: Failed to process {filename_str}: {e}")
                return None, None
        image, label = tf.py_function(_read_image_and_label, [filename], [tf.float32, tf.float32])
        if image is None or label is None:
            return None, None
        image.set_shape([bounding_square, bounding_square, 1])
        label.set_shape([len(labels_in_order)])
        return image, label
    dataset = file_names.map(jpg_from_zip_list)
    dataset = dataset.filter(lambda img, lbl: tf.logical_and(tf.not_equal(tf.size(img), 0),
                                                             tf.not_equal(tf.size(lbl), 0)))
    return dataset, len(filenames)


def vector_encoding(labels):
    vector = []
    for label in labels_in_order:
        if label in labels:
            value = labels[label]
            if value is not None:
                vector.append(value)
            else:
                vector.append(0.0)
        else:
            print(f"Warning: Label '{label}' not found in the input data. Defaulting to 0.")
            vector.append(0.0)
    # Return as a TensorFlow constant for compatibility with the dataset
    return tf.constant(vector, dtype=tf.float32, shape=(len(labels_in_order),))

def load_labels(label_file):
    """
    Load the labels from a JSON file.
    :param label_file: Path to the label file.
    :return: A dictionary mapping image paths to their label vectors.
    """
    label_data = {}
    with open(label_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                if 'path_to_image' in entry:
                    # convert to an encoded vector for findings
                    path = entry['path_to_image']
                    label_data[path] = vector_encoding(entry)
                else:
                    print(f"Skipping entry without 'path_to_image' on line {f.tell()}: {entry}")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
    return label_data


def create_model():
    model = Sequential([
        Input(shape=(bounding_square,bounding_square,1)),
        Conv2D(64, (5,5), activation='relu', padding='valid'),
        MaxPooling2D(pool_size=(2,2), strides=2),

        Conv2D(128, (5,5), activation='relu', padding='valid'),
        MaxPooling2D(pool_size=(2,2), strides=2),

        Conv2D(256, (5,5), activation='relu', padding='valid'),
        MaxPooling2D(pool_size=(2,2), strides=2),

        Flatten(),

        #Dense(512, activation='relu'),
        Dense(128, activation='relu'),
        # Dropout(0.5),
        Dense(14, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss=BinaryCrossentropy(),
                  metrics=[BinaryAccuracy()])
    model.summary()
    return model

def plot_training(history):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.subplot(1, 2, 2)
    if 'binary_accuracy' in history.history:
        plt.plot(history.history['binary_accuracy'], label='Training Accuracy')
    if 'val_binary_accuracy' in history.history:
        plt.plot(history.history['val_binary_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over epochs')
    plt.legend()
    plt.tight_layout()
    plt.show()

def process_dataset(model, dataset, length, validate_dataset, vlength):
    steps_per_epoch = length // batch_size
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    validation_steps = vlength // batch_size
    validate_dataset = validate_dataset.repeat()
    validate_dataset = validate_dataset.batch(batch_size, drop_remainder=True)
    validate_dataset = validate_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    it = iter(dataset)

    #time the epochs to see how long it takes to train
    start_time = tf.timestamp()  # Start time for timing the training
    history = model.fit(dataset,
                        batch_size=batch_size,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=validate_dataset,
                        validation_steps=validation_steps,
                        verbose=1)
    end_time = tf.timestamp()  # End time for timing the training
    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time.numpy()} seconds.")
    # Save the model
    # model.save('model.keras')
    # print("Model saved as 'model.keras'.")
    print("Model not saved")
    return model

def validate_ds(dataset, num_batches=3):
    it = iter(dataset)
    for i in range(num_batches):
        try:
            batch = next(it)
            images, labels = batch
            tf.print(f"Batch {i}: images shape {tf.shape(images)}, labels shape {tf.shape(labels)}")
        except StopIteration:
            tf.print("Dataset exhausted early at batch", i)
            break

if __name__ == "__main__":
    # -l for the label file, -i for the input file
    parser = argparse.ArgumentParser(description="Train a model from zipped JPGs.")
    parser.add_argument('-l', '--label_file', type=str, default=default_label_file,
                        help='Path to the label file (default: %(default)s)')
    parser.add_argument('-v', '--validate_file', type=str, default=default_validate_file,
                        help='Path to the label file (default: %(default)s)')
    parser.add_argument('input_files', nargs='*', default=[default_input_file],
                        help='Path to the input zip file (default: %(default)s)')
    args = parser.parse_args()
    input_files = args.input_files
    label_file = args.label_file
    validate_file = args.validate_file

    print(f"Input files: {input_files}")
    print(f"Validation file: {validate_file}")
    print(f"Label file: {label_file}")

    for f in input_files:
        if not os.path.exists(f):
            print(f"File {f} does not exist.")
            exit(1)

    # read each line from the input file and parse it as JSON
    label_data = load_labels(label_file)
    
    # print the number of entries in the dictionary
    print(f"Total labels processed: {len(label_data)}")

    dataset, length = dataset_from_zips(input_files, label_data)
    print(f"Dataset length: {length}")
    validate_dataset, vlength = dataset_from_zips([validate_file], label_data)

    model = create_model()
    model = process_dataset(model, dataset, length, validate_dataset, vlength)

