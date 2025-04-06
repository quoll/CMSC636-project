#!/usr/bin/env python3
import sys
import json
import os
import argparse
import zipfile
import math
import re
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy

default_label_file = 'label/findings_fixed.json'
default_input_file = 'CheXpert-v1.0 batch 2 (train 1).zip'
default_validate_file = 'CheXpert-v1.0 batch 1 (validate & csv).zip'
epochs = 10  # Number of epochs for training
batch_size = 4  # Adjust based on your system's memory
bounding_square = 2880
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
    

def count_jpg_in_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        return sum(1 for name in zf.namelist() if name.endswith(image_ext))

def jpg_from_zip_generator(zip_path, labels):
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith(image_ext):
                nname = normalize_path(name)
                if nname not in labels:
                    print(f"Warning: Found JPG '{name}' in zip but no corresponding label entry. Skipping.")
                    continue
                with zf.open(name) as f:
                    yield f.read(), labels[nname]


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
    padded.set_shape([bounding_square, bounding_square, 1])
    tf.print("Padded image from (", current_height, ",", current_width, ") to", tf.shape(padded))
    return padded


def dataset_from_zip(zip_path, labels):
    length = count_jpg_in_zip(zip_path)
    dataset = tf.data.Dataset.from_generator(
        lambda: jpg_from_zip_generator(zip_path, labels),
        output_types=(tf.string, tf.float32),  # output types of the generator
        output_shapes=((), (len(labels_in_order),))
    )
    def pair_images_and_labels(x, y):
        image = tf.image.convert_image_dtype(tf.io.decode_jpeg(x, channels=1), dtype=tf.float32)  # Decode the JPEG
        return pad_to_fixed_size(image), y

    return dataset.map(pair_images_and_labels), length


def dataset_from_zips(zip_paths, labels):
    datasets_and_lengths = [dataset_from_zip(path, labels) for path in zip_paths]
    datasets, lengths = zip(*datasets_and_lengths)
    full_dataset = datasets[0]
    for dataset in datasets[1:]:
        full_dataset = full_dataset.concatenate(dataset)

    return full_dataset, sum(lengths)


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

        # Dense(512, activation='relu'),
        Dense(128, activation='relu'),
        # Dropout(0.5),
        Dense(14, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss=BinaryCrossentropy(),
                  metrics=[BinaryAccuracy()])
    model.summary()
    return model


def process_dataset(model, dataset, length, validate_dataset, vlength):
    steps_per_epoch = math.ceil(length / batch_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()  # Repeat the dataset for multiple epochs
    validation_steps = math.ceil(vlength / batch_size)
    validate_dataset = validate_dataset.batch(batch_size)

    #time the epochs to see how long it takes to train
    start_time = tf.timestamp()  # Start time for timing the training
    model.fit(dataset,
              batch_size=batch_size,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=validate_dataset,
              validation_steps=validation_steps)
    end_time = tf.timestamp()  # End time for timing the training
    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time.numpy()} seconds.")
    # Save the model
    model.save('model.h5')
    print("Model saved as 'model.h5'.")
    return model


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
    print(f"Total entries processed: {len(label_data)}")

    dataset, length = dataset_from_zips(input_files, label_data)
    validate_dataset, vlength = dataset_from_zip(validate_file, label_data)
    model = create_model()
    model = process_dataset(model, dataset, length, validate_dataset, vlength)

