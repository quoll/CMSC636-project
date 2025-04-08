import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy
from keras.callbacks import CSVLogger

from config.constants import bounding_square, batch_size, epochs
import os
import math


def build_model():
    model = Sequential([
        Input(shape=(bounding_square, bounding_square, 1)),
        Conv2D(8, (3, 3), activation='relu'),  # solo 8 filtros
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(16, activation='relu'),         # capa densa muy pequeña
        Dense(14, activation='sigmoid')       # salida igual
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=BinaryCrossentropy(),
        metrics=[BinaryAccuracy()]
    )
    return model


def train_model(model, train_dataset, train_size, val_dataset, val_size, args):
    train_dataset = train_dataset.batch(batch_size).repeat()
    val_dataset = val_dataset.batch(batch_size)

    steps_per_epoch = math.ceil(train_size / batch_size)
    validation_steps = math.ceil(val_size / batch_size)

    os.makedirs("results/logs", exist_ok=True)

    log_file = f"results/logs/{args.model_id}_trainlog.csv"
    csv_logger = CSVLogger(log_file, append=True)

    print(f"[{args.model_id}] Starting training... Logging to {log_file}")
    start = tf.timestamp()

    history = model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=[csv_logger],
        verbose=1
    )

    duration = tf.timestamp() - start

    os.makedirs("results", exist_ok=True)
    model_path = f"results/{args.model_id}_model.h5"
    model.save(model_path)

    print(f"[{args.model_id}] Training completed in {duration.numpy():.2f} seconds.")
    print(f"[{args.model_id}] Model saved to '{model_path}'")
    print(f"[{args.model_id}] Training log saved to '{log_file}'")