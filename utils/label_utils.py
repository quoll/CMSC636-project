import json
import tensorflow as tf
from config.constants import labels_in_order

def vector_encoding(labels):
    vector = []
    for label in labels_in_order:
        value = labels.get(label, 0.0)
        if value is None:
            value = 0.0
        vector.append(value)
    return tf.constant(vector, dtype=tf.float32, shape=(len(labels_in_order),))

def load_labels(label_file):
    label_data = {}
    with open(label_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                if 'path_to_image' in entry:
                    path = entry['path_to_image']
                    label_data[path] = vector_encoding(entry)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
    return label_data
