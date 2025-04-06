import re
import zipfile
import tensorflow as tf
from config.constants import image_ext, bounding_square, labels_in_order
from .preprocessing import pad_to_fixed_size

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
                    continue
                with zf.open(name) as f:
                    yield f.read(), labels[nname]

def dataset_from_zip(zip_path, labels):
    length = count_jpg_in_zip(zip_path)
    dataset = tf.data.Dataset.from_generator(
        lambda: jpg_from_zip_generator(zip_path, labels),
        output_types=(tf.string, tf.float32),
        output_shapes=((), (len(labels_in_order),))
    )

    def pair_images_and_labels(x, y):
        image = tf.image.convert_image_dtype(tf.io.decode_jpeg(x, channels=1), dtype=tf.float32)
        return pad_to_fixed_size(image), y

    return dataset.map(pair_images_and_labels), length

def dataset_from_zips(zip_paths, labels):
    datasets_and_lengths = [dataset_from_zip(path, labels) for path in zip_paths]
    datasets, lengths = zip(*datasets_and_lengths)
    full_dataset = datasets[0]
    for dataset in datasets[1:]:
        full_dataset = full_dataset.concatenate(dataset)
    return full_dataset, sum(lengths)
