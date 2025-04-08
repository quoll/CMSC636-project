# main.py

import argparse
import os
import random
import numpy as np
import tensorflow as tf

from utils.label_utils import load_labels
from utils.model_selector import get_model_components
from utils.data_loader import dataset_from_zips, dataset_from_zip
from utils.evaluator import evaluate_model_on_test
from config.constants import (
    default_label_file,
    default_input_file,
    default_validate_file,
    default_test_file
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run selected model on CheXpert dataset.")
    parser.add_argument('--model_id', type=str, default="base_cnn",
                        help='Identifier of the model to run (e.g., base_cnn)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--train', action='store_true', help='Flag to train the model')
    parser.add_argument('--test', action='store_true', help='Flag to test the model')
    return parser.parse_args()


def set_seed(seed):
    print(f"[main] Setting seed: {seed}")
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    input_files = [default_input_file]
    validate_file = default_validate_file
    test_file = default_test_file
    label_file = default_label_file

    required_files = [label_file]
    if args.train:
        required_files += input_files + [validate_file]
    if args.test:
        required_files += [test_file]

    for f in required_files:
        if not os.path.exists(f):
            print(f"[main] Required file not found: {f}")
            return

    print(f"[main] Loading labels from {label_file}")
    label_data = load_labels(label_file)

    if args.train:
        print("[main] Preparing training and validation datasets...")
        train_ds, train_len = dataset_from_zips(input_files, label_data)
        val_ds, val_len = dataset_from_zip(validate_file, label_data)

        print(f"[main] Dataset sizes -> Train: {train_len}, Validation: {val_len}")

        build_model_fn, train_model_fn = get_model_components(args.model_id)
        model = build_model_fn()
        train_model_fn(model, train_ds, train_len, val_ds, val_len, args)

    if args.test:
        print("[main] Preparing test dataset...")
        model_path = f"results/{args.model_id}_model.h5"
        if not os.path.exists(model_path):
            print(f"[main] Model file not found at {model_path}. Cannot run test.")
            return

        evaluate_model_on_test(
            model_path=model_path,
            test_file=test_file,
            label_data=label_data,
            model_id=args.model_id
        )


if __name__ == "__main__":
    main()
