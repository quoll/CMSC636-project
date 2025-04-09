# utils/evaluator.py

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from sklearn.metrics import roc_auc_score
from utils.data_loader import dataset_from_zip
from utils.metrics import log_metrics
from config.constants import labels_in_order
import time

def compute_auroc_per_class(y_true, y_pred, class_names):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    aurocs = {}
    for i, name in enumerate(class_names):
        try:
            auroc = roc_auc_score(y_true[:, i], y_pred[:, i])
            aurocs[f"test_auroc_{name}"] = auroc
        except ValueError:
            aurocs[f"test_auroc_{name}"] = None
    return aurocs


def evaluate_model_on_test(model_path, test_file, label_data, model_id):
    print(f"[test] Loading model from {model_path}")
    model = keras.models.load_model(model_path)

    print("[test] Preparing test dataset...")
    test_ds, test_len = dataset_from_zip(test_file, label_data)
    test_ds = test_ds.batch(50)  

    steps = int(np.ceil(test_len / 50))

    print(f"[test] Evaluating on {test_len} instances...")

    start = time.time()
    loss, acc = model.evaluate(test_ds, steps=steps)
    eval_time = time.time() - start

    # AUROC manual
    y_true, y_pred = [], []
    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch)
        y_true.extend(y_batch.numpy())
        y_pred.extend(preds)

    aurocs = compute_auroc_per_class(y_true, y_pred, labels_in_order)
    mean_auroc = np.nanmean([v for v in aurocs.values() if v is not None])

    log_metrics(model_id, metrics_dict={
        "test_loss": float(loss),
        "test_binary_accuracy": float(acc),
        "test_mean_auroc": float(mean_auroc),
        "test_eval_time_seconds": float(eval_time),
        **aurocs
    })

    print(f"[test] Evaluation complete. Mean AUROC: {mean_auroc:.4f}")
