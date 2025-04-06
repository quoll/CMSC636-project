# utils/metrics.py

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score


def compute_auroc_per_class(y_true, y_pred, class_names):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    aurocs = {}
    for i, name in enumerate(class_names):
        try:
            score = roc_auc_score(y_true[:, i], y_pred[:, i])
            aurocs[f"auroc_{name}"] = score
        except ValueError:
            
            aurocs[f"auroc_{name}"] = None
    return aurocs


def log_metrics(model_id, metrics_dict, csv_path="results/metrics_log.csv"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = [
        {
            "model_id": model_id,
            "metric_name": metric,
            "value": value,
            "timestamp": timestamp
        }
        for metric, value in metrics_dict.items()
    ]

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)

    print(f"[metrics] Logged {len(rows)} metrics to {csv_path}")
