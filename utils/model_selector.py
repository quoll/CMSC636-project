# utils/model_selector.py

from models import base_cnn
# from models import ag_cnn, etc.

def get_model_components(model_id):
    if model_id == "base_cnn":
        return base_cnn.build_model, base_cnn.train_model
    # elif model_id == "ag_cnn":
    #     return ag_cnn.build_model, ag_cnn.train_model
    else:
        raise ValueError(f"Unknown model_id: {model_id}")
