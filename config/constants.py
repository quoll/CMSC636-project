# config/constants.py


default_label_file = "data/sample_label.json"
default_input_file = "data/sample.zip"
default_validate_file = "data/sample_valid.zip"
default_test_file = "data/test.zip"

bounding_square = 2880
batch_size = 4
epochs = 10
image_ext = '.jpg'

labels_in_order = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices", "No Finding"
]