import numpy as np
import torch
import ml_collections
import json


def load_config(filename: str) -> ml_collections.ConfigDict:
    """
    Load a ConfigDict from a file.

    Args:
        filename (str): The file name to load the configuration from.

    Returns:
        ConfigDict: The loaded configuration dictionary.
    """
    # Load the JSON file
    with open(filename, 'r') as f:
        config_dict = json.load(f)

    # Convert the dictionary to a ConfigDict
    config = ml_collections.ConfigDict(config_dict)
    return config


def load_txt_features(d_path, npz_key='arr_0', feature_keys=('ernie_content_only', 'pulse_content_only')):
    d_feature = {}
    for k, p in d_path.items():
        if feature_keys and k not in feature_keys:
            continue
        try:
            t = np.load(p, allow_pickle=True)[npz_key]
        except Exception as e:
            print(f'Error loading {p}')
            raise e
        # to tensor
        t = torch.from_numpy(t).float()

        d_feature[k] = t

    return d_feature


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    else:
        return obj


def calc_surv_risk(model_output):
    pass


def predict_survival(model, input_data):
    """
    Predict survival using the pre-trained model.

    Args:
        model: The pre-trained model.
        input_data: Input data for prediction.

    Returns:
        Prediction results.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output = model(input_data)
    return output


def load_data(file_path):
    """
    Load patient data from a JSON file.

    Args:
        file_path: Path to the JSON file containing patient data.

    Returns:
        Loaded patient data.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for event in data['trajectory']:
        event['txt_features'] = load_txt_features(event['txt_features'])

    traj = data['trajectory']

    date_origin = traj[0]['date']

    clean_text = []
    for event in traj:
        event = event.copy()
        if 'txt_features' in event:
            del event['txt_features']
        del event['md5-content']
        clean_text.append(event)

    return [traj], [date_origin], clean_text


def hazards_to_risk(hazards):
    S = torch.cumprod(1 - hazards, dim=1)
    risk = torch.sum(1 - S, dim=1)
    return risk


def logits_to_risk(logits):
    hazards = torch.sigmoid(logits)
    return hazards_to_risk(hazards)


def compute_risk_percentile(pred_risk: float):
    risk_npy = np.load('data/risks.npy')
    pencentile = np.sum(risk_npy < pred_risk) / len(risk_npy) * 100
    return pencentile
