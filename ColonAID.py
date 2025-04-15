from modeling.trajectory_transformer import get_default_trajectory_transformer
from config.config_v0 import get_default_config
from ml_collections import ConfigDict
from data.example_data import data
from modeling.utils import to_device, summarize_structure, logits_to_risk


"""
This is an example illustrating the model architecture of COLON-AID, with random data and weights. 
"""


args = ConfigDict()

config = get_default_config(args)
model = get_default_trajectory_transformer(config)

print(model)

trajectory_batch, date_origin_batch = data

trajectory_batch = to_device(trajectory_batch, config.device)
date_origin_batch = to_device(date_origin_batch, config.device)
model.to(config.device)

prediction = model(trajectory_batch, date_origin_batch)

predicted_risk = logits_to_risk(prediction['survival'])

normalized_risk = predicted_risk / 12

print(f'Predicted risk: {predicted_risk.item()}')
