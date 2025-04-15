import torch
import numpy as np
import torch.nn as nn


trajectory_batch, date_origin_batch = [], []

# Example data
admit_note = {
    'date': '2023-01-01',
    'event_type': 'admission note',
    'content': 'This is an admit note',
    'txt_features': {
        'pulse': torch.randn(1, 713, 5120),
        'ernie': torch.randn(1, 851, 768),
    }
}

def generate_meta(num_patches, num_features):
    meta = torch.randn(num_patches, num_features)
    return meta


ct_exam = {
    'date': '2023-01-02',
    'event_type': 'imaging exam',
    'content': 'CT',
    'img_features': {
            "universal-clipdriven": [
                {
                    "features": torch.randn(24, 384, 6, 6, 6),
                    "meta": np.random.randn(24, 5)  # num_patches, coordinates and other meta info
                }
            ],
            "fmcib": [
                {
                    "features": torch.randn(500, 4096),
                    "meta": np.random.randn(500, 5)
                }
            ],
            "SAMMed3D": [
                {
                    "features": torch.randn(24, 384, 8, 8, 8),
                    "meta": np.random.randn(24, 5)
                }
            ]
    }
}

lab_event = {
    'date': '2023-01-03',
    'event_type': 'lab',
    'lab_features': torch.randn(1, 3 * 55),
}

op_note = {
    'date': '2023-01-04',
    'event_type': 'operative note',
    'content': 'This is an operative note',
    'txt_features': {
        'pulse': torch.randn(1, 629, 5120),
        'ernie': torch.randn(1, 784, 768),
    }
}

# Example trajectory
trajectory = [
        admit_note,
        ct_exam,
        lab_event,
        op_note,
]

trajectory_batch = [trajectory]
date_origin_batch = [admit_note['date']]


data = trajectory_batch, date_origin_batch