import numpy as np
import torch
import torch.nn as nn
from modeling.gated_attention import get_gated_attnetion_ct, \
    get_gated_attnetion_video
from modeling.event_embeding import AvgEventEmbedding
from modeling.constants import FEATURE_DIMS_V3


class EventVectorizer(nn.Module):
    """
    EventVectorizer class:
    - Embeds different types of events using configurable transformers and linear layers.
    - Dynamically creates linear layers for image and text features based on their respective keys.
    - Uses separate transformer decoder layers to fuse multiple image or text features if necessary.
    - Integrates video event processing with a specific transformer model.
    """

    def __init__(self, ct_event_former, video_event_former, wsi_former, text_former, config):
        super(EventVectorizer, self).__init__()
        self.ct_event_former = ct_event_former
        self.video_event_former = video_event_former
        self.wsi_former = wsi_former
        self.text_former = text_former
        self.device = config.device
        self.config = config
        dim = config.hidden_size
        self.hidden_size = dim
        self.max_sequence_len = config.model.max_sequence_len

        # Single linear layer for lab features
        self.lab_linear = nn.Linear(FEATURE_DIMS_V3['lab'], dim)
        self.gelu = nn.GELU()

    def forward(self, event):
        event_type = event['event_type']

        if event_type == '影像检查':
            if 'CT' == event['content']:
                res_dict = self.ct_event_former(event)  # 1, dim
            else:
                res_dict = self.video_event_former(event)  # 1, dim

        elif event_type == '化验':
            # Directly process lab features
            lab_features = event['lab_features']
            lab_shape = lab_features.shape
            lab_features = self.lab_linear(lab_features.reshape(1, -1))  # 1, dim
            weights = self.lab_linear.weight.detach().cpu()  # dim, 3*55
            attn = weights.mean(dim=0).reshape(1, 1, *lab_shape)  # 1, 1, 3, 55
            attn_abs = weights.abs().mean(dim=0).reshape(1, 1, *lab_shape)  # 1, 1, 3, 55
            attn = torch.cat([attn, attn_abs], dim=1)  # 1, 2, 3, 55, two heads
            res_dict = {'cls': lab_features, 'attn_list': [attn]}
        else:
            res_dict = self.text_former(event)
        res_dict['cls'] = self.gelu(res_dict['cls'])
        return res_dict


def compute_n_param(model):
    return sum(p.numel() for p in model.parameters())


def get_event_vectorizer(config):
    ct_sformer = get_gated_attnetion_ct(config)
    video_sformer = get_gated_attnetion_video(config)
    text_sformer = AvgEventEmbedding(config, legacy_mode=False)

    model = EventVectorizer(ct_sformer, video_sformer, None, text_sformer, config)
    return model
