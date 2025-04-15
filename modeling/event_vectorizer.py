import torch
import torch.nn as nn
from modeling.gated_attention import get_gated_attnetion_ct, get_gated_attention_text, \
    get_gated_attnetion_video
from modeling.event_embeding import AvgEventEmbedding
from config.constants import FEATURE_DIMS


class EventVectorizer(nn.Module):
    """
    EventVectorizer class:
    - Embeds different types of events using configurable transformers and linear layers.
    - Dynamically creates linear layers for image and text features based on their respective keys.
    """

    def __init__(self, ct_event_former, video_event_former, text_former, config):
        super(EventVectorizer, self).__init__()
        self.ct_event_former = ct_event_former
        self.video_event_former = video_event_former
        self.text_former = text_former
        self.device = config.device
        dim = config.hidden_size
        self.hidden_size = dim
        self.max_sequence_len = config.model.max_sequence_len
        self.config = config

        # Single linear layer for lab features
        self.lab_linear = nn.Linear(FEATURE_DIMS['lab'], dim)
        self.gelu = nn.GELU()

    def forward(self, event):
        event_type = event['event_type']

        if event_type == 'imaging exam':
            if 'CT' == event['content']:
                res_dict = self.ct_event_former(event)  # 1, dim
            else:
                res_dict = self.video_event_former(event)  # 1, dim

        elif 'lab' == event_type:
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


def get_event_vectorizer_average(config, legacy_mode=False):
    model = AvgEventEmbedding(config, legacy_mode=legacy_mode)
    return model


def get_event_vectorizer_gated_attn(config):
    ct_sformer = get_gated_attnetion_ct(config)
    video_sformer = get_gated_attnetion_video(config)
    text_sformer = get_gated_attention_text(config)

    model = EventVectorizer(ct_sformer, video_sformer, text_sformer, config)
    return model


def get_event_vectorizer(config, type):
    if type == 'average':
        return get_event_vectorizer_average(config)
    if type == 'gated_attn':
        return get_event_vectorizer_gated_attn(config)
    raise ValueError(f'Invalid type: {type}')

