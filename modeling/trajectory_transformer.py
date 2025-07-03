import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint

from modeling.pre_norm_transformer import TransformerEncoder


def pad_and_truncate(sequences, max_len, padding_value=0., batch_first=True):
    for seq in sequences:
        if len(seq) > max_len:
            print(f"Sequence length {len(seq)} is greater than max_len {max_len}")
    # Truncate and pad sequences
    truncated_sequences = [seq[:max_len] for seq in sequences]
    padded_sequences = pad_sequence(truncated_sequences, batch_first=batch_first, padding_value=padding_value)  # (B, max_len, d)
    return padded_sequences


class TrajectoryTransformer(nn.Module):
    """
    This module processes the sequence of encoded event tokens using transformer architecture.
    It handles complex dependencies between events and extracts features useful for downstream tasks like survival
    prediction and diagnosis.
    """
    def __init__(self, config, *, event_vectorizer, event_sequence_encoder):
        super(TrajectoryTransformer, self).__init__()
        self.config = config
        self.dim = config.hidden_size
        self.num_layers = config.model.num_layers
        self.nhead = config.model.nhead
        self.dim_feedforward = config.model.dim_feedforward
        self.dropout = 0.2
        self.map_task_ncls = config.map_task_ncls
        self.gpu_checkpoint = config.model.gpu_checkpoint
        self.survival_task_names = config.survival_task_names

        self.event_vectorizer = event_vectorizer
        self.event_sequence_encoder = event_sequence_encoder

        self.encoder = TransformerEncoder(self.dim, self.num_layers, self.nhead, self.dim_feedforward // self.dim,
                                          self.dropout, return_q_attn=True)

        self.traj_cls_token = nn.Parameter(torch.randn(1, self.config.model.num_traj_cls_tokens, self.dim))

        self.linear_tasks = nn.ModuleList([
            nn.Linear(self.dim, self.map_task_ncls[k]) for k in self.map_task_ncls.keys()
        ])

    def forward(self, trajectory_batch, date_origin_batch):
        # tj_batch = batch['trajectory']
        event_sequence_batch = []
        event_attn_list_batch = []
        traj_len_batch = []
        for traj, date_origin in zip(trajectory_batch, date_origin_batch):
            event_embeddings = []
            # date_origin = target['first_large_surgery_date']
            for event in traj:
                if self.gpu_checkpoint:
                    embd = checkpoint(self.event_vectorizer, event, use_reentrant=False)  # a dict {'cls': embd, 'attn_list': [attn1, attn2, ...]}
                else:
                    embd = self.event_vectorizer(event)  # a dict {'cls': embd, 'attn_list': [attn1, attn2, ...]}
                event_embeddings.append(embd)
            event_sequence = self.event_sequence_encoder(traj, event_embeddings, date_origin)  # (seq_len, embd_dim)
            attn_list = [embd['attn_list'] for embd in event_embeddings]  # O(n_events x n_feature_keys x (1, num_heads, *S))
            event_attn_list_batch.append(attn_list)  # O(batchsize x n_events x n_feature_keys x (1, num_heads, *S))
            traj_len_batch.append(len(traj))

            event_sequence_batch.append(event_sequence)  # O(batchsize x n_events x embd_dim)

        event_sequence_batch = pad_and_truncate(event_sequence_batch, max_len=self.config.model.max_sequence_len, padding_value=0,
                                                batch_first=True)  # N, max_len, d

        traj_cls_token = self.traj_cls_token.expand(event_sequence_batch.size(0), -1, -1)
        event_sequence_batch = torch.cat([traj_cls_token, event_sequence_batch], dim=1)
        tj_embd_batch, q_attn = self.encoder(event_sequence_batch)

        cls = tj_embd_batch[:, 1]

        tasks_logit = []  # [(N, d_t1), (N, d_t2), ...]
        for l_t in self.linear_tasks:
            tasks_logit.append(l_t(cls))

        output_dict = {k: v[-1].unsqueeze(0) for k, v in zip(self.map_task_ncls.keys(), tasks_logit) if k in self.survival_task_names}  # make up for the batch dimension
        return output_dict


def get_default_trajectory_transformer(config):
    from modeling.event_vectorizer import get_event_vectorizer
    from modeling.event_sequence_encoder import get_default_event_sequence_encoder
    event_vectorizer = get_event_vectorizer(config)
    event_sequence_encoder = get_default_event_sequence_encoder(config)
    model = TrajectoryTransformer(config, event_vectorizer=event_vectorizer, event_sequence_encoder=event_sequence_encoder)
    return model


def build_model(model_path='model.pth'):
    from fvcore.common.checkpoint import Checkpointer
    from utils import load_config
    config = load_config('modeling/config.json')
    model = get_default_trajectory_transformer(config)
    ckpt = Checkpointer(model, 'data')

    model_pt = model_path
    ckpt.load(model_pt, [])
    return model
