import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from modeling.pre_norm_transformer import TransformerEncoder
from modeling.utils import to_device, pad_and_truncate


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
        print(self.map_task_ncls)
        self.gpu_checkpoint = config.model.gpu_checkpoint

        self.event_vectorizer = event_vectorizer
        self.event_sequence_encoder = event_sequence_encoder

        self.encoder = TransformerEncoder(self.dim, self.num_layers, self.nhead, self.dim_feedforward // self.dim,
                                          self.dropout, return_q_attn=True)

        self.n_cls_tokens = self.config.model.num_traj_cls_tokens
        if self.config.model.num_traj_cls_tokens:
            self.traj_cls_token = nn.Parameter(torch.randn(1, self.config.model.num_traj_cls_tokens, self.dim))
        else:
            self.traj_cls_token = None

        self.need_linear_back = self.config.model.need_linear_back
        if self.need_linear_back:
            self.linear_back = nn.Linear(self.dim, self.dim * len(self.map_task_ncls))  # each section for a task
        else:
            self.linear_back = None

        self.linear_tasks = nn.ModuleList([
            nn.Linear(self.dim, self.map_task_ncls[k]) for k in self.map_task_ncls.keys()
        ])

    def forward(self, trajectory_batch, date_origin_batch):
        # batch = to_device(batch, self.config.device)
        # tj_batch = batch['trajectory']
        event_sequence_batch = []
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
            traj_len_batch.append(len(traj))

            event_sequence_batch.append(event_sequence)  # O(batchsize x n_events x embd_dim)

        event_sequence_batch = pad_and_truncate(event_sequence_batch, max_len=self.config.model.max_sequence_len, padding_value=0,
                                                batch_first=True)  # N, max_len, d

        if self.n_cls_tokens > 0:
            traj_cls_token = self.traj_cls_token.expand(event_sequence_batch.size(0), -1, -1)
            event_sequence_batch = torch.cat([traj_cls_token, event_sequence_batch], dim=1)
        tj_embd_batch, q_attn = self.encoder(event_sequence_batch)  # (N, n_cls + max_len, d), (N, nhead, n_cls + max_len)

        if self.n_cls_tokens == 1:
            if self.config.model.cls_at_0:
                cls = tj_embd_batch[:, 0]
            else:
                cls = tj_embd_batch[:, 1]
        elif self.n_cls_tokens <= 0:
            cls = torch.mean(tj_embd_batch, dim=1)  # N, d
        else:
            raise ValueError(f'Invalid number of trajectory cls tokens: {self.n_cls_tokens}')

        tasks_logit = []  # [(N, d_t1), (N, d_t2), ...]
        if self.need_linear_back:
            cls = self.linear_back(cls)  # N, d*num_tasks
            for i, l_t in enumerate(self.linear_tasks):
                tasks_logit.append(l_t(cls[..., i * self.d_model:(i + 1) * self.d_model]))
        else:
            for l_t in self.linear_tasks:
                tasks_logit.append(l_t(cls))

        output_dict = {k: v for k, v in zip(self.map_task_ncls.keys(), tasks_logit)}

        assert 'CLS' not in output_dict, f'Key "cls" is reserved for trajectory cls token. Please rename the key.'
        output_dict['CLS'] = cls

        return output_dict


def get_default_trajectory_transformer(config):
    from modeling.event_vectorizer import get_event_vectorizer
    from modeling.event_sequence_encoder import get_default_event_sequence_encoder
    event_vectorizer = get_event_vectorizer(config, config.model.event_vectorizer_type)
    event_sequence_encoder = get_default_event_sequence_encoder(config)
    model = TrajectoryTransformer(config, event_vectorizer=event_vectorizer, event_sequence_encoder=event_sequence_encoder)
    if config.logger is not None:
        config.logger.info(f'Using event vectorizer: {config.model.event_vectorizer_type}'
                           f'{type(event_vectorizer).__name__} and '
                           f'event sequence encoder: {type(event_sequence_encoder).__name__} and'
                           f'Trajectory Transformer: {type(model).__name__}')
    return model
