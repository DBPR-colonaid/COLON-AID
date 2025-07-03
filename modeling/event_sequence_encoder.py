import torch
import torch.nn as nn
import pandas as pd
from modeling.constants import ALL_EVENT_TYPES
from modeling.time_embd import RelativeTimeEmbedding
from modeling.join import join


class EventSequenceEncoder(nn.Module):
    """
    Encode a list of events into a event sequence.
    This module adds event type and event date embeddings to form a complete representation of each event as a token.
    "Event Sequence Encoder" reflects the module’s role in encoding and enhancing event vectors with additional contextual embeddings to prepare them for sequential processing.
    """
    def __init__(self, config):
        super(EventSequenceEncoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.use_event_time_embd = config.model.event_time_embd
        self.use_event_type_embd = config.model.event_type_embd
        self.use_event_separator = config.model.event_separator

        # Event type embedding
        self.event_embedding = nn.Embedding(len(ALL_EVENT_TYPES), self.hidden_size)

        # Time encoding
        self.time_encoding = RelativeTimeEmbedding(config)

        # Learnable separator
        if self.use_event_separator:
            self.event_separator = nn.Parameter(torch.zeros(1, self.hidden_size))

        # # Event content embedding
        # self.event_content_embedding = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, trajectory, embeddings, date_origin):
        """
        Forward pass of the EventSequenceEncoder module.
        :param events: list, a list of events, each represented as a dictionary with keys 'event_type', 'event_date', and 'event_content'.
        :return: torch.Tensor, the encoded event sequence.
        """
        assert len(trajectory) == len(embeddings), f"The number of events and embeddings must match. Got {len(trajectory)} events and {len(embeddings)} embeddings."
        # event_embedding_list = []

        content_embeddings = [embedding['cls'] for embedding in embeddings]
        if len(content_embeddings) > 0:
            embeddings = torch.concat(content_embeddings, dim=0)  # (L, d)
        else:
            return torch.zeros(1, self.hidden_size, device=self.event_embedding.weight.device)

        if self.use_event_type_embd:
            type_ids = [ALL_EVENT_TYPES.index(event['event_type']) for event in trajectory]
            type_embeddings = self.event_embedding(torch.tensor(type_ids, device=self.event_embedding.weight.device))  # (L, d)
            embeddings = embeddings + type_embeddings

        if self.use_event_time_embd:
            dates = [event['date'] if event['event_type'] != 'basic info' else date_origin for event in trajectory]
            time_embeddings = self.time_encoding(dates, date_origin)  # (L, d)
            time_embeddings[0] = 0  # zero out the time embedding for the basic info event
            embeddings = embeddings + time_embeddings

        return embeddings


def get_default_event_sequence_encoder(config):
    return EventSequenceEncoder(config)

