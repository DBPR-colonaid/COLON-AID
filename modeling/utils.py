import torch
from torch.nn.utils.rnn import pad_sequence


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    else:
        return obj


def pad_and_truncate(sequences, max_len, padding_value=0., batch_first=True):
    for seq in sequences:
        if len(seq) > max_len:
            print(f"Sequence length {len(seq)} is greater than max_len {max_len}")
    # Truncate and pad sequences
    truncated_sequences = [seq[:max_len] for seq in sequences]
    padded_sequences = pad_sequence(truncated_sequences, batch_first=batch_first, padding_value=padding_value)  # (B, max_len, d)
    return padded_sequences


def summarize_structure(data):
    if hasattr(data, 'shape'):
        # Replace object with its type and shape description
        return f'{type(data).__name__}: {tuple(data.shape)}'
    elif isinstance(data, dict):
        # Recursively apply to dictionary values
        return {key: summarize_structure(value) for key, value in data.items()}
    elif isinstance(data, list):
        # Recursively apply to each item in the list
        return [summarize_structure(item) for item in data]
    elif isinstance(data, tuple):
        # Recursively apply to each item in the tuple, convert result to tuple
        return tuple(summarize_structure(item) for item in data)
    else:
        # Return the data unchanged if it doesn't meet any of the above conditions
        return data


def hazards_to_risk(hazards):
    S = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(S, dim=1)
    return risk


def logits_to_risk(logits, shift=12):
    hazards = torch.sigmoid(logits)
    return hazards_to_risk(hazards) + shift

