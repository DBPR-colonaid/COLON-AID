import torch
import torch.nn as nn


def update_dropout(model, new_dropout_rate=0.5):
    """
    Traverse and update dropout layers' dropout rate in a PyTorch model.

    Args:
    model (torch.nn.Module): The model to traverse.
    new_dropout_rate (float): The new dropout rate to set.
    """
    for child_name, child in model.named_children():
        if isinstance(child, nn.Dropout):
            # Update the dropout rate
            child.p = new_dropout_rate
            print(f"Updated {child_name} to dropout rate {new_dropout_rate}")
        elif hasattr(child, 'children') and len(list(child.children())) > 0:
            # Recursive call for nested modules
            update_dropout(child, new_dropout_rate)
