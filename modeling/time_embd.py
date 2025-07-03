import torch
import torch.nn as nn
import math
import pandas as pd


def get_scaled_time_since_origin(time, origin, norm_factor=1.):
    """
    Get the time since origin in days, scaled by 1/1000
    :param time: str, time in format 'yyyy-mm-dd'
    :param origin: str, time in format 'yyyy-mm-dd'
    :return: float, time since origin in days, scaled by 1 / norm_factor
    """
    time = pd.to_datetime(time[:10])
    origin = pd.to_datetime(origin)
    return (time - origin).days / norm_factor


class RelativeTimeEmbedding(nn.Module):
    def __init__(self, config):
        super(RelativeTimeEmbedding, self).__init__()
        self.config = config
        self.time_embedding = SinusoidalTimeEmbedding(config.hidden_size)

    def forward(self, dates: str | list, origins: str | list):
        if isinstance(dates, str):
            dates = [dates]
        if isinstance(origins, str):
            origins = [origins]
        times = []
        for date, origin in zip(dates, origins):
            time = get_scaled_time_since_origin(date, origin)
            times.append(time)
        times = torch.tensor(times, dtype=torch.float32)
        return self.time_embedding(times)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(SinusoidalTimeEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.register_buffer('device_holder', torch.empty(0))  # Placeholder for device

        # Precompute the division term that will be used in the forward pass
        self.register_buffer('div_term', torch.exp(torch.arange(0, embedding_dim, 2).float() *
                                                   (-math.log(10000.0) / embedding_dim)))

    def forward(self, times):
        device = self.device_holder.device
        times = times.to(device)  # Ensure the input times tensor is on the correct device
        position = times.unsqueeze(1)
        sinusoidal_embedding = torch.zeros((len(times), self.embedding_dim), device=device)

        sinusoidal_embedding[:, 0::2] = torch.sin(position * self.div_term)
        sinusoidal_embedding[:, 1::2] = torch.cos(position * self.div_term)

        return sinusoidal_embedding


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    device = 'cuda'
    config = nn.Module()
    config.hidden_size = 768
    # Parameters
    embedding_dim = 768  # Dimension 'd' of the embedding

    # Create the embedding module on the specified device
    time_embedding = SinusoidalTimeEmbedding(embedding_dim)
    time_embedding.to(device)

    rte = RelativeTimeEmbedding(config)
    rte.to(device)

    # Example usage
    # times = torch.tensor([0, 100, 1000])  # Move the example time points tensor to the device
    # embedded_times = time_embedding(times)
    dates = ['2021-01-02', '2021-01-02']
    origins = ['2021-01-01', '2020-01-01']
    embedded_times = rte(dates, origins)

    print("Embedded Times:", embedded_times)
    print(embedded_times.shape)
    # plt.plot(embedded_times[0].cpu().numpy(), label='Time 0')
    for i in range(embedded_times.shape[0]):
        plt.plot(embedded_times[i].cpu().numpy(), label=f'Date {dates[i]} - Origin {origins[i]}')
    plt.legend()
    plt.show()
