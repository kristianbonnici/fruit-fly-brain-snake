"""Small learned two-image perception encoder, without game geometry or actions."""
import numpy as np
import mlx.core as mx
import mlx.nn as nn

LEVELS = (0., .25, 1/3, .5, 1.)
PARAMETERS = 82561


def values(raw):
    if raw.ndim != 2 or raw.shape[1] != 17:
        raise ValueError('Two food outputs and three five-class obstacle outputs required')
    food = mx.tanh(raw[:, :2])
    obstacles = mx.sum(mx.softmax(raw[:, 2:].reshape(-1, 3, 5), axis=-1) * mx.array(LEVELS, mx.float32), axis=-1)
    return mx.concatenate((food, mx.clip(obstacles, 0., 1.)), axis=1)


class PerceptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(nn.Conv2d(2, 16, 5, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU())
        self.hidden = nn.Linear(512, 128)
        self.output = nn.Linear(128, 17)

    def __call__(self, images):
        if images.ndim != 4 or images.shape[1:] != (16, 16, 2):
            raise ValueError('Only two rendered16x16images per observation required')
        return self.output(nn.relu(self.hidden(self.convs(images).reshape(images.shape[0], -1))))

    def sensors(self, images):
        return values(self(images))
