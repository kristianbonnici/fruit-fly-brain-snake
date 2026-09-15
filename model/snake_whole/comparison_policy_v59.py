"""Small autodiff comparison models; inference has no teacher or game-state input."""
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


class ImageModel(nn.Module):
    """Conventional diagnostic only: two rendered images to three relative actions."""
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(nn.Conv2d(2, 16, 5, padding=2), nn.ReLU(),
                                  nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU())
        self.hidden = nn.Linear(512, 128)
        self.output = nn.Linear(128, 3)

    def __call__(self, images):
        if images.ndim != 4 or images.shape[1:] != (16, 16, 2):
            raise ValueError('Only a batch of two rendered standard-board images is accepted')
        x = self.convs(images).reshape(images.shape[0], -1)
        return self.output(nn.relu(self.hidden(x)))


class NeuralDecoder(nn.Module):
    """Small nonlinear decoder from the same 2129 declared output-neuron rates."""
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2129, 64)
        self.output = nn.Linear(64, 3)

    def __call__(self, features):
        if features.ndim != 2 or features.shape[1] != 2129:
            raise ValueError('Only the declared neural output population is accepted')
        return self.output(nn.relu(self.hidden(features)))


def make_model(kind):
    if kind == 'pixels':
        return ImageModel()
    if kind == 'neural':
        return NeuralDecoder()
    raise ValueError('Explicit conventional-image or neural-decoder comparison required')


def parameter_count(model):
    return sum(value.size for _, value in tree_flatten(model.parameters()))


class ImagePolicy:
    """Teacher-free inference state consists only of the previous actual image."""
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.previous = np.zeros((16, 16), np.float32)
        self.observations = 0

    def choose(self, image):
        image = np.asarray(image, np.float32)
        if image.shape != (16, 16) or not np.isfinite(image).all() or np.any((image < 0) | (image > 1)):
            raise ValueError('A finite rendered standard-board image is required')
        pair = np.stack((self.previous, image), axis=-1)[None]
        logits = np.asarray(self.model(mx.array(pair)), np.float64)[0]
        if not np.isfinite(logits).all():
            raise FloatingPointError('Nonfinite image-policy logits')
        self.previous = image.copy()
        self.observations += 1
        return int(logits.argmax()), logits

    def snapshot(self):
        return dict(previous=self.previous.copy(), observations=self.observations)

    def restore(self, state):
        previous = np.asarray(state['previous'])
        if (set(state) != {'previous', 'observations'} or previous.shape != (16, 16)
                or previous.dtype != np.float32 or not np.isfinite(previous).all()
                or type(state['observations']) is not int or state['observations'] < 0):
            raise ValueError('Complete finite image-history state required')
        self.previous = previous.copy()
        self.observations = state['observations']
