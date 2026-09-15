"""Condition the declared output population without changing neural transmission."""
import mlx.nn as nn

from .reduced_model_v63 import ReducedModel as PreviousModel, load_anatomy

HEAD_SPEC = dict(normalization='LayerNorm over64declared outputs, no affine parameters',
    epsilon=1e-5, hidden=32, activation='LeakyReLU', negative_slope=.1)


class ReducedModel(PreviousModel):
    def __init__(self, arrays, train_internal=True):
        super().__init__(arrays,train_internal)
        # Reuse the exact seeded linear parameters; no extra random draws or
        # trainable parameters. Only decoder computation is a new version.
        first,last=self.decoder.layers[0],self.decoder.layers[2]
        self.decoder=nn.Sequential(nn.LayerNorm(64,eps=HEAD_SPEC['epsilon'],affine=False),
            first,nn.LeakyReLU(HEAD_SPEC['negative_slope']),last)
        self._head_spec=dict(HEAD_SPEC)
