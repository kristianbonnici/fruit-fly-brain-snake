"""Global image encoder plus learned full-resolution local obstacle residual."""
import mlx.core as mx
import mlx.nn as nn
from .perception_model_v80 import PerceptionModel as GlobalPerceptionModel

PARAMETERS=99600
LOCAL_PARAMETERS=17039


def attention_pool(features,logits):
    if features.ndim!=4 or logits.shape!=(*features.shape[:3],1):
        raise ValueError('Aligned image features and one attention logit per location required')
    flat=features.reshape(features.shape[0],-1,features.shape[-1])
    weights=mx.softmax(logits.reshape(features.shape[0],-1),axis=1)
    return mx.sum(flat*weights[...,None],axis=1),weights


class PerceptionModel(GlobalPerceptionModel):
    def __init__(self):
        super().__init__()
        self.local_convs=nn.Sequential(nn.Conv2d(2,16,3,padding=1),nn.ReLU(),
            nn.Conv2d(16,16,3,padding=1),nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),nn.ReLU())
        self.local_attention=nn.Linear(32,1,bias=False)
        self.local_output=nn.Linear(32,15)
        self.local_output.weight=mx.zeros_like(self.local_output.weight)
        self.local_output.bias=mx.zeros_like(self.local_output.bias)

    def __call__(self,images):
        global_raw=super().__call__(images)
        features=self.local_convs(images)
        pooled,_=attention_pool(features,self.local_attention(features))
        residual=self.local_output(pooled)
        return mx.concatenate((global_raw[:,:2],global_raw[:,2:]+residual),axis=1)
