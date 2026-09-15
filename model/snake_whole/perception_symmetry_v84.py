"""Fixed all-eight-view image perception; no training or geometry observations."""
import json
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from .common import ROOT,digest,stable
from .comparison_archive_v59 import read,install
from .comparison_contract_v59 import environment
from .gru_contract_v67 import precision
from .perception_model_v83 import PerceptionModel,PARAMETERS
from .perception_model_v80 import values

CHECKPOINTS={0:'2c1443115b99d52cb930d78fe54a33be192d561beba82897b9473cc9c36d586b',60:'af398097ec7256cad3b01afb919c3cb1a974e1d364e035adf6e6559b51dd707c'}


def transform(images,code):
    images=np.asarray(images,np.float32)
    if (images.ndim!=4 or images.shape[1:]!=(16,16,2) or not np.isfinite(images).all()
            or type(code) is not int or code not in range(8)):
        raise ValueError('Finite paired images and a fixed D4 code required')
    result=np.rot90(images,code%4,axes=(1,2))
    if code>=4: result=result[:,:,::-1,:]
    return np.ascontiguousarray(result)


def pool(raw_views):
    if raw_views.ndim!=3 or raw_views.shape[0]!=8 or raw_views.shape[2]!=17:
        raise ValueError('All eight ordered17-output views required')
    mapped=[]
    for code in range(8):
        raw=raw_views[code]
        if code>=4:
            raw=mx.concatenate((raw[:,:1],-raw[:,1:2],raw[:,12:17],raw[:,7:12],raw[:,2:7]),axis=1)
        mapped.append(raw)
    return mx.mean(mx.stack(mapped,axis=0),axis=0)


class SymmetricPerception:
    def __init__(self,encoder):
        if type(encoder) is not PerceptionModel: raise ValueError('Exact frozen local/global encoder required')
        self.encoder=encoder; self.encoder.eval()

    def predict(self,images):
        views=mx.stack([self.encoder(mx.array(transform(images,c))) for c in range(8)],axis=0)
        raw=pool(views); mx.eval(views,raw)
        return raw,views

    def __call__(self,images): return self.predict(images)[0]

    def sensors(self,images): return values(self(images))


def load_encoder(epoch):
    precision()
    if type(epoch) is not int or epoch not in (0,60): raise ValueError('Only frozen V83initial/final checkpoints required')
    directory=ROOT/'data/snake-whole-v83/runs/perception-91001'; path=directory/f'checkpoint-{epoch:02d}.npz'
    report=json.loads((directory.parent.parent/'main-accounting.json').read_text())
    expected=report['checkpoint_chain'][epoch]['sha256']
    if digest(path)!=expected or expected!=CHECKPOINTS[epoch]: raise ValueError('Exact frozen V83checkpoint required')
    arrays,meta=read(path); cfg=meta['configuration']
    if (meta['format']!='perception-epoch-v80' or meta['binding']!=stable(cfg) or meta['epoch']!=epoch
            or cfg.get('model_version')!=83 or cfg.get('parameters')!=PARAMETERS or cfg.get('seed')!=91001
            or cfg['runtime']!=environment() or cfg['device']!=str(mx.default_device()) or cfg['precision']!=precision()):
        raise ValueError('Exact V83source model, runtime and precision required')
    for p,sha in cfg['sources'].items():
        if digest(ROOT/p)!=sha: raise ValueError('Frozen encoder source changed: '+p)
    mx.random.seed(91001); model=PerceptionModel(); install(model,arrays)
    if sum(v.size for _,v in tree_flatten(model.parameters()))!=PARAMETERS: raise ValueError('Exact encoder capacity required')
    return model,meta
