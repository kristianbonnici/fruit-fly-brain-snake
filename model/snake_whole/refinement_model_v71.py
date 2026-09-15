"""Fixed trained interfaces around the unchanged V69 anatomical dynamics."""
import mlx.core as mx
from .common import ROOT,digest,stable
from .comparison_archive_v59 import read,install,model_arrays,array_hash
from .structured_model_v69 import Model,Policy,load_anatomy
from .structured_contract_v69 import precision

BASE=ROOT/'data/snake-whole-v71'
WARM=ROOT/'data/snake-whole-v69/runs/frozen-91001/checkpoint-10.npz'
WARM_SHA='a56040aec8b503e23a18f0abc2fe959ffdb36e7828f056434f7c1231a3a2118f'


def interfaces(model):
    return stable({k:array_hash(v) for k,v in model_arrays(model).items() if k.startswith(('model.encoder.','model.decoder.'))})


def make_model(seed):
    precision();mx.random.seed(seed);model=Model(load_anatomy()[0],train_internal=True)
    model.encoder.freeze();model.decoder.freeze();mx.eval(model.parameters())
    if model.capacity()['trainable']!=23193:raise ValueError('Only distributed internal efficacies may learn')
    return model


def warm_model(seed):
    if seed!=91001 or digest(WARM)!=WARM_SHA:raise ValueError('Exact useful frozen-network initialization required')
    arrays,meta=read(WARM)
    if meta['arm']!='frozen' or meta['completed_epoch']!=10 or meta['seed']!=seed:raise ValueError('Preserve strong baseline')
    for path,sha in meta['configuration']['sources'].items():
        if digest(ROOT/path)!=sha:raise ValueError('Warm-start source changed: '+path)
    model=make_model(seed);install(model,arrays)
    if not bool(mx.all(model.core.log_e==0).item()):raise ValueError('Warm start must retain E1')
    return model
