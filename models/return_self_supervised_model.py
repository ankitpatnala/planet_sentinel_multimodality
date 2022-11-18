import joblib
import optuna
from self_supervised_models.backbones import MLP,ResMLP
#from self_supervised_models.transformer_encoder import TransformerEncoder
from self_supervised_models.temporal_vit_time import TransformerEncoder

from collections import OrderedDict
import re
import torch

def return_self_supervised_model_sentinel2(ckpt_path,pretrain_type='temporal_transformer',config=None,hyper_param_file=None,**kwargs):
    if config is None and hyper_param_file is not None:
        config = {}
        trial = joblib.load(hyper_param_file)
        best_param = trial.best_params 
        for param in best_param.keys():
            config[param] = best_param[param] 
    if pretrain_type == "mlp":
        sentinel_mlp = MLP(12,
                config['num_layers'] if config is not None else 4,
                config['hidden_dim'] if config is not None else 256)
        backbone_model_name = 'backbone_sentinel'
        emb_dim = config['hidden_dim'] if config is not None else 256
    if pretrain_type == "resmlp":
        sentinel_mlp = ResMLP(12,
                config['num_layers'] if config is not None else 4,
                config['hidden_dim'] if config is not None else 256)
        backbone_model_name = 'backbone_sentinel'
        emb_dim = config['hidden_dim'] if config is not None else 256
    if pretrain_type == 'temporal_transformer':
        sentinel_mlp = TransformerEncoder(
                12,
                config['d_model'] if config is not None else 64,
                config['n_head'] if config is not None else 4,
                config['num_layer'] if config is not None else 4,
                config['mlp_dim'] if config is not None else 128,
                config['dropout'] if config is not None else 0.0,
                config['projector_layer'] if config is not None else 2)
        backbone_model_name = 'sentinel_transformer_encoder'
        #emb_dim = config['mlp_dim'] if config is not None else 256
        emb_dim = config['d_model'] if config is not None else 64
    ckpt = torch.load(ckpt_path)
    new_ckpt = OrderedDict()
    for key in ckpt['state_dict'].keys():
        if backbone_model_name in key:
            mlp_key = re.sub(f'{backbone_model_name}.',"",key)
            new_ckpt[mlp_key] = ckpt['state_dict'][key]
    sentinel_mlp.load_state_dict(new_ckpt)
    return sentinel_mlp,emb_dim

def run_time_series_with_mlp(model,x):
    b,t,n = x.shape
    if hasattr(model,'return_embeddings'):
        x =  model.return_embeddings(x)
        return x
    x = model(x.reshape(-1,n)).reshape(b,t,-1)
    return x

