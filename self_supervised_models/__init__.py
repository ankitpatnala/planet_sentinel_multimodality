from self_supervised_models.backbones import MLP,ResMLP
from self_supervised_models.multimodal import Multimodal
from self_supervised_models.temporal_contrastive_transformers import TemporalContrastiveLearning

BACKBONE_POINT = {'mlp':MLP,
           'resmlp':ResMLP}

SELF_SUPERVISED_TYPE = {'point':Multimodal,
                        'time':TemporalContrastiveLearning}












