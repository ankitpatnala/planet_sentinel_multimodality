from self_supervised_models.backbones import MLP,ResMLP
from self_supervised_models.multimodal import Multimodal
from self_supervised_models.sentinel2_self_supervised import Sentinel2Scarf
from self_supervised_models.temporal_contrastive_transformers import TemporalContrastiveLearning
from self_supervised_models.bert_style_temporal_transformer import BertStyleTemporalTransformer

BACKBONE_POINT = {'mlp':MLP,
           'resmlp':ResMLP}

SELF_SUPERVISED_TYPE = {'sentinel2': Sentinel2Scarf,
                        'point':Multimodal,
                        'time':TemporalContrastiveLearning,
                        'bert':BertStyleTemporalTransformer}












