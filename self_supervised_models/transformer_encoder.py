import torch
from torch import nn

import math
import pickle

from self_supervised_models.backbones import MLP

with open("../utils/h5_folder/time_stamp_sentinel_list.pkl",'rb') as pickle_reader:
    sentinel_days = (pickle.load(pickle_reader))

with open("../utils/h5_folder/time_stamp_planet_list.pkl",'rb') as pickle_reader:
    planet_days = (pickle.load(pickle_reader))*5
    
    
def pos_embedding(input_dim,days):
    doy_array = torch.unsqueeze(torch.arange(0,365),dim=0)
    div_array = torch.pow(10000,torch.arange(input_dim/2)/input_dim)
    pos_array = torch.einsum('ij,k->jk',doy_array,1/div_array)
    sin_cos_pos_array = torch.zeros((365,input_dim))
    sin_cos_pos_array[:,0::2] = torch.sin(pos_array)
    sin_cos_pos_array[:,1::2] = torch.cos(pos_array)
    return sin_cos_pos_array[days,:]


def pos_embedding2(input_dim,days):
    doy_array = torch.unsqueeze(torch.arange(len(days)),dim=0)
    div_array = torch.pow(10000,torch.arange(input_dim/2)/input_dim)
    pos_array = torch.einsum('ij,k->jk',doy_array,1/div_array)
    sin_cos_pos_array = torch.zeros((len(days),input_dim))
    sin_cos_pos_array[:,0::2] = torch.sin(pos_array)
    sin_cos_pos_array[:,1::2] = torch.cos(pos_array)
    return sin_cos_pos_array

# _trunc_normal_ taken from timm library
def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)

class TransformerEncoder(nn.Module):
    def __init__(self,num_inputs,d_model,n_head,num_layer,mlp_dim,dropout,mode_type='sentinel',activation=nn.GELU()):
        super(TransformerEncoder,self).__init__()
        self.patch_embed = nn.Conv1d(num_inputs,d_model,1,1)
        self.class_token = nn.parameter.Parameter(torch.randn(1,1,d_model,requires_grad=True))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model,n_head,dim_feedforward=mlp_dim,dropout=dropout)
        self.embedding = nn.Linear(d_model,mlp_dim)
        self.projector = MLP(mlp_dim,2,mlp_dim,activation=nn.GELU())
        self.pos_embedding = pos_embedding2(d_model,sentinel_days if mode_type=="sentinel" else planet_days).to(torch.device('cuda'))
        trunc_normal_(self.class_token,std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self,m):
        if isinstance(m,nn.Linear):
            trunc_normal_(m.weight,std=0.02)
            if isinstance(m,nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.LayerNorm):
                nn.init.constant(m.bias,0)
                nn.init.constant_(m.weight,1.0)
    def forward(self,x):
        n,t,c = x.shape
        repeat_class_token = torch.repeat_interleave(self.class_token,n,dim=0)
        x = torch.permute(self.patch_embed(torch.permute(x,[0,2,1])),dims=[0,2,1])
        x  = x + self.pos_embedding
        x = torch.cat([repeat_class_token,x],dim=1)
        x = self.encoder_layer(x)
        x_class_token = x[:,0,:]
        x_embedding = self.embedding(x_class_token)
        x_projector = self.projector(x_embedding)
        return x_embedding,x_projector

    def return_embeddings(self,x):
        n,t,c = x.shape
        repeat_class_token = torch.repeat_interleave(self.class_token,n,dim=0)
        x = torch.permute(self.patch_embed(torch.permute(x,[0,2,1])),dims=[0,2,1])
        x  = x + self.pos_embedding
        x = torch.cat([repeat_class_token,x],dim=1)
        x = self.encoder_layer(x)
        x = x[:,1:,:]
        n,t,c = x.shape
        x = self.embedding(x.reshape(-1,c)).reshape(n,t,-1)
        return x 

