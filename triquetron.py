import math

import torch
from torch.nn import init
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from torch import Tensor


class FCLinear(Module):
    '''
    Birth, Death 뉴런 모두 가중치 보유
    '''
    __constants__ = ('in_features', 'out_features')
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(FCLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.BD_weight = Parameter(torch.empty((2*out_features, in_features), **factory_kwargs))
        self.RE_weight = Parameter(torch.empty((out_features, 2), **factory_kwargs))
        if bias:
            self.BD_bias = Parameter(torch.empty(2*out_features, **factory_kwargs)) #2(Birth, Death) x out_feature
            self.RE_bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('BD_bias', None)
            self.register_parameter('RE_bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        def init_bias(weight: Parameter, bias: Parameter) -> None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(bias, -bound, bound)

        init.kaiming_uniform_(self.BD_weight, a=math.sqrt(5))
        if self.BD_bias is not None:
            init_bias(self.BD_weight, self.BD_bias)
            init_bias(self.RE_weight, self.RE_bias)

    def forward(self, input: Tensor) -> Tensor:
        out = F.linear(input, self.BD_weight, self.BD_bias) #Batch x 2(DB) * out_feature

        out = out.view(-1, self.out_features, 2) #Batch x out_feature x 2(DB)
        o1 = F.leaky_relu(out, negative_slope=0.01)

        out = torch.mul(o1, self.RE_weight).sum(dim=2) #Batch x out_feature
        o2 = torch.add(out, self.RE_bias) #Batch x out_feature
        return o2
 
class OneWeightLinear(Module):
    '''
    + Birth, Death는 bias만 소유
    + 입력은 하나의 가중치를 적용한 후 Birth와 Death 뉴런에 각각 입력
    '''
    __constants__ = ('in_features', 'out_features')
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(OneWeightLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.BD_weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.RE_weight = Parameter(torch.empty((out_features, 2), **factory_kwargs))
        if bias:
            self.BD_bias = Parameter(torch.empty((2, 1, out_features), **factory_kwargs)) #2(Birth, Death) x Batch x out_feature
            self.RE_bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('BD_bias', None)
            self.register_parameter('RE_bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        def init_bias(weight: Parameter, bias: Parameter) -> None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(bias, -bound, bound)

        init.kaiming_uniform_(self.BD_weight, a=math.sqrt(5))
        if self.BD_bias is not None:
            init_bias(self.BD_weight, self.BD_bias)
            init_bias(self.RE_weight, self.RE_bias)

    def forward(self, input: Tensor) -> Tensor:
        out = F.linear(input, self.BD_weight) #Batch x out_feature
        out = torch.add(out, self.BD_bias) #2(BD) x Batch x out_feature
        out = out.permute(1, 2, 0) #Batch x out_feature x 2(BD)

        out = out.view(-1, self.out_features, 2) #Batch x out_feature x 2(DB)
        o1 = F.leaky_relu(out, negative_slope=0.01)

        out = torch.mul(o1, self.RE_weight).sum(dim=2) #Batch x out_feature
        o2 = torch.add(out, self.RE_bias) #Batch x out_feature
        return o2   